"""
=============================================================================
RAG (Retrieval-Augmented Generation) 知识库模块
=============================================================================
功能：
  1. 文档加载与分块  - 支持 .txt / .md / .pdf 文件
  2. 向量嵌入         - 优先使用智谱 Embedding API，备选 ChromaDB 默认
  3. 向量存储与检索   - 基于 ChromaDB 持久化
  4. 增强生成         - 检索相关文档 + LLM 生成回答（含引用）

依赖: pip install chromadb sentence-transformers
=============================================================================
"""

import os
import re
import json
import time
import hashlib
from typing import List, Dict, Optional, Tuple, Any

import chromadb
from chromadb.utils import embedding_functions

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================
# 路径常量
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
KNOWLEDGE_DIR = os.path.join(ROOT_DIR, "knowledge_base")
CHROMA_PERSIST_DIR = os.path.join(ROOT_DIR, "chroma_db")

# ============================
# 配置
# ============================
CHUNK_SIZE = 500          # 每个文本块的最大字符数
CHUNK_OVERLAP = 50        # 相邻块的重叠字符数
DEFAULT_TOP_K = 5         # 默认检索返回条数

# ============================
# LLM 客户端（与 agent.py 共用配置）
# ============================
_llm_client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 智谱 Embedding 模型名
ZHIPU_EMBEDDING_MODEL = "embedding-2"

# ChromaDB collection 名称
COLLECTION_NAME = "knowledge_base"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     1. 嵌入函数（Embedding Function）                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ZhipuEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    基于智谱 API 的自定义嵌入函数

    调用智谱 Embedding API（embedding-2 模型）将文本转为向量。
    如果 API 不可用，自动降级为 ChromaDB 的默认本地嵌入。
    """

    def __init__(self, api_key: Optional[str] = None, batch_size: int = 16):
        self.api_key = api_key or os.getenv("API_KEY")
        self.batch_size = batch_size
        self._fallback = None
        self._available = bool(self.api_key)

    def name(self) -> str:
        """ChromaDB 要求的接口，返回嵌入函数标识"""
        return "zhipu-embedding-2"

    def _get_fallback(self):
        """延迟初始化备选嵌入函数"""
        if self._fallback is None:
            try:
                self._fallback = embedding_functions.DefaultEmbeddingFunction()
            except Exception:
                self._fallback = None
        return self._fallback

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        调用智谱 Embedding API 获取向量

        文档: https://open.bigmodel.cn/dev/api/normal-model/embedding
        """
        import requests

        url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        all_embeddings = []
        # 分批请求，每批最多 batch_size 条
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            payload = {
                "model": ZHIPU_EMBEDDING_MODEL,
                "input": batch,
            }
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    # 按 index 排序后取出 embedding
                    items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                    all_embeddings.extend([item["embedding"] for item in items])
                else:
                    raise RuntimeError(f"Embedding API 返回 {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"[RAG] 智谱 Embedding 失败: {e}")
                raise
        return all_embeddings

    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB 要求的调用接口"""
        if not input:
            return []

        # 尝试使用智谱 API
        if self._available:
            try:
                return self._embed_batch(input)
            except Exception as e:
                print(f"[RAG] 智谱 Embedding 不可用，降级为本地模型: {e}")
                self._available = False  # 后续不再重试

        # 降级：使用 ChromaDB 默认嵌入（sentence-transformers）
        fallback = self._get_fallback()
        if fallback:
            return fallback(input)

        # 极端情况：无任何嵌入可用，返回零向量（保证不崩溃）
        print("[RAG] ⚠ 无可用嵌入模型，使用零向量占位")
        return [[0.0] * 384 for _ in input]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     2. 文本分块（Text Splitter）                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    将长文本切分为有重叠的块。

    切分策略：
      1. 优先按段落（双换行）切分
      2. 段落过长时按句子（。！？\n）切分
      3. 句子仍过长时强制按字符截断

    参数:
        text: 原始文本
        chunk_size: 每个块最大字符数
        chunk_overlap: 相邻块重叠字符数

    返回:
        [{"text": str, "index": int, "char_start": int, "char_end": int}, ...]
    """
    # 第一步：按段落切分
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # 第二步：对超长段落按句子切分
    segments = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            segments.append(para)
        else:
            # 按句子分隔符切分
            sentences = re.split(r"(?<=[。！？；\n])", para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= chunk_size:
                    current += sent
                else:
                    if current.strip():
                        segments.append(current.strip())
                    current = sent
            if current.strip():
                segments.append(current.strip())

    # 第三步：生成重叠块
    chunks = []
    current_chunk = ""
    current_start = 0
    char_cursor = 0
    index = 0

    for seg in segments:
        seg_len = len(seg)
        if len(current_chunk) + seg_len <= chunk_size:
            current_chunk += seg
        else:
            # 保存当前块
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "index": index,
                    "char_start": current_start,
                    "char_end": current_start + len(current_chunk),
                })
                index += 1
                # 重叠：保留尾部 overlap 字符
                overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 and len(current_chunk) > chunk_overlap else ""
                current_chunk = overlap_text + seg
                current_start = current_start + len(current_chunk) - len(overlap_text) - seg_len
            else:
                current_chunk = seg

    # 保存最后一块
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "index": index,
            "char_start": current_start,
            "char_end": current_start + len(current_chunk),
        })

    return chunks


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     3. 文档加载器（Document Loader）                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_documents(directory: str = KNOWLEDGE_DIR) -> List[Dict[str, Any]]:
    """
    加载指定目录下的所有文档，按文件切分后返回。

    支持格式: .txt, .md, .py（纯文本）; .pdf（需要 PyPDF2）

    返回:
        [{"text": str, "metadata": {"source": str, "chunk_index": int}}, ...]
    """
    all_chunks = []

    if not os.path.isdir(directory):
        print(f"[RAG] 知识库目录不存在: {directory}")
        return all_chunks

    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)

        # 跳过隐藏文件和非文件
        if filename.startswith(".") or not os.path.isfile(filepath):
            continue

        ext = os.path.splitext(filename)[1].lower()
        content = None

        try:
            if ext in (".txt", ".md", ".py", ".yaml", ".yml", ".json", ".xml", ".html"):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

            elif ext == ".pdf":
                # 尝试导入 PyPDF2
                try:
                    from PyPDF2 import PdfReader
                except ImportError:
                    print(f"[RAG] ⚠ 跳过 PDF 文件（需安装 PyPDF2）: {filename}")
                    continue
                reader = PdfReader(filepath)
                pages = []
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
                content = "\n\n".join(pages)

            else:
                # 尝试当纯文本读取
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    print(f"[RAG] ⚠ 跳过不支持的文件格式: {filename}")
                    continue

        except Exception as e:
            print(f"[RAG] ⚠ 读取文件失败 {filename}: {e}")
            continue

        if not content or not content.strip():
            continue

        # 分块
        chunks = split_text(content)
        for ch in chunks:
            all_chunks.append({
                "text": ch["text"],
                "metadata": {
                    "source": filename,
                    "chunk_index": ch["index"],
                    "char_start": ch["char_start"],
                    "char_end": ch["char_end"],
                },
            })

        print(f"[RAG] 已加载: {filename} → {len(chunks)} 块")

    return all_chunks


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     4. RAG 知识库核心类                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class RAGKnowledgeBase:
    """
    RAG 知识库管理类

    用法:
        kb = RAGKnowledgeBase()
        kb.rebuild()                         # 首次运行：加载文档并建库
        results = kb.search("安全规范")       # 语义检索
        answer = kb.query("如何应对火灾?")     # RAG 问答
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        os.makedirs(persist_dir, exist_ok=True)

        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # 初始化 ChromaDB 持久化客户端
        self._client = chromadb.PersistentClient(path=persist_dir)

        # 嵌入函数：优先智谱 API，备选本地
        self._embed_fn = ZhipuEmbeddingFunction()

        # 获取或创建集合（兼容旧版本嵌入函数冲突）
        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embed_fn,
                metadata={"description": "边云协同系统知识库"},
            )
        except (AttributeError, ValueError):
            # 嵌入函数不兼容（如旧集合用了不同嵌入），删除后重建
            try:
                self._client.delete_collection(name=collection_name)
            except Exception:
                pass
            self._collection = self._client.create_collection(
                name=collection_name,
                embedding_function=self._embed_fn,
                metadata={"description": "边云协同系统知识库"},
            )

    # ── 增 ────────────────────────────────────────────────────────────

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        将文档块添加到向量数据库（自动去重：按文本 hash）

        参数:
            chunks: load_documents() 的返回值

        返回:
            新增的块数量
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for ch in chunks:
            text = ch["text"]
            doc_id = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
            ids.append(doc_id)
            documents.append(text)
            metadatas.append(ch.get("metadata", {}))

        # ChromaDB upsert：相同 ID 会覆盖，实现去重
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"[RAG] 已索引 {len(chunks)} 个文档块")
        return len(chunks)

    def add_text(self, text: str, source: str = "manual") -> str:
        """
        添加单条文本到知识库

        参数:
            text: 文本内容
            source: 来源标识

        返回:
            文档 ID
        """
        doc_id = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[{"source": source, "chunk_index": 0}],
        )
        return doc_id

    # ── 删 ────────────────────────────────────────────────────────────

    def delete_by_source(self, source: str) -> int:
        """按来源文件名删除文档块"""
        results = self._collection.get(where={"source": source})
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        print(f"[RAG] 已删除来源 '{source}' 的 {len(ids)} 个块")
        return len(ids)

    def clear(self):
        """清空知识库"""
        count = self._collection.count()
        if count > 0:
            all_ids = self._collection.get()["ids"]
            self._collection.delete(ids=all_ids)
        print(f"[RAG] 已清空知识库（原 {count} 块）")

    # ── 查 ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义检索：返回与查询最相关的文档块

        参数:
            query: 查询文本
            top_k: 返回数量
            where: ChromaDB 元数据过滤条件，如 {"source": "安全规范.txt"}

        返回:
            [{"text": str, "score": float, "source": str, "chunk_index": int}, ...]
        """
        if self._collection.count() == 0:
            return []

        query_kwargs = {
            "query_texts": [query],
            "n_results": top_k,
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        items = []
        if results.get("ids") and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                items.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results.get("documents") else "",
                    "score": round(1.0 - results["distances"][0][i], 4) if results.get("distances") else 0.0,
                    "source": results["metadatas"][0][i].get("source", "") if results.get("metadatas") else "",
                    "chunk_index": results["metadatas"][0][i].get("chunk_index", 0) if results.get("metadatas") else 0,
                })

        return items

    def generate(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        model: str = "glm-4-flash",
    ) -> Dict[str, Any]:
        """
        RAG 增强生成：检索相关文档，拼接为上下文，交给 LLM 生成回答。

        参数:
            question: 用户问题
            top_k: 检索数量
            model: 生成模型名（纯文本模型即可）

        返回:
            {
                "answer": str,          # LLM 生成的回答
                "sources": [...]        # 引用的文档块
            }
        """
        # 1. 检索
        retrieved = self.search(question, top_k=top_k)
        if not retrieved:
            return {
                "answer": "知识库中暂无相关信息，无法回答该问题。",
                "sources": [],
            }

        # 2. 构建上下文
        context_parts = []
        for i, item in enumerate(retrieved, 1):
            context_parts.append(
                f"[文档{i}] 来源: {item['source']}\n{item['text']}"
            )
        context = "\n\n".join(context_parts)

        # 3. 调用 LLM 生成回答
        system_prompt = (
            "你是边云协同智能检测系统的知识库助手。"
            "请严格基于以下【参考资料】回答用户问题。\n"
            "要求：\n"
            "  - 如果资料中有明确答案，直接引用并注明 [文档N]\n"
            "  - 如果资料不足以回答，请明确说明'知识库中暂无相关信息'\n"
            "  - 不要编造资料中没有的内容\n"
            "  - 中文回答，简洁清晰"
        )

        try:
            response = _llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"【参考资料】\n{context}\n\n【用户问题】\n{question}"},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"RAG 生成失败: {e}"

        return {
            "answer": answer,
            "sources": [
                {"source": r["source"], "text_preview": r["text"][:200]}
                for r in retrieved
            ],
        }

    # ── 维护 ──────────────────────────────────────────────────────────

    def rebuild(self, directory: str = KNOWLEDGE_DIR) -> int:
        """一键重建知识库：清空 → 加载 → 索引"""
        self.clear()
        chunks = load_documents(directory)
        if chunks:
            self.add_documents(chunks)
        return len(chunks)

    def stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        count = self._collection.count()
        sources = set()
        if count > 0:
            all_meta = self._collection.get()["metadatas"]
            for m in all_meta:
                if m and m.get("source"):
                    sources.add(m["source"])
        return {
            "collection": self.collection_name,
            "total_chunks": count,
            "sources": sorted(sources),
            "persist_dir": self.persist_dir,
        }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     5. 便捷函数（供外部调用）                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# 全局单例（懒加载）
_kb_instance: Optional[RAGKnowledgeBase] = None


def _get_kb() -> RAGKnowledgeBase:
    """获取全局知识库实例（单例）"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = RAGKnowledgeBase()
    return _kb_instance


def query_knowledge_base(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """
    对外暴露的 RAG 查询接口（供 agent.py 或 API 调用）

    用法:
        from cloud.rag import query_knowledge_base
        result = query_knowledge_base("如何进行安全巡检?")
        print(result["answer"])
    """
    kb = _get_kb()
    return kb.generate(question, top_k=top_k)


def search_knowledge_base(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    对外暴露的语义检索接口

    用法:
        from cloud.rag import search_knowledge_base
        docs = search_knowledge_base("消防规范")
        for d in docs:
            print(d["text"][:100])
    """
    kb = _get_kb()
    return kb.search(query, top_k=top_k)


def rebuild_knowledge_base(directory: str = KNOWLEDGE_DIR) -> int:
    """重建知识库（加载目录下所有文档）"""
    kb = _get_kb()
    return kb.rebuild(directory)


def get_knowledge_base_stats() -> Dict[str, Any]:
    """获取知识库统计"""
    kb = _get_kb()
    return kb.stats()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     6. 命令行入口                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  RAG 知识库模块 — 命令行工具")
    print("=" * 60)

    kb_instance = RAGKnowledgeBase()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
    else:
        cmd = ""

    if cmd == "rebuild":
        # 重建知识库
        directory = sys.argv[2] if len(sys.argv) > 2 else KNOWLEDGE_DIR
        print(f"\n📂 从目录加载文档: {directory}")
        count = kb_instance.rebuild(directory)
        print(f"\n✅ 知识库重建完成，共 {count} 个文档块")

    elif cmd == "search":
        # 语义检索
        if len(sys.argv) < 3:
            print("用法: python rag.py search <查询文本> [top_k]")
            sys.exit(1)
        query = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_TOP_K
        print(f"\n🔍 检索: {query}")
        results = kb_instance.search(query, top_k=top_k)
        if results:
            for i, r in enumerate(results, 1):
                print(f"\n── 结果 {i} (相关度: {r['score']}, 来源: {r['source']}) ──")
                print(r["text"][:300])
        else:
            print("⚠ 未找到相关文档（知识库可能为空，请先执行 rebuild）")

    elif cmd == "query":
        # RAG 问答
        if len(sys.argv) < 3:
            print("用法: python rag.py query <问题>")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        print(f"\n❓ 问题: {question}")
        result = kb_instance.generate(question)
        print(f"\n📝 回答:\n{result['answer']}")
        if result["sources"]:
            print(f"\n📚 参考来源: {[s['source'] for s in result['sources']]}")

    elif cmd == "stats":
        # 统计信息
        stats = kb_instance.stats()
        print(f"\n📊 知识库统计")
        print(f"   集合名:   {stats['collection']}")
        print(f"   文档块数: {stats['total_chunks']}")
        print(f"   来源文件: {stats['sources']}")
        print(f"   持久目录: {stats['persist_dir']}")

    elif cmd == "clear":
        # 清空
        kb_instance.clear()
        print("\n✅ 知识库已清空")

    else:
        print(f"""
用法:
  python rag.py rebuild [目录]    从目录加载文档并重建知识库
  python rag.py search <查询> [k]  语义检索（默认 top 5）
  python rag.py query  <问题>      RAG 增强问答
  python rag.py stats             查看知识库统计
  python rag.py clear             清空知识库

示例:
  python rag.py rebuild
  python rag.py search "安全巡检规范"
  python rag.py query "发现可疑人员应该怎么办?"
""")
