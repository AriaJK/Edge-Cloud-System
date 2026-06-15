"""
=============================================================================
联网搜索工具模块 (Search Tool)
=============================================================================
功能：
  1. ZhipuAI 内置 web_search 工具（优先，通过 glm-4-flash + tools 参数）
  2. DuckDuckGo 搜索（备选，无需 API Key）
  3. 统一搜索接口 search_web()

依赖: pip install duckduckgo-search  (备选方案)
=============================================================================
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv()

# ============================
# 客户端
# ============================

_client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# ============================
# 数据结构
# ============================


@dataclass
class SearchResult:
    """单条搜索结果"""
    title: str
    url: str
    snippet: str = ""


@dataclass
class SearchResponse:
    """搜索返回"""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    summary: str = ""       # LLM 对搜索结果的总结
    source: str = ""        # "zhipu" / "duckduckgo" / "fallback"
    error: str = ""


# ============================
# 方式一：ZhipuAI 内置 web_search
# ============================

_WEB_SEARCH_TOOL = {
    "type": "web_search",
    "web_search": {
        "enable": True,
    },
}


def search_via_zhipu(query: str) -> SearchResponse:
    """
    使用 ZhipuAI 内置 web_search 工具搜索。

    调用 glm-4-flash 模型，传入 web_search tool。
    模型会自主决定是否搜索、搜索什么关键词，并返回带来源链接的回答。
    """
    try:
        response = _client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个信息检索助手。请根据用户问题，使用联网搜索获取最新信息。"
                        "回答时请引用信息来源，分点说明。使用中文。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"请搜索以下问题的相关信息，并给出详细回答：\n{query}",
                },
            ],
            tools=[_WEB_SEARCH_TOOL],
            temperature=0.3,
            max_tokens=2048,
        )

        choice = response.choices[0]
        message = choice.message

        # 情况1: 模型直接返回了答案（搜索已自动完成，结果嵌入 content）
        if message.content:
            return SearchResponse(
                query=query,
                summary=message.content.strip(),
                source="zhipu",
            )

        # 情况2: 模型返回了 tool_calls（需要手动处理）
        if message.tool_calls:
            # ZhipuAI 的 web_search 通常是服务端自动处理，
            # 但以防万一需要二次调用
            tool_results = []
            for tc in message.tool_calls:
                if tc.function.name == "web_search":
                    # 解析搜索结果
                    try:
                        result_data = json.loads(tc.function.arguments)
                        tool_results.append(result_data)
                    except json.JSONDecodeError:
                        pass

            # 用搜索结果再次调用模型生成总结
            if tool_results:
                follow_up = _client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[
                        {
                            "role": "system",
                            "content": "请根据搜索结果，用中文给出简洁准确的回答。",
                        },
                        {
                            "role": "user",
                            "content": f"问题：{query}\n\n搜索结果：{json.dumps(tool_results, ensure_ascii=False)}",
                        },
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )
                return SearchResponse(
                    query=query,
                    summary=follow_up.choices[0].message.content.strip(),
                    source="zhipu",
                )

        # 兜底
        return SearchResponse(
            query=query,
            summary=message.content or "搜索未返回有效结果",
            source="zhipu",
        )

    except Exception as e:
        return SearchResponse(
            query=query,
            error=str(e),
            source="zhipu",
        )


# ============================
# 方式二：DuckDuckGo 搜索（备选，免费无需 API Key）
# ============================

def search_via_duckduckgo(query: str, max_results: int = 5) -> SearchResponse:
    """
    使用 DuckDuckGo 搜索（备选方案，无需 API Key）
    需要: pip install duckduckgo-search
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                ))

        if not results:
            return SearchResponse(
                query=query,
                error="DuckDuckGo 未返回结果",
                source="duckduckgo",
            )

        # 将搜索结果拼接为上下文
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[{i}] {r.title}\n{r.snippet}\n来源: {r.url}")

        context = "\n\n".join(context_parts)

        # 用 LLM 总结搜索结果
        try:
            summary_resp = _client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {
                        "role": "system",
                        "content": "请基于以下搜索结果，用中文给出简洁准确的回答，注明信息来源编号。",
                    },
                    {
                        "role": "user",
                        "content": f"问题：{query}\n\n搜索结果：\n{context}",
                    },
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            summary = summary_resp.choices[0].message.content.strip()
        except Exception:
            summary = f"搜索到 {len(results)} 条结果，但 LLM 总结失败。"
            for r in results:
                summary += f"\n- {r.title}: {r.snippet[:100]}..."

        return SearchResponse(
            query=query,
            results=results,
            summary=summary,
            source="duckduckgo",
        )

    except ImportError:
        return SearchResponse(
            query=query,
            error="DuckDuckGo 搜索不可用（需安装 duckduckgo-search 包）",
            source="duckduckgo",
        )
    except Exception as e:
        return SearchResponse(
            query=query,
            error=str(e),
            source="duckduckgo",
        )


# ============================
# 统一搜索接口
# ============================

def search_web(
    query: str,
    max_results: int = 5,
    prefer: str = "zhipu",
) -> SearchResponse:
    """
    统一联网搜索接口（供 agent.py 调用）

    策略：优先 ZhipuAI 内置搜索，失败时自动降级到 DuckDuckGo

    参数:
        query: 搜索查询词
        max_results: 最大结果数（仅 DuckDuckGo 方式有效）
        prefer: 首选搜索方式 "zhipu" / "duckduckgo"

    返回:
        SearchResponse: 包含搜索结果和 LLM 总结
    """
    if prefer == "duckduckgo":
        return search_via_duckduckgo(query, max_results)

    # 优先 ZhipuAI
    result = search_via_zhipu(query)
    if result.error:
        # 降级
        fallback = search_via_duckduckgo(query, max_results)
        if not fallback.error:
            return fallback
    return result


# ============================
# 命令行测试
# ============================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python search_tool.py <查询关键词>")
        print("示例: python search_tool.py \"今天天气怎么样\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"🔍 搜索: {query}\n")

    # 先尝试 ZhipuAI
    print("--- ZhipuAI 内置搜索 ---")
    result = search_via_zhipu(query)
    if result.error:
        print(f"❌ 失败: {result.error}")
        print("\n--- 降级到 DuckDuckGo ---")
        result = search_via_duckduckgo(query)
        if result.error:
            print(f"❌ DuckDuckGo 也失败: {result.error}")
        else:
            print(f"✅ 来源: {result.source}")
            print(result.summary)
    else:
        print(f"✅ 来源: {result.source}")
        print(result.summary)
