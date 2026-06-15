import cv2
import base64
import json
import os
import re
import shutil
import time
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ==========================
# ROOT统一
# ==========================

ROOT_DIR = Path(__file__).resolve().parent.parent


load_dotenv()


client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)



# ==========================
# 图片base64
# ==========================

def image_to_base64(image):

    if image is None:
        return None


    _, buffer = cv2.imencode(
        ".jpg",
        image
    )


    return base64.b64encode(
        buffer
    ).decode()



# ==========================
# YOLO人数修正
# ==========================

def normalize_analysis_person_count(
        text,
        person_count
):

    if not text:
        return text


    lines=text.splitlines()

    flag=False


    for i,line in enumerate(lines):

        if (
            "人数" in line
            and not flag
        ):

            lines[i]=(
                f"1. 人数统计：当前检测到{person_count}人"
            )

            flag=True


    if not flag:

        lines.insert(
            0,
            f"1. 人数统计：当前检测到{person_count}人"
        )


    return "\n".join(lines)



# ==========================
# 云端视觉分析
# ==========================

def analyze_scene(
        image,
        person_count
):

    try:

        _,buffer=cv2.imencode(
            ".jpg",
            image
        )


        img64=base64.b64encode(
            buffer
        ).decode()



        resp=client.chat.completions.create(

            model="glm-4v-flash",

            messages=[
                {
                    "role":"user",

                    "content":[

                        {
                            "type":"image_url",
                            "image_url":{
                                "url":
                                f"data:image/jpeg;base64,{img64}"
                            }
                        },

                        {
                            "type":"text",

                            "text":
                            f"""
当前YOLO检测人数:
{person_count}人

请分析监控画面。

要求:
1.人数
2.场景
3.人员行为
4.风险
5.建议

人数必须使用YOLO结果。
中文回答。
"""
                        }

                    ]
                }
            ]
        )


        answer=resp.choices[0].message.content


        return normalize_analysis_person_count(
            answer,
            person_count
        )


    except Exception as e:

        return f"AI分析失败:{e}"





# ==========================
# 视觉问答
# ==========================

def chat_with_agent(
        question,
        history=None
):

    try:

        image_path=os.path.join(
            ROOT_DIR,
            "dashboard",
            "latest.jpg"
        )


        if not os.path.exists(image_path):

            return "暂无监控画面"



        tmp=image_path+".tmp"


        image=None


        for _ in range(10):

            try:

                shutil.copy2(
                    image_path,
                    tmp
                )


                data=np.fromfile(
                    tmp,
                    dtype=np.uint8
                )


                image=cv2.imdecode(
                    data,
                    cv2.IMREAD_COLOR
                )


                if image is not None:
                    break


            except:
                pass


            time.sleep(0.1)



        try:
            os.remove(tmp)
        except:
            pass



        if image is None:

            return "无法读取最新图片"



        img64=image_to_base64(
            image
        )



        messages=[

            {
                "role":"system",

                "content":
                """
你是边云协同智能检测系统助手。

根据图片回答。

不能确认身份。

不要编造。
"""
            }

        ]



        if history:

            messages.extend(
                history[-6:]
            )



        messages.append(

            {
                "role":"user",

                "content":[

                    {
                        "type":"image_url",

                        "image_url":{
                            "url":
                            f"data:image/jpeg;base64,{img64}"
                        }
                    },

                    {
                        "type":"text",

                        "text":question
                    }

                ]
            }

        )



        resp=client.chat.completions.create(

            model="glm-4v-flash",

            messages=messages

        )


        return resp.choices[0].message.content



    except Exception as e:

        return f"视觉问答失败:{e}"


# ==========================
# RAG + 搜索配置
# ==========================

_WEB_SEARCH_TOOL = {

    "type":"web_search",

    "web_search":{
        "enable":True
    }

}



_SEARCH_TRIGGERS=[

    "搜索",
    "查一下",
    "网上",
    "最新",
    "新闻",
    "今天",
    "当前",
    "实时",
    "天气",
    "股价",
    "比赛",
    "事件",
    "政策",
    "法规",
    "标准",
    "更新"

]



def _should_search(question):

    question=question.lower()

    return any(
        x in question
        for x in _SEARCH_TRIGGERS
    )





# ==========================
# 构建上下文
# ==========================

def _build_tool_context(question):

    context=[]


    # -------- RAG --------

    try:

        from cloud.rag import search_knowledge_base


        results=search_knowledge_base(
            question,
            top_k=3
        )


        if results:

            rag="\n".join(

                [
                    f"""
[{i}]
来源:{r.get('source')}

{r.get('text')[:400]}
"""
                    for i,r in enumerate(
                        results,
                        1
                    )
                ]

            )


            context.append(
                "【知识库】\n"+rag
            )


    except Exception as e:

        print(
            "[RAG失败]",
            e
        )




    # -------- 状态 --------

    try:

        status_path=os.path.join(
            ROOT_DIR,
            "status.json"
        )


        if os.path.exists(status_path):

            with open(
                status_path,
                encoding="utf-8"
            ) as f:

                status=json.load(f)



            context.append(

                f"""
【系统状态】

人数:
{status.get('person_count')}

任务:
{status.get('task_type')}

FPS:
{status.get('fps')}
"""

            )


    except:

        pass



    return "\n\n".join(context)






# ==========================
# 搜索后回答
# ==========================

def _search_then_answer(question, history, context):
    """先通过 DuckDuckGo 搜索，再用 LLM 总结搜索结果"""
    search_text = ""
    try:
        from cloud.search_tool import search_via_duckduckgo
        sr = search_via_duckduckgo(question, max_results=5)

        # 优先用原始结果拼接（信息量更大），LLM摘要作为备选
        if sr.results:
            parts = []
            for i, r in enumerate(sr.results, 1):
                parts.append(f"[{i}] 标题:{r.title}\n内容:{r.snippet}\n链接:{r.url}")
            search_text = "\n\n".join(parts)
        elif sr.summary and not sr.error:
            search_text = sr.summary
    except Exception as e:
        print("[搜索失败]", e)

    if not search_text:
        return _simple_llm(question, history, context, enable_search=False)

    print("[搜索]", f"结果长度={len(search_text)}")

    # 用专门的总结 prompt
    summary_messages = [
        {
            "role": "system",
            "content": (
                "你是信息检索助手。请严格基于【联网搜索结果】回答用户问题。"
                "引用来源时注明 [编号]。中文回答，简洁完整。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"【用户问题】\n{question}\n\n"
                f"【联网搜索结果】\n{search_text}\n\n"
                "请基于以上搜索结果回答用户问题。"
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=summary_messages,
            temperature=0.3,
            max_tokens=2048,
        )
        print("[搜索总结]", f"回答长度={len(resp.choices[0].message.content)}")
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("[搜索总结失败]", e)
        return f"搜索完成但总结失败: {e}"


# ==========================
# Agent总入口
# ==========================

def agent_chat_full(question, history=None):
    try:
        # -- 视觉关键词 --
        visual = any(
            word in question
            for word in ["画面","图片","图像","看到","监控","摄像头"]
        )
        # 上下文推断：跳过最后一条（当前问题），看上上条用户消息
        if not visual and history:
            user_msgs = [m for m in history if m.get("role") == "user"]
            if len(user_msgs) >= 2:
                prev_user = str(user_msgs[-2].get("content", ""))
                visual = any(
                    word in prev_user
                    for word in ["画面","图片","几个人","看到","摄像头","监控"]
                )

        image_exists = os.path.exists(
            os.path.join(ROOT_DIR, "dashboard", "latest.jpg")
        )
        search = _should_search(question)
        context = _build_tool_context(question)

        print("[agent]", "visual=", visual, "search=", search)

        # 图片问题 → 视觉模型
        if visual and image_exists:
            return chat_with_agent(question, history)

        # 搜索问题 → 先用 DuckDuckGo 搜，再用 LLM 总结
        if search:
            return _search_then_answer(question, history, context)

        # 默认 → RAG + LLM
        return _simple_llm(question, history, context, enable_search=False)

    except Exception as e:
        return f"Agent异常:{e}"







# ==========================
# 普通LLM
# ==========================

def _simple_llm(

        question,

        history=None,

        context="",

        enable_search=False

):


    # 动态 system prompt：搜索模式下强制使用 web_search
    system_text = (
        "你是边云协同智能检测系统助手。回答必须中文。"
        "如果有参考信息，优先基于参考信息回答。"
    )
    if enable_search:
        system_text += (
            "【重要】你必须使用 web_search 工具联网搜索用户问题，"
            "然后基于搜索结果给出回答，不要跳过搜索步骤。"
        )

    messages = [{"role": "system", "content": system_text}]




    if history:


        for h in history[-6:]:

            if h.get("role") in [
                "user",
                "assistant"
            ]:

                messages.append(
                    {
                        "role":h["role"],
                        "content":h["content"]
                    }
                )




    user_text=question



    if context:


        user_text=f"""

参考信息:

{context}


问题:

{question}

"""



    messages.append(

        {
            "role":"user",

            "content":user_text

        }

    )




    kwargs={


        "model":"glm-4-flash",

        "messages":messages,

        "temperature":0.3,

        "max_tokens":4096

    }




    if enable_search:

        kwargs["tools"]=[

            _WEB_SEARCH_TOOL

        ]



    print(
        "[LLM调用]",
        "search=",
        enable_search
    )



    resp=client.chat.completions.create(
        **kwargs
    )



    msg=resp.choices[0].message




    # 正常回答

    if msg.content:

        return msg.content.strip()




    # ======================
    # 搜索工具返回
    # ======================

    if msg.tool_calls:


        print(
            "触发搜索工具"
        )


        messages.append(

            {
                "role":"assistant",

                "content":None,

                "tool_calls":[

                    {

                        "id":tc.id,

                        "type":"function",

                        "function":{

                            "name":
                            tc.function.name,

                            "arguments":
                            tc.function.arguments

                        }

                    }

                    for tc in msg.tool_calls

                ]

            }

        )



        # 给工具结果

        for tc in msg.tool_calls:


            messages.append(

                {

                    "role":"tool",

                    "tool_call_id":
                    tc.id,

                    "content":
                    "搜索已经完成，请根据搜索结果回答用户。"

                }

            )




        # 二次生成

        resp2=client.chat.completions.create(

            model="glm-4-flash",

            messages=messages,

            temperature=0.3

        )



        answer=resp2.choices[0].message.content



        if answer:

            return answer.strip()



    return "暂时无法生成回答"


# ==========================
# 兼容旧接口
# ==========================

def chat(question, history=None):

    return agent_chat_full(
        question,
        history
    )



# ==========================
# 测试入口
# ==========================

if __name__ == "__main__":

    print("="*60)

    print(
        "边云协同智能检测系统 Agent"
    )

    print("="*60)



    while True:

        try:

            q=input(
                "\n用户:"
            )


            if q in [
                "exit",
                "quit"
            ]:

                break



            answer=agent_chat_full(
                q
            )


            print(
                "\nAgent:"
            )


            print(answer)



        except Exception as e:


            print(
                "错误:",
                e
            )