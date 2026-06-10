import cv2
import base64
import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ROOT_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        ".."
    )
)

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ==========================
# 图片转Base64
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
    ).decode("utf-8")


def normalize_analysis_person_count(analysis, person_count):

    lines = analysis.splitlines()

    replaced = False

    for index, line in enumerate(lines):

        stripped_line = line.strip()

        if (
                not replaced
                and re.match(r"^\d+\.", stripped_line)
                and "人数" in stripped_line
        ):

            lines[index] = f"1. 人数统计：当前检测到{person_count}人。"

            replaced = True

    if not replaced:

        lines.insert(0, f"1. 人数统计：当前检测到{person_count}人。")

    return "\n".join(lines)


# ==========================
# 云端视觉分析
# ==========================
def analyze_scene(image, person_count):

    try:

        _, buffer = cv2.imencode(".jpg", image)

        image_base64 = base64.b64encode(
            buffer
        ).decode("utf-8")

        response = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":
                                f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text":
                            f"""
                            当前YOLO实时检测人数：

                            {person_count}人

                            请以YOLO人数为准分析画面。

                            输出：

                            1. 当前人数
                            2. 场景类型
                            3. 人员活动
                            4. 重要目标
                            5. 风险评估
                            6. 管理建议

                            不允许重新估计人数。

                            必须使用：
                            当前检测人数：{person_count}人

                            作为最终人数。

                            中文回答。
                            """
                        }
                    ]
                }
            ]
        )

        analysis = response.choices[0].message.content.strip()

        return normalize_analysis_person_count(
            analysis,
            person_count
        )

    except Exception as e:

        return f"AI分析失败：{e}"


# ==========================
# Agent视觉问答
# ==========================
def chat_with_agent(question, history=None):

    try:

        image_path = os.path.join(
            ROOT_DIR,
            "dashboard",
            "latest.jpg"
        )

        if not os.path.exists(image_path):

            return "暂无监控画面"

        image = None

        for _ in range(5):

            image = cv2.imread(image_path)

            if image is not None and image.size > 0:
                break

            time.sleep(0.08)

        if image is None or image.size == 0:

            return f"问答失败：无法读取最新监控画面 {image_path}"

        image_base64 = image_to_base64(
            image
        )

        if not image_base64:

            return "问答失败：监控画面为空，无法进行视觉分析"

        status_path = os.path.join(
            ROOT_DIR,
            "status.json"
        )

        status_context = ""

        if os.path.exists(status_path):

            try:

                with open(
                        status_path,
                        "r",
                        encoding="utf-8"
                ) as f:

                    status = json.load(f)

                status_context = (
                    f"当前检测人数：{status.get('person_count', '未知')}人\n"
                    f"当前任务类型：{status.get('task_type', '未知')}\n"
                    f"当前FPS：{status.get('fps', '未知')}\n"
                    f"最近更新时间：{status.get('last_update', '未知')}\n"
                )

            except:

                status_context = ""

        conversation_messages = [

            {
                "role": "system",

                "content":
                """
                你是边云协同智能检测系统助手，擅长结合监控画面做视觉问答。

                你必须优先依据最新监控画面回答问题，不要凭空猜测。

                如果画面里能看到人数、场景、物体、行为或风险，请明确描述。

                如果画面不清晰或信息不足，要直接说明“不确定”，不要编造。

                回答风格要求：

                - 先给出直接结论
                - 再给出依据
                - 如果问题涉及人数，请以画面和系统检测结果综合判断，但优先说明画面可见内容

                可分析：

                - 人数
                - 场景
                - 行为
                - 物体
                - 风险
                - 异常情况

                如果涉及真实身份识别，
                必须说明无法确认身份。
                """
            }
        ]

        if history:

            for msg in history[-8:]:

                role = msg.get("role")

                content = str(msg.get("content", "")).strip()

                if role in {"user", "assistant"} and content:

                    conversation_messages.append(
                        {
                            "role": role,
                            "content": content
                        }
                    )

        conversation_messages.append(
            {
                "role": "user",

                "content": [

                    {
                        "type": "image_url",

                        "image_url": {
                            "url":
                            f"data:image/jpeg;base64,{image_base64}"
                        }
                    },

                    {
                        "type": "text",

                        "text": (
                            f"{status_context}\n"
                            f"用户问题：{question}\n\n"
                            "请结合图片直接回答，必要时分点说明。"
                        )
                    }
                ]
            }
        )

        response = client.chat.completions.create(

            model="glm-4v-flash",

            messages=conversation_messages
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"问答失败：{e}"