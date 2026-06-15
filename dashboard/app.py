import streamlit as st
import json
import requests
import pandas as pd
import os

from streamlit_autorefresh import st_autorefresh

# ==========================
# 页面配置
# ==========================
st.set_page_config(
    page_title="边云协同智能检测系统",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# ✅ 统一图片路径（关键修复）
LATEST_IMAGE = os.path.join(ROOT_DIR, "dashboard", "latest.jpg")

# ==========================
# 自动刷新
# ==========================
st_autorefresh(interval=3000, key="refresh")

# ==========================
# 读取状态
# ==========================
def load_status():
    status_path = os.path.join(ROOT_DIR, "status.json")

    default = {
        "person_count": 0,
        "task_type": "未知",
        "fps": 0,
        "analysis": "暂无分析结果",
        "last_update": "--",
        "history": [],
        "logs": [],
        "capture_time": "--"
    }

    try:
        if not os.path.exists(status_path):
            return default

        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print("[UI LOAD ERROR]", e)
        return default


status = load_status()

# ==========================
# analysis 解析（防乱码）
# ==========================
def parse_analysis(raw):
    if not raw:
        return "暂无分析结果"

    if isinstance(raw, dict):
        raw = raw.get("analysis", str(raw))

    if isinstance(raw, str):
        # 去掉 JSON 形式干扰
        raw = raw.replace("\\n", "\n")

        # 如果是 dict string（非常常见坑）
        if raw.strip().startswith("{"):
            try:
                data = eval(raw)
                if isinstance(data, dict):
                    raw = data.get("analysis", str(data))
            except:
                pass

    return raw


# ==========================
# UI
# ==========================
st.title("边云协同智能检测系统")

col1, col2 = st.columns([2, 1])

# --------------------------
# 左侧
# --------------------------
with col1:

    st.subheader("检测画面")

    # ✅ 修复点：统一路径 + 防炸
    if os.path.exists(LATEST_IMAGE):
        st.image(LATEST_IMAGE, width=760)
    else:
        st.warning(f"暂无检测画面: {LATEST_IMAGE}")

    st.subheader("Agent 问答")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("请输入问题")

    if question:

        # 保存用户消息
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # 显示用户输入
        with st.chat_message("user"):
            st.write(question)

        try:

            with st.chat_message("assistant"):

                with st.spinner("Agent分析中..."):

                    response = requests.post(
                        "http://127.0.0.1:8000/chat/full",
                        json={
                            "question": question,
                            "history": st.session_state.messages[-10:]
                        },
                        timeout=60
                    )

                    if response.status_code == 200:

                        data = response.json()

                        answer = data.get(
                            "answer",
                            "Agent没有返回内容"
                        )

                    else:

                        answer = (
                            f"Agent错误:"
                            f"{response.status_code}"
                        )

                    st.write(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


        except Exception as e:

            answer = f"Agent暂时不可用:{e}"

            st.error(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


# --------------------------
# 右侧
# --------------------------
with col2:

    st.subheader("实时信息")

    st.metric("检测人数", status.get("person_count", 0))
    st.write("任务类型:", status.get("task_type", "未知"))
    st.write("FPS:", status.get("fps", 0))
    st.write("画面采集时间:", status.get("capture_time", "--"))

    st.subheader("AI分析结果")

    analysis_text = parse_analysis(status.get("analysis"))
    st.markdown(analysis_text)


# ==========================
# 趋势
# ==========================
st.subheader("人数变化趋势")

history_data = status.get("history", [])

if history_data:
    df = pd.DataFrame(history_data)
    st.line_chart(df.set_index("time"))
else:
    st.info("暂无历史数据")


# ==========================
# logs
# ==========================
st.subheader("系统日志")

logs = status.get("logs", [])

if logs:
    for log in reversed(logs):
        st.write(log)
else:
    st.info("暂无日志")