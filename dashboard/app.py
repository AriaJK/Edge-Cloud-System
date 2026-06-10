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

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ROOT_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        ".."
    )
)

# ==========================
# 自动刷新
# ==========================

st_autorefresh(
    interval=3000,
    key="refresh"
)

# ==========================
# 读取状态
# ==========================

def load_status():

    try:

        status_path = os.path.join(
            ROOT_DIR,
            "status.json"
        )

        with open(
                status_path,
                "r",
                encoding="utf-8"
        ) as f:

            return json.load(f)

    except:

        return {

            "person_count": 0,

            "task_type": "未知",

            "fps": 0,

            "analysis": "暂无分析结果",

            "last_update": "--",

            "history": [],

            "logs": []
        }


status = load_status()

# ==========================
# 标题
# ==========================

st.title("边云协同智能检测系统")

# --------------------------
# 侧边栏导航
# --------------------------
st.sidebar.title("导航")
page = st.sidebar.radio(
    "页面",
    ("实时监控", "任务日志", "系统状态")
)


if page == "实时监控":

    st.header("实时监控")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.subheader("检测画面")

        try:

            image_path = os.path.join(
                BASE_DIR,
                "latest.jpg"
            )

            st.image(image_path, width=760)

        except:

            st.warning("暂无检测画面")

        st.subheader("Agent 问答")

        if "messages" not in st.session_state:

            st.session_state.messages = []

        for msg in st.session_state.messages:

            with st.chat_message(msg["role"]):

                st.write(msg["content"])

        question = st.chat_input("请输入问题")

        if question:

            st.session_state.messages.append({"role": "user", "content": question})

            try:

                response = requests.post(

                    "http://127.0.0.1:8000/chat",

                    json={

                        "question": question,

                        "history": st.session_state.messages[-10:]

                    },

                    timeout=30

                )

                answer = response.json().get("answer", "无返回结果")

            except Exception as e:

                answer = f"Agent暂时不可用：{e}"

            st.session_state.messages.append({"role": "assistant", "content": answer})

            st.rerun()

    with col2:

        st.subheader("实时信息")

        st.metric("检测人数", status.get("person_count", 0))

        st.write("任务类型: ", status.get("task_type", "未知"))

        st.write("FPS: ", status.get("fps", 0))

        st.write("画面采集时间: ", status.get("capture_time", "--"))

        st.subheader("AI分析结果")

        st.info(status.get("analysis", "暂无分析结果"))


elif page == "任务日志":

    st.header("任务日志")

    logs = status.get("logs", [])

    if logs:

        for log in reversed(logs):

            st.write(log)

    else:

        st.info("暂无日志")


elif page == "系统状态":

    st.header("系统状态")

    task_type = str(status.get("task_type", "未知"))

    if task_type == "edge":

        st.success("当前模式：边缘端在线")

    elif task_type.startswith("cloud"):

        st.success("当前模式：云端在线")

    else:

        st.info(f"当前模式：{task_type}")

    try:

        requests.get("http://127.0.0.1:8000/health", timeout=1)

        st.success("云端在线")

    except:

        st.error("云端离线")

# ==========================
# 当前检测结果
# ==========================

st.subheader("当前检测结果")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "检测人数",
    status.get(
        "person_count",
        0
    )
)

c2.metric(
    "任务类型",
    status.get(
        "task_type",
        "未知"
    )
)

c3.metric(
    "FPS",
    status.get(
        "fps",
        0
    )
)

c4.metric(
    "更新时间",
    status.get(
        "last_update",
        "--"
    )
)

# ==========================
# AI分析
# ==========================

st.subheader("AI分析结果")

analysis_text = status.get(
    "analysis",
    "暂无分析结果"
)

if not str(
        status.get(
            "task_type",
            ""
        )
).startswith("cloud"):

    analysis_text = "当前为边缘模式，暂无云端分析。"

st.info(
    analysis_text
)

# ==========================
# 人数变化趋势图
# ==========================

st.subheader("人数变化趋势")

history_data = status.get(
    "history",
    []
)

if history_data:

    history_df = pd.DataFrame(
        history_data
    )

    st.line_chart(
        history_df.set_index(
            "time"
        )
    )

else:

    st.info(
        "暂无历史数据"
    )

# ==========================
# 系统日志
# ==========================

st.subheader("系统日志")

logs = status.get(
    "logs",
    []
)

if logs:

    for log in reversed(logs):

        st.write(log)

else:

    st.info(
        "暂无日志"
    )

# ==========================
# Agent问答
# ==========================

st.subheader("Agent问答")

if "messages" not in st.session_state:

    st.session_state.messages = []

# 显示历史消息

for msg in st.session_state.messages:

    with st.chat_message(
            msg["role"]
    ):

        st.write(
            msg["content"]
        )

# 输入框

question = st.chat_input(
    "请输入问题"
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    try:

        response = requests.post(

            "http://127.0.0.1:8000/chat",

            json={
                "question": question,
                "history": st.session_state.messages[-10:]
            },

            timeout=30
        )

        answer = response.json().get(
            "answer",
            "无返回结果"
        )

    except Exception as e:

        answer = f"Agent暂时不可用：{e}"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()