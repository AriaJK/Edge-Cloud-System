from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

from cloud.agent import (
    analyze_scene,
    chat_with_agent,
    agent_chat_full,
)

import numpy as np
import cv2
import traceback


app = FastAPI()


# =============================
# 图片分析
# =============================

@app.post("/analyze")
async def analyze(
        file: UploadFile = File(...),
        person_count: int = Form(...)
):

    try:

        content = await file.read()

        img_array = np.frombuffer(
            content,
            np.uint8
        )

        image = cv2.imdecode(
            img_array,
            cv2.IMREAD_COLOR
        )


        if image is None:

            return {
                "status":"error",
                "message":"图片解析失败"
            }


        result = analyze_scene(
            image,
            person_count
        )


        return {
            "status":"success",
            "analysis":result
        }


    except Exception as e:

        traceback.print_exc()

        return {
            "status":"error",
            "message":str(e)
        }



# =============================
# Chat
# =============================

class ChatRequest(BaseModel):

    question:str

    history:Optional[
        List[Dict[str,Any]]
    ] = None



@app.post("/chat")
async def chat(
        request:ChatRequest
):

    answer = chat_with_agent(
        request.question,
        request.history
    )

    return {
        "answer":answer
    }



# =============================
# 增强Agent
# =============================

@app.post("/chat/full")
async def chat_full(request: ChatRequest):

    print("==========================")
    print("进入 /chat/full")
    print("问题:", request.question)
    print("==========================")

    try:

        answer = agent_chat_full(
            request.question,
            request.history
        )

        print("返回:", answer[:100])

        return {
            "answer": answer
        }


    except Exception as e:

        print("chat/full错误:", e)

        return {
            "answer": f"错误:{e}"
        }


@app.get("/health")
async def health():
    print("[health] ping", flush=True)
    return {"status": "online"}