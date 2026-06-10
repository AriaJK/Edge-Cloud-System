from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi import Form
from typing import List, Dict, Any, Optional

from cloud.agent import (
    analyze_scene,
    chat_with_agent
)

import numpy as np
import cv2

app = FastAPI()


@app.post("/analyze")
async def analyze(
        file: UploadFile = File(...),
        person_count: int = Form(...)
):

    try:

        content = await file.read()

        np_arr = np.frombuffer(
            content,
            np.uint8
        )

        image = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return {
                "status": "error",
                "message": "无法解析图片"
            }

        result = analyze_scene(
            image,
            person_count
        )

        return {
            "status": "success",
            "analysis": result
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, Any]]] = None


@app.post("/chat")
async def chat(request: ChatRequest):

    answer = chat_with_agent(
        request.question,
        request.history
    )

    return {
        "answer": answer
    }


@app.get("/health")
async def health():

    return {
        "status": "online"
    }