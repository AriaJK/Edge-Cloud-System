import json
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def update_status(
        person_count,
        task_type,
        fps,
        analysis=""
):
    status = {
        "person_count": person_count,
        "task_type": task_type,
        "fps": fps,
        "last_update": str(datetime.now()),
        "analysis": analysis,
        "logs": [
            f"{datetime.now()} 检测到{person_count}人",
            f"{datetime.now()} {task_type}"
        ]
    }

    with open(
        "status.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            status,
            f,
            ensure_ascii=False,
            indent=4
        )