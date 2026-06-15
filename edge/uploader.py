import requests
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CLOUD_URL = "http://127.0.0.1:8000/analyze"


def upload_image(image_path, person_count):
    """
    上传YOLO图像到云端
    """

    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {"person_count": person_count}

            response = requests.post(
                CLOUD_URL,
                files=files,
                data=data,
                timeout=30
            )

        print("[UPLOAD] 状态码:", response.status_code)
        print("[UPLOAD] 返回:", response.text)

        return response.json()

    except Exception as e:
        print("[UPLOAD ERROR]", e)
        return None