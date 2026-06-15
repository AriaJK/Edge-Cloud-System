import cv2
from ultralytics import YOLO
import requests
import time
import os
import threading
from datetime import datetime
import json
import uuid
import shutil

# =========================
# 统一路径（核心修复）
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DASHBOARD_DIR = os.path.join(ROOT_DIR, "dashboard")
os.makedirs(DASHBOARD_DIR, exist_ok=True)

LATEST_IMAGE_PATH = os.path.join(DASHBOARD_DIR, "latest.jpg")
LATEST_TMP_PATH = os.path.join(DASHBOARD_DIR, "latest.tmp.jpg")

# =========================
# 参数
# =========================
last_upload_time = 0
UPLOAD_INTERVAL = 10

MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
CLOUD_URL = "http://127.0.0.1:8000/analyze"
SIMPLE_TASK_THRESHOLD = 3

TEMP_IMAGE_PATH = os.path.join(ROOT_DIR, "temp.jpg")
UPLOAD_QUEUE_DIR = os.path.join(ROOT_DIR, "upload_queue")
os.makedirs(UPLOAD_QUEUE_DIR, exist_ok=True)

WINDOW_NAME = "YOLO Edge Detection"

STATUS_INTERVAL = 0.3
last_status_time = 0

# =========================
# AI结果缓存
# =========================
latest_ai_analysis = {
    "analysis": "等待云端分析...",
    "person_count": 0,
    "capture_time": ""
}

# =========================
# 上传函数（修复路径）
# =========================
def upload_image(image_path, person_count):

    global latest_ai_analysis

    if not os.path.exists(image_path):
        return

    capture_time = datetime.now().isoformat()

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

        result = response.json()
        print(result)

        if result.get("status") == "success":

            latest_ai_analysis = {
                "analysis": "[云端分析时间 "
                            + datetime.now().strftime('%H:%M:%S')
                            + "]\n\n"
                            + result.get("analysis", "暂无分析结果"),
                "person_count": person_count,
                "capture_time": capture_time
            }

        return result

    except Exception as e:
        print("[上传错误]", e)

        # 离线队列
        try:
            uid = uuid.uuid4().hex

            dst_image = os.path.join(UPLOAD_QUEUE_DIR, f"{uid}.jpg")
            shutil.copy2(image_path, dst_image)

            meta = {
                "image": os.path.basename(dst_image),
                "person_count": person_count,
                "capture_time": capture_time,
                "retries": 0,
                "timestamp": datetime.now().isoformat()
            }

            with open(os.path.join(UPLOAD_QUEUE_DIR, f"{uid}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)

        except Exception as e2:
            print("[队列保存失败]", e2)

        return None


# =========================
# 异步上传
# =========================
def async_upload(image_path, person_count):
    threading.Thread(
        target=upload_image,
        args=(image_path, person_count),
        daemon=True
    ).start()


# =========================
# 队列处理
# =========================
def process_upload_queue():

    while True:
        try:
            items = [f for f in os.listdir(UPLOAD_QUEUE_DIR) if f.endswith(".json")]

            for meta_file in items:
                meta_path = os.path.join(UPLOAD_QUEUE_DIR, meta_file)

                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except:
                    os.remove(meta_path)
                    continue

                image_name = meta.get("image")
                image_path = os.path.join(UPLOAD_QUEUE_DIR, image_name)
                person_count = meta.get("person_count", 0)
                retries = meta.get("retries", 0)

                try:
                    with open(image_path, "rb") as f:
                        files = {"file": f}
                        data = {"person_count": person_count}

                        resp = requests.post(CLOUD_URL, files=files, data=data, timeout=20)

                    if resp.status_code == 200:
                        os.remove(image_path)
                        os.remove(meta_path)
                        print("[队列] 上传成功")
                    else:
                        raise Exception(resp.status_code)

                except Exception as e:
                    retries += 1
                    meta["retries"] = retries

                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False)

                    time.sleep(min(60, 2 ** retries))

        except Exception as e:
            print("[队列异常]", e)

        time.sleep(5)


# =========================
# 调度（修复关键：统一写图路径）
# =========================
def schedule_task(person_count, frame):

    global last_upload_time

    current_time = time.time()

    if person_count <= SIMPLE_TASK_THRESHOLD:
        return "edge"

    if current_time - last_upload_time < UPLOAD_INTERVAL:
        return "cloud(wait)"

    last_upload_time = current_time

    # ✅ 关键修复：只写统一 dashboard 路径
    success = cv2.imwrite(LATEST_TMP_PATH, frame)
    print("写入状态:", success, LATEST_TMP_PATH)

    if success:
        os.replace(LATEST_TMP_PATH, LATEST_IMAGE_PATH)

    async_upload(TEMP_IMAGE_PATH, person_count)

    return "cloud"


# =========================
# 保存状态（UI读取核心）
# =========================
def save_status(person_count, task_type, fps, analysis, frame):

    global last_status_time
    global latest_ai_analysis

    now = time.time()
    if now - last_status_time < STATUS_INTERVAL:
        return

    last_status_time = now

    status_path = os.path.join(ROOT_DIR, "status.json")

    status = {}
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
        except:
            status = {}

    status.setdefault("history", [])
    status.setdefault("logs", [])

    current_time = datetime.now().strftime("%H:%M:%S")

    status["person_count"] = person_count
    status["task_type"] = task_type
    status["fps"] = fps
    status["analysis"] = analysis
    status["last_update"] = current_time

    status["history"].append({
        "time": current_time,
        "person_count": person_count
    })
    status["history"] = status["history"][-50:]

    status["logs"].append(f"{current_time} 人数:{person_count} 状态:{task_type}")
    status["logs"] = status["logs"][-50:]

    # ✅ UI唯一读取图片源
    cv2.imwrite(LATEST_IMAGE_PATH, frame)

    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=4)


# =========================
# 主函数（不改逻辑）
# =========================
def main():

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    threading.Thread(target=process_upload_queue, daemon=True).start()

    prev_time = time.time()

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated = results.plot()

        person_count = sum(1 for c in results.boxes.cls if int(c) == 0)

        fps = 1 / (time.time() - prev_time + 1e-6)
        prev_time = time.time()

        task_type = schedule_task(person_count, frame)

        save_status(person_count, task_type, round(fps, 2), latest_ai_analysis, annotated)

        cv2.imshow(WINDOW_NAME, annotated)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()