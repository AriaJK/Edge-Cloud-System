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

# ---------------------------
# 配置
# ---------------------------
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ROOT_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        ".."
    )
)

last_upload_time = 0
UPLOAD_INTERVAL = 10

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolov8n.pt"
)

CLOUD_URL = "http://127.0.0.1:8000/analyze"

SIMPLE_TASK_THRESHOLD = 3

TEMP_IMAGE_PATH = os.path.join(
    ROOT_DIR,
    "temp.jpg"
)

UPLOAD_QUEUE_DIR = os.path.join(
    ROOT_DIR,
    "upload_queue"
)

os.makedirs(UPLOAD_QUEUE_DIR, exist_ok=True)

WINDOW_NAME = "YOLO Edge Detection"

latest_ai_analysis = "等待云端分析..."

STATUS_INTERVAL = 0.3
last_status_time = 0


# ---------------------------
# 上传函数
# ---------------------------
def upload_image(
        image_path,
        person_count
):

    global latest_ai_analysis

    if not os.path.exists(image_path):
        return

    try:

        with open(image_path, "rb") as f:

            files = {
                "file": f
            }

            data = {
                "person_count": person_count
            }

            response = requests.post(
                CLOUD_URL,
                files=files,
                data=data,
                timeout=30
            )

        result = response.json()

        print(result)

        if result.get("status") == "success":

            latest_ai_analysis = (
                f"[云端分析时间 "
                f"{datetime.now().strftime('%H:%M:%S')}]\n\n"
                +
                result.get(
                    "analysis",
                    "暂无分析结果"
                )
            )

        return result

    except Exception as e:

        print("[上传错误]", e)

        # 保存到离线队列（原子复制图片 + metadata）
        try:

            uid = uuid.uuid4().hex

            dst_image = os.path.join(
                UPLOAD_QUEUE_DIR,
                f"{uid}.jpg"
            )

            shutil.copy2(
                image_path,
                dst_image
            )

            meta = {
                "image": os.path.basename(dst_image),
                "person_count": person_count,
                "retries": 0,
                "timestamp": datetime.now().isoformat()
            }

            with open(
                    os.path.join(
                        UPLOAD_QUEUE_DIR,
                        f"{uid}.json"
                    ),
                    "w",
                    encoding="utf-8"
            ) as mf:

                json.dump(meta, mf, ensure_ascii=False)

            print(f"[队列] 已加入上传队列: {dst_image}")

        except Exception as e2:

            print("[队列保存失败]", e2)

        return None


# ---------------------------
# 异步上传
# ---------------------------
def async_upload(
        image_path,
        person_count
):
    threading.Thread(
        target=upload_image,
        args=(
            image_path,
            person_count
        ),
        daemon=True
    ).start()


def process_upload_queue():

    while True:

        try:

            items = [f for f in os.listdir(UPLOAD_QUEUE_DIR) if f.endswith('.json')]

            for meta_file in items:

                meta_path = os.path.join(UPLOAD_QUEUE_DIR, meta_file)

                try:

                    with open(meta_path, 'r', encoding='utf-8') as mf:

                        meta = json.load(mf)

                except Exception:

                    os.remove(meta_path)

                    continue

                image_name = meta.get('image')

                image_path = os.path.join(UPLOAD_QUEUE_DIR, image_name)

                person_count = meta.get('person_count', 0)

                retries = meta.get('retries', 0)

                try:

                    with open(image_path, 'rb') as f:

                        files = {'file': f}

                        data = {'person_count': person_count}

                        resp = requests.post(CLOUD_URL, files=files, data=data, timeout=20)

                    if resp.status_code == 200:

                        print(f"[队列] 上传成功: {image_name}")

                        # 删除 files
                        try:

                            os.remove(image_path)

                        except:

                            pass

                        try:

                            os.remove(meta_path)

                        except:

                            pass

                    else:

                        raise Exception(f"状态码{resp.status_code}")

                except Exception as e:

                    retries += 1

                    meta['retries'] = retries

                    with open(meta_path, 'w', encoding='utf-8') as mf:

                        json.dump(meta, mf, ensure_ascii=False)

                    backoff = min(60, (2 ** retries))

                    print(f"[队列] 上传失败 {image_name} 重试 {retries}，等待{backoff}s: {e}")

                    time.sleep(backoff)

        except Exception as e:

            print('[队列处理异常]', e)

        time.sleep(5)


# ---------------------------
# 调度策略
# ---------------------------
def schedule_task(person_count, frame):

    global last_upload_time

    current_time = time.time()

    if person_count <= SIMPLE_TASK_THRESHOLD:

        return "edge"

    if current_time - last_upload_time < UPLOAD_INTERVAL:

        return "cloud(wait)"

    last_upload_time = current_time

    cv2.imwrite(
        TEMP_IMAGE_PATH,
        frame
    )

    # 上传图片 + 同步人数
    async_upload(
        TEMP_IMAGE_PATH,
        person_count
    )

    return "cloud"

# ---------------------------
# 保存状态
# ---------------------------
def save_status(
        person_count,
        task_type,
        fps,
        analysis,
        frame
):

    global last_status_time

    now = time.time()

    if now - last_status_time < STATUS_INTERVAL:
        return

    last_status_time = now

    status_path = os.path.join(
        ROOT_DIR,
        "status.json"
    )

    if os.path.exists(status_path):

        try:

            with open(
                    status_path,
                    "r",
                    encoding="utf-8"
            ) as f:

                status = json.load(f)

        except:

            status = {}

    else:

        status = {}

    status.setdefault(
        "history",
        []
    )

    status.setdefault(
        "logs",
        []
    )

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    capture_time_iso = datetime.now().isoformat()

    status["person_count"] = person_count

    status["task_type"] = task_type

    status["fps"] = fps

    status["analysis"] = analysis

    status["last_update"] = current_time
    status["capture_time"] = capture_time_iso

    status["history"].append({

        "time": current_time,

        "person_count": person_count
    })

    status["history"] = status["history"][-50:]

    status["logs"].append(
        f"{current_time} "
        f"人数:{person_count} "
        f"状态:{task_type}"
    )

    status["logs"] = status["logs"][-50:]

    os.makedirs(
        os.path.join(
            ROOT_DIR,
            "dashboard"
        ),
        exist_ok=True
    )

    latest_image_path = os.path.join(
        ROOT_DIR,
        "dashboard",
        "latest.jpg"
    )

    latest_tmp_path = os.path.join(
        ROOT_DIR,
        "dashboard",
        "latest.tmp.jpg"
    )

    if cv2.imwrite(
            latest_tmp_path,
            frame
    ):

        os.replace(
            latest_tmp_path,
            latest_image_path
        )

    with open(
            status_path,
            "w",
            encoding="utf-8"
    ) as f:

        json.dump(
            status,
            f,
            ensure_ascii=False,
            indent=4
        )


# ---------------------------
# 主函数
# ---------------------------
def main():

    global latest_ai_analysis

    model = YOLO(
        MODEL_PATH
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("无法打开摄像头")

        return

    # 创建窗口
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        960,
        540
    )

    try:

        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_TOPMOST,
            1
        )

    except:
        pass

    print("按ESC退出")

    prev_time = time.time()

    try:

        # start upload queue processor
        threading.Thread(
            target=process_upload_queue,
            daemon=True
        ).start()

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # YOLO推理
            results = model(frame)[0]

            annotated_frame = results.plot()

            # 人数统计
            person_count = sum(
                1
                for cls in results.boxes.cls
                if int(cls) == 0
            )

            # FPS
            now = time.time()

            fps = 1 / (
                    now -
                    prev_time +
                    1e-6
            )

            prev_time = now

            # 调度
            task_type = schedule_task(
                person_count,
                frame
            )

            # AI分析
            analysis = latest_ai_analysis

            save_status(
                person_count,
                task_type,
                round(fps, 2),
                analysis,
                annotated_frame
            )

            cv2.putText(
                annotated_frame,
                f"Task:{task_type}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                f"FPS:{fps:.1f}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            cv2.imshow(
                WINDOW_NAME,
                annotated_frame
            )

            try:

                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_TOPMOST,
                    1
                )

            except:

                pass

            # 点击X关闭窗口
            try:

                window_visible = cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE
                )

            except cv2.error:

                window_visible = -1

            if window_visible < 1:

                print("检测窗口已关闭")

                try:

                    cv2.destroyWindow(
                        WINDOW_NAME
                    )

                except:

                    pass

                break

            key = cv2.waitKey(1)

            if key == 27:
                print("ESC退出")
                break

            time.sleep(0.03)

    except KeyboardInterrupt:

        print("手动中断")

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print("程序结束")


if __name__ == "__main__":
    main()