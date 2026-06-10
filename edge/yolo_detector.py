from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow(
        "YOLO Detection",
        annotated_frame
    )

    if cv2.waitKey(1) == 27:
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()