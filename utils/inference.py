import cv2
import numpy as np
from ultralytics import YOLO #type:ignore

# Load YOLO model
yolo_model = YOLO("models/best_yolo_model.pt")

CLASS_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','truck',
    'traffic light','stop sign','bench','bird','cat','dog','horse',
    'cow','elephant','bottle','cup','bowl','pizza','cake',
    'chair','couch','bed','potted plant'
]
def detect_objects(image_np, conf_threshold=0.5):
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = yolo_model.predict(
        source=image_bgr,
        conf=conf_threshold,
        iou=0.5,
        imgsz=640,
        verbose=False
    )[0]

    detections = []

    for box in results.boxes: #type: ignore
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        score = float(box.conf[0])

        label = CLASS_NAMES[cls_id]
        detections.append((label, score))

        cv2.rectangle(image_bgr, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(
            image_bgr,
            f"{label} {score:.2f}",
            (x1, y1-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), detections
