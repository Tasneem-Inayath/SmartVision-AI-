import sys, os
sys.path.append(os.path.abspath("."))
import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO #type:ignore

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.title("📷 Live Webcam Detection")
st.caption("Real-time object detection using YOLOv8")

st.info(
    "⚠️ Live webcam works only on **local execution**.\n\n"
    "Cloud deployments do not allow direct webcam access."
)

# -------------------------------
# LOAD MODEL (ONCE)
# -------------------------------
@st.cache_resource
def load_yolo():
    return YOLO("models/best_yolo_model.pt")

yolo_model = load_yolo()

# -------------------------------
# USER CONTROLS
# -------------------------------
conf_threshold = st.slider(
    "Detection Confidence",
    0.1, 1.0, 0.5, 0.05
)

start = st.button("▶ Start Webcam")
stop = st.button("⏹ Stop Webcam")

frame_placeholder = st.empty()
fps_placeholder = st.empty()

# -------------------------------
# WEBCAM LOOP
# -------------------------------
if start:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Webcam not accessible")
    else:
        prev_time = 0

        while cap.isOpened():
            if stop:
                break

            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Frame not received")
                break

            # YOLO inference
            results = yolo_model.predict(
                source=frame,
                conf=conf_threshold,
                imgsz=640,
                verbose=False
            )[0]

            # Draw boxes
            for box in results.boxes: #type: ignore
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                score = float(box.conf[0])

                label = yolo_model.names[cls_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {score:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            fps_placeholder.metric("FPS", f"{fps:.2f}")

            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

        cap.release()
