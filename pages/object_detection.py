import sys, os
sys.path.append(os.path.abspath("."))
import streamlit as st
import numpy as np
from PIL import Image
from utils.inference import detect_objects
from utils.cnn_predict import predict_with_cnn
from collections import Counter

st.title("Object Detection")
st.caption("YOLOv8 multi-object detection")

yolo_conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Reset CNN state on new upload
if uploaded:
    st.session_state["run_cnn"] = False

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    st.session_state["uploaded_image"] = image_np

    with st.spinner("Running YOLO detection..."):
        detected_img, detections = detect_objects(image_np, yolo_conf)

    st.session_state["detections"] = detections

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.image(detected_img, caption="Detection Result", use_container_width=True)

    with col2:
        st.subheader("Detection Results")

        if detections:
            st.success(f"Detected {len(detections)} object(s)")

            st.markdown("### 🧾 Detected Objects")
            for i, (label, score) in enumerate(detections, 1):
                st.write(f"**{i}. {label}** — Confidence: `{score:.2f}`")
        else:
            st.warning("No objects detected at this confidence level")

        if st.button("🔍 Run CNN Classification"):
            st.session_state["run_cnn"] = True

# CNN block
if st.session_state.get("run_cnn", False):

    st.divider()
    st.subheader("CNN Classification Result")

    image_np = st.session_state["uploaded_image"]

    cnn_label, cnn_conf = predict_with_cnn(image_np)

    st.metric("Predicted Class", cnn_label)
    st.progress(cnn_conf)
    st.write(f"Confidence: **{cnn_conf:.2f}**")

# Detection summary
if "detections" in st.session_state and st.session_state["detections"]:

    labels_only = [d[0] for d in st.session_state["detections"]]
    counts = Counter(labels_only)

    st.markdown("### 📊 Detection Summary")
    for cls, cnt in counts.items():
        st.write(f"- **{cls}** × {cnt}")
