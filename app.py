import streamlit as st
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image
import io

# -------------------------------
# Title and Intro
# -------------------------------
st.title("SmartVision – Intelligent Visual Understanding Platform")

st.markdown("""
SmartVision is an **end-to-end computer vision system** combining  
**Deep Learning image classification** and **YOLO-based object detection**.

Designed for **accuracy**, **performance**, and **real-world usability**.
""")

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Supported Classes", "25")
col2.metric("Detection Model", "YOLOv8")
col3.metric("Classification Models", "4 CNNs")

st.divider()

st.subheader("🚀 Key Features")

st.markdown("""
✔ Single-object image classification  
✔ Multi-object detection with bounding boxes  
✔ CNN & YOLO hybrid inference  
✔ Confidence-based prediction refinement  
✔ Interactive performance analysis  
""")

st.subheader("🧭 How It Works")

st.markdown("""
1. Upload an image  
2. YOLO detects objects  
3. CNN models classify objects  
4. Predictions are compared  
5. Results visualized clearly  
""")

st.divider()

# -------------------------------
# Load YOLO model from Hugging Face Model repo
# -------------------------------
@st.cache_resource
def load_yolo():
    model_path = hf_hub_download(
        repo_id="TasneemFirdhosh/SmartVision-Models",  # your Hugging Face Model repo
        filename="yolov8s.pt"                          # adjust to your file name
    )
    return YOLO(model_path)

yolo_model = load_yolo()

# -------------------------------
# Upload + Detection
# -------------------------------
st.subheader("🔍 Try it yourself")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO detection
    results = yolo_model(image)

    # Show results
    st.subheader("📊 Detection Results")
    for r in results:
        st.write(r.boxes)  # bounding boxes, confidence, class IDs

    # Save and display annotated image
    annotated = results[0].plot()  # numpy array with boxes drawn
    st.image(annotated, caption="YOLOv8 Detection", use_column_width=True)