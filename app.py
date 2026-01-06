import streamlit as st

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
