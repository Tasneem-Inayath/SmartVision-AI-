import streamlit as st

st.title("Live Webcam Detection")

st.warning("Webcam functionality depends on deployment environment.")

st.markdown("""
### Supported Scenarios

**Local Machine**
- OpenCV webcam
- Real-time YOLO inference

**Cloud Deployment**
- Streamlit camera input
- Web-based image capture
""")

st.info("This module is optional and environment-dependent.")
