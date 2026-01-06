import sys, os
sys.path.append(os.path.abspath("."))
import streamlit as st
import numpy as np
from PIL import Image
from utils.cnn_predict import predict_with_all_models

st.title("Image Classification")
st.caption("Single-object classification — CNN model comparison")

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    with col2:
        st.subheader("Model Predictions")

        with st.spinner("Running inference on all models..."):
            results = predict_with_all_models(image_np)

        for model_name, preds in results.items():
            st.markdown(f"### {model_name}")
            for label, conf in preds:
                st.progress(conf)
                st.write(f"**{label}** — {conf:.2f}")
