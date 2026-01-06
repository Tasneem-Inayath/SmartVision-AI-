import sys, os
sys.path.append(os.path.abspath("."))
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils.metrics import load_model_metrics

st.title("📊 Model Performance Dashboard")
st.caption("Comparison of CNN architectures used in SmartVision")

df = load_model_metrics()

st.subheader("📋 Overall Performance Summary")
st.dataframe(df, use_container_width=True)

# Highlight best model
best_model = df.loc[df["Accuracy"].idxmax()]

st.success(
    f"🏆 Best Model: **{best_model['Model']}** "
    f"(Accuracy: {best_model['Accuracy']:.2f})"
)

st.divider()

# Accuracy Bar Chart
st.subheader("📈 Accuracy Comparison")

fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.barplot(
    data=df,
    x="Model",
    y="Accuracy",
    ax=ax1
)
ax1.set_ylim(0, 1)
ax1.set_title("Model Accuracy")

st.pyplot(fig1)

# Inference Speed
st.subheader("⚡ Inference Time Comparison")

fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.barplot(
    data=df,
    x="Model",
    y="Inference Time (sec)",
    ax=ax2
)
ax2.set_title("Inference Time (Lower is Better)")

st.pyplot(fig2)

st.divider()

st.subheader("🧠 Technical Observations")

st.markdown("""
- **MobileNetV2** offers the best balance between accuracy and inference speed  
- Heavy models (VGG16, ResNet50) underperform on limited datasets  
- EfficientNet requires larger datasets and advanced tuning  
- Lightweight architectures generalize better for real-time systems  
""")
