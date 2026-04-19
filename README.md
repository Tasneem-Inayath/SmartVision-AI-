# 👁️ SmartVision – Intelligent Visual Understanding Platform

SmartVision is an end-to-end **Computer Vision application** that integrates  
**image classification** and **object detection** using deep learning.

The system combines **CNN-based classification models** with **YOLOv8 object detection**  
and is deployed as an interactive **Streamlit web application**.

---

## 🚀 Key Features

- 📸 Image Classification (25 classes)
- 🎯 Object Detection with YOLOv8
- 🔄 Hybrid CNN + YOLO inference pipeline
- 📊 Model performance comparison dashboard
- 🎥 Live webcam detection (optional)
- ☁️ Deployed on Hugging Face Spaces

---

## 🧠 Models Used

### 🔹 Classification Models
- MobileNetV2 (Fine-tuned) ✅ **Best performing**
- VGG16
- ResNet50
- EfficientNetB0

### 🔹 Detection Model
- YOLOv8 (custom trained on COCO 25-class subset)

---

## 📂 Dataset

- Source: **COCO Dataset**
- Classes: 25 common objects  
  (`person, car, bicycle, dog, cat, airplane, bus, truck, bottle, chair, ...`)
- Balanced test set used for evaluation

---

## 📊 Model Performance Summary

| Model            | Accuracy | Inference Time |
|------------------|----------|----------------|
| MobileNetV2      | **59%**  | ~10 sec |
| VGG16            | 34%      | ~4.5 sec |
| ResNet50         | 24%      | ~9 sec |
| EfficientNetB0   | 4%       | ~14 sec |
| YOLOv8 (Detection) | mAP@0.5 ≈ 0.78 | Real-time |

> 📌 MobileNetV2 was selected as the **primary CNN** due to best accuracy–speed tradeoff.

---

## 🧪 Application Pages

1. **Home** – Project overview and workflow  
2. **Image Classification** – Single image classification with CNN models  
3. **Object Detection** – YOLO detection with confidence control  
4. **Model Performance** – Metrics, accuracy comparison, charts  
5. **Live Webcam** – Real-time detection (optional)  
6. **About** – Project & technical details  

---
#Link for Review
[Link](https://huggingface.co/spaces/TasneemFirdhosh/smartvision_ai)
---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Streamlit
- NumPy, Pandas, Matplotlib

---

## 🖥️ Run Locally

### 1️⃣ Create virtual environment (optional)
```powershell
python -m venv venv
venv\Scripts\activate
# 👁️ SmartVision – Intelligent Visual Understanding Platform

SmartVision is an end-to-end **Computer Vision application** that integrates  
**image classification** and **object detection** using deep learning.

The system combines **CNN-based classification models** with **YOLOv8 object detection**  
and is deployed as an interactive **Streamlit web application**.

---

## 🚀 Key Features

- 📸 Image Classification (25 classes)
- 🎯 Object Detection with YOLOv8
- 🔄 Hybrid CNN + YOLO inference pipeline
- 📊 Model performance comparison dashboard
- 🎥 Live webcam detection (optional)
- ☁️ Deployed on Hugging Face Spaces

---

## 🧠 Models Used

### 🔹 Classification Models
- MobileNetV2 (Fine-tuned) ✅ **Best performing**
- VGG16
- ResNet50
- EfficientNetB0

### 🔹 Detection Model
- YOLOv8 (custom trained on COCO 25-class subset)

---

## 📂 Dataset

- Source: **COCO Dataset**
- Classes: 25 common objects  
  (`person, car, bicycle, dog, cat, airplane, bus, truck, bottle, chair, ...`)
- Balanced test set used for evaluation

---

## 📊 Model Performance Summary

| Model            | Accuracy | Inference Time |
|------------------|----------|----------------|
| MobileNetV2      | **59%**  | ~10 sec |
| VGG16            | 34%      | ~4.5 sec |
| ResNet50         | 24%      | ~9 sec |
| EfficientNetB0   | 4%       | ~14 sec |
| YOLOv8 (Detection) | mAP@0.5 ≈ 0.78 | Real-time |

> 📌 MobileNetV2 was selected as the **primary CNN** due to best accuracy–speed tradeoff.

---

## 🧪 Application Pages

1. **Home** – Project overview and workflow  
2. **Image Classification** – Single image classification with CNN models  
3. **Object Detection** – YOLO detection with confidence control  
4. **Model Performance** – Metrics, accuracy comparison, charts  
5. **Live Webcam** – Real-time detection (optional)  
6. **About** – Project & technical details  

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Streamlit
- NumPy, Pandas, Matplotlib

---

## 🖥️ Run Locally

### 1️⃣ Create virtual environment (optional)
```powershell
python -m venv venv
venv\Scripts\activate
2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run application
streamlit run app.py
