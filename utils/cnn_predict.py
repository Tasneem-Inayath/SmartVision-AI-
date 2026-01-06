import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input #type: ignore

IMG_SIZE = 224

# --------------------------------------------------
# Load class names EXACTLY from training dataset
# --------------------------------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory( #type: ignore
    "smartvision_dataset/classification/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    shuffle=False
)
cnn_model = tf.keras.models.load_model("models/mobilenet_finetuned_final.h5") #type: ignore

CLASS_NAMES = train_ds.class_names

# --------------------------------------------------
# Load all CNN models
# ------------------------------------------------__
MODELS = {
    "MobileNetV2": load_model("models/mobilenet_finetuned_final.h5"),
    "EfficientNet": load_model("models/efficientnetb0_best_model.h5"),
    "ResNet50": load_model("models/resnet50_best_model.h5"),
    "VGG16": load_model("models/vgg16_best_model.h5"),
}
def predict_with_cnn(image_np):
    img = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    preds = cnn_model.predict(img, verbose=0)[0]

    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return CLASS_NAMES[class_id], confidence
# --------------------------------------------------
# Predict function (shared)
# --------------------------------------------------
def predict_with_model(model, image_np, top_k=5):
    img = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:top_k]

    return [(CLASS_NAMES[i], float(preds[i])) for i in top_idx]


# --------------------------------------------------
# Predict with ALL models
# --------------------------------------------------
def predict_with_all_models(image_np, top_k=5):
    results = {}
    for name, model in MODELS.items():
        results[name] = predict_with_model(model, image_np, top_k)
    return results
