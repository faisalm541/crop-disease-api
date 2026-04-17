from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("smart_krishi_vision_model.h5")

class_names = [
    "Corn_Blight",
    "Corn_Rust",
    "Corn_Healthy",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_Healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Mosaic",
    "Tomato_Yellow_Leaf",
    "Tomato_Healthy"
]

solutions = {
    "Potato_Early_blight": "Use fungicide (Mancozeb). Remove infected leaves. Avoid overhead watering.",
    "Potato_Late_blight": "Apply copper-based fungicide. Improve air circulation.",
    "Potato_Healthy": "Your crop is healthy.",

    "Corn_Blight": "Use resistant varieties. Apply fungicide.",
    "Corn_Rust": "Use fungicide (Triazole). Monitor humidity.",
    "Corn_Healthy": "Your crop is healthy.",

    "Tomato_Bacterial_spot": "Use copper spray.",
    "Tomato_Early_blight": "Apply fungicide. Remove infected leaves.",
    "Tomato_Late_blight": "Use fungicide. Avoid moisture.",
    "Tomato_Leaf_Mold": "Increase ventilation.",
    "Tomato_Septoria": "Remove infected leaves.",
    "Tomato_Spider_mites": "Use neem oil.",
    "Tomato_Target_Spot": "Apply fungicide.",
    "Tomato_Mosaic": "Remove infected plants.",
    "Tomato_Yellow_Leaf": "Control whiteflies.",
    "Tomato_Healthy": "Your crop is healthy."
}

IMG_SIZE = 224

def format_name(name):
    return name.replace("_", " ")

def preprocess(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return "Disease API Running 🚀"

@app.route("/predict-disease", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image provided"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "status": "error",
                "message": "Empty file selected"
            }), 400

        image = Image.open(file).convert("RGB")
        processed = preprocess(image)

        preds = model.predict(processed)[0]
        top_index = np.argmax(preds)
        confidence = preds[top_index]

        if confidence < 0.5:
            return jsonify({
                "status": "not_recognized",
                "message": "Unable to recognize the disease. Upload a clear crop image."
            })

        full_label = class_names[top_index]
        parts = full_label.split("_", 1)

        crop = parts[0]
        disease = parts[1] if len(parts) > 1 else "Healthy"

        return jsonify({
            "crop": crop,
            "disease": format_name(disease),
            "solution": solutions.get(full_label, "Consult agricultural expert")
        })

    except Exception:
        return jsonify({
            "status": "error",
            "message": "Invalid image or server error"
        }), 500


# 🔥 IMPORTANT FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)