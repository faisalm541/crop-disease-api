from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model safely (IMPORTANT FIX)
try:
    model = load_model("smart_krishi_vision_model.h5", compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", str(e))
    model = None


@app.route("/")
def home():
    return "API is running 🚀"


@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Accept both 'image' and 'file'
        if "image" in request.files:
            file = request.files["image"]
        elif "file" in request.files:
            file = request.files["file"]
        else:
            return jsonify({"error": "No file provided"}), 400

        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        # Preprocess image (adjust size if needed)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)
        predicted_class = int(np.argmax(prediction))

        return jsonify({
            "status": "success",
            "prediction": predicted_class
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
