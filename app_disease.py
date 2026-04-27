from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os
import io
import traceback

app = Flask(__name__)
model = None

def get_model():
    global model
    if model is None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")

        print(f"[Model] TF={tf.__version__}  Keras={tf.keras.__version__}")
        print("[Model] Loading smart_krishi_vision_model.h5 ...")

        # Try 1: Direct load (works if TF version matches save version)
        try:
            model = tf.keras.models.load_model(
                "smart_krishi_vision_model.h5",
                compile=False
            )
            print("[Model] ✅ Loaded successfully.")
            return model
        except Exception as e1:
            print(f"[Model] ❌ Direct load failed: {e1}")

        # Try 2: Load via SavedModel format (most version-agnostic)
        try:
            model = tf.saved_model.load("smart_krishi_saved_model")
            print("[Model] ✅ Loaded via SavedModel format.")
            return model
        except Exception as e2:
            print(f"[Model] ❌ SavedModel load failed: {e2}")

        # Try 3: weights-only — rebuild architecture manually then load weights
        try:
            print("[Model] Attempting weights-only load with hardcoded MobileNetV2 base...")
            base = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights=None
            )
            x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
            x = tf.keras.layers.Dense(16, activation="softmax")(x)
            model = tf.keras.Model(inputs=base.input, outputs=x)
            model.load_weights("smart_krishi_vision_model_weights.h5")
            print("[Model] ✅ Loaded via weights-only fallback.")
            return model
        except Exception as e3:
            print(f"[Model] ❌ Weights-only load failed: {e3}")

        raise RuntimeError(
            "All model loading attempts failed. "
            "Please run convert_model.py locally and re-upload the converted model."
        )

    return model


CLASS_NAMES = [
    "Corn_Blight", "Corn_Rust", "Corn_Healthy",
    "Potato_Early_blight", "Potato_Late_blight", "Potato_Healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria", "Tomato_Spider_mites",
    "Tomato_Target_Spot", "Tomato_Mosaic", "Tomato_Yellow_Leaf",
    "Tomato_Healthy"
]

SOLUTIONS = {
    "Potato_Early_blight":   "Use fungicide (Mancozeb). Remove infected leaves. Avoid overhead watering.",
    "Potato_Late_blight":    "Apply copper-based fungicide. Improve air circulation.",
    "Potato_Healthy":        "Your crop is healthy! Keep monitoring regularly.",
    "Corn_Blight":           "Use resistant varieties. Apply fungicide at early signs.",
    "Corn_Rust":             "Use fungicide (Triazole). Monitor humidity levels.",
    "Corn_Healthy":          "Your crop is healthy! Keep monitoring regularly.",
    "Tomato_Bacterial_spot": "Use copper spray. Remove infected plant debris.",
    "Tomato_Early_blight":   "Apply fungicide. Remove infected leaves immediately.",
    "Tomato_Late_blight":    "Use fungicide. Avoid overhead watering.",
    "Tomato_Leaf_Mold":      "Increase ventilation. Reduce leaf wetness.",
    "Tomato_Septoria":       "Remove infected leaves. Apply fungicide.",
    "Tomato_Spider_mites":   "Use neem oil or miticide spray.",
    "Tomato_Target_Spot":    "Apply fungicide. Remove crop debris.",
    "Tomato_Mosaic":         "Remove and destroy infected plants. Control aphids.",
    "Tomato_Yellow_Leaf":    "Control whiteflies. Use reflective mulch.",
    "Tomato_Healthy":        "Your crop is healthy! Keep monitoring regularly.",
}

IMG_SIZE = 224


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr   = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route("/")
def home():
    return "Disease API Running 🚀", 200

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict-disease", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"status": "error",
                            "message": "No image provided. Field name must be 'image'."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "Empty file selected."}), 400

        raw_bytes = file.read()
        if len(raw_bytes) == 0:
            return jsonify({"status": "error", "message": "Uploaded file is empty."}), 400

        try:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            return jsonify({"status": "error",
                            "message": "Cannot open image. Send a valid JPEG or PNG."}), 400

        processed = preprocess(image)
        m         = get_model()
        preds     = m.predict(processed, verbose=0)[0]

        top_index  = int(np.argmax(preds))
        confidence = float(preds[top_index])

        if confidence < 0.5:
            return jsonify({
                "status":     "not_recognized",
                "confidence": round(confidence, 4),
                "message":    "Could not identify disease. Upload a clear, well-lit crop leaf image."
            }), 200

        full_label = CLASS_NAMES[top_index]
        parts      = full_label.split("_", 1)
        crop       = parts[0]
        disease    = parts[1].replace("_", " ") if len(parts) > 1 else "Healthy"

        return jsonify({
            "crop":       crop,
            "disease":    disease,
            "solution":   SOLUTIONS.get(full_label, "Consult an agricultural expert."),
            "confidence": round(confidence, 4),
            "label":      full_label
        }), 200

    except MemoryError:
        traceback.print_exc()
        return jsonify({"status": "error",
                        "message": "Server out of memory. Try a smaller image."}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
