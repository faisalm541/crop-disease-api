from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os
import io
import traceback

app = Flask(__name__)

# ── Lazy-load model (NEVER load at module level on Render) ────
# Loading at module level causes Render to:
# 1. Hit 512MB RAM limit during startup → worker killed → 502
# 2. Timeout the health check → deploy fails → 500 on every request
model = None

def get_model():
    global model
    if model is None:
        # Suppress noisy TF logs
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf

        # Disable GPU (Render has none) — prevents TF from wasting memory searching
        tf.config.set_visible_devices([], "GPU")

        print("[Model] Loading smart_krishi_vision_model.h5 ...")
        model = tf.keras.models.load_model(
            "smart_krishi_vision_model.h5",
            compile=False   # inference only — skips optimizer, saves ~50MB RAM
        )
        print("[Model] Loaded successfully.")
    return model


# ── Class names ───────────────────────────────────────────────
CLASS_NAMES = [
    "Corn_Blight", "Corn_Rust", "Corn_Healthy",
    "Potato_Early_blight", "Potato_Late_blight", "Potato_Healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria", "Tomato_Spider_mites",
    "Tomato_Target_Spot", "Tomato_Mosaic", "Tomato_Yellow_Leaf",
    "Tomato_Healthy"
]

SOLUTIONS = {
    "Potato_Early_blight":    "Use fungicide (Mancozeb). Remove infected leaves. Avoid overhead watering.",
    "Potato_Late_blight":     "Apply copper-based fungicide. Improve air circulation.",
    "Potato_Healthy":         "Your crop is healthy! Keep monitoring regularly.",
    "Corn_Blight":            "Use resistant varieties. Apply fungicide at early signs.",
    "Corn_Rust":              "Use fungicide (Triazole). Monitor humidity levels.",
    "Corn_Healthy":           "Your crop is healthy! Keep monitoring regularly.",
    "Tomato_Bacterial_spot":  "Use copper spray. Remove infected plant debris.",
    "Tomato_Early_blight":    "Apply fungicide. Remove infected leaves immediately.",
    "Tomato_Late_blight":     "Use fungicide. Avoid overhead watering.",
    "Tomato_Leaf_Mold":       "Increase ventilation. Reduce leaf wetness.",
    "Tomato_Septoria":        "Remove infected leaves. Apply fungicide.",
    "Tomato_Spider_mites":    "Use neem oil or miticide spray.",
    "Tomato_Target_Spot":     "Apply fungicide. Remove crop debris.",
    "Tomato_Mosaic":          "Remove and destroy infected plants. Control aphids.",
    "Tomato_Yellow_Leaf":     "Control whiteflies. Use reflective mulch.",
    "Tomato_Healthy":         "Your crop is healthy! Keep monitoring regularly.",
}

IMG_SIZE = 224


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize, normalize and batch the image."""
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr   = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)


# ── Health check ──────────────────────────────────────────────
@app.route("/")
def home():
    return "Disease API Running 🚀", 200


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ── Main prediction endpoint ──────────────────────────────────
@app.route("/predict-disease", methods=["POST"])
def predict():
    try:
        # 1. Validate file presence
        if "image" not in request.files:
            return jsonify({
                "status":  "error",
                "message": "No image provided. Send field name: 'image'"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "status":  "error",
                "message": "Empty file selected"
            }), 400

        # 2. Read image safely
        raw_bytes = file.read()
        if len(raw_bytes) == 0:
            return jsonify({
                "status":  "error",
                "message": "Uploaded file is empty"
            }), 400

        try:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            return jsonify({
                "status":  "error",
                "message": "Cannot open image. Send a valid JPEG or PNG."
            }), 400

        # 3. Preprocess
        processed = preprocess(image)

        # 4. Run inference
        m    = get_model()
        preds = m.predict(processed, verbose=0)[0]   # verbose=0 silences progress bar

        top_index  = int(np.argmax(preds))
        confidence = float(preds[top_index])

        # 5. Confidence gate
        if confidence < 0.5:
            return jsonify({
                "status":     "not_recognized",
                "confidence": round(confidence, 4),
                "message":    "Could not identify disease. Upload a clear, well-lit crop leaf image."
            }), 200

        # 6. Build response
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
        # Render free tier OOM — tell the caller clearly
        traceback.print_exc()
        return jsonify({
            "status":  "error",
            "message": "Server ran out of memory. Try a smaller image."
        }), 500

    except Exception as e:
        # Print FULL traceback to Render logs so you can debug
        traceback.print_exc()
        return jsonify({
            "status":  "error",
            "message": f"Server error: {str(e)}"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
