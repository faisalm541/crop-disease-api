from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
import io
import logging
import sys
import traceback

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Lazy-load model (prevents 502 on Render free tier) ───────
model = None

def get_model():
    global model
    if model is None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        logger.info(f"TF={tf.__version__}  Keras={tf.keras.__version__}")
        logger.info("Loading smart_krishi_vision_model.h5 ...")
        model = tf.keras.models.load_model(
            "smart_krishi_vision_model.h5",
            compile=False
        )
        logger.info(f"✅ Model loaded. Input shape: {model.input_shape}")
    return model


CLASS_MAP = {
    0: {"crop": "Corn (Maize)", "disease": "Cercospora Leaf Spot (Gray Leaf Spot)",
        "solution": "Apply fungicide containing strobilurin. Use resistant hybrids. Practice crop rotation."},

    1: {"crop": "Corn (Maize)", "disease": "Common Rust",
        "solution": "Apply foliar fungicide at early stage. Use resistant maize varieties."},

    2: {"crop": "Corn (Maize)", "disease": "Northern Leaf Blight",
        "solution": "Apply fungicide at early tassel stage. Rotate crops. Use resistant hybrids."},

    3: {"crop": "Corn (Maize)", "disease": "Healthy",
        "solution": "No treatment needed. Maintain proper nitrogen and irrigation."},

    4: {"crop": "Potato", "disease": "Early Blight",
        "solution": "Apply fungicide like chlorothalonil. Remove infected leaves. Practice crop rotation."},

    5: {"crop": "Potato", "disease": "Late Blight",
        "solution": "Apply metalaxyl immediately. Destroy infected plants. Avoid overhead watering."},

    6: {"crop": "Potato", "disease": "Healthy",
        "solution": "No treatment needed. Maintain proper soil moisture and spacing."},

    7: {"crop": "Tomato", "disease": "Bacterial Spot",
        "solution": "Apply copper-based bactericide. Avoid working in wet conditions."},

    8: {"crop": "Tomato", "disease": "Early Blight",
        "solution": "Apply fungicide like chlorothalonil. Remove lower infected leaves."},

    9: {"crop": "Tomato", "disease": "Late Blight",
        "solution": "Apply fungicide immediately. Remove infected plants."},

    10: {"crop": "Tomato", "disease": "Leaf Mold",
         "solution": "Improve ventilation. Apply fungicide like chlorothalonil."},

    11: {"crop": "Tomato", "disease": "Septoria Leaf Spot",
         "solution": "Apply fungicide. Remove infected leaves. Use mulch to prevent spread."},

    12: {"crop": "Tomato", "disease": "Spider Mites",
         "solution": "Apply neem oil or miticide. Increase humidity. Remove heavily infected leaves."},

    13: {"crop": "Tomato", "disease": "Target Spot",
         "solution": "Apply fungicide like azoxystrobin. Improve air circulation."},

    14: {"crop": "Tomato", "disease": "Yellow Leaf Curl Virus",
         "solution": "No cure. Remove infected plants. Control whiteflies."},

    15: {"crop": "Tomato", "disease": "Mosaic Virus",
         "solution": "No cure. Remove infected plants. Disinfect tools."},

    16: {"crop": "Tomato", "disease": "Healthy",
         "solution": "No treatment needed. Maintain balanced nutrients and watering."}
}

IMG_SIZE = 224


def preprocess(raw_bytes: bytes) -> np.ndarray:
    """Decode bytes → cv2 → resize → normalize → batch."""
    file_bytes = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm    = img_rgb.astype(np.float32) / 255.0
    return np.expand_dims(img_norm, axis=0)


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def home():
    return "Disease API Running 🚀", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "model_loaded":  model is not None,
        "total_classes": len(CLASS_MAP)
    }), 200


@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    try:
        # Accept both 'image' and 'file' keys for compatibility
        file = request.files.get("image") or request.files.get("file")
        if file is None:
            return jsonify({
                "status": "error",
                "error":  "No file provided. Send image as multipart/form-data with key 'image'."
            }), 400

        raw_bytes = file.read()
        if len(raw_bytes) == 0:
            return jsonify({"status": "error", "error": "Uploaded file is empty."}), 400

        logger.info(f"Image received: {len(raw_bytes) / 1024:.1f} KB")

        # Preprocess with cv2
        img_batch = preprocess(raw_bytes)
        if img_batch is None:
            return jsonify({
                "status": "error",
                "error":  "Invalid image. Could not decode. Send JPEG or PNG."
            }), 400

        # Predict
        m               = get_model()
        prediction      = m.predict(img_batch, verbose=0)
        predicted_class = int(np.argmax(prediction))
        confidence      = float(np.max(prediction)) * 100

        logger.info(f"Predicted class={predicted_class}  confidence={confidence:.1f}%")

        # Low confidence gate
        if confidence < 30.0:
            return jsonify({
                "status":     "not_recognized",
                "crop":       None,
                "disease":    None,
                "solution":   None,
                "confidence": round(confidence, 1),
                "message":    "Could not identify disease. Upload a clear, well-lit crop leaf image."
            }), 200

        # Map class index → labels
        label = CLASS_MAP.get(predicted_class)
        if label is None:
            logger.warning(f"Class index {predicted_class} not in CLASS_MAP")
            return jsonify({
                "status":     "not_recognized",
                "crop":       None,
                "disease":    None,
                "solution":   None,
                "confidence": round(confidence, 1)
            }), 200

        return jsonify({
            "status":     "success",
            "crop":       label["crop"],
            "disease":    label["disease"],
            "solution":   label["solution"],
            "confidence": round(confidence, 1)
        }), 200

    except MemoryError:
        traceback.print_exc()
        return jsonify({"status": "error", "error": "Server out of memory. Try a smaller image."}), 500
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
