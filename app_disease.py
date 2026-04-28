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


# ── 38-class label map ────────────────────────────────────────
CLASS_MAP = {
    0:  {"crop": "Apple",       "disease": "Apple Scab",             "solution": "Apply fungicide containing captan or myclobutanil. Remove and destroy infected leaves. Avoid overhead irrigation."},
    1:  {"crop": "Apple",       "disease": "Black Rot",              "solution": "Prune infected branches. Apply copper-based fungicide. Ensure good air circulation around the tree."},
    2:  {"crop": "Apple",       "disease": "Cedar Apple Rust",       "solution": "Apply fungicide at early pink stage. Remove nearby juniper trees if possible. Use resistant apple varieties."},
    3:  {"crop": "Apple",       "disease": "Healthy",                "solution": "No treatment needed. Maintain regular watering and balanced fertilization."},
    4:  {"crop": "Blueberry",   "disease": "Healthy",                "solution": "No treatment needed. Ensure acidic soil pH between 4.5 and 5.5."},
    5:  {"crop": "Cherry",      "disease": "Powdery Mildew",         "solution": "Apply sulfur-based fungicide. Improve air circulation. Avoid excess nitrogen fertilization."},
    6:  {"crop": "Cherry",      "disease": "Healthy",                "solution": "No treatment needed. Maintain proper pruning and irrigation schedule."},
    7:  {"crop": "Corn (Maize)","disease": "Cercospora Leaf Spot",   "solution": "Apply fungicide containing strobilurin. Use resistant hybrids. Practice crop rotation."},
    8:  {"crop": "Corn (Maize)","disease": "Common Rust",            "solution": "Apply foliar fungicide at early rust development. Use resistant maize varieties."},
    9:  {"crop": "Corn (Maize)","disease": "Northern Leaf Blight",   "solution": "Apply fungicide at early tassel stage. Rotate crops. Use resistant hybrids."},
    10: {"crop": "Corn (Maize)","disease": "Healthy",                "solution": "No treatment needed. Ensure adequate nitrogen supply for healthy growth."},
    11: {"crop": "Grape",       "disease": "Black Rot",              "solution": "Apply fungicide from bud break. Remove mummified fruit. Prune for air circulation."},
    12: {"crop": "Grape",       "disease": "Esca (Black Measles)",   "solution": "No curative treatment available. Remove infected wood. Apply wound sealant after pruning."},
    13: {"crop": "Grape",       "disease": "Leaf Blight",            "solution": "Apply copper-based fungicide. Remove infected leaves. Avoid overhead irrigation."},
    14: {"crop": "Grape",       "disease": "Healthy",                "solution": "No treatment needed. Maintain regular canopy management."},
    15: {"crop": "Orange",      "disease": "Citrus Greening",        "solution": "No cure available. Remove infected trees. Control psyllid vector with insecticide."},
    16: {"crop": "Peach",       "disease": "Bacterial Spot",         "solution": "Apply copper-based bactericide. Avoid overhead irrigation. Use resistant varieties."},
    17: {"crop": "Peach",       "disease": "Healthy",                "solution": "No treatment needed. Ensure good drainage and balanced fertilization."},
    18: {"crop": "Bell Pepper", "disease": "Bacterial Spot",         "solution": "Apply copper hydroxide spray. Avoid working in wet fields. Use disease-free seeds."},
    19: {"crop": "Bell Pepper", "disease": "Healthy",                "solution": "No treatment needed. Maintain consistent moisture and adequate calcium supply."},
    20: {"crop": "Potato",      "disease": "Early Blight",           "solution": "Apply fungicide containing chlorothalonil or mancozeb. Remove infected leaves. Practice crop rotation."},
    21: {"crop": "Potato",      "disease": "Late Blight",            "solution": "Apply metalaxyl or cymoxanil fungicide immediately. Destroy infected plants. Avoid overhead irrigation."},
    22: {"crop": "Potato",      "disease": "Healthy",                "solution": "No treatment needed. Hill soil around plants and maintain consistent watering."},
    23: {"crop": "Raspberry",   "disease": "Healthy",                "solution": "No treatment needed. Prune old canes after harvest."},
    24: {"crop": "Soybean",     "disease": "Healthy",                "solution": "No treatment needed. Ensure proper inoculation with Rhizobium bacteria."},
    25: {"crop": "Squash",      "disease": "Powdery Mildew",         "solution": "Apply potassium bicarbonate or sulfur spray. Improve air circulation. Avoid overhead watering."},
    26: {"crop": "Strawberry",  "disease": "Leaf Scorch",            "solution": "Apply fungicide containing captan. Remove infected leaves. Ensure good drainage."},
    27: {"crop": "Strawberry",  "disease": "Healthy",                "solution": "No treatment needed. Mulch around plants to retain moisture."},
    28: {"crop": "Tomato",      "disease": "Bacterial Spot",         "solution": "Apply copper-based bactericide. Avoid working in wet conditions. Use certified disease-free seeds."},
    29: {"crop": "Tomato",      "disease": "Early Blight",           "solution": "Apply chlorothalonil fungicide. Remove lower infected leaves. Practice crop rotation."},
    30: {"crop": "Tomato",      "disease": "Late Blight",            "solution": "Apply metalaxyl immediately. Destroy infected plants. Avoid overhead irrigation."},
    31: {"crop": "Tomato",      "disease": "Leaf Mold",              "solution": "Improve greenhouse ventilation. Apply fungicide containing chlorothalonil."},
    32: {"crop": "Tomato",      "disease": "Septoria Leaf Spot",     "solution": "Apply fungicide at first sign. Remove infected leaves. Mulch to prevent soil splash."},
    33: {"crop": "Tomato",      "disease": "Spider Mites",           "solution": "Apply miticide or neem oil. Increase humidity. Remove heavily infested leaves."},
    34: {"crop": "Tomato",      "disease": "Target Spot",            "solution": "Apply fungicide containing azoxystrobin. Improve air circulation. Rotate crops."},
    35: {"crop": "Tomato",      "disease": "Yellow Leaf Curl Virus", "solution": "No cure. Remove infected plants. Control whitefly vector. Use resistant varieties."},
    36: {"crop": "Tomato",      "disease": "Mosaic Virus",           "solution": "No cure. Remove infected plants. Disinfect tools. Use virus-free seeds."},
    37: {"crop": "Tomato",      "disease": "Healthy",                "solution": "No treatment needed. Maintain consistent watering and calcium supply."},
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
