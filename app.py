"""Flask web app for handwritten digit recognition."""

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MODEL_PATH
from src.image_utils import preprocess_canvas_pixels
from src.predict import load_trained_model, predict_digit

app = Flask(__name__)
model = None


def get_model():
    """Load model once and reuse across requests."""
    global model
    if model is None:
        model = load_trained_model()
    return model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "pixels" not in data:
            return jsonify({"error": "Missing pixel data"}), 400

        pixels = data["pixels"]
        if len(pixels) != 784:
            return jsonify({"error": "Expected 784 pixels (28x28)"}), 400

        image = preprocess_canvas_pixels(pixels)
        result = predict_digit(image, model=get_model())

        return jsonify(
            {
                "digit": result["digit"],
                "confidence": round(result["confidence"] * 100, 2),
                "probabilities": {
                    str(k): round(v * 100, 2) for k, v in result["probabilities"].items()
                },
            }
        )
    except FileNotFoundError:
        return jsonify(
            {
                "error": "Model not found. Run: python main.py --train",
            }
        ), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})


# === FIXED SECTION: Added the runner function wrapper ===
def run_web_app(port=5000):
    """Function called from main.py to pass the dynamic port configuration."""
    app.run(debug=True, host="0.0.0.0", port=port)


if __name__ == "__main__":
    # Fallback to keep backwards compatibility if app.py is executed directly
    run_web_app()