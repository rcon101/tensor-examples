from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

from src.character_recognition.labels import LETTER_LABELS
from src.character_recognition.preprocessing import preprocess_image

APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / "models" / "character_model.keras"

app = Flask(__name__)
model = None


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            return None
        # Load once so each request does not pay TensorFlow startup cost.
        model = keras.models.load_model(MODEL_PATH)
    return model


@app.get("/")
def index():
    return render_template("index.html", has_model=MODEL_PATH.exists())


@app.post("/predict")
def predict():
    loaded_model = get_model()
    if loaded_model is None:
        return (
            jsonify(
                {
                    "error": "Model not found. Train it first with: python train.py --epochs 8"
                }
            ),
            503,
        )

    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return jsonify({"error": "Upload a JPG or PNG image."}), 400

    try:
        image = Image.open(upload.stream)
    except UnidentifiedImageError:
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    tensor = preprocess_image(image)
    probabilities = loaded_model.predict(tensor, verbose=0)[0]
    # Return the best label plus nearby alternatives for debugging misses.
    top_indices = np.argsort(probabilities)[::-1][:5]
    predictions = [
        {
            "label": LETTER_LABELS[index],
            "confidence": round(float(probabilities[index]) * 100, 2),
        }
        for index in top_indices
    ]

    return jsonify(
        {
            "prediction": predictions[0]["label"],
            "confidence": predictions[0]["confidence"],
            "top_predictions": predictions,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
