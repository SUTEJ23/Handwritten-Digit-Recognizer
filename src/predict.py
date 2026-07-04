"""Run inference on single images or batches."""

from pathlib import Path

import numpy as np
from tensorflow import keras

from src.config import INPUT_SHAPE, MODEL_PATH


def load_trained_model(model_path: Path | str | None = None) -> keras.Model:
    """Load the saved CNN model."""
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train first: python main.py --train"
        )
    return keras.models.load_model(path)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess a single image for prediction.

    Accepts 28x28 grayscale arrays or flattened 784-length vectors.
    """
    image = np.asarray(image, dtype="float32")

    if image.max() > 1.0:
        image = image / 255.0

    if image.ndim == 1:
        image = image.reshape(28, 28)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    if image.shape != INPUT_SHAPE:
        raise ValueError(f"Expected shape {INPUT_SHAPE}, got {image.shape}")

    return np.expand_dims(image, axis=0)


def predict_digit(
    image: np.ndarray,
    model: keras.Model | None = None,
    model_path: Path | str | None = None,
) -> dict:
    """
    Predict the digit in a handwritten image.

    Args:
        image: 28x28 grayscale image (numpy array).
        model: Optional pre-loaded model.
        model_path: Path to model if model is not provided.

    Returns:
        Dictionary with predicted digit, confidence, and all class probabilities.
    """
    if model is None:
        model = load_trained_model(model_path)

    processed = preprocess_image(image)
    probabilities = model.predict(processed, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return {
        "digit": predicted_digit,
        "confidence": confidence,
        "probabilities": {i: float(probabilities[i]) for i in range(10)},
    }


def predict_batch(
    images: np.ndarray,
    model: keras.Model | None = None,
    model_path: Path | str | None = None,
) -> list[dict]:
    """Predict digits for a batch of images."""
    if model is None:
        model = load_trained_model(model_path)

    results = []
    for image in images:
        results.append(predict_digit(image, model=model))
    return results
