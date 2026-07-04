"""Preprocess canvas drawings to match MNIST format."""

import numpy as np

from src.config import INPUT_SHAPE


def preprocess_canvas_pixels(pixels: list | np.ndarray) -> np.ndarray:
    """
    Convert a 28x28 canvas drawing to MNIST-style input.

    Canvas uses black strokes on white background; MNIST uses white strokes
    on black background. Also centers the digit for better accuracy.
    """
    image = np.asarray(pixels, dtype="float32").reshape(28, 28)

    if image.max() > 1.0:
        image = image / 255.0

    # Invert: white digit on black background (MNIST style)
    image = 1.0 - image
    image = _center_digit(image)
    image = np.clip(image, 0.0, 1.0)
    image = np.expand_dims(image, axis=-1)

    if image.shape != INPUT_SHAPE:
        raise ValueError(f"Expected shape {INPUT_SHAPE}, got {image.shape}")

    return image


def _center_digit(image: np.ndarray) -> np.ndarray:
    """Crop to digit bounding box and center in a 28x28 canvas."""
    rows = np.where(image.max(axis=1) > 0.05)[0]
    cols = np.where(image.max(axis=0) > 0.05)[0]

    if len(rows) == 0 or len(cols) == 0:
        return image

    cropped = image[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    height, width = cropped.shape

    max_dim = max(height, width)
    padded = np.zeros((max_dim, max_dim), dtype="float32")
    row_offset = (max_dim - height) // 2
    col_offset = (max_dim - width) // 2
    padded[row_offset : row_offset + height, col_offset : col_offset + width] = cropped

    from PIL import Image

    resized = np.array(
        Image.fromarray((padded * 255).astype("uint8")).resize((20, 20), Image.LANCZOS),
        dtype="float32",
    ) / 255.0

    centered = np.zeros((28, 28), dtype="float32")
    offset = (28 - 20) // 2
    centered[offset : offset + 20, offset : offset + 20] = resized
    return centered
