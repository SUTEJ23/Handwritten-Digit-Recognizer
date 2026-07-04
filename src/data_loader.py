"""Load and preprocess the MNIST dataset."""

import numpy as np
from tensorflow import keras

from src.config import INPUT_SHAPE, RANDOM_SEED


def load_mnist_data():
    """
    Download and load the MNIST dataset.

    Returns:
        tuple: (x_train, y_train, x_test, y_test) with normalized images
               and one-hot encoded labels.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = _preprocess_images(x_train)
    x_test = _preprocess_images(x_test)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def _preprocess_images(images: np.ndarray) -> np.ndarray:
    """Normalize pixel values and reshape for CNN input."""
    images = images.astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1)
    return images


def get_data_summary(x_train, y_train, x_test, y_test) -> None:
    """Print dataset statistics."""
    print("=" * 50)
    print("MNIST Dataset Summary")
    print("=" * 50)
    print(f"Training samples:   {x_train.shape[0]:,}")
    print(f"Test samples:       {x_test.shape[0]:,}")
    print(f"Image shape:        {INPUT_SHAPE}")
    print(f"Training labels:    {y_train.shape}")
    print(f"Test labels:        {y_test.shape}")
    print("=" * 50)
