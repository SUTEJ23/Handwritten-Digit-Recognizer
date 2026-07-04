"""Convolutional Neural Network architecture for digit recognition."""

from tensorflow import keras
from tensorflow.keras import layers

from src.config import INPUT_SHAPE, LEARNING_RATE, NUM_CLASSES


def build_cnn_model() -> keras.Model:
    """
    Build a CNN model for MNIST digit classification.

    Architecture:
        Conv2D -> MaxPooling -> Conv2D -> MaxPooling ->
        Flatten -> Dense -> Dropout -> Output

    Returns:
        Compiled Keras model ready for training.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="digit_cnn",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_summary(model: keras.Model) -> None:
    """Print model architecture and parameter count."""
    print("\n" + "=" * 50)
    print("CNN Model Architecture")
    print("=" * 50)
    model.summary()
