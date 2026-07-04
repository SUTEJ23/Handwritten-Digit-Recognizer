"""Training pipeline for the digit recognition CNN."""

from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow import keras

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_SEED,
    VALIDATION_SPLIT,
)
from src.data_loader import get_data_summary, load_mnist_data
from src.model import build_cnn_model, get_model_summary


def train_model(save_plot: bool = True) -> tuple[keras.Model, keras.callbacks.History]:
    """
    Load data, build model, train, and save weights.

    Args:
        save_plot: Whether to save training history plot to models/.

    Returns:
        Tuple of (trained_model, training_history).
    """
    keras.utils.set_random_seed(RANDOM_SEED)

    print("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist_data()
    get_data_summary(x_train, y_train, x_test, y_test)

    print("\nBuilding CNN model...")
    model = build_cnn_model()
    get_model_summary(model)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    print(f"\nTraining for up to {EPOCHS} epochs...")
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    if save_plot:
        _plot_training_history(history)

    return model, history


def _plot_training_history(history: keras.callbacks.History) -> None:
    """Save accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = MODEL_DIR / "training_history.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training plot saved to: {plot_path}")
