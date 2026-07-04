"""Evaluate the trained CNN on the MNIST test set."""

from pathlib import Path

import numpy as np
from tensorflow import keras

from src.config import MODEL_PATH
from src.data_loader import load_mnist_data


def evaluate_model(model_path: Path | str | None = None) -> dict:
    """
    Evaluate saved model on MNIST test data.

    Args:
        model_path: Path to saved .keras model. Uses default if None.

    Returns:
        Dictionary with loss, accuracy, and per-class accuracy.
    """
    path = Path(model_path) if model_path else MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run training first: python main.py --train"
        )

    print(f"Loading model from: {path}")
    model = keras.models.load_model(path)

    _, _, x_test, y_test = load_mnist_data()

    print("\nEvaluating on test set...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    predictions = model.predict(x_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    per_class_accuracy = {}
    for digit in range(10):
        mask = true_classes == digit
        if mask.sum() > 0:
            per_class_accuracy[digit] = (predicted_classes[mask] == digit).mean()

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nPer-digit accuracy:")
    for digit, acc in per_class_accuracy.items():
        print(f"  Digit {digit}: {acc * 100:.2f}%")
    print("=" * 50)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
    }


def show_sample_predictions(model_path: Path | str | None = None, num_samples: int = 10) -> None:
    """Display sample predictions with confidence scores."""
    import matplotlib.pyplot as plt

    path = Path(model_path) if model_path else MODEL_PATH
    model = keras.models.load_model(path)
    _, _, x_test, y_test = load_mnist_data()

    indices = np.random.choice(len(x_test), num_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        image = x_test[idx]
        true_label = np.argmax(y_test[idx])
        probs = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        pred_label = np.argmax(probs)
        confidence = probs[pred_label] * 100

        axes[i].imshow(image.squeeze(), cmap="gray")
        color = "green" if pred_label == true_label else "red"
        axes[i].set_title(
            f"True: {true_label} | Pred: {pred_label}\n({confidence:.1f}%)",
            color=color,
            fontsize=10,
        )
        axes[i].axis("off")

    plt.suptitle("Sample Predictions (Green = Correct, Red = Wrong)", fontsize=12)
    plt.tight_layout()
    plt.savefig(MODEL_PATH.parent / "sample_predictions.png", dpi=150)
    plt.show()
    print(f"Sample predictions saved to: {MODEL_PATH.parent / 'sample_predictions.png'}")
