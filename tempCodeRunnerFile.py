"""
Handwritten Digit Recognizer - Main Entry Point

Train, evaluate, or run predictions using the MNIST CNN.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluate import evaluate_model, show_sample_predictions
from src.predict import load_trained_model, predict_digit
from src.train import train_model
from app import run_web_app


def demo_predict():
    """Run prediction demo on random MNIST test images."""
    import numpy as np

    from src.data_loader import load_mnist_data

    print("Loading model and test data...")
    model = load_trained_model()
    _, _, x_test, y_test = load_mnist_data()

    indices = np.random.choice(len(x_test), 5, replace=False)
    print("\n" + "=" * 50)
    print("Prediction Demo")
    print("=" * 50)

    for idx in indices:
        true_label = int(np.argmax(y_test[idx]))
        result = predict_digit(x_test[idx], model=model)
        status = "✓" if result["digit"] == true_label else "✗"
        print(
            f"{status} True: {true_label} | "
            f"Predicted: {result['digit']} | "
            f"Confidence: {result['confidence'] * 100:.1f}%"
        )
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Handwritten Digit Recognizer using CNN on MNIST"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the CNN model on MNIST",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the trained model on test set",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run prediction demo on sample images",
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Show sample predictions visualization (requires --evaluate)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the web app to draw digits and get predictions",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for web app (default: 5000)",
    )

    args = parser.parse_args()

    if not any([args.train, args.evaluate, args.predict, args.web]):
        parser.print_help()
        print("\nExample: python main.py --train --evaluate")
        sys.exit(0)

    if args.train:
        train_model()

    if args.evaluate:
        evaluate_model()
        if args.samples:
            show_sample_predictions()

    if args.predict:
        demo_predict()

    if args.web:
        print(f"Launching web application on port {args.port}...")
        run_web_app(port=args.port)


if __name__ == "__main__":
    main()