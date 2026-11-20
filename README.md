# Handwritten Digit Recognizer

A **Convolutional Neural Network (CNN)** trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset to recognize handwritten digits (0–9). Built with Python and TensorFlow/Keras.

## Project Structure

```
handwritten_digit_recogniser/
├── main.py                 # Entry point (train / evaluate / predict)
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── models/                 # Saved model & training plots (after training)
│   └── .gitkeep
└── src/
    ├── config.py           # Hyperparameters & paths
    ├── data_loader.py      # MNIST loading & preprocessing
    ├── model.py            # CNN architecture
    ├── train.py            # Training pipeline
    ├── evaluate.py         # Test set evaluation
    └── predict.py          # Inference on new images
```

## Features

- CNN with two convolutional blocks, dropout, and Adam optimizer
- Automatic MNIST download via Keras
- Early stopping and learning-rate scheduling
- Training history plots saved to `models/`
- Per-digit accuracy reporting
- Single-image and batch prediction API

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- NumPy, Matplotlib

## Installation

```bash


# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python main.py --train
```

This downloads MNIST, trains the CNN for up to 10 epochs, saves the model to `models/digit_cnn.keras`, and writes `models/training_history.png`.

### Evaluate on test set

```bash
python main.py --evaluate
```

### Run prediction demo

```bash
python main.py --predict
```

### Train and evaluate in one run

```bash
python main.py --train --evaluate
```

### Use prediction in your own code

```python
import numpy as np
from src.predict import predict_digit, load_trained_model

# Load a 28x28 grayscale image (values 0–255 or 0–1)
image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

result = predict_digit(image)
print(f"Digit: {result['digit']}, Confidence: {result['confidence']:.2%}")
```

## Model Architecture

| Layer        | Output Shape | Parameters |
|-------------|--------------|------------|
| Conv2D (32) | 28×28×32     | 320        |
| MaxPooling  | 14×14×32     | 0          |
| Conv2D (64) | 14×14×64     | 18,496     |
| MaxPooling  | 7×7×64       | 0          |
| Flatten     | 3136         | 0          |
| Dense (128) | 128          | 401,536    |
| Dropout     | 128          | 0          |
| Dense (10)  | 10           | 1,290      |

**Expected test accuracy:** ~98–99% after training.

## Configuration

Edit `src/config.py` to change:

- `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`
- Model save path
- Validation split


## Author

SUTEJ RAJ — [GitHub](https://github.com/SUTEJ23)
