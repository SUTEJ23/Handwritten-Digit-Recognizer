"""Project configuration and hyperparameters."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "digit_cnn.keras"

# Dataset
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

# Training hyperparameters
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42
