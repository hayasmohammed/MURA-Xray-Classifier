import os

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "MURA-v1.1")

TRAIN_CSV = os.path.join(DATASET_DIR, "train_image_paths.csv")
VALID_CSV = os.path.join(DATASET_DIR, "valid_image_paths.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")

PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")

LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ==========================================================
# TRAINING SETTINGS
# ==========================================================

IMAGE_SIZE = (224, 224)

CHANNELS = 3

BATCH_SIZE = 16

EPOCHS_HEAD = 5

EPOCHS_FINE = 10

LEARNING_RATE = 1e-4

FINE_TUNE_LR = 1e-5

SEED = 42

NUM_CLASSES = 1

# ==========================================================
# MODEL
# ==========================================================

MODEL_NAME = "DenseNet121"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_model.keras"
)

# ==========================================================
# GPU
# ==========================================================

USE_MIXED_PRECISION = True

ENABLE_XLA = False

# ==========================================================
# OUTPUT
# ==========================================================

ACCURACY_PLOT = os.path.join(
    PLOTS_DIR,
    "accuracy.png"
)

LOSS_PLOT = os.path.join(
    PLOTS_DIR,
    "loss.png"
)