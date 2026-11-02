import os
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import mixed_precision

from config import *

# ==========================================================
# GPU INITIALIZATION
# ==========================================================

def initialize_gpu():

    if ENABLE_XLA:
        tf.config.optimizer.set_jit(True)

    if USE_MIXED_PRECISION:
        mixed_precision.set_global_policy("mixed_float16")

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:

        try:

            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            print("\nGPU Detected:", gpus)

        except RuntimeError as e:

            print(e)

# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

def create_directories():

    folders = [

        MODEL_DIR,

        OUTPUT_DIR,

        PLOTS_DIR,

        GRADCAM_DIR,

        PREDICTION_DIR,

        LOG_DIR

    ]

    for folder in folders:

        os.makedirs(folder, exist_ok=True)

# ==========================================================
# LOAD IMAGE PATHS
# ==========================================================

def load_csv(csv_file):

    images = []

    labels = []

    with open(csv_file, "r") as f:

        for line in f:

            line = line.strip()

            image_path = os.path.join(BASE_DIR, "dataset", line)

            images.append(image_path)

            if "positive" in line.lower():

                labels.append(1)

            else:

                labels.append(0)

    return images, labels

# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_png(image, channels=3)

    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.05),

    tf.keras.layers.RandomZoom(0.10),

    tf.keras.layers.RandomContrast(0.10)

])

# ==========================================================
# CREATE DATASET
# ==========================================================

def create_dataset(images, labels, training=True):

    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:

        ds = ds.shuffle(5000, seed=SEED)

    ds = ds.map(

        preprocess,

        num_parallel_calls=tf.data.AUTOTUNE

    )

    if training:

        ds = ds.map(

            lambda x, y: (augmentation(x, training=True), y),

            num_parallel_calls=tf.data.AUTOTUNE

        )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# ==========================================================
# LOAD TRAIN + VALID
# ==========================================================

def get_datasets():

    train_images, train_labels = load_csv(TRAIN_CSV)

    valid_images, valid_labels = load_csv(VALID_CSV)

    train_ds = create_dataset(

        train_images,

        train_labels,

        training=True

    )

    valid_ds = create_dataset(

        valid_images,

        valid_labels,

        training=False

    )

    return train_ds, valid_ds

# ==========================================================
# PLOT HISTORY
# ==========================================================

def plot_history(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"])

    plt.plot(history.history["val_accuracy"])

    plt.title("Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend(["Train","Validation"])

    plt.savefig(ACCURACY_PLOT)

    plt.close()

    plt.figure(figsize=(8,5))

    plt.plot(history.history["loss"])

    plt.plot(history.history["val_loss"])

    plt.title("Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend(["Train","Validation"])

    plt.savefig(LOSS_PLOT)

    plt.close()