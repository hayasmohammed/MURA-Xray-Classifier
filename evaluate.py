import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc
)

from config import *
from utils import *

print("=" * 50)
print("Loading Model...")
print("=" * 50)

model = tf.keras.models.load_model(BEST_MODEL_PATH)

_, valid_ds = get_datasets()

y_true = []
y_pred = []
y_prob = []

print("\nRunning predictions...\n")

for images, labels in valid_ds:

    probs = model.predict(images, verbose=0)

    preds = (probs > 0.5).astype(int)

    y_true.extend(labels.numpy())

    y_pred.extend(preds.flatten())

    y_prob.extend(probs.flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

print("\nClassification Report\n")

print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Abnormal"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(
    OUTPUT_DIR + "/confusion_matrix.png",
    dpi=300
)

plt.close()

fpr, tpr, _ = roc_curve(y_true, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    OUTPUT_DIR + "/roc_curve.png",
    dpi=300
)

plt.close()

loss, accuracy, auc_score = model.evaluate(valid_ds)

print("\nValidation Accuracy :", accuracy)

print("Validation AUC      :", auc_score)

print("\nSaved")

print("Confusion Matrix")

print("ROC Curve")