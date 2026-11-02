import tkinter as tk
from tkinter import filedialog

import tensorflow as tf
import numpy as np

from PIL import Image, ImageTk

from config import *

print("Loading model...")

model = tf.keras.models.load_model(BEST_MODEL_PATH)

print("Model Loaded")


def predict():

    global photo

    file_path = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg")]
    )

    if file_path == "":
        return

    image = Image.open(file_path)

    preview = image.resize((350, 350))

    photo = ImageTk.PhotoImage(preview)

    image_label.config(image=photo)

    img = image.convert("RGB")
    img = img.resize(IMAGE_SIZE)

    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    probability = model.predict(img, verbose=0)[0][0]

    if probability >= 0.5:
        prediction = "Abnormal"
        confidence = probability * 100
    else:
        prediction = "Normal"
        confidence = (1 - probability) * 100

    result.config(
        text=f"Prediction : {prediction}"
    )

    conf.config(
        text=f"Confidence : {confidence:.2f}%"
    )

    prob.config(
        text=f"Probability : {probability:.4f}"
    )


root = tk.Tk()

root.title("MURA X-Ray Classifier")

root.geometry("650x650")

title = tk.Label(
    root,
    text="MURA X-Ray Classification",
    font=("Arial", 22, "bold")
)

title.pack(pady=15)

btn = tk.Button(
    root,
    text="Select X-Ray Image",
    font=("Arial", 14),
    command=predict
)

btn.pack()

image_label = tk.Label(root)

image_label.pack(pady=20)

result = tk.Label(
    root,
    text="Prediction : -",
    font=("Arial", 16, "bold")
)

result.pack()

conf = tk.Label(
    root,
    text="Confidence : -",
    font=("Arial", 15)
)

conf.pack()

prob = tk.Label(
    root,
    text="Probability : -",
    font=("Arial", 15)
)

prob.pack()

root.mainloop()