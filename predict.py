import os
import tensorflow as tf

from config import *

model = tf.keras.models.load_model(BEST_MODEL_PATH)

image_path = input("Enter image path: ").strip().strip('"')

# Convert forward slashes/backslashes automatically
image_path = os.path.normpath(image_path)

if not os.path.exists(image_path):
    print("\nFile not found!")
    print(image_path)
    exit()

image = tf.io.read_file(image_path)
image = tf.image.decode_png(image, channels=3)
image = tf.image.resize(image, IMAGE_SIZE)
image = tf.cast(image, tf.float32) / 255.0

image = tf.expand_dims(image, 0)

prob = model.predict(image, verbose=0)[0][0]

prediction = "Abnormal" if prob >= 0.5 else "Normal"
confidence = prob if prob >= 0.5 else (1 - prob)

print("\n=========================")
print("Prediction :", prediction)
print(f"Confidence : {confidence*100:.2f}%")
print(f"Probability: {prob:.4f}")
print("=========================")