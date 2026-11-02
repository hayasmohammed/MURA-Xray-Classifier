import tensorflow as tf

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)

from config import *
from utils import *

# ==========================================================
# INITIALIZATION
# ==========================================================

initialize_gpu()

create_directories()

train_ds, valid_ds = get_datasets()

# ==========================================================
# BUILD MODEL
# ==========================================================

base_model = DenseNet121(

    include_top=False,

    weights="imagenet",

    input_shape=(224,224,3)

)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224,224,3))

x = base_model(inputs, training=False)

x = GlobalAveragePooling2D()(x)

x = Dropout(0.30)(x)

outputs = Dense(
    1,
    activation="sigmoid",
    dtype="float32"
)(x)

model = Model(inputs, outputs)

model.compile(

    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),

    loss="binary_crossentropy",

    metrics=[

        "accuracy",

        tf.keras.metrics.AUC(name="auc")

    ]

)

model.summary()

# ==========================================================
# CALLBACKS
# ==========================================================

callbacks = [

    ModelCheckpoint(

        BEST_MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1

    ),

    EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True

    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=2,

        verbose=1

    ),

    TensorBoard(

        log_dir=LOG_DIR

    )

]

# ==========================================================
# STAGE 1
# ==========================================================

print("\n==============================")
print("Stage 1 : Training Classifier")
print("==============================\n")

history1 = model.fit(

    train_ds,

    validation_data=valid_ds,

    epochs=EPOCHS_HEAD,

    callbacks=callbacks

)

# ==========================================================
# STAGE 2
# ==========================================================

print("\n==============================")
print("Stage 2 : Fine Tuning")
print("==============================\n")

base_model.trainable = True

for layer in base_model.layers[:-40]:

    layer.trainable = False

model.compile(

    optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LR),

    loss="binary_crossentropy",

    metrics=[

        "accuracy",

        tf.keras.metrics.AUC(name="auc")

    ]

)

history2 = model.fit(

    train_ds,

    validation_data=valid_ds,

    epochs=EPOCHS_FINE,

    callbacks=callbacks

)

model.save(FINAL_MODEL_PATH)

plot_history(history2)

print("\n===================================")
print("Training Finished Successfully")
print("===================================")