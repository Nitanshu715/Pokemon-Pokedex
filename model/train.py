import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------- PATHS --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "PokemonData")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "pokemon_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "..", "class_names.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

print("Loading dataset from:", DATASET_PATH)

# -------- DATA GENERATORS --------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation",
    shuffle=False
)

print("Classes found:", train_gen.class_indices)

# -------- MODEL --------
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------- TRAIN --------
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# -------- SAVE MODEL --------
model.save(MODEL_SAVE_PATH)
print("MODEL SAVED AT:", MODEL_SAVE_PATH)

# -------- SAVE CLASS NAMES --------
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)

print("CLASS NAMES SAVED AT:", CLASS_NAMES_PATH)
