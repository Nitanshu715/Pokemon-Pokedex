import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

# -------- PATHS (LOCKED) --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "PokemonData")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "pokemon_model.keras")

IMG_SIZE = (224, 224)
MAX_PER_CLASS = 20
EPOCHS = 5

X = []
y = []
class_names = []

print("Loading dataset from:", DATASET_PATH)

for class_name in sorted(os.listdir(DATASET_PATH)):
    class_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_path):
        continue

    label = len(class_names)
    class_names.append(class_name)

    images = os.listdir(class_path)[:MAX_PER_CLASS]
    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            X.append(np.array(img) / 255.0)
            y.append(label)
        except:
            continue

X = np.array(X)
y = np.array(y)

print("Images loaded:", len(X))
print("Classes:", len(class_names))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=32
)

model.save(MODEL_SAVE_PATH)

print("MODEL SAVED AT:", MODEL_SAVE_PATH)
