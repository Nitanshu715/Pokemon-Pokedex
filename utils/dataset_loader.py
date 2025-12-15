import os
import numpy as np
from PIL import Image


def load_dataset(dataset_path, image_size=(224, 224), max_per_class=None):
    X = []
    y = []
    class_names = []

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        class_index = len(class_names)
        class_names.append(class_name)

        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if max_per_class:
            images = images[:max_per_class]

        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(image_size)
                img = np.array(img) / 255.0

                X.append(img)
                y.append(class_index)
            except:
                continue

    return np.array(X), np.array(y), class_names
