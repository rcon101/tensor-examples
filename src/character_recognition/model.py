from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.RandomRotation(0.08),
            layers.RandomTranslation(0.12, 0.12),
            layers.RandomZoom(0.1),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(96, 3, activation="relu"),
            layers.Flatten(),
            layers.Dropout(0.25),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(26, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
