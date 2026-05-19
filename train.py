import argparse
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

from src.character_recognition.model import build_model
from src.character_recognition.synthetic import generate_synthetic_characters

MODEL_PATH = Path("models/character_model.keras")


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    # EMNIST stores letters rotated/mirrored relative to normal display.
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)
    # EMNIST letters are labeled 1..26; the model expects 0..25.
    label = label - 1
    return image, tf.cast(label, tf.int32)


def prepare_dataset(dataset, batch_size, shuffle=False, repeat=False):
    if shuffle:
        dataset = dataset.shuffle(20_000)
    if repeat:
        dataset = dataset.repeat()
    return (
        dataset.batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def main():
    parser = argparse.ArgumentParser(description="Train an EMNIST letter classifier.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=Path, default=Path("data/tfds"))
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--synthetic-samples-per-font", type=int, default=12)
    args = parser.parse_args()

    train_data, test_data = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        data_dir=args.data_dir,
    )

    train_data = train_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    # Add rendered fonts so typed single-character JPGs are in-distribution.
    synthetic_data = tf.data.Dataset.from_generator(
        lambda: generate_synthetic_characters(args.synthetic_samples_per_font),
        output_signature=(
            tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    mixed_train_data = train_data.concatenate(synthetic_data)
    train_batches = prepare_dataset(
        mixed_train_data,
        args.batch_size,
        shuffle=True,
    )
    test_batches = prepare_dataset(test_data, args.batch_size)

    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        )
    ]

    model.fit(
        train_batches,
        validation_data=test_batches,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    print(f"Saved model to {args.model_path}")


if __name__ == "__main__":
    main()
