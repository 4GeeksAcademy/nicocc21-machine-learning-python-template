import hashlib
import json
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "dogs-vs-cats" / "train"
SPLIT_DIR = PROJECT_ROOT / "data" / "interim" / "image_classifier"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "efficientnet_b0_dogs_vs_cats.keras"
IMAGE_SIZE = (200, 200)
BATCH_SIZE = 16
SEED = 42


def prepare_data(validation_fraction=0.1, test_fraction=0.1, seed=SEED):
    image_paths = sorted(RAW_IMAGES_DIR.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(
            f"No se encontraron imágenes JPG en {RAW_IMAGES_DIR}. "
            "Comprueba la ubicación del dataset."
        )

    labels = [path.name.split(".", maxsplit=1)[0] for path in image_paths]
    if set(labels) != {"cat", "dog"}:
        raise ValueError("El dataset debe contener imágenes con prefijo cat. o dog.")
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction y test_fraction deben ser mayores que 0.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("La suma de validation_fraction y test_fraction debe ser menor que 1.")

    digest = hashlib.sha256("\n".join(path.name for path in image_paths).encode())
    manifest = {
        "dataset_signature": digest.hexdigest(),
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
    }
    manifest_path = SPLIT_DIR / "split.json"
    train_dir = SPLIT_DIR / "train"
    validation_dir = SPLIT_DIR / "validation"
    test_dir = SPLIT_DIR / "test"

    if manifest_path.exists():
        saved_manifest = json.loads(manifest_path.read_text())
        saved_count = sum(
            len(list((SPLIT_DIR / split / label).glob("*.jpg")))
            for split in ("train", "validation", "test")
            for label in ("cat", "dog")
        )
        if saved_manifest == manifest and saved_count == len(image_paths):
            return train_dir, validation_dir, test_dir

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    train_paths, remaining_paths = train_test_split(
        image_paths,
        test_size=validation_fraction + test_fraction,
        random_state=seed,
        stratify=labels,
    )
    remaining_labels = [path.name.split(".", maxsplit=1)[0] for path in remaining_paths]
    test_share = test_fraction / (validation_fraction + test_fraction)
    validation_paths, test_paths = train_test_split(
        remaining_paths,
        test_size=test_share,
        random_state=seed,
        stratify=remaining_labels,
    )

    for split_dir, split_paths in (
        (train_dir, train_paths),
        (validation_dir, validation_paths),
        (test_dir, test_paths),
    ):
        for source_path in split_paths:
            label = source_path.name.split(".", maxsplit=1)[0]
            target_dir = split_dir / label
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_dir / source_path.name)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return train_dir, validation_dir, test_dir


def create_generators(train_dir, validation_dir, test_dir, batch_size=BATCH_SIZE):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    validation_datagen = ImageDataGenerator()

    train_data = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )
    validation_data = validation_datagen.flow_from_directory(
        str(validation_dir),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_data = validation_datagen.flow_from_directory(
        str(test_dir),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_data, validation_data, test_data


def build_model():
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=(*IMAGE_SIZE, 3),
            ),
            GlobalAveragePooling2D(),
            Dense(units=128, activation="relu"),
            Dense(units=2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, train_data, validation_data, epochs=20):
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    return model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
    )


def load_best_model():
    from tensorflow.keras.models import load_model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado en {MODEL_PATH}.")
    return load_model(MODEL_PATH)


def evaluate_model(model, validation_data):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report

    validation_data.reset()
    loss, accuracy = model.evaluate(validation_data, verbose=1)
    validation_data.reset()
    predictions = model.predict(validation_data, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    class_names = [
        name
        for name, _ in sorted(
            validation_data.class_indices.items(), key=lambda item: item[1]
        )
    ]
    true_classes = validation_data.classes

    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(
        classification_report(
            true_classes,
            predicted_classes,
            target_names=class_names,
            zero_division=0,
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        true_classes,
        predicted_classes,
        display_labels=class_names,
        cmap="Blues",
        values_format="d",
    )
    plt.title("Matriz de confusion: perros y gatos")
    plt.tight_layout()
    plt.show()
    return {"loss": loss, "accuracy": accuracy}


def main():
    train_dir, validation_dir, test_dir = prepare_data()
    train_data, validation_data, test_data = create_generators(
        train_dir, validation_dir, test_dir
    )
    model = build_model()
    model.summary()
    train_model(model, train_data, validation_data)
    best_model = load_best_model()
    evaluate_model(best_model, test_data)
    print(f"Modelo guardado en: {MODEL_PATH}")