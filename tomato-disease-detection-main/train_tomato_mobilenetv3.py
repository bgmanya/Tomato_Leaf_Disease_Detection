from __future__ import annotations

import shutil
from pathlib import Path

import setuptools as _setuptools
import tensorflow as tf

from tomato_model_utils import CLASS_NAMES

REPO_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path('/Users/admin/Documents/Research_paper/Tomato_Leaf_Dataset')
TEMP_DATASET_ROOT = REPO_ROOT / 'temp_training_dataset'
MODEL_PATH = REPO_ROOT / 'models' / 'tomato-mobilenetv3.keras'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 8

SOURCE_MAPPING = {
    'Tomato_Early_blight': 'Early-blight',
    'Tomato_Healthy': 'Healthy',
    'Tomato_leaf_late_blight': 'Late-blight',
    'Tomato_mold_leaf': 'Leaf-mold',
    'Tomato_septora_leaf_spot': 'Septoria-leaf-spot',
    'Tomato_leaf_yellow_curl_virus': 'Yellow-leaf-curl-virus',
}


def prepare_dataset() -> None:
    if TEMP_DATASET_ROOT.exists():
        shutil.rmtree(TEMP_DATASET_ROOT, ignore_errors=True)
    TEMP_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    for source_name, target_name in SOURCE_MAPPING.items():
        source_dir = DATASET_ROOT / source_name
        target_dir = TEMP_DATASET_ROOT / target_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_file in sorted(source_dir.iterdir()):
            if image_file.is_file() and image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                shutil.copy2(image_file, target_dir / image_file.name)


def build_model() -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling='avg',
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # Apply MobileNetV3 preprocessing exactly once to the image tensor in the
    # dataset pipeline and at inference time. Do not preprocess again inside the
    # model graph or the same image will be normalized twice, which distorts the
    # six-class predictions and lowers confidence on visually similar diseases.
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    return model


def main() -> None:
    prepare_dataset()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEMP_DATASET_ROOT),
        validation_split=0.2,
        subset='training',
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        class_names=CLASS_NAMES,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEMP_DATASET_ROOT),
        validation_split=0.2,
        subset='validation',
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        class_names=CLASS_NAMES,
    )

    train_ds = train_ds.map(lambda x, y: (tf.keras.applications.mobilenet_v3.preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (tf.keras.applications.mobilenet_v3.preprocess_input(x), y))
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True),
        ],
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH), save_format='keras')
    print(f'\nSaved model to {MODEL_PATH}')
    print(f'Best val accuracy: {max(history.history.get("val_accuracy", [0])):.4f}')


if __name__ == '__main__':
    main()
