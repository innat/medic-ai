"""End-to-end TensorFlow ``tf.data`` coverage for migrated transforms."""

import keras
import numpy as np
import pytest

from medicai.transforms import Compose, Flip, ScaleIntensityRange


def _require_tensorflow():
    if keras.config.backend() != "tensorflow":
        pytest.skip("TensorFlow tf.data coverage requires the TensorFlow backend.")
    import tensorflow as tf

    return tf


def _build_model(input_shape, *, segmentation):
    inputs = keras.Input(shape=input_shape)
    spatial_rank = len(input_shape) - 1
    conv = keras.layers.Conv2D if spatial_rank == 2 else keras.layers.Conv3D
    x = conv(4, 3, padding="same", activation="relu")(inputs)
    if segmentation:
        outputs = conv(1, 1, padding="same", activation="sigmoid")(x)
    else:
        pooling = keras.layers.GlobalAveragePooling2D if spatial_rank == 2 else keras.layers.GlobalAveragePooling3D
        outputs = keras.layers.Dense(1, activation="sigmoid")(pooling()(x))
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        jit_compile=False,
    )
    return model


def _make_samples(spatial_shape, num_samples=4):
    size = num_samples * int(np.prod(spatial_shape))
    images = np.linspace(0.0, 1.0, num=size, dtype=np.float32).reshape(
        (num_samples, *spatial_shape)
    )
    images = images[..., None]
    labels = (images > 0.5).astype(np.float32)
    class_labels = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
    return images, labels, class_labels[:num_samples]


def _pipeline(input_layout, keys=("image",)):
    return Compose(
        [
            ScaleIntensityRange(
                keys=["image"],
                source_value_range=(0.0, 1.0),
                target_value_range=(0.0, 1.0),
                clip=True,
                input_layout=input_layout,
            ),
            Flip(keys=list(keys), spatial_axis=0, input_layout=input_layout),
        ]
    )


def _fit_tfdata(images, targets, *, input_layout, input_shape, segmentation):
    tf = _require_tensorflow()
    keys = ("image", "label") if segmentation else ("image",)
    pipeline = _pipeline(input_layout, keys=keys)

    def map_sample(image, target):
        sample = {"image": image}
        if segmentation:
            sample["label"] = target
        result = pipeline(sample)
        return result["image"], result["label"] if segmentation else target

    dataset = tf.data.Dataset.from_tensor_slices((images, targets))
    dataset = dataset.map(map_sample, num_parallel_calls=1).batch(2)
    model = _build_model(input_shape, segmentation=segmentation)
    history = model.fit(dataset, epochs=1, verbose=0)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


@pytest.mark.integration
def test_tensorflow_tfdata_2d_classification_accepts_migrated_transforms():
    """Train a 2D classifier after transforms run inside ``Dataset.map``."""
    images, _, class_labels = _make_samples((12, 12))
    _fit_tfdata(
        images,
        class_labels,
        input_layout="HWC",
        input_shape=(12, 12, 1),
        segmentation=False,
    )


@pytest.mark.integration
def test_tensorflow_tfdata_2d_segmentation_accepts_migrated_transforms():
    """Train a 2D segmentation model with aligned image and label transforms."""
    images, labels, _ = _make_samples((12, 12))
    _fit_tfdata(
        images,
        labels,
        input_layout="HWC",
        input_shape=(12, 12, 1),
        segmentation=True,
    )


@pytest.mark.integration
def test_tensorflow_tfdata_3d_classification_accepts_migrated_transforms():
    """Train a 3D classifier after transforms run inside ``Dataset.map``."""
    images, _, class_labels = _make_samples((6, 6, 6))
    _fit_tfdata(
        images,
        class_labels,
        input_layout="DHWC",
        input_shape=(6, 6, 6, 1),
        segmentation=False,
    )


@pytest.mark.integration
def test_tensorflow_tfdata_3d_segmentation_accepts_migrated_transforms():
    """Train a 3D segmentation model with aligned image and label transforms."""
    images, labels, _ = _make_samples((6, 6, 6))
    _fit_tfdata(
        images,
        labels,
        input_layout="DHWC",
        input_shape=(6, 6, 6, 1),
        segmentation=True,
    )
