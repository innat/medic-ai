"""End-to-end TensorFlow training coverage for migrated transforms.

The tests keep dataset mapping separate from model distribution. This makes
the same transform pipeline usable with ordinary ``tf.data`` training,
``OneDeviceStrategy``, and ``MirroredStrategy``.
"""

import keras
import numpy as np
import pytest

from medicai.transforms import Compose, Flip, ScaleIntensityRange
from test.training.data import (
    make_classification_2d_samples,
    make_classification_3d_samples,
    make_segmentation_2d_samples,
    make_segmentation_3d_samples,
)


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
        pooling = (
            keras.layers.GlobalAveragePooling2D
            if spatial_rank == 2
            else keras.layers.GlobalAveragePooling3D
        )
        outputs = keras.layers.Dense(1, activation="sigmoid")(pooling()(x))
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        jit_compile=False,
    )
    return model


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


def _fit_tfdata_classification(images, labels, *, input_layout, input_shape, strategy=None):
    tf = _require_tensorflow()
    pipeline = _pipeline(input_layout)

    def map_sample(image, label):
        result = pipeline({"image": image})
        return result["image"], label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(map_sample, num_parallel_calls=1).batch(2)
    if strategy is None:
        model = _build_model(input_shape, segmentation=False)
        history = model.fit(dataset, epochs=1, verbose=0)
    else:
        with strategy.scope():
            model = _build_model(input_shape, segmentation=False)
        history = model.fit(dataset, epochs=1, verbose=0)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


def _fit_tfdata_segmentation(images, labels, *, input_layout, input_shape, strategy=None):
    tf = _require_tensorflow()
    pipeline = _pipeline(input_layout, keys=("image", "label"))

    def map_sample(image, label):
        result = pipeline({"image": image, "label": label})
        return result["image"], result["label"]

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(map_sample, num_parallel_calls=1).batch(2)
    if strategy is None:
        model = _build_model(input_shape, segmentation=True)
        history = model.fit(dataset, epochs=1, verbose=0)
    else:
        with strategy.scope():
            model = _build_model(input_shape, segmentation=True)
        history = model.fit(dataset, epochs=1, verbose=0)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


@pytest.mark.integration
def test_tensorflow_tfdata_2d_classification_accepts_migrated_transforms():
    """Train a 2D classifier after transforms run inside ``Dataset.map``."""
    images, class_labels = make_classification_2d_samples()
    _fit_tfdata_classification(
        images,
        class_labels,
        input_layout="HWC",
        input_shape=(12, 12, 1),
    )


@pytest.mark.integration
def test_tensorflow_tfdata_2d_segmentation_accepts_migrated_transforms():
    """Train a 2D segmentation model with aligned image and label transforms."""
    images, labels = make_segmentation_2d_samples()
    _fit_tfdata_segmentation(
        images,
        labels,
        input_layout="HWC",
        input_shape=(12, 12, 1),
    )


@pytest.mark.integration
def test_tensorflow_tfdata_3d_classification_accepts_migrated_transforms():
    """Train a 3D classifier after transforms run inside ``Dataset.map``."""
    images, class_labels = make_classification_3d_samples()
    _fit_tfdata_classification(
        images,
        class_labels,
        input_layout="DHWC",
        input_shape=(6, 6, 6, 1),
    )


@pytest.mark.integration
def test_tensorflow_tfdata_3d_segmentation_accepts_migrated_transforms():
    """Train a 3D segmentation model with aligned image and label transforms."""
    images, labels = make_segmentation_3d_samples()
    _fit_tfdata_segmentation(
        images,
        labels,
        input_layout="DHWC",
        input_shape=(6, 6, 6, 1),
    )


@pytest.mark.integration
def test_tensorflow_tfdata_2d_classification_accepts_single_device_strategy():
    """Train through an explicit single-device TensorFlow strategy."""
    tf = _require_tensorflow()
    images, labels = make_classification_2d_samples()
    strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    _fit_tfdata_classification(
        images,
        labels,
        input_layout="HWC",
        input_shape=(12, 12, 1),
        strategy=strategy,
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_tensorflow_tfdata_2d_classification_accepts_multi_device_strategy():
    """Train through ``MirroredStrategy`` when at least two GPUs are available."""
    tf = _require_tensorflow()
    devices = [device.name for device in tf.config.list_logical_devices("GPU")]
    if len(devices) < 2:
        pytest.skip("Multi-device TensorFlow coverage requires at least two GPUs.")

    images, labels = make_classification_2d_samples()
    strategy = tf.distribute.MirroredStrategy(devices=devices[:2])
    _fit_tfdata_classification(
        images,
        labels,
        input_layout="HWC",
        input_shape=(12, 12, 1),
        strategy=strategy,
    )
