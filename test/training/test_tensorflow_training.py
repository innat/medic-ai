"""End-to-end TensorFlow training coverage for migrated transforms."""

import keras
import numpy as np
import pytest

from medicai.losses import BinaryDiceLoss
from medicai.metrics import BinaryDiceMetric
from test.training.common import (
    DatasetBuilder as make_dataset,
    apply_classification_pipeline,
    apply_segmentation_pipeline,
    build_classification_model,
    build_segmentation_model,
    build_transform_pipelines,
)


class GPUAugmentedModel(keras.Model):
    """Wrap a model and apply a batch transform inside ``train_step``."""

    def __init__(self, model, augment_data, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.augment_data = augment_data

    def train_step(self, data):
        x, y = data
        x, y = self.augment_data(x, y)
        return super().train_step((x, y))

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)


def _require_tensorflow():
    if keras.config.backend() != "tensorflow":
        pytest.skip("TensorFlow tf.data coverage requires the TensorFlow backend.")
    import tensorflow as tf

    return tf


def _fit_tfdata_classification(
    images,
    labels,
    *,
    input_layout: str,
    input_shape: tuple[int, ...],
    pipeline_index: int,
    strategy=None,
):
    tf = _require_tensorflow()
    pipeline = build_transform_pipelines(input_layout, segmentation=False)[pipeline_index]
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(
        lambda image, label: apply_classification_pipeline(pipeline, image, label),
        num_parallel_calls=1,
    ).batch(2)

    if strategy is None:
        model = build_classification_model(input_shape)
    else:
        with strategy.scope():
            model = build_classification_model(input_shape)
    history = model.fit(dataset, epochs=1, verbose=0)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


def _fit_tfdata_segmentation(
    images,
    labels,
    *,
    input_layout: str,
    input_shape: tuple[int, ...],
    pipeline_index: int,
    strategy=None,
):
    tf = _require_tensorflow()
    pipeline = build_transform_pipelines(input_layout, segmentation=True)[pipeline_index]

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(
        lambda image, label: apply_segmentation_pipeline(pipeline, image, label),
        num_parallel_calls=1,
    ).batch(2)

    if strategy is None:
        model = build_segmentation_model(input_shape)
    else:
        with strategy.scope():
            model = build_segmentation_model(input_shape)
    history = model.fit(dataset, epochs=1, verbose=0)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


def _fit_gpu_augmented_model(
    images,
    labels,
    *,
    input_layout: str,
    input_shape: tuple[int, ...],
    segmentation: bool,
):
    tf = _require_tensorflow()
    pipeline = build_transform_pipelines(input_layout, segmentation=segmentation)[4]

    def augment_data(image, label):
        if segmentation:
            result = pipeline({"image": image, "label": label})
            return result["image"], result["label"]
        result = pipeline({"image": image})
        return result["image"], label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)
    base_model = (
        build_segmentation_model(input_shape)
        if segmentation
        else build_classification_model(input_shape)
    )
    model = GPUAugmentedModel(base_model, augment_data)
    if segmentation:
        loss = BinaryDiceLoss(from_logits=False, num_classes=1)
        metrics = [BinaryDiceMetric(from_logits=False, num_classes=1)]
    else:
        loss = "binary_crossentropy"
        metrics = [keras.metrics.BinaryAccuracy()]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=metrics,
        jit_compile=False,
    )
    history = model.fit(dataset, epochs=1, verbose=0)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


@pytest.mark.integration
def test_tensorflow_tfdata_2d_classification_accepts_migrated_transforms():
    """Train a 2D classifier after transforms run inside ``Dataset.map``."""
    images, labels = make_dataset().classification_2d()
    _fit_tfdata_classification(
        images, labels, input_layout="HWC", input_shape=(12, 12, 1), pipeline_index=0
    )


@pytest.mark.integration
def test_tensorflow_tfdata_2d_segmentation_accepts_crop_flip_pipeline():
    """Train a 2D segmenter with a crop and synchronized flip pipeline."""
    images, labels = make_dataset().segmentation_2d()
    _fit_tfdata_segmentation(
        images, labels, input_layout="HWC", input_shape=(8, 8, 1), pipeline_index=2
    )


@pytest.mark.integration
def test_tensorflow_tfdata_3d_classification_accepts_rotation_pipeline():
    """Train a 3D classifier with a flip and 90-degree rotation pipeline."""
    images, labels = make_dataset().classification_3d()
    _fit_tfdata_classification(
        images, labels, input_layout="DHWC", input_shape=(6, 6, 6, 1), pipeline_index=3
    )


@pytest.mark.integration
def test_tensorflow_tfdata_3d_segmentation_accepts_random_choice_pipeline():
    """Train a 3D segmenter with synchronized ``RandomChoice`` geometry."""
    images, labels = make_dataset().segmentation_3d()
    _fit_tfdata_segmentation(
        images, labels, input_layout="DHWC", input_shape=(6, 6, 6, 1), pipeline_index=4
    )


@pytest.mark.integration
def test_tensorflow_tfdata_anisotropic_2d_segmentation_accepts_crop_pipeline():
    """Train a 2D segmenter with unequal height and width dimensions."""
    images, labels = make_dataset().segmentation_2d(spatial_shape=(8, 12))
    _fit_tfdata_segmentation(
        images, labels, input_layout="HWC", input_shape=(8, 8, 1), pipeline_index=2
    )


@pytest.mark.integration
def test_tensorflow_tfdata_anisotropic_3d_classification_accepts_rotation_pipeline():
    """Train a 3D classifier with unequal depth, height, and width dimensions."""
    images, labels = make_dataset().classification_3d(spatial_shape=(4, 6, 8))
    _fit_tfdata_classification(
        images,
        labels,
        input_layout="DHWC",
        input_shape=(4, 8, 6, 1),
        pipeline_index=3,
    )


@pytest.mark.integration
def test_tensorflow_tfdata_2d_classification_accepts_random_choice_pipeline():
    """Train a 2D classifier through the fifth reusable pipeline."""
    images, labels = make_dataset().classification_2d()
    _fit_tfdata_classification(
        images, labels, input_layout="HWC", input_shape=(12, 12, 1), pipeline_index=4
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_tensorflow_gpu_augmented_model_trains_2d_classification():
    """Apply a batch ``RandomChoice`` pipeline inside a 2D model's train step."""
    images, labels = make_dataset().classification_2d()
    _fit_gpu_augmented_model(
        images,
        labels,
        input_layout="BHWC",
        input_shape=(12, 12, 1),
        segmentation=False,
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_tensorflow_gpu_augmented_model_trains_3d_segmentation():
    """Apply synchronized batch geometry inside a 3D segmenter's train step."""
    images, labels = make_dataset().segmentation_3d()
    _fit_gpu_augmented_model(
        images,
        labels,
        input_layout="BDHWC",
        input_shape=(6, 6, 6, 1),
        segmentation=True,
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_tensorflow_tfdata_2d_classification_accepts_multi_device_strategy():
    """Train through ``MirroredStrategy`` when at least two GPUs are available."""
    tf = _require_tensorflow()
    devices = [device.name for device in tf.config.list_logical_devices("GPU")]
    if len(devices) < 2:
        pytest.skip("Multi-device TensorFlow coverage requires at least two GPUs.")

    images, labels = make_dataset().classification_2d()
    strategy = tf.distribute.MirroredStrategy(devices=devices[:2])
    _fit_tfdata_classification(
        images,
        labels,
        input_layout="HWC",
        input_shape=(12, 12, 1),
        pipeline_index=1,
        strategy=strategy,
    )
