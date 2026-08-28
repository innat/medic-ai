"""End-to-end TensorFlow training coverage for migrated transforms."""

import keras
import numpy as np
import pytest

from medicai.losses import BinaryDiceLoss
from medicai.metrics import BinaryDiceMetric
from test.training.common import (
    DatasetBuilder as make_dataset,
    GPUAugmentedModel,
    PyGrainSource,
    apply_classification_pipeline,
    apply_segmentation_pipeline,
    build_classification_model,
    build_segmentation_model,
    build_gpu_random_pipeline,
    build_transform_pipelines,
    build_volume_geometry_pipeline,
)


def _require_tensorflow():
    if keras.config.backend() != "tensorflow":
        pytest.skip("TensorFlow tf.data coverage requires the TensorFlow backend.")
    import tensorflow as tf

    return tf


def _require_pygrain():
    _require_tensorflow()
    try:
        import grain.python as pygrain
    except ImportError:
        pytest.skip("PyGrain is not installed.")
    return pygrain


def _make_pygrain_loader(
    images,
    labels,
    pipeline,
    affines=None,
    *,
    worker_count=0,
    num_threads=None,
):
    pygrain = _require_pygrain()
    read_options = None
    if num_threads is not None:
        read_options = pygrain.ReadOptions(
            num_threads=num_threads,
            prefetch_buffer_size=2,
        )
    return pygrain.load(
        PyGrainSource(images, labels, pipeline, affines=affines),
        batch_size=2,
        num_epochs=1,
        worker_count=worker_count,
        read_options=read_options,
    )


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
    history = model.fit(dataset, epochs=1, verbose=0, shuffle=False)

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
    history = model.fit(dataset, epochs=1, verbose=0, shuffle=False)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


def _fit_gpu_augmented_model(
    images,
    labels,
    *,
    input_layout: str,
    input_shape: tuple[int, ...],
    segmentation: bool,
    strategy=None,
):
    tf = _require_tensorflow()
    pipeline = build_gpu_random_pipeline(input_layout, segmentation=segmentation)

    def augment_data(image, label):
        if segmentation:
            result = pipeline({"image": image, "label": label})
            return result["image"], result["label"]
        result = pipeline({"image": image})
        return result["image"], label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)
    def build_base_model():
        return (
            build_segmentation_model(input_shape)
            if segmentation
            else build_classification_model(input_shape)
        )

    if segmentation:
        loss = BinaryDiceLoss(from_logits=False, num_classes=1)
        metrics = [BinaryDiceMetric(from_logits=False, num_classes=1)]
    else:
        loss = "binary_crossentropy"
        metrics = [keras.metrics.BinaryAccuracy()]

    def build_and_compile_model():
        model = GPUAugmentedModel(build_base_model(), augment_data)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=loss,
            metrics=metrics,
            jit_compile=False,
        )
        return model

    if strategy is None:
        model = build_and_compile_model()
    else:
        with strategy.scope():
            model = build_and_compile_model()
    history = model.fit(dataset, epochs=1, verbose=0, shuffle=False)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


@pytest.mark.integration
def test_tensorflow_tfdata_2d_classification_accepts_migrated_transforms():
    """Train a 2D classifier after transforms run inside ``Dataset.map``."""
    images, labels = make_dataset().classification_2d()
    _fit_tfdata_classification(
        images, labels, input_layout="HWC", input_shape=(32, 48, 1), pipeline_index=0
    )


@pytest.mark.integration
def test_tensorflow_tfdata_2d_segmentation_accepts_crop_flip_pipeline():
    """Train a 2D segmenter with a crop and synchronized flip pipeline."""
    images, labels = make_dataset().segmentation_2d()
    _fit_tfdata_segmentation(
        images, labels, input_layout="HWC", input_shape=(24, 24, 1), pipeline_index=2
    )


@pytest.mark.integration
def test_tensorflow_tfdata_3d_classification_accepts_rotation_pipeline():
    """Train a 3D classifier with a flip and 90-degree rotation pipeline."""
    images, labels = make_dataset().classification_3d()
    _fit_tfdata_classification(
        images, labels, input_layout="DHWC", input_shape=(8, 16, 16, 1), pipeline_index=3
    )


@pytest.mark.integration
def test_tensorflow_tfdata_3d_segmentation_accepts_random_choice_pipeline():
    """Train a 3D segmenter with synchronized ``RandomChoice`` geometry."""
    images, labels = make_dataset().segmentation_3d()
    _fit_tfdata_segmentation(
        images, labels, input_layout="DHWC", input_shape=(8, 16, 16, 1), pipeline_index=4
    )


@pytest.mark.integration
def test_tensorflow_tfdata_anisotropic_2d_segmentation_accepts_crop_pipeline():
    """Train a 2D segmenter with unequal height and width dimensions."""
    images, labels = make_dataset().segmentation_2d(spatial_shape=(32, 48))
    _fit_tfdata_segmentation(
        images, labels, input_layout="HWC", input_shape=(24, 24, 1), pipeline_index=2
    )


@pytest.mark.integration
def test_tensorflow_tfdata_anisotropic_3d_classification_accepts_rotation_pipeline():
    """Train a 3D classifier with unequal depth, height, and width dimensions."""
    images, labels = make_dataset().classification_3d(spatial_shape=(8, 12, 16))
    _fit_tfdata_classification(
        images,
        labels,
        input_layout="DHWC",
        input_shape=(8, 16, 12, 1),
        pipeline_index=3,
    )


@pytest.mark.integration
def test_tensorflow_tfdata_2d_classification_accepts_random_choice_pipeline():
    """Train a 2D classifier through the fifth reusable pipeline."""
    images, labels = make_dataset().classification_2d()
    _fit_tfdata_classification(
        images, labels, input_layout="HWC", input_shape=(32, 48, 1), pipeline_index=4
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_tensorflow_gpu_augmented_model_trains_2d_classification():
    """Apply a batch ``RandomChoice`` pipeline inside a 2D model's train step."""
    images, labels = make_dataset().classification_2d(spatial_shape=(32, 32))
    _fit_gpu_augmented_model(
        images,
        labels,
        input_layout="BHWC",
        input_shape=(32, 32, 1),
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
        input_shape=(8, 16, 16, 1),
        segmentation=True,
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_tensorflow_gpu_augmented_model_uses_all_available_gpus():
    """Train model-side augmentation through all available GPUs."""
    tf = _require_tensorflow()
    devices = [device.name for device in tf.config.list_logical_devices("GPU")]
    if len(devices) < 2:
        pytest.skip("Multi-device TensorFlow coverage requires at least two GPUs.")

    images, labels = make_dataset().classification_2d(spatial_shape=(32, 32))
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    _fit_gpu_augmented_model(
        images,
        labels,
        input_layout="BHWC",
        input_shape=(32, 32, 1),
        segmentation=False,
        strategy=strategy,
    )


@pytest.mark.integration
def test_tensorflow_pygrain_accepts_classification_samples():
    """Train a classifier from PyGrain samples transformed before batching."""
    images, labels = make_dataset().classification_2d()
    loader = _make_pygrain_loader(
        images,
        labels,
        build_transform_pipelines("HWC", segmentation=False)[0],
    )
    model = build_classification_model((32, 48, 1))

    history = model.fit(loader, epochs=1, verbose=0, shuffle=False)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


@pytest.mark.integration
def test_tensorflow_pygrain_accepts_segmentation_samples():
    """Train a segmenter from PyGrain samples with aligned image/mask transforms."""
    images, labels = make_dataset().segmentation_2d()
    loader = _make_pygrain_loader(
        images,
        labels,
        build_transform_pipelines("HWC", segmentation=True)[2],
    )
    model = build_segmentation_model((24, 24, 1))

    history = model.fit(loader, epochs=1, verbose=0, shuffle=False)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


@pytest.mark.integration
def test_tensorflow_pygrain_accepts_orientation_and_spacing():
    """Run affine-aware geometry transforms per sample in a PyGrain loader."""
    images, labels, affines = make_dataset().segmentation_3d_with_affine()
    loader = _make_pygrain_loader(
        images,
        labels,
        build_volume_geometry_pipeline(),
        affines=affines,
    )
    model = build_segmentation_model((8, 16, 8, 1))

    history = model.fit(loader, epochs=1, verbose=0, shuffle=False)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


@pytest.mark.integration
def test_tensorflow_pygrain_accepts_model_side_random_transforms():
    """Feed PyGrain batches into a model with random transforms in ``train_step``."""
    images, labels = make_dataset().segmentation_2d()
    loader = _make_pygrain_loader(
        images,
        labels,
        build_transform_pipelines("HWC", segmentation=True)[0],
    )
    pipeline = build_gpu_random_pipeline("BHWC", segmentation=True)

    def augment_data(image, label):
        result = pipeline({"image": image, "label": label})
        return result["image"], result["label"]

    model = GPUAugmentedModel(build_segmentation_model((32, 48, 1)), augment_data)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy()],
        jit_compile=False,
    )

    history = model.fit(loader, epochs=1, verbose=0, shuffle=False)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


def _assert_pygrain_classification_batches(loader):
    """Consume PyGrain batches and validate transformed host arrays."""
    batches = list(loader)

    assert len(batches) == 2
    for images, labels in batches:
        assert isinstance(images, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert images.shape == (2, 32, 48, 1)
        assert labels.shape == (2, 1)
        assert np.isfinite(images).all()


@pytest.mark.integration
@pytest.mark.slow
def test_tensorflow_pygrain_supports_multiple_reader_threads():
    """Run CPU transforms concurrently in multiple threads in one process."""
    _assert_pygrain_classification_batches(
        _make_pygrain_loader(
            *make_dataset().classification_2d(),
            build_transform_pipelines("HWC", segmentation=False)[0],
            num_threads=2,
        )
    )


@pytest.mark.integration
@pytest.mark.slow
def test_tensorflow_pygrain_supports_multiple_worker_processes():
    """Run CPU transforms in multiple PyGrain worker processes."""
    _assert_pygrain_classification_batches(
        _make_pygrain_loader(
            *make_dataset().classification_2d(),
            build_transform_pipelines("HWC", segmentation=False)[0],
            worker_count=2,
            num_threads=1,
        )
    )
