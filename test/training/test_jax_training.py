"""End-to-end JAX training coverage for transforms and Keras distribution."""

import keras
import numpy as np
import pytest

from test.training.common import (
    DatasetBuilder as make_dataset,
    GPUAugmentedModel,
    build_classification_model,
    build_gpu_random_pipeline,
    build_segmentation_model,
    build_transform_pipelines,
)


def _require_jax():
    """Skip this module unless the JAX Keras backend is active."""
    if keras.config.backend() != "jax":
        pytest.skip("JAX training coverage requires the JAX Keras backend.")
    try:
        import jax
    except ImportError:
        pytest.skip("JAX is not installed.")
    return jax


def _make_pygrain_loader(images, labels, pipeline):
    """Build a shuffled PyGrain loader for JAX sample-level transforms."""
    try:
        import grain.python as pygrain
    except ImportError:
        pytest.skip("PyGrain is not installed.")
    records = [
        {"image": image, "label": label}
        for image, label in zip(images, labels)
    ]

    def apply_transform(record):
        with keras.device("cpu:0"):
            result = pipeline(record)
        return {
            "image": keras.ops.convert_to_numpy(result["image"]),
            "label": keras.ops.convert_to_numpy(result["label"]),
        }

    dataset = (
        pygrain.MapDataset.source(records)
        .shuffle(seed=7)
        .map(apply_transform)
        .batch(batch_size=2, drop_remainder=True)
        .to_iter_dataset()
    )
    return dataset


def _fit_sample_transformed(*, segmentation, input_layout, input_shape, pipeline_index):
    """Train from a shuffled PyGrain loader with sample-level transforms."""
    pipeline = build_transform_pipelines(input_layout, segmentation=segmentation)[pipeline_index]
    if segmentation:
        images, labels = make_dataset().segmentation_2d()
        model = build_segmentation_model(input_shape)
    else:
        images, labels = make_dataset().classification_3d()
        model = build_classification_model(input_shape)
    loader = _make_pygrain_loader(images, labels, pipeline)

    history = model.fit(loader, epochs=1, verbose=0, shuffle=False)
    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


@pytest.mark.integration
def test_jax_training_uses_sample_transforms_for_2d_segmentation():
    """Train a 2D segmenter from PyGrain ``HWC`` samples."""
    _require_jax()
    _fit_sample_transformed(
        segmentation=True,
        input_layout="HWC",
        input_shape=(24, 24, 1),
        pipeline_index=2,
    )


@pytest.mark.integration
def test_jax_training_uses_sample_transforms_for_3d_classification():
    """Train a 3D classifier from PyGrain ``DHWC`` samples."""
    _require_jax()
    _fit_sample_transformed(
        segmentation=False,
        input_layout="DHWC",
        input_shape=(8, 16, 16, 1),
        pipeline_index=3,
    )


def _fit_model_augmented(*, segmentation, input_layout, input_shape):
    """Train with random transforms executed inside the JAX train step."""
    _require_jax()
    pipeline = build_gpu_random_pipeline(input_layout, segmentation=segmentation)
    if segmentation:
        images, labels = make_dataset().segmentation_2d(spatial_shape=(32, 48))
        base_model = build_segmentation_model(input_shape)
    else:
        images, labels = make_dataset().classification_3d()
        base_model = build_classification_model(input_shape)

    def augment_data(image, label):
        result = pipeline(
            {"image": image, "label": label}
            if segmentation
            else {"image": image}
        )
        if segmentation:
            return result["image"], result["label"]
        return result["image"], label

    model = GPUAugmentedModel(base_model, augment_data)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy()],
        jit_compile="auto",
    )
    history = model.fit(images, labels, epochs=1, verbose=0, shuffle=False)
    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


@pytest.mark.integration
def test_jax_training_applies_random_transforms_inside_train_step():
    """Apply batch random geometry inside a JAX classifier's train step."""
    _fit_model_augmented(
        segmentation=False,
        input_layout="BDHWC",
        input_shape=(8, 16, 16, 1),
    )


@pytest.mark.integration
def test_jax_training_applies_random_transforms_to_segmentation_batches():
    """Apply synchronized batch random geometry inside a JAX segmenter's train step."""
    _fit_model_augmented(
        segmentation=True,
        input_layout="BHWC",
        input_shape=(32, 48, 1),
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_jax_distribution_data_parallel_trains_on_multiple_devices():
    """Use Keras ``DataParallel`` when multiple JAX GPU devices are available."""
    jax = _require_jax()
    devices = jax.devices("gpu")
    if len(devices) < 2:
        pytest.skip("JAX multi-device coverage requires at least two GPUs.")

    keras.distribution.set_distribution(
        keras.distribution.DataParallel(devices=devices)
    )
    images, labels = make_dataset().classification_2d(spatial_shape=(32, 32))
    pipeline = build_transform_pipelines("HWC", segmentation=False)[1]
    loader = _make_pygrain_loader(images, labels, pipeline)
    model = build_classification_model((32, 32, 1))
    history = model.fit(loader, epochs=1, verbose=0, shuffle=False)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])
