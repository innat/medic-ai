"""PyGrain training coverage for sample- and model-side transforms."""

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


def _require_pygrain():
    try:
        import grain.python as pygrain
    except ImportError:
        pytest.skip("PyGrain is not installed.")
    return pygrain


class _ArraySource:
    """Small PyGrain source that applies a sample transform in ``__getitem__``."""

    def __init__(self, images, labels, pipeline):
        self.images = images
        self.labels = labels
        self.pipeline = pipeline

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        result = self.pipeline(
            {"image": self.images[index], "label": self.labels[index]}
        )
        return result["image"], result["label"]

    def __repr__(self):
        return f"_ArraySource(size={len(self)})"


def _make_loader(images, labels, pipeline):
    pygrain = _require_pygrain()
    return pygrain.load(
        _ArraySource(images, labels, pipeline),
        batch_size=2,
        num_epochs=1,
        worker_count=0,
    )


def _assert_finite_history(history):
    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


@pytest.mark.integration
def test_pygrain_training_accepts_sample_level_transforms():
    """Train a segmentation model from PyGrain-transformed samples."""
    images, labels = make_dataset().segmentation_2d()
    pipeline = build_transform_pipelines("HWC", segmentation=True)[2]
    loader = _make_loader(images, labels, pipeline)
    model = build_segmentation_model((8, 8, 1))

    history = model.fit(loader, epochs=1, verbose=0)

    _assert_finite_history(history)


@pytest.mark.integration
def test_pygrain_training_accepts_classification_samples():
    """Train a classifier from PyGrain samples transformed before batching."""
    images, labels = make_dataset().classification_2d()
    pipeline = build_transform_pipelines("HWC", segmentation=False)[0]
    loader = _make_loader(images, labels, pipeline)
    model = build_classification_model((12, 12, 1))

    history = model.fit(loader, epochs=1, verbose=0)

    _assert_finite_history(history)


@pytest.mark.integration
def test_pygrain_training_accepts_model_side_random_transforms():
    """Train through PyGrain while random transforms run in ``train_step``."""
    images, labels = make_dataset().segmentation_2d()
    loader = _make_loader(
        images,
        labels,
        build_transform_pipelines("HWC", segmentation=True)[0],
    )
    pipeline = build_gpu_random_pipeline("BHWC", segmentation=True)

    def augment_data(image, label):
        result = pipeline({"image": image, "label": label})
        return result["image"], result["label"]

    model = GPUAugmentedModel(build_segmentation_model((12, 12, 1)), augment_data)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy()],
        jit_compile=False,
    )

    history = model.fit(loader, epochs=1, verbose=0)

    _assert_finite_history(history)
