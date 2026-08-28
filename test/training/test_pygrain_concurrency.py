"""Concurrency coverage for PyGrain-backed TensorFlow input pipelines."""

import keras
import numpy as np
import pytest

from test.training.common import DatasetBuilder, build_transform_pipelines
from test.training.test_tensorflow_training import _ArraySource, _require_pygrain


def _load_with_concurrency(*, worker_count: int, num_threads: int):
    """Create a small PyGrain loader with explicit thread/process settings."""
    pygrain = _require_pygrain()
    images, labels = DatasetBuilder().classification_2d()
    pipeline = build_transform_pipelines("HWC", segmentation=False)[0]
    return pygrain.load(
        _ArraySource(images, labels, pipeline),
        batch_size=2,
        num_epochs=1,
        worker_count=worker_count,
        read_options=pygrain.ReadOptions(
            num_threads=num_threads,
            prefetch_buffer_size=2,
        ),
    )


def _assert_classification_batches(loader):
    """Consume all batches and validate the transformed host arrays."""
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
def test_pygrain_dataloader_supports_multiple_reader_threads():
    """Run CPU transforms concurrently in multiple threads in one process."""
    if keras.config.backend() != "tensorflow":
        pytest.skip("This training coverage targets the TensorFlow PyGrain path.")

    _assert_classification_batches(_load_with_concurrency(worker_count=0, num_threads=2))


@pytest.mark.integration
@pytest.mark.slow
def test_pygrain_dataloader_supports_multiple_worker_processes():
    """Run CPU transforms in multiple PyGrain worker processes."""
    if keras.config.backend() != "tensorflow":
        pytest.skip("This training coverage targets the TensorFlow PyGrain path.")

    _assert_classification_batches(_load_with_concurrency(worker_count=2, num_threads=1))
