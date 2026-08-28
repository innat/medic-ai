"""End-to-end Torch training coverage for migrated transforms."""

import os

import keras
import numpy as np
import pytest

from test.training.common import (
    DatasetBuilder as make_dataset,
    PyGrainSource,
    apply_classification_pipeline,
    apply_segmentation_pipeline,
    build_classification_model,
    build_segmentation_model,
    build_transform_pipelines,
)


def _require_torch():
    if keras.config.backend() != "torch":
        pytest.skip("Torch training coverage requires the Torch Keras backend.")
    try:
        import torch
    except ImportError:
        pytest.skip("Torch is not installed.")
    return torch


def _require_pygrain():
    _require_torch()
    try:
        import grain.python as pygrain
    except ImportError:
        pytest.skip("PyGrain is not installed.")
    return pygrain


class _TorchDataset:
    """CPU dataset that applies a Medicai sample transform before batching."""

    def __init__(self, images, labels, pipeline, *, segmentation):
        self.images = images
        self.labels = labels
        self.pipeline = pipeline
        self.segmentation = segmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        torch = _require_torch()
        image = self.images[index]
        label = self.labels[index]
        with keras.device("cpu:0"):
            if self.segmentation:
                result = apply_segmentation_pipeline(self.pipeline, image, label)
            else:
                result = apply_classification_pipeline(self.pipeline, image, label)
        return torch.as_tensor(result[0]), torch.as_tensor(result[1])


def _fit_torch_dataset(
    images,
    labels,
    *,
    input_layout: str,
    input_shape: tuple[int, ...],
    pipeline_index: int,
    segmentation: bool,
    num_workers: int = 0,
    fit: bool = True,
):
    torch = _require_torch()
    from torch.utils.data import DataLoader

    pipeline = build_transform_pipelines(input_layout, segmentation=segmentation)[
        pipeline_index
    ]
    loader_kwargs = {
        "batch_size": 2,
        "shuffle": False,
        "num_workers": num_workers,
    }
    if num_workers:
        torch.set_num_threads(1)
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["timeout"] = 30
    dataset = DataLoader(
        _TorchDataset(images, labels, pipeline, segmentation=segmentation),
        **loader_kwargs,
    )
    if not fit:
        batch_images, batch_labels = next(iter(dataset))
        assert tuple(batch_images.shape) == (2, 24, 24, 1)
        assert tuple(batch_labels.shape) == (2, 24, 24, 1)
        return

    model = (
        build_segmentation_model(input_shape)
        if segmentation
        else build_classification_model(input_shape)
    )
    history = model.fit(dataset, epochs=1, verbose=0, shuffle=False)

    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])
    assert torch.cuda.is_available() or not any(
        parameter.is_cuda for parameter in model.parameters()
    )


@pytest.mark.integration
def test_torch_dataloader_trains_2d_classification_with_migrated_transforms():
    """Train a 2D classifier from CPU Torch DataLoader samples."""
    images, labels = make_dataset().classification_2d()
    _fit_torch_dataset(
        images,
        labels,
        input_layout="HWC",
        input_shape=(32, 48, 1),
        pipeline_index=0,
        segmentation=False,
    )


@pytest.mark.integration
def test_torch_dataloader_trains_2d_segmentation_with_crop_pipeline():
    """Train a 2D segmenter with synchronized crop and flip transforms."""
    images, labels = make_dataset().segmentation_2d()
    _fit_torch_dataset(
        images,
        labels,
        input_layout="HWC",
        input_shape=(24, 24, 1),
        pipeline_index=2,
        segmentation=True,
    )


@pytest.mark.integration
def test_torch_dataloader_trains_3d_classification_with_rotation_pipeline():
    """Train a 3D classifier with synchronized deterministic geometry."""
    images, labels = make_dataset().classification_3d()
    _fit_torch_dataset(
        images,
        labels,
        input_layout="DHWC",
        input_shape=(8, 16, 16, 1),
        pipeline_index=3,
        segmentation=False,
    )


@pytest.mark.integration
def test_torch_dataloader_trains_3d_segmentation_with_random_choice_pipeline():
    """Train a 3D segmenter with synchronized ``RandomChoice`` geometry."""
    images, labels = make_dataset().segmentation_3d()
    _fit_torch_dataset(
        images,
        labels,
        input_layout="DHWC",
        input_shape=(8, 16, 16, 1),
        pipeline_index=4,
        segmentation=True,
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("MEDICAI_RUN_TORCH_MULTIPROCESS") != "1",
    reason="Opt in to Torch multiprocessing coverage with MEDICAI_RUN_TORCH_MULTIPROCESS=1.",
)
def test_torch_dataloader_supports_multiple_worker_processes():
    """Apply CPU transforms from multiple Torch DataLoader worker processes."""
    images, labels = make_dataset().segmentation_2d()
    _fit_torch_dataset(
        images,
        labels,
        input_layout="HWC",
        input_shape=(24, 24, 1),
        pipeline_index=2,
        segmentation=True,
        num_workers=2,
        fit=False,
    )


def _make_pygrain_loader(images, labels, pipeline):
    pygrain = _require_pygrain()
    return pygrain.load(
        PyGrainSource(images, labels, pipeline),
        batch_size=2,
        num_epochs=1,
        worker_count=0,
    )


@pytest.mark.integration
def test_torch_pygrain_trains_2d_classification():
    """Train a classifier from PyGrain samples under the Torch backend."""
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
def test_torch_pygrain_trains_2d_segmentation():
    """Train a segmenter from PyGrain samples under the Torch backend."""
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
