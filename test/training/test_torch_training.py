"""End-to-end Torch training coverage for migrated transforms."""

import keras
import numpy as np
import pytest

from test.training.common import (
    DatasetBuilder as make_dataset,
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
):
    torch = _require_torch()
    from torch.utils.data import DataLoader

    pipeline = build_transform_pipelines(input_layout, segmentation=segmentation)[
        pipeline_index
    ]
    dataset = DataLoader(
        _TorchDataset(images, labels, pipeline, segmentation=segmentation),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
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
