"""Backend-neutral fixtures and model builders for training integration tests."""

import numpy as np
import keras

from medicai.losses import BinaryDiceLoss
from medicai.metrics import BinaryDiceMetric
from medicai.transforms import (
    Compose,
    Flip,
    RandomChoice,
    Rotate90,
    ScaleIntensityRange,
    SpatialCrop,
)


class DatasetBuilder:
    """Build small deterministic classification and segmentation fixtures."""

    def __init__(self, num_samples: int = 4):
        self.num_samples = num_samples

    def classification_2d(self, spatial_shape: tuple[int, int] = (12, 12)):
        """Return ``(N, H, W, C)`` images and binary targets."""
        height, width = spatial_shape
        images = np.linspace(
            0.0, 1.0, num=self.num_samples * height * width, dtype=np.float32
        ).reshape(self.num_samples, height, width, 1)
        labels = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
        return images, labels[: self.num_samples]

    def classification_3d(self, spatial_shape: tuple[int, int, int] = (6, 6, 6)):
        """Return ``(N, D, H, W, C)`` volumes and binary targets."""
        depth, height, width = spatial_shape
        images = np.linspace(
            0.0, 1.0, num=self.num_samples * depth * height * width, dtype=np.float32
        ).reshape(self.num_samples, depth, height, width, 1)
        labels = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
        return images, labels[: self.num_samples]

    def segmentation_2d(self, spatial_shape: tuple[int, int] = (12, 12)):
        """Return ``(N, H, W, C)`` images and aligned binary masks."""
        height, width = spatial_shape
        images = np.linspace(
            0.0, 1.0, num=self.num_samples * height * width, dtype=np.float32
        ).reshape(self.num_samples, height, width, 1)
        return images, (images > 0.5).astype(np.float32)

    def segmentation_3d(self, spatial_shape: tuple[int, int, int] = (6, 6, 6)):
        """Return ``(N, D, H, W, C)`` volumes and aligned binary masks."""
        depth, height, width = spatial_shape
        images = np.linspace(
            0.0, 1.0, num=self.num_samples * depth * height * width, dtype=np.float32
        ).reshape(self.num_samples, depth, height, width, 1)
        return images, (images > 0.5).astype(np.float32)


def build_classification_model(input_shape):
    """Build and compile a small rank-aware binary classifier."""
    inputs = keras.Input(shape=input_shape)
    spatial_rank = len(input_shape) - 1
    conv = keras.layers.Conv2D if spatial_rank == 2 else keras.layers.Conv3D
    x = conv(4, 3, padding="same", activation="relu")(inputs)
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
        metrics=[keras.metrics.BinaryAccuracy()],
        jit_compile=False,
    )
    return model


def build_segmentation_model(input_shape, *, use_medicai_objectives: bool = True):
    """Build a rank-aware binary segmenter with optional Medicai objectives."""
    inputs = keras.Input(shape=input_shape)
    spatial_rank = len(input_shape) - 1
    conv = keras.layers.Conv2D if spatial_rank == 2 else keras.layers.Conv3D
    x = conv(4, 3, padding="same", activation="relu")(inputs)
    outputs = conv(1, 1, padding="same", activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    if use_medicai_objectives:
        loss = BinaryDiceLoss(from_logits=False, num_classes=1)
        metrics = [BinaryDiceMetric(from_logits=False, num_classes=1)]
    else:
        loss = "binary_crossentropy"
        metrics = []
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=metrics,
        jit_compile=False,
    )
    return model


def build_transform_pipelines(input_layout: str, *, segmentation: bool):
    """Return five representative pipelines for use inside training maps.

    The pipelines keep image and label geometry synchronized when
    ``segmentation=True``. They intentionally contain no data-loader logic;
    the backend-specific test decides how the returned samples are consumed.
    """
    keys = ["image", "label"] if segmentation else ["image"]
    return [
        Compose(
            [
                ScaleIntensityRange(
                    keys=["image"],
                    source_value_range=(-1.0, 1.0),
                    target_value_range=(0.0, 1.0),
                    clip=True,
                    input_layout=input_layout,
                )
            ]
        ),
        Compose(
            [Flip(keys=keys, spatial_axis=0, input_layout=input_layout)]
        ),
        Compose(
            [
                SpatialCrop(
                    keys=keys,
                    crop_size=(8, 8) if input_layout == "HWC" else (4, 4, 4),
                    crop_start=(2, 2) if input_layout == "HWC" else (1, 1, 1),
                    input_layout=input_layout,
                ),
                Flip(keys=keys, spatial_axis=0, input_layout=input_layout),
            ]
        ),
        Compose(
            [
                Flip(keys=keys, spatial_axis=0, input_layout=input_layout),
                Rotate90(
                    keys=keys,
                    k=1,
                    spatial_axis=(0, 1)
                    if input_layout == "HWC"
                    else (1, 2),
                    input_layout=input_layout,
                ),
            ]
        ),
        Compose(
            [
                RandomChoice(
                    transforms=[
                        Flip(keys=keys, spatial_axis=0, input_layout=input_layout),
                        Rotate90(
                            keys=keys,
                            k=2,
                            spatial_axis=(0, 1)
                            if input_layout == "HWC"
                            else (1, 2),
                            input_layout=input_layout,
                        ),
                    ],
                    num_choices=1,
                    prob=1.0,
                    seed=7,
                )
            ]
        ),
    ]


def apply_classification_pipeline(pipeline, image, label):
    """Apply one image-only pipeline and keep its classification target."""
    result = pipeline({"image": image})
    return result["image"], label


def apply_segmentation_pipeline(pipeline, image, label):
    """Apply one synchronized image/mask pipeline exactly once."""
    result = pipeline({"image": image, "label": label})
    return result["image"], result["label"]
