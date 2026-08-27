"""Backend-neutral fixtures and model builders for training integration tests."""

import numpy as np
import keras

from medicai.losses import BinaryDiceLoss
from medicai.metrics import BinaryDiceMetric
from medicai.transforms import (
    Compose,
    Flip,
    RandomChoice,
    RandomFlip,
    RandomRotate,
    RandomRotate90,
    Rotate90,
    ScaleIntensityRange,
    Orientation,
    SpatialCrop,
    Spacing,
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

    def segmentation_3d_with_affine(
        self, spatial_shape: tuple[int, int, int] = (6, 6, 6)
    ):
        """Return 3D segmentation samples and one identity affine per sample."""
        images, labels = self.segmentation_3d(spatial_shape=spatial_shape)
        affine = np.eye(4, dtype=np.float32)
        affines = np.repeat(affine[None, ...], self.num_samples, axis=0)
        return images, labels, affines


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
    is_2d = input_layout in {"HWC", "BHWC"}
    first_spatial_axis = 0 if input_layout in {"HWC", "DHWC"} else 1
    rotation_axes = (
        (0, 1)
        if input_layout == "HWC"
        else (1, 2)
        if input_layout in {"DHWC", "BHWC"}
        else (2, 3)
    )
    crop_size = (8, 8) if is_2d else (4, 4, 4)
    crop_start = (2, 2) if is_2d else (1, 1, 1)
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
            [Flip(keys=keys, spatial_axis=first_spatial_axis, input_layout=input_layout)]
        ),
        Compose(
            [
                SpatialCrop(
                    keys=keys,
                    crop_size=crop_size,
                    crop_start=crop_start,
                    input_layout=input_layout,
                ),
                Flip(
                    keys=keys,
                    spatial_axis=first_spatial_axis,
                    input_layout=input_layout,
                ),
            ]
        ),
        Compose(
            [
                Flip(
                    keys=keys,
                    spatial_axis=first_spatial_axis,
                    input_layout=input_layout,
                ),
                Rotate90(
                    keys=keys,
                    k=1,
                    spatial_axis=rotation_axes,
                    input_layout=input_layout,
                ),
            ]
        ),
        Compose(
            [
                RandomChoice(
                    transforms=[
                        Flip(
                            keys=keys,
                            spatial_axis=first_spatial_axis,
                            input_layout=input_layout,
                        ),
                        Rotate90(
                            keys=keys,
                            k=2,
                            spatial_axis=rotation_axes,
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


def build_gpu_random_pipeline(input_layout: str, *, segmentation: bool):
    """Build a batch-layout pipeline for model-side random augmentation."""
    keys = ["image", "label"] if segmentation else ["image"]
    is_2d = input_layout == "BHWC"
    flip_axis = 1
    rotation_axes = (1, 2) if is_2d else (2, 3)
    return Compose(
        [
            RandomFlip(
                keys=keys,
                prob=1.0,
                spatial_axis=flip_axis,
                input_layout=input_layout,
                seed=11,
            ),
            RandomRotate90(
                keys=keys,
                prob=1.0,
                max_k=3,
                spatial_axis=rotation_axes,
                input_layout=input_layout,
                seed=13,
            ),
            RandomRotate(
                keys=keys,
                factor=0.1,
                prob=1.0,
                input_layout=input_layout,
                seed=17,
            ),
        ]
    )


def build_volume_geometry_pipeline():
    """Build the sample-level affine-aware pipeline used by dataloader tests."""
    return Compose(
        [
            Orientation(keys=["image", "label"], axcodes="RAS", input_layout="DHWC"),
            Spacing(
                keys=["image", "label"],
                pixdim=(2.0, 1.0, 1.0),
                interpolation=("trilinear", "nearest"),
                input_layout="DHWC",
            ),
        ]
    )
