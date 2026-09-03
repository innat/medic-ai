"""Backend-neutral fixtures and model builders for training integration tests."""

import keras
import numpy as np

from medicai.losses import BinaryDiceLoss
from medicai.metrics import BinaryDiceMetric
from medicai.transforms import (
    Compose,
    CropForeground,
    Flip,
    NormalizeIntensity,
    Orientation,
    RandomChoice,
    RandomCropByPosNegLabel,
    RandomCutOut,
    RandomFlip,
    RandomRotate,
    RandomRotate90,
    RandomShiftIntensity,
    RandomSpatialCrop,
    Resize,
    Rotate90,
    ScaleIntensityRange,
    ShiftIntensity,
    Spacing,
    SpatialCrop,
)


class GPUAugmentedModel(keras.Model):
    """Wrap a model and apply a batch transform inside backend-specific steps."""

    def __init__(self, model, augment_data, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.augment_data = augment_data

    def train_step(self, *args, **kwargs):
        """Dispatch augmentation while preserving each backend's signature."""
        backend = keras.config.backend()
        if backend == "jax":
            return self._jax_train_step(*args, **kwargs)
        if backend == "tensorflow":
            return self._tensorflow_train_step(*args, **kwargs)
        if backend == "torch":
            return self._torch_train_step(*args, **kwargs)
        raise ValueError(f"Unsupported Keras backend: {backend!r}")

    def _jax_train_step(self, state, data):
        x, y = data
        x, y = self.augment_data(x, y)
        return super().train_step(state, (x, y))

    def _tensorflow_train_step(self, data):
        x, y = data
        x, y = self.augment_data(x, y)
        return super().train_step((x, y))

    def _torch_train_step(self, data):
        x, y = data
        x, y = self.augment_data(x, y)
        return super().train_step((x, y))

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)


class DatasetBuilder:
    """Build small deterministic classification and segmentation fixtures."""

    def __init__(self, num_samples: int = 4):
        self.num_samples = num_samples

    def classification_2d(self, spatial_shape: tuple[int, int] = (32, 48)):
        """Return ``(N, H, W, C)`` images and binary targets."""
        height, width = spatial_shape
        images = np.linspace(
            0.0, 1.0, num=self.num_samples * height * width, dtype=np.float32
        ).reshape(self.num_samples, height, width, 1)
        labels = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
        return images, labels[: self.num_samples]

    def classification_3d(self, spatial_shape: tuple[int, int, int] = (8, 16, 16)):
        """Return ``(N, D, H, W, C)`` volumes and binary targets."""
        depth, height, width = spatial_shape
        images = np.linspace(
            0.0, 1.0, num=self.num_samples * depth * height * width, dtype=np.float32
        ).reshape(self.num_samples, depth, height, width, 1)
        labels = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
        return images, labels[: self.num_samples]

    def segmentation_2d(self, spatial_shape: tuple[int, int] = (32, 48)):
        """Return ``(N, H, W, C)`` images and aligned binary masks."""
        height, width = spatial_shape
        images = np.linspace(
            0.0, 1.0, num=self.num_samples * height * width, dtype=np.float32
        ).reshape(self.num_samples, height, width, 1)
        return images, (images > 0.5).astype(np.float32)

    def segmentation_3d(self, spatial_shape: tuple[int, int, int] = (8, 16, 16)):
        """Return ``(N, D, H, W, C)`` volumes and aligned binary masks."""
        depth, height, width = spatial_shape
        images = np.linspace(
            0.0, 1.0, num=self.num_samples * depth * height * width, dtype=np.float32
        ).reshape(self.num_samples, depth, height, width, 1)
        return images, (images > 0.5).astype(np.float32)

    def segmentation_3d_with_affine(self, spatial_shape: tuple[int, int, int] = (8, 16, 16)):
        """Return 3D segmentation samples and one identity affine per sample."""
        images, labels = self.segmentation_3d(spatial_shape=spatial_shape)
        affine = np.eye(4, dtype=np.float32)
        affines = np.repeat(affine[None, ...], self.num_samples, axis=0)
        return images, labels, affines


class PyGrainSource:
    """Small backend-neutral PyGrain source for transformed CPU samples."""

    def __init__(self, images, labels, pipeline, affines=None):
        self.images = images
        self.labels = labels
        self.pipeline = pipeline
        self.affines = affines

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data = {"image": self.images[index], "label": self.labels[index]}
        meta = None if self.affines is None else {"affine": self.affines[index]}
        with keras.device("cpu:0"):
            result = self.pipeline(data, meta)
        return result["image"], result["label"]

    def __repr__(self):
        return f"PyGrainSource(size={len(self)})"


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
        jit_compile="auto",
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
        jit_compile="auto",
    )
    return model


def build_multi_output_classification_model(input_shape):
    """Build a classifier with two named outputs from one image input."""
    image = keras.Input(shape=input_shape, name="image")
    spatial_rank = len(input_shape) - 1
    conv = keras.layers.Conv2D if spatial_rank == 2 else keras.layers.Conv3D
    pooling = (
        keras.layers.GlobalAveragePooling2D
        if spatial_rank == 2
        else keras.layers.GlobalAveragePooling3D
    )
    features = pooling()(conv(4, 3, padding="same", activation="relu")(image))
    outputs = {
        "class_output": keras.layers.Dense(1, activation="sigmoid", name="class_output")(features),
        "intensity_output": keras.layers.Dense(1, name="intensity_output")(features),
    }
    model = keras.Model(image, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            "class_output": "binary_crossentropy",
            "intensity_output": "mse",
        },
        jit_compile="auto",
    )
    return model


def build_multi_input_classification_model(input_shape):
    """Build a classifier that consumes two named image inputs."""
    image_1 = keras.Input(shape=input_shape, name="image_1")
    image_2 = keras.Input(shape=input_shape, name="image_2")
    spatial_rank = len(input_shape) - 1
    conv = keras.layers.Conv2D if spatial_rank == 2 else keras.layers.Conv3D
    pooling = (
        keras.layers.GlobalAveragePooling2D
        if spatial_rank == 2
        else keras.layers.GlobalAveragePooling3D
    )

    def encode(image):
        features = conv(4, 3, padding="same", activation="relu")(image)
        return pooling()(features)

    features = keras.layers.Concatenate()([encode(image_1), encode(image_2)])
    output = keras.layers.Dense(1, activation="sigmoid")(features)
    model = keras.Model({"image_1": image_1, "image_2": image_2}, output)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy()],
        jit_compile="auto",
    )
    return model


def build_multi_input_output_classification_model(input_shape):
    """Build a model with two named image inputs and two named outputs."""
    image_1 = keras.Input(shape=input_shape, name="image_1")
    image_2 = keras.Input(shape=input_shape, name="image_2")
    spatial_rank = len(input_shape) - 1
    conv = keras.layers.Conv2D if spatial_rank == 2 else keras.layers.Conv3D
    pooling = (
        keras.layers.GlobalAveragePooling2D
        if spatial_rank == 2
        else keras.layers.GlobalAveragePooling3D
    )

    def encode(image):
        features = conv(4, 3, padding="same", activation="relu")(image)
        return pooling()(features)

    features = keras.layers.Concatenate()([encode(image_1), encode(image_2)])
    outputs = {
        "class_output": keras.layers.Dense(1, activation="sigmoid", name="class_output")(features),
        "intensity_output": keras.layers.Dense(1, name="intensity_output")(features),
    }
    model = keras.Model({"image_1": image_1, "image_2": image_2}, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            "class_output": "binary_crossentropy",
            "intensity_output": "mse",
        },
        jit_compile="auto",
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
        (0, 1) if input_layout == "HWC" else (1, 2) if input_layout in {"DHWC", "BHWC"} else (2, 3)
    )
    crop_size = (24, 24) if is_2d else (6, 12, 12)
    crop_start = (4, 4) if is_2d else (1, 2, 2)
    pipelines = [
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
        Compose([Flip(keys=keys, spatial_axis=first_spatial_axis, input_layout=input_layout)]),
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

    # Keep the original five pipeline indices stable; append focused cases for
    # the remaining concrete transforms so backend training tests can opt into
    # them without changing existing callers.
    pipelines.extend(
        [
            Compose(
                [
                    NormalizeIntensity(
                        keys=["image"],
                        channel_wise=True,
                        input_layout=input_layout,
                    )
                ]
            ),
            Compose([ShiftIntensity(keys=["image"], offset=0.1, input_layout=input_layout)]),
            Compose(
                [
                    RandomFlip(
                        keys=keys,
                        spatial_axis=first_spatial_axis,
                        prob=1.0,
                        seed=11,
                        input_layout=input_layout,
                    )
                ]
            ),
            Compose(
                [
                    RandomShiftIntensity(
                        keys=["image"],
                        offset=0.1,
                        prob=1.0,
                        seed=13,
                        input_layout=input_layout,
                    )
                ]
            ),
            Compose(
                [
                    RandomSpatialCrop(
                        keys=keys,
                        crop_size=crop_size,
                        random_center=False,
                        input_layout=input_layout,
                        seed=17,
                    )
                ]
            ),
            Compose(
                [
                    Resize(
                        keys=keys,
                        interpolation=(
                            (("bilinear", "nearest") if is_2d else ("trilinear", "nearest"))
                            if segmentation
                            else ("bilinear" if is_2d else "trilinear")
                        ),
                        target_shape=crop_size,
                        input_layout=input_layout,
                    )
                ]
            ),
        ]
    )

    if input_layout in {"HWC", "DHWC"}:
        pipelines.append(
            Compose(
                [
                    CropForeground(
                        keys=keys,
                        source_key="image",
                        k_divisible=(8, 8) if is_2d else (2, 4, 4),
                        input_layout=input_layout,
                    )
                ]
            )
        )

    if segmentation:
        pipelines.append(
            Compose(
                [
                    RandomCropByPosNegLabel(
                        keys=["image", "label"],
                        target_shape=crop_size,
                        pos=1,
                        neg=1,
                        input_layout=input_layout,
                        seed=19,
                    )
                ]
            )
        )
        if is_2d:
            pipelines.append(
                Compose(
                    [
                        RandomCutOut(
                            keys=["image"],
                            mask_size=(4, 4),
                            num_cuts=1,
                            prob=1.0,
                            input_layout=input_layout,
                            seed=23,
                        )
                    ]
                )
            )

    if is_2d:
        # This case is intended for square 2D fixtures because quarter-turn
        # rotation preserves shape only when its spatial plane is square.
        pipelines.append(
            Compose(
                [
                    RandomRotate90(
                        keys=keys,
                        max_k=3,
                        prob=1.0,
                        spatial_axis=rotation_axes,
                        input_layout=input_layout,
                        seed=29,
                    )
                ]
            )
        )

    return pipelines


def apply_classification_pipeline(pipeline, image, label):
    """Apply one image-only pipeline and keep its classification target."""
    with keras.device("cpu:0"):
        result = pipeline({"image": image})
    return result["image"], label


def apply_segmentation_pipeline(pipeline, image, label):
    """Apply one synchronized image/mask pipeline exactly once."""
    with keras.device("cpu:0"):
        result = pipeline({"image": image, "label": label})
    return result["image"], result["label"]


def build_gpu_random_pipeline(
    input_layout: str,
    *,
    segmentation: bool,
):
    """Build a batch-layout pipeline for model-side random augmentation."""
    keys = ["image", "label"] if segmentation else ["image"]
    flip_axis = 1
    transforms = [
        RandomFlip(
            keys=keys,
            prob=1.0,
            spatial_axis=flip_axis,
            input_layout=input_layout,
            seed=11,
        )
    ]
    transforms.append(
        RandomRotate(
            keys=keys,
            factor=0.1,
            prob=1.0,
            input_layout=input_layout,
            seed=17,
        )
    )
    return Compose(transforms)


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
