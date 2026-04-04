"""
Configuration classes and JSON helpers for the nnU-Net pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path


class IntensityStats:
    """Per-modality intensity statistics."""

    def __init__(
        self,
        mean=0.0,
        std=1.0,
        min=0.0,
        max=1.0,
        percentile_00_5=0.0,
        percentile_99_5=1.0,
    ):
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.percentile_00_5 = percentile_00_5
        self.percentile_99_5 = percentile_99_5

    def to_dict(self):
        return vars(self)


class DatasetFingerprint:
    """
    Summary statistics extracted from a raw dataset.
    """

    def __init__(
        self,
        dataset_name="unknown",
        n_cases=0,
        n_classes=2,
        class_names=None,
        modalities=None,
        spacings=None,
        sizes=None,
        median_spacing=None,
        median_size=None,
        intensity_stats=None,
        class_stats=None,
        anisotropy_ratio=1.0,
        is_anisotropic=False,
        image_type="CT",
        task_type="multi-class",
        ignore_class_ids=None,
        target_class_ids=None,
        output_channels=None,
        spatial_dims=3,
    ):
        self.dataset_name = dataset_name
        self.n_cases = n_cases
        self.n_classes = n_classes
        self.class_names = (
            class_names
            if class_names is not None
            else [
                "background",
                "foreground",
            ]
        )
        self.modalities = modalities if modalities is not None else ["CT"]

        self.spacings = spacings if spacings is not None else []
        self.sizes = sizes if sizes is not None else []
        self.median_spacing = median_spacing if median_spacing is not None else [1.0, 1.0, 1.0]
        self.median_size = median_size if median_size is not None else [128, 128, 128]

        self.intensity_stats = intensity_stats if intensity_stats is not None else {}
        self.class_stats = class_stats if class_stats is not None else {}

        self.anisotropy_ratio = anisotropy_ratio
        self.is_anisotropic = is_anisotropic
        self.image_type = image_type

        self.task_type = task_type
        self.ignore_class_ids = ignore_class_ids if ignore_class_ids is not None else []
        self.target_class_ids = target_class_ids if target_class_ids is not None else []
        self.output_channels = (
            output_channels if output_channels is not None else self._default_output_channels()
        )
        self.spatial_dims = spatial_dims

    def _default_output_channels(self):
        if self.task_type == "binary":
            return 1
        if self.task_type == "multi-label":
            if self.target_class_ids:
                return len(self.target_class_ids)
            return max(1, len(self.class_names) - 1)
        return self.n_classes

    def to_dict(self):
        return vars(self)

    def to_json(self, path):
        save_json(self.to_dict(), path)

    @classmethod
    def from_json(cls, path):
        data = load_json(path)
        return cls(**data)


class NetworkConfig:
    """Architecture parameters for one U-Net configuration."""

    def __init__(
        self,
        spatial_dims=3,
        patch_size=None,
        batch_size=2,
        n_pooling=5,
        base_filters=32,
        max_filters=320,
        kernel_size=None,
        pool_op_kernel_sizes=None,
        conv_per_stage=2,
        deep_supervision=True,
        n_classes=2,
        n_modalities=1,
        output_activation="softmax",
    ):
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size if patch_size is not None else [128, 128, 128]
        self.batch_size = batch_size
        self.n_pooling = n_pooling
        self.base_filters = base_filters
        self.max_filters = max_filters
        self.kernel_size = kernel_size if kernel_size is not None else [3, 3, 3]
        self.pool_op_kernel_sizes = pool_op_kernel_sizes if pool_op_kernel_sizes is not None else []
        self.conv_per_stage = conv_per_stage
        self.deep_supervision = deep_supervision
        self.n_classes = n_classes
        self.n_modalities = n_modalities
        self.output_activation = output_activation

    def to_dict(self):
        return vars(self)


class PreprocessedCaseProperties:
    """
    Metadata required to invert preprocessing and drive patch sampling.
    """

    def __init__(
        self,
        case_id,
        original_spacing,
        target_spacing,
        original_shape,
        shape_after_resampling,
        shape_after_cropping,
        bbox,
        modalities,
        normalization_schemes,
        image_files,
        label_file=None,
        class_locations=None,
        item_type="multi-class",
        spatial_dims=3,
    ):
        self.case_id = case_id
        self.original_spacing = original_spacing
        self.target_spacing = target_spacing
        self.original_shape = original_shape
        self.shape_after_resampling = shape_after_resampling
        self.shape_after_cropping = shape_after_cropping
        self.bbox = bbox
        self.modalities = modalities
        self.normalization_schemes = normalization_schemes
        self.image_files = image_files
        self.label_file = label_file
        self.class_locations = class_locations if class_locations is not None else {}
        self.item_type = item_type
        self.spatial_dims = spatial_dims

    def to_dict(self):
        return vars(self)

    def to_json(self, path):
        save_json(self.to_dict(), path)

    @classmethod
    def from_json(cls, path):
        data = load_json(path)
        return cls(**data)


class nnUNetPlan:
    """
    Complete experiment plan produced by the planner.
    """

    def __init__(
        self,
        dataset_name="unknown",
        network_type="3d_fullres",
        target_spacing=None,
        plan_3d_fullres=None,
        plan_3d_lowres=None,
        plan_2d=None,
        use_cascade=False,
        normalization_schemes=None,
        splits_file="splits_final.json",
        preprocessed_properties_dirname="properties",
        task_type="multi-class",
        ignore_class_ids=None,
        target_class_ids=None,
        output_channels=None,
    ):
        self.dataset_name = dataset_name
        self.network_type = network_type
        self.target_spacing = target_spacing if target_spacing is not None else [1.0, 1.0, 1.0]

        self.plan_3d_fullres = plan_3d_fullres
        self.plan_3d_lowres = plan_3d_lowres
        self.plan_2d = plan_2d

        self.use_cascade = use_cascade
        self.normalization_schemes = (
            normalization_schemes if normalization_schemes is not None else ["z_score"]
        )
        self.splits_file = splits_file
        self.preprocessed_properties_dirname = preprocessed_properties_dirname
        self.task_type = task_type
        self.ignore_class_ids = ignore_class_ids if ignore_class_ids is not None else []
        self.target_class_ids = target_class_ids if target_class_ids is not None else []
        self.output_channels = output_channels

    def to_dict(self):
        d = vars(self).copy()
        if self.plan_3d_fullres:
            d["plan_3d_fullres"] = self.plan_3d_fullres.to_dict()
        if self.plan_3d_lowres:
            d["plan_3d_lowres"] = self.plan_3d_lowres.to_dict()
        if self.plan_2d:
            d["plan_2d"] = self.plan_2d.to_dict()
        return d

    def to_json(self, path):
        save_json(self.to_dict(), path)

    @classmethod
    def from_json(cls, path):
        data = load_json(path)
        for key in ("plan_3d_fullres", "plan_3d_lowres", "plan_2d"):
            if data.get(key) is not None:
                data[key] = NetworkConfig(**data[key])
        return cls(**data)


class TrainingConfig:
    """
    Hyperparameters and paths for the training pipeline.
    """

    def __init__(
        self,
        optimizer="sgd",
        lr=1e-2,
        weight_decay=3e-5,
        momentum=0.99,
        nesterov=True,
        lr_schedule="poly",
        poly_exp=0.9,
        n_epochs=1000,
        iters_per_epoch=250,
        gradient_accumulation_steps=1,
        deep_supervision=True,
        ds_loss_weights=None,
        use_ema=True,
        ema_momentum=0.9999,
        use_fg_oversampling=True,
        checkpoint_dir="outputs",
        save_every_n_epochs=50,
        do_rotation=True,
        do_scaling=True,
        do_elastic=True,
        do_mirror=True,
        do_gamma=True,
        do_gaussian_noise=True,
        val_every_n_epochs=1,
    ):
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        self.lr_schedule = lr_schedule
        self.poly_exp = poly_exp

        self.n_epochs = n_epochs
        self.iters_per_epoch = iters_per_epoch
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.deep_supervision = deep_supervision
        self.ds_loss_weights = ds_loss_weights if ds_loss_weights is not None else []

        self.use_ema = use_ema
        self.ema_momentum = ema_momentum

        self.use_fg_oversampling = use_fg_oversampling

        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_epochs = save_every_n_epochs

        self.do_rotation = do_rotation
        self.do_scaling = do_scaling
        self.do_elastic = do_elastic
        self.do_mirror = do_mirror
        self.do_gamma = do_gamma
        self.do_gaussian_noise = do_gaussian_noise

        self.val_every_n_epochs = val_every_n_epochs

    def to_dict(self):
        return vars(self)

    def to_json(self, path):
        save_json(self.to_dict(), path)

    @classmethod
    def from_json(cls, path):
        data = load_json(path)
        for legacy_key in (
            "use_mixed_precision",
            "use_wandb",
            "wandb_project",
            "wandb_run_name",
            "early_stopping_patience",
        ):
            data.pop(legacy_key, None)
        return cls(**data)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
