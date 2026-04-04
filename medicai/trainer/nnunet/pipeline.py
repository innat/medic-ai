"""
High-level API for the self-configuring nnU-Net workflow.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import keras
import numpy as np

from medicai.models.nnunet.dynamic_unet import build_unet_from_plan
from medicai.trainer.nnunet.cross_validation import (
    generate_splits,
    load_splits,
    normalize_case_id,
    save_splits,
)
from medicai.trainer.nnunet.data.dataset_fingerprint import fingerprint_dataset
from medicai.trainer.nnunet.data.manifest import DatasetManifest
from medicai.trainer.nnunet.data.preprocessing import preprocess_dataset
from medicai.trainer.nnunet.planning.planners import (
    nnUNetPlanner,
    nnUNetPlannerResEncL,
    nnUNetPlannerResEncM,
)
from medicai.trainer.nnunet.training.augmentations import (
    AugmentationConfig,
    AugmentationPipeline,
)
from medicai.trainer.nnunet.training.trainer import nnUNetTrainer
from medicai.trainer.nnunet.utils.config import (
    DatasetFingerprint,
    TrainingConfig,
    nnUNetPlan,
)
from medicai.trainer.nnunet.utils.io import (
    collapse_single_channel,
    infer_spatial_dims,
    load_medical_image,
    load_npz,
    normalize_layout,
    save_medical_image,
)
from medicai.utils.inference import sliding_window_inference


def _auto_detect_gpu_memory(default=8.0):
    """
    Query nvidia-smi for total memory in GB.
    Returns the minimum memory across all visible GPUs to be safe for planning.
    Falls back to `default` if the query fails or on non-NVIDIA accelerators.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        memories = [float(x) / 1024.0 for x in result.stdout.strip().split("\n") if x.strip()]
        if memories:
            return min(memories)
    except Exception:
        pass
    return float(default)


class nnUNetDataset(keras.utils.PyDataset):
    """
    Keras 3 PyDataset for loading and augmenting nnU-Net batches on the fly.
    """

    def __init__(
        self,
        case_files,
        batch_size,
        patch_size,
        augmentor,
        train_cfg,
        net_cfg,
        task_type="multi-class",
        augment=True,
        ensure_channel_last=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.case_files = case_files
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.augmentor = augmentor
        self.train_cfg = train_cfg
        self.net_cfg = net_cfg
        self.task_type = task_type
        self.augment = augment
        self.ensure_channel_last = ensure_channel_last

    def __len__(self):
        # nnU-Net defines epoch length directly in config, independent of actual dataset size
        return self.train_cfg.iters_per_epoch

    def __getitem__(self, index):
        # We sample randomly from the entire dataset for `batch_size` items
        # nnU-Net traditionally randomly samples indefinitely per iter.
        batch_images = []
        batch_labels = []

        while len(batch_images) < self.batch_size:
            case_file = random.choice(self.case_files)
            data = load_npz(case_file)
            image = data["data"]
            label = data.get("seg", None)

            if not self.ensure_channel_last:
                # Transpose from [C, ...] to [..., C]
                image_cl = np.transpose(
                    image,
                    list(range(1, len(image.shape))) + [0],
                )
                label_cl = label
                if label_cl is not None and label_cl.ndim == image.ndim:
                    label_cl = np.transpose(
                        label_cl,
                        list(range(1, len(label_cl.shape))) + [0],
                    )
            else:
                image_cl = image
                label_cl = label

            if self.augment:
                if (
                    label_cl is not None
                    and self.task_type != "multi-label"
                    and label_cl.ndim == len(image_cl.shape) - 1
                ):
                    label_cl = label_cl[..., np.newaxis]

                image_cl, label_cl = self.augmentor(
                    image_cl,
                    label_cl,
                    patch_size=self.patch_size,
                )

                if (
                    label_cl is not None
                    and self.task_type != "multi-label"
                    and label_cl.ndim == image_cl.ndim
                ):
                    label_cl = np.squeeze(label_cl, axis=-1)

            batch_images.append(np.asarray(image_cl, dtype=np.float32))

            if label_cl is None:
                shape = batch_images[-1].shape[:-1] if hasattr(batch_images[-1], "shape") else []
                batch_labels.append(np.zeros(shape, dtype=np.int64))
            else:
                label_dtype = np.float32 if self.task_type == "multi-label" else np.int64
                batch_labels.append(np.asarray(label_cl, dtype=label_dtype))

        image_batch = np.stack(batch_images, axis=0)
        label_batch = np.stack(batch_labels, axis=0)

        if self.train_cfg.deep_supervision and self.net_cfg and self.net_cfg.deep_supervision:
            y_dict = {"final": label_batch}
            for i in range(self.net_cfg.n_pooling - 1):
                y_dict[f"aux_{i}"] = label_batch
            return image_batch, y_dict

        return image_batch, label_batch


class nnUNetPipeline:
    """
    High-level API to run the end-to-end nnU-Net segmentation pipeline.
    """

    def __init__(
        self,
        dataset_dir,
        output_dir=None,
        manifest_file=None,
        ensure_channel_last=True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir) if output_dir else self.dataset_dir / "outputs"

        self.manifest_file = (
            Path(manifest_file) if manifest_file else self.dataset_dir / "manifest.json"
        )
        self.ensure_channel_last = ensure_channel_last
        self.configuration = "3d_fullres"
        self.custom_splits = None

        self.fingerprint_path = self.dataset_dir / "dataset_fingerprint.json"
        self.plan_path = self.dataset_dir / "nnunet_plans.json"
        self.preprocessed_dir = self.dataset_dir / "preprocessed"

    def setup(
        self,
        gpu_memory_gb="auto",
        modalities=None,
        class_names=None,
        image_type=None,
    ):
        self.fingerprint(modalities, class_names, image_type)
        self.plan(gpu_memory_gb)
        self.preprocess()

    def fingerprint(
        self,
        modalities=None,
        class_names=None,
        image_type=None,
    ):
        if not self.manifest_file.exists():
            raise FileNotFoundError(
                f"A valid manifest.json is strictly required outlining data topologies to build pipeline fingerprints. "
                f"Missing: {self.manifest_file}"
            )

        print(f"Using manifest configuration from {self.manifest_file}")
        manifest = DatasetManifest.from_json(self.manifest_file)

        return fingerprint_dataset(
            manifest=manifest,
            output_file=self.fingerprint_path,
            ensure_channel_last=self.ensure_channel_last,
        )

    def plan(self, gpu_memory_gb="auto", planner_name="nnUNetPlanner"):
        if gpu_memory_gb == "auto":
            gpu_memory_gb = _auto_detect_gpu_memory()

        if not self.fingerprint_path.exists():
            raise FileNotFoundError(f"Fingerprint not found: {self.fingerprint_path}")

        fp = DatasetFingerprint.from_json(self.fingerprint_path)
        planners_map = {
            "nnUNetPlanner": nnUNetPlanner,
            "nnUNetPlannerResEncM": nnUNetPlannerResEncM,
            "nnUNetPlannerResEncL": nnUNetPlannerResEncL,
        }
        planner_class = planners_map.get(planner_name, nnUNetPlanner)
        planner = planner_class(fingerprint=fp, gpu_memory_gb=gpu_memory_gb)
        plan = planner.plan(output_path=self.plan_path)
        available_configs = {
            name
            for name, cfg in {
                "3d_fullres": plan.plan_3d_fullres,
                "3d_lowres": plan.plan_3d_lowres,
                "2d": plan.plan_2d,
            }.items()
            if cfg is not None
        }
        if self.configuration not in available_configs and available_configs:
            self.configuration = (
                plan.network_type
                if plan.network_type in available_configs
                else sorted(available_configs)[0]
            )
        return plan

    def preprocess(self):
        if not self.plan_path.exists():
            raise FileNotFoundError(f"Plan not found: {self.plan_path}")

        fp = DatasetFingerprint.from_json(self.fingerprint_path)
        plan = nnUNetPlan.from_json(self.plan_path)
        preprocess_dataset(
            manifest_file=self.manifest_file,
            fingerprint=fp,
            plan=plan,
            output_dir=self.preprocessed_dir,
            configuration=self.configuration,
            ensure_channel_last=self.ensure_channel_last,
        )

    def _build_datasets(self, plan, train_cfg, n_folds):
        prep_dir = self.preprocessed_dir / self.configuration
        case_files = sorted(prep_dir.glob("*.npz"))
        if not case_files:
            raise RuntimeError(f"No preprocessed files found in {prep_dir}")

        case_ids = [normalize_case_id(f.stem) for f in case_files]
        splits_path = self.plan_path.parent / plan.splits_file

        if self.custom_splits is not None:
            splits = self.custom_splits
        elif splits_path.exists():
            splits = load_splits(splits_path)
        else:
            splits = generate_splits(case_ids, n_folds=n_folds)
            save_splits(splits, splits_path)

        net_cfg_map = {
            "3d_fullres": plan.plan_3d_fullres,
            "3d_lowres": plan.plan_3d_lowres,
            "2d": plan.plan_2d,
        }
        net_cfg = net_cfg_map.get(self.configuration)
        patch_size = net_cfg.patch_size if net_cfg else [128, 128, 128]
        batch_size = net_cfg.batch_size if net_cfg else 1

        augmentor = AugmentationPipeline(
            AugmentationConfig(),
            patch_size=patch_size,
            use_fg_oversampling=train_cfg.use_fg_oversampling,
        )

        def _create_pydataset(file_list, augment=True):
            return nnUNetDataset(
                case_files=list(file_list),
                batch_size=batch_size,
                patch_size=patch_size,
                augmentor=augmentor,
                train_cfg=train_cfg,
                net_cfg=net_cfg,
                task_type=plan.task_type,
                augment=augment,
                ensure_channel_last=self.ensure_channel_last,
            )

        return case_files, splits, _create_pydataset

    def train(
        self,
        fold=0,
        n_folds=5,
        epochs=1000,
        iters_per_epoch=250,
        lr=1e-2,
        lr_scheduler="poly",
        optimizer="sgd",
        callbacks=None,
        loss=None,
        metrics=None,
        train_dataset=None,
        val_dataset=None,
    ):
        plan = nnUNetPlan.from_json(self.plan_path)
        train_cfg = TrainingConfig(
            n_epochs=epochs,
            iters_per_epoch=iters_per_epoch,
            optimizer=optimizer,
            lr=lr,
            lr_schedule=lr_scheduler,
            checkpoint_dir=str(self.output_dir),
        )

        model = build_unet_from_plan(plan, configuration=self.configuration)

        if train_dataset is None or val_dataset is None:
            case_files, splits, create_pydataset = self._build_datasets(
                plan=plan,
                train_cfg=train_cfg,
                n_folds=n_folds,
            )
            if fold >= len(splits):
                raise ValueError(f"Requested fold {fold}, but only {len(splits)} splits exist.")

            fold_split = splits[fold]
            train_ids = set(fold_split["train"])
            val_ids = set(fold_split["val"])

            if train_dataset is None:
                train_dataset = create_pydataset(
                    [f for f in case_files if normalize_case_id(f.stem) in train_ids],
                    augment=True,
                )
            if val_dataset is None:
                val_dataset = create_pydataset(
                    [f for f in case_files if normalize_case_id(f.stem) in val_ids],
                    augment=False,
                )

        trainer = nnUNetTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            plan=plan,
            train_config=train_cfg,
            fold=fold,
            configuration=self.configuration,
            loss=loss,
            metrics=metrics,
        )
        return trainer.run(callbacks=callbacks)

    def predict(
        self,
        input_path,
        output_path,
        fold=0,
        model_weights_path=None,
        overlap=0.5,
        mode="gaussian",
        padding_mode="constant",
        cval=0.0,
    ):
        plan = nnUNetPlan.from_json(self.plan_path)
        model = build_unet_from_plan(plan, configuration=self.configuration)

        if not model_weights_path:
            model_weights_path = (
                self.output_dir
                / plan.dataset_name
                / plan.network_type
                / self.configuration
                / f"fold_{fold}"
                / "best_model.weights.h5"
            )

        if not Path(model_weights_path).exists():
            raise FileNotFoundError(f"Weights not found: {model_weights_path}")

        manifest = (
            DatasetManifest.from_json(self.manifest_file) if self.manifest_file.exists() else None
        )
        image_layout = None if manifest is None else manifest.image_layout

        net_cfg_map = {
            "3d_fullres": plan.plan_3d_fullres,
            "3d_lowres": plan.plan_3d_lowres,
            "2d": plan.plan_2d,
        }
        net_cfg = net_cfg_map.get(self.configuration)
        patch_size = net_cfg.patch_size if net_cfg else [128, 128, 128]
        n_mod = net_cfg.n_modalities if net_cfg else 1
        n_outputs = net_cfg.n_classes if net_cfg else 2

        dummy = np.zeros([1] + patch_size + [n_mod], dtype=np.float32)
        _ = model(dummy, training=False)
        model.load_weights(str(model_weights_path))

        image, affine, header, spacing = load_medical_image(
            input_path,
            ensure_channel_last=self.ensure_channel_last,
        )
        spatial_dims = infer_spatial_dims(image, spacing=spacing)
        image = normalize_layout(
            image,
            spatial_dims=spatial_dims,
            ensure_channel_last=self.ensure_channel_last,
            layout=image_layout,
        )
        image = collapse_single_channel(
            image,
            spatial_dims=spatial_dims,
            ensure_channel_last=self.ensure_channel_last,
        )
        if image.ndim == spatial_dims:
            image = image[..., np.newaxis]
        image = image[np.newaxis].astype(np.float32)

        if self.configuration == "2d":
            pred_probs = model.predict(image, verbose=1)
        else:
            pred_probs = sliding_window_inference(
                inputs=image,
                model=model,
                num_classes=n_outputs,
                roi_size=patch_size,
                sw_batch_size=1,
                overlap=overlap,
                mode=mode,
                padding_mode=padding_mode,
                cval=cval,
            )

        pred = self._postprocess_prediction(pred_probs[0], plan)

        save_medical_image(
            pred.astype(np.int16),
            affine,
            Path(output_path),
            header=header,
            dtype=np.int16,
        )

    def _postprocess_prediction(self, pred_probs, plan):
        if plan.task_type == "multi-label":
            return (pred_probs > 0.5).astype(np.int16)

        if plan.task_type == "binary":
            pred = (pred_probs[..., 0] > 0.5).astype(np.int16)
            if plan.target_class_ids:
                pred[pred > 0] = int(plan.target_class_ids[0])
            return pred

        return np.argmax(pred_probs, axis=-1).astype(np.int16)
