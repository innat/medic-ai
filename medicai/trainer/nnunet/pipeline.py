import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from medicai.trainer.nnunet.data.augmentations import (
    AugmentationConfig,
    AugmentationPipeline,
)
from medicai.trainer.nnunet.data.dataset import nnUNetDataset
from medicai.trainer.nnunet.data.dataset_fingerprint import fingerprint_dataset
from medicai.trainer.nnunet.data.manifest import DatasetManifest
from medicai.trainer.nnunet.data.preprocessing import (
    _load_image_channels,
    _validate_label_alignment,
    preprocess_dataset,
)
from medicai.models.nnunet.dynamic_unet import build_unet_from_plan
from medicai.trainer.nnunet.data.cross_validation import (
    generate_custom_splits,
    generate_splits,
    load_splits,
    normalize_case_id,
    save_splits,
    validate_splits,
)
from medicai.trainer.nnunet.planning.planners import (
    nnUNetPlanner,
    nnUNetPlannerResEncL,
    nnUNetPlannerResEncM,
)
from medicai.trainer.nnunet.analysis import AnalysisReport
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
    normalize_layout_and_spacing,
    save_medical_image,
)
from medicai.utils.inference import sliding_window_inference

logger = logging.getLogger(__name__)
_DEFAULT_STEPS_PER_EPOCH = 250
_DEFAULT_NUM_THREADS = 4
_DEFAULT_SEED = 12345


class nnUNetPipeline:
    """
    High-level API to run the end-to-end nnU-Net segmentation pipeline.
    """

    def __init__(
        self,
        input_path,
        output_path=None,
        manifest_file=None,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else self.input_path / "outputs"

        self.manifest_file = (
            Path(manifest_file) if manifest_file else self.input_path / "manifest.json"
        )
        self.configuration = "3d_fullres"
        self._compiled_trainer = None
        self._compile_configuration = "auto"
        self._compile_kwargs = {}

        self.fingerprint_path = self.input_path / "dataset_fingerprint.json"
        self.plan_path = self.input_path / "nnunet_plans.json"
        self.preprocessed_dir = self.input_path / "preprocessed"

    def setup(self) -> None:
        report = self.analyze()
        report.raise_if_errors()
        if report.fingerprint is None:
            raise RuntimeError("Dataset analysis did not produce a fingerprint.")
        report.fingerprint.to_json(self.fingerprint_path)
        self.plan()
        self.preprocess()

    def analyze(self) -> AnalysisReport:
        """Validate a manifest-backed dataset and compute a non-persisted fingerprint.

        This method reads source images and labels but does not write into the
        dataset directory. Invalid cases are collected into the report; a
        fingerprint is computed only when all validation checks pass.
        """
        if not self.manifest_file.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_file}")

        manifest = DatasetManifest.from_json(self.manifest_file)
        errors = []
        warnings = []
        declared_label_ids = set(manifest.task.labels.values())
        allowed_label_ids = declared_label_ids | set(manifest.ignore_class_ids)
        class_voxel_counts = {label_id: 0 for label_id in declared_label_ids}
        region_voxel_counts = {name: 0 for name in manifest.task.regions}
        labeled_voxel_count = 0

        for item in manifest.items:
            missing_images = [path for path in item.images if not Path(path).is_file()]
            if missing_images:
                errors.append(f"Case {item.case_id}: missing image file(s): {missing_images}.")
                continue
            label_paths = (
                item.labels if isinstance(item.labels, list) else [item.labels]
            ) if item.labels is not None else []
            missing_labels = [path for path in label_paths if not Path(path).is_file()]
            if missing_labels:
                errors.append(f"Case {item.case_id}: missing label file(s): {missing_labels}.")
                continue

            try:
                _, spacing, shape, spatial_dims, affine = _load_image_channels(
                    item.images,
                    original_spacing_override=item.spacing,
                    image_layout=item.image_layout or manifest.image_layout,
                )
                _validate_label_alignment(
                    label_paths=item.labels,
                    image_shape=shape,
                    image_affine=affine,
                    spatial_dims=spatial_dims,
                    original_spacing=spacing,
                    original_spacing_override=item.spacing,
                    label_layout=item.label_layout or manifest.label_layout,
                )
            except Exception as exc:
                errors.append(f"Case {item.case_id}: {exc}")
                continue

            try:
                actual_label_ids = set()
                case_label_counts = {}
                for label_path in label_paths:
                    label, _, _, label_spacing = load_medical_image(label_path)
                    label_dims = infer_spatial_dims(
                        label,
                        spacing=item.spacing if item.spacing is not None else label_spacing,
                        is_3d=spatial_dims == 3,
                    )
                    label, _ = normalize_layout_and_spacing(
                        label,
                        label_dims,
                        spacing=item.spacing if item.spacing is not None else label_spacing,
                        layout=item.label_layout or manifest.label_layout,
                    )
                    label = collapse_single_channel(label, label_dims)
                    if not np.issubdtype(label.dtype, np.number) or not np.all(np.isfinite(label)):
                        raise ValueError("label values must be finite numeric values.")
                    if not np.all(label == np.floor(label)):
                        raise ValueError("categorical label maps must contain integer-valued IDs.")
                    label_ids, label_counts = np.unique(label, return_counts=True)
                    actual_label_ids.update(label_ids.astype(np.int64).tolist())
                    for label_id, count in zip(label_ids, label_counts):
                        key = int(label_id)
                        case_label_counts[key] = case_label_counts.get(key, 0) + int(count)
                unexpected = sorted(actual_label_ids - allowed_label_ids)
                if unexpected:
                    errors.append(
                        f"Case {item.case_id}: label {item.labels} contains undeclared "
                        f"label ID(s) {unexpected}; declared IDs are "
                        f"{sorted(allowed_label_ids)}."
                    )
                elif not (actual_label_ids - {0, *manifest.ignore_class_ids}):
                    warnings.append(f"Case {item.case_id}: label contains no foreground voxels.")
                if not unexpected:
                    labeled_voxel_count += sum(
                        count
                        for label_id, count in case_label_counts.items()
                        if label_id not in manifest.ignore_class_ids
                    )
                    for label_id, count in case_label_counts.items():
                        if label_id in class_voxel_counts:
                            class_voxel_counts[label_id] += count
                    if manifest.task.task_type == "region_based":
                        for name, member_ids in manifest.task.regions.items():
                            region_voxel_counts[name] += sum(
                                count
                                for label_id, count in case_label_counts.items()
                                if label_id in member_ids
                            )
            except Exception as exc:
                errors.append(f"Case {item.case_id}: unable to validate label values: {exc}")

        fingerprint = None
        recommendations = []
        if labeled_voxel_count:
            missing_classes = [
                name
                for name, label_id in manifest.task.labels.items()
                if label_id != 0 and class_voxel_counts[label_id] == 0
            ]
            if missing_classes:
                warnings.append(
                    "No train/validation voxels found for declared class(es): "
                    f"{missing_classes}."
                )
            missing_regions = [
                name for name, count in region_voxel_counts.items() if count == 0
            ]
            if missing_regions:
                warnings.append(
                    "No train/validation voxels found for declared region(s): "
                    f"{missing_regions}."
                )
        if not errors:
            try:
                fingerprint = fingerprint_dataset(manifest=manifest)
                if fingerprint.is_anisotropic:
                    recommendations.append(
                        "The dataset is anisotropic; compare the planned 2D and 3D configurations."
                    )
                if manifest.task.task_type == "region_based":
                    recommendations.append(
                        "Inspect per-region validation metrics and preserve overlapping region probabilities."
                    )
            except Exception as exc:
                errors.append(f"Fingerprint computation failed: {exc}")

        return AnalysisReport(
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            fingerprint=fingerprint,
            class_prevalence={
                name: class_voxel_counts[label_id] / max(labeled_voxel_count, 1)
                for name, label_id in manifest.task.labels.items()
            },
            region_prevalence={
                name: count / max(labeled_voxel_count, 1)
                for name, count in region_voxel_counts.items()
            },
        )

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
        )

    def plan(
        self,
        planner: str | None = None,
    ) -> nnUNetPlan:
        """Derive and save every suitable configuration for this dataset.

        Most users should leave ``planner`` unset. Advanced users can provide
        a registered planner name. Configuration choice is deferred to
        training; preprocessing caches all planned configurations by default.
        """
        if not self.fingerprint_path.exists():
            report = self.analyze()
            report.raise_if_errors()
            if report.fingerprint is None:
                raise RuntimeError("Dataset analysis did not produce a fingerprint.")
            report.fingerprint.to_json(self.fingerprint_path)

        fp = DatasetFingerprint.from_json(self.fingerprint_path)
        planners_map = {
            "nnUNetPlanner": nnUNetPlanner,
            "nnUNetPlannerResEncM": nnUNetPlannerResEncM,
            "nnUNetPlannerResEncL": nnUNetPlannerResEncL,
        }
        planner_name = planner or "nnUNetPlanner"
        try:
            planner_class = planners_map[planner_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown planner {planner_name!r}; available planners: {', '.join(planners_map)}."
            ) from exc
        # TODO: Add backend/device-aware memory estimation when reliable
        # discovery is available across GPU, TPU, and other accelerators.
        planner_instance = planner_class(fingerprint=fp)
        plan = planner_instance.plan()
        # Region targets use multiple sigmoid channels internally, but remain
        # a distinct public task contract from independent multi-label data.
        manifest = DatasetManifest.from_json(self.manifest_file)
        plan.task_type = manifest.task.task_type
        plan.regions = manifest.task.regions
        plan.regions_class_order = manifest.task.regions_class_order
        preferred = plan.network_type
        if preferred == "3d_cascade":
            preferred = "3d_fullres"
        if preferred not in plan.configurations:
            preferred = next(iter(plan.configurations))
        self.configuration = preferred
        plan.selected_configuration = preferred
        plan.to_json(self.plan_path)
        self._compiled_trainer = None
        return plan

    def _sync_configuration(self, plan: nnUNetPlan) -> None:
        """Use the configuration saved in the plan across pipeline restarts."""
        if self.manifest_file.is_file():
            task = DatasetManifest.from_json(self.manifest_file).task
            plan.task_type = task.task_type
            plan.regions = task.regions
            plan.regions_class_order = task.regions_class_order
        selected = plan.selected_configuration
        if selected is None:
            selected = plan.network_type
            if selected == "3d_cascade":
                selected = "3d_fullres"
        if selected not in plan.configurations:
            raise ValueError(
                f"Plan selects unavailable configuration {selected!r}; "
                f"available configurations: {', '.join(plan.configurations)}."
            )
        self.configuration = selected
        # Keep subsequent stages aligned with the configuration saved in the plan.
    def preprocess(
        self,
        configurations: str | list[str] | None = None,
        incremental: bool = True,
        num_workers: int | None = None,
    ) -> dict[str, Any]:
        """Preprocess selected plan configurations and report cache actions.

        Cache entries are reused only when source-file hashes and all
        preprocessing-relevant settings match the saved per-case properties.
        """
        if not self.plan_path.exists():
            raise FileNotFoundError(f"Plan not found: {self.plan_path}")

        fp = DatasetFingerprint.from_json(self.fingerprint_path)
        plan = nnUNetPlan.from_json(self.plan_path)
        self._sync_configuration(plan)
        if configurations is None:
            selected_configurations = list(plan.configurations)
        elif isinstance(configurations, str):
            selected_configurations = [configurations]
        else:
            selected_configurations = list(configurations)
        if not selected_configurations:
            raise ValueError("Select at least one configuration to preprocess.")
        invalid = [name for name in selected_configurations if name not in plan.configurations]
        if invalid:
            raise ValueError(
                f"Unavailable configuration(s) {invalid}; "
                f"available configurations: {', '.join(plan.configurations)}."
            )

        reports = {}
        for configuration in selected_configurations:
            reports[configuration] = preprocess_dataset(
                manifest_file=self.manifest_file,
                fingerprint=fp,
                plan=plan,
                output_dir=self.preprocessed_dir,
                configuration=configuration,
                incremental=incremental,
                max_cases=None,
                num_workers=num_workers,
            )
        return {"configurations": reports}

    def _build_datasets(
        self,
        plan,
        train_cfg,
        n_folds,
        *,
        configuration=None,
        splits=None,
        splitter=None,
        groups=None,
        num_threads=_DEFAULT_NUM_THREADS,
        seed=_DEFAULT_SEED,
    ):
        configuration = configuration or self.configuration
        prep_dir = self.preprocessed_dir / configuration
        case_files = sorted(prep_dir.glob("*.npz"))
        if not case_files:
            raise RuntimeError(f"No preprocessed files found in {prep_dir}")

        case_ids = [normalize_case_id(f.stem) for f in case_files]
        splits_path = self.plan_path.parent / plan.splits_file

        if splits is not None and splitter is not None:
            raise ValueError("Pass either explicit splits or a splitter, not both.")
        if groups is not None and splitter is None and splits is None:
            raise ValueError("groups require explicit splits or a splitter.")

        if splits is not None:
            splits = validate_splits(splits, case_ids, groups=groups)
        elif splitter is not None:
            splits = generate_custom_splits(case_ids, splitter, groups=groups)
        else:
            requested_n_folds = 5 if n_folds is None else n_folds
            if isinstance(requested_n_folds, bool) or not isinstance(requested_n_folds, int):
                raise ValueError("n_folds must be an integer of at least 2.")
            if requested_n_folds < 2:
                raise ValueError("n_folds must be an integer of at least 2.")
            splits = None
            if splits_path.exists():
                try:
                    cached_splits = validate_splits(load_splits(splits_path), case_ids)
                    if len(cached_splits) == requested_n_folds:
                        splits = cached_splits
                    else:
                        logger.info(
                            "Regenerating saved cross-validation splits: requested %d folds, "
                            "found %d.",
                            requested_n_folds,
                            len(cached_splits),
                        )
                except (OSError, TypeError, ValueError) as exc:
                    logger.warning("Ignoring invalid saved splits at %s: %s", splits_path, exc)
            if splits is None:
                splits = generate_splits(case_ids, n_folds=requested_n_folds, seed=seed)
                save_splits(splits, splits_path)

        if n_folds is not None and len(splits) != n_folds:
            raise ValueError(
                f"n_folds={n_folds} does not match the {len(splits)} supplied cross-validation folds."
            )

        net_cfg_map = {
            "3d_fullres": plan.plan_3d_fullres,
            "3d_lowres": plan.plan_3d_lowres,
            "2d": plan.plan_2d,
        }
        net_cfg = net_cfg_map.get(configuration)
        patch_size = net_cfg.patch_size if net_cfg else [128, 128, 128]
        planned_batch_size = net_cfg.batch_size if net_cfg else 1
        batch_size = planned_batch_size
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if num_threads < 1:
            raise ValueError("num_threads must be positive.")

        augmentor = AugmentationPipeline(
            AugmentationConfig(),
            patch_size=patch_size,
        )

        def _create_dataset(file_list, augment=True):
            sampler = nnUNetDataset(
                case_files=list(file_list),
                batch_size=batch_size,
                patch_size=patch_size,
                augmentor=augmentor,
                train_cfg=train_cfg,
                net_cfg=net_cfg,
                task_type=plan.task_type,
                augment=augment,
            )
            return sampler.to_pygrain(
                shuffle=augment,
                seed=seed,
                num_threads=num_threads,
            )

        return case_files, splits, _create_dataset

    def compile(self, configuration: str = "auto", **kwargs: Any) -> None:
        """Build and compile the planned segmentation model.

        Args:
            configuration: Planned configuration name, or ``"auto"`` to use
                the planner's recommended configuration.
            **kwargs: Arguments forwarded to :meth:`keras.Model.compile`, such
                as ``optimizer``, ``loss``, ``metrics``, ``jit_compile``, and
                ``run_eagerly``. Omitted objectives and optimizer use MedicAI's
                nnU-Net defaults.

        Raises:
            FileNotFoundError: If no saved plan is available.
            ValueError: If the requested configuration is not in the plan.
        """
        if not self.plan_path.is_file():
            raise FileNotFoundError(f"Plan not found: {self.plan_path}")
        plan = nnUNetPlan.from_json(self.plan_path)
        self._sync_configuration(plan)
        if configuration != "auto":
            if configuration not in plan.configurations:
                raise ValueError(
                    f"Configuration {configuration!r} is not in the saved plan; "
                    f"available configurations: {', '.join(plan.configurations)}."
                )
            self.configuration = configuration

        train_config = TrainingConfig(
            n_epochs=1000,
            iters_per_epoch=_DEFAULT_STEPS_PER_EPOCH,
            checkpoint_dir=str(self.output_path),
            use_ema="optimizer" not in kwargs or kwargs["optimizer"] is None,
        )
        model = build_unet_from_plan(plan, configuration=self.configuration)
        compile_kwargs = dict(kwargs)
        if compile_kwargs.get("optimizer") is None:
            compile_kwargs.pop("optimizer", None)
        if compile_kwargs.get("loss") is None:
            compile_kwargs.pop("loss", None)
        self._compile_configuration = configuration
        self._compile_kwargs = dict(compile_kwargs)
        self._compiled_trainer = nnUNetTrainer(
            model=model,
            train_dataset=None,
            val_dataset=None,
            plan=plan,
            train_config=train_config,
            fold=0,
            configuration=self.configuration,
            compile_kwargs=compile_kwargs,
        )

    def dataset(
        self,
        *,
        split: str = "train",
        fold: int = 0,
        n_folds: int | None = None,
        splits: list[dict[str, list[str]]] | None = None,
        splitter: Any | None = None,
        groups: dict[str, Any] | None = None,
        configuration: str | None = None,
        num_threads: int = _DEFAULT_NUM_THREADS,
        seed: int = _DEFAULT_SEED,
    ) -> Any:
        """Create a PyGrain patch stream from preprocessed cases.

        Batch size is fixed by the selected plan. The sampler exposes the
        nnU-Net training iteration count as its finite epoch length. The
        selected configuration must already have a preprocessed cache.

        Args:
            split: ``"train"`` for augmented patches or ``"validation"`` for
                unaugmented patches.
            fold: Cross-validation fold index.
            n_folds: Number of folds when splits are generated automatically.
                Defaults to five for the built-in KFold splitter; inferred from
                explicit folds or a custom splitter when omitted.
            splits: Explicit K-fold partitions with ``train`` and ``val`` case-ID lists.
            splitter: Scikit-learn-style splitter implementing ``split(X, y=None, groups=None)``.
            groups: Mapping from each case ID to its group, required by group-aware splitters.
            configuration: Planned configuration such as ``"3d_fullres"``.
            num_threads: PyGrain reader threads.
            seed: Seed used to shuffle training batches.

        Returns:
            PyGrain iterator yielding ``(image_batch, target_batch)`` pairs.

        Raises:
            FileNotFoundError: If no plan or preprocessed cache is available.
            ImportError: If PyGrain is not installed.
            ValueError: If the split or fold settings are invalid.
        """
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")
        if not self.plan_path.is_file():
            raise FileNotFoundError(f"Plan not found: {self.plan_path}")

        plan = nnUNetPlan.from_json(self.plan_path)
        if configuration is None and self._compiled_trainer is not None:
            selected_configuration = self._compiled_trainer.configuration
        else:
            self._sync_configuration(plan)
            selected_configuration = configuration or self.configuration
        if selected_configuration not in plan.configurations:
            raise ValueError(
                f"Configuration {selected_configuration!r} is not in the saved plan."
            )
        train_cfg = TrainingConfig(
            iters_per_epoch=_DEFAULT_STEPS_PER_EPOCH,
            checkpoint_dir=str(self.output_path),
        )
        case_files, splits, create_dataset = self._build_datasets(
            plan,
            train_cfg,
            n_folds,
            configuration=selected_configuration,
            splits=splits,
            splitter=splitter,
            groups=groups,
            num_threads=num_threads,
            seed=seed,
        )
        if fold < 0 or fold >= len(splits):
            raise ValueError(f"Requested fold {fold}, but only {len(splits)} splits exist.")
        split_key = "val" if split == "validation" else "train"
        ids = set(splits[fold][split_key])
        selected_files = [f for f in case_files if normalize_case_id(f.stem) in ids]
        if not selected_files:
            raise ValueError(f"Fold {fold} has no cases in the {split!r} split.")
        return create_dataset(selected_files, augment=split == "train")

    def train(
        self,
        fold: int = 0,
        n_folds: int | None = None,
        splits: list[dict[str, list[str]]] | None = None,
        splitter: Any | None = None,
        groups: dict[str, Any] | None = None,
        epochs: int = 1000,
        callbacks=None,
        resume: bool = False,
        x=None,
        validation_data=None,
        **fit_kwargs: Any,
    ) -> dict[str, Any]:
        """Fit the compiled model using an internal cross-validation fold.

        The patch batch size and steps per epoch are determined internally by
        the plan and nnU-Net sampler. Keras ``Model.fit`` options such as
        ``verbose``, ``initial_epoch`` and ``validation_freq`` may be forwarded
        with ``fit_kwargs``. Use :meth:`compile` to customize optimizer, loss,
        metrics, or compilation options.

        Args:
            fold: Generated cross-validation fold to train.
            n_folds: Number of default folds (five by default), or expected number
                of folds for supplied splits/a custom splitter. Inferred from
                supplied splits or a splitter when omitted.
            splits: Explicit K-fold partitions, each with ``train`` and ``val`` case-ID lists.
            splitter: Scikit-learn-style splitter implementing ``split(X, y=None, groups=None)``.
            groups: Mapping from each case ID to its group, required by group-aware splitters.
            epochs: Maximum number of training epochs.
            callbacks: Keras callbacks, including learning-rate and user checkpoint callbacks.
            resume: Resume an interrupted run from its latest training-state backup.
            x: Optional custom Keras training input. If omitted, the selected
                cross-validation training partition is sampled internally.
            validation_data: Optional custom validation data used with ``x``.
            **fit_kwargs: Additional supported Keras ``Model.fit`` keyword
                arguments. ``batch_size`` and ``steps_per_epoch`` are managed
                internally and cannot be overridden.

        Returns:
            dict: Keras training history keyed by metric name.

        Raises:
            ValueError: If fold, epoch, resume, or fit options are invalid.
        """
        forbidden_fit_args = {"batch_size", "steps_per_epoch", "callbacks", "epochs"}
        invalid_fit_args = forbidden_fit_args.intersection(fit_kwargs)
        if invalid_fit_args:
            names = ", ".join(sorted(invalid_fit_args))
            raise TypeError(f"{names} are managed by nnUNetPipeline.train().")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
            raise ValueError("fold must be a non-negative integer.")
        if n_folds is not None and (
            isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2
        ):
            raise ValueError("n_folds must be an integer of at least 2.")
        if splits is not None and splitter is not None:
            raise ValueError("Pass either splits or splitter, not both.")
        if groups is not None and splitter is None and splits is None:
            raise ValueError("groups require explicit splits or a splitter.")
        if not isinstance(resume, bool):
            raise ValueError("resume must be a boolean.")

        if self._compiled_trainer is None or getattr(
            self._compiled_trainer, "has_run", False
        ):
            self.compile(configuration=self._compile_configuration, **self._compile_kwargs)
        trainer = self._compiled_trainer
        plan = trainer.plan
        configuration = self.configuration
        train_config = trainer.cfg
        train_config.n_epochs = epochs
        train_config.iters_per_epoch = _DEFAULT_STEPS_PER_EPOCH

        external_x = x is not None
        if external_x and (splits is not None or splitter is not None or groups is not None):
            raise ValueError(
                "Cross-validation split arguments cannot be combined with custom x; "
                "provide already-split x and validation_data instead."
            )
        fold_split = None
        if x is None:
            case_files, splits, create_dataset = self._build_datasets(
                plan=plan,
                train_cfg=train_config,
                n_folds=n_folds,
                splits=splits,
                splitter=splitter,
                groups=groups,
                num_threads=_DEFAULT_NUM_THREADS,
                seed=_DEFAULT_SEED,
            )
            if fold >= len(splits):
                raise ValueError(f"Requested fold {fold}, but only {len(splits)} splits exist.")
            fold_split = splits[fold]
            train_ids = set(fold_split["train"])
            val_ids = set(fold_split["val"])
            x = create_dataset(
                [f for f in case_files if normalize_case_id(f.stem) in train_ids],
                augment=True,
            )
            if validation_data is None:
                validation_data = create_dataset(
                    [f for f in case_files if normalize_case_id(f.stem) in val_ids],
                    augment=False,
                )
        elif validation_data is None:
            raise ValueError("Provide validation_data when supplying a custom x dataset.")

        net_config = plan.configurations[configuration]
        planned_batch_size = net_config.batch_size
        if isinstance(x, (tuple, list)) and x and hasattr(x[0], "shape"):
            inferred_steps = math.ceil(int(x[0].shape[0]) / planned_batch_size)
        elif hasattr(x, "shape"):
            inferred_steps = math.ceil(int(x.shape[0]) / planned_batch_size)
        else:
            try:
                inferred_steps = len(x)
            except (TypeError, AttributeError):
                if external_x:
                    raise ValueError(
                        "Custom training data must have a known finite batch count; "
                        "steps_per_epoch is intentionally managed internally."
                    ) from None
                inferred_steps = train_config.iters_per_epoch
        if inferred_steps < 1:
            raise ValueError("Training data must contain at least one batch.")
        train_config.iters_per_epoch = inferred_steps
        if (isinstance(x, (tuple, list)) and x and hasattr(x[0], "shape")) or hasattr(
            x, "shape"
        ):
            fit_kwargs["batch_size"] = planned_batch_size

        trainer.fold = fold
        trainer.train_dataset = x
        trainer.val_dataset = validation_data
        trainer.output_dir = (
            Path(train_config.checkpoint_dir)
            / plan.dataset_name
            / plan.network_type
            / configuration
            / f"fold_{fold}"
        )
        trainer.run_metadata.update(
            {
                "n_folds": len(splits) if fold_split is not None else None,
                "steps_per_epoch": train_config.iters_per_epoch,
                "batch_size": planned_batch_size,
            }
        )
        if fold_split is not None:
            trainer.run_metadata.update(
                {
                    "fold_train_cases": fold_split["train"],
                    "fold_validation_cases": fold_split["val"],
                }
            )
        history = trainer.run(callbacks=callbacks, resume=resume, fit_kwargs=fit_kwargs)
        trainer.has_run = True
        return history

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
        configuration: str = "auto",
    ):
        plan = nnUNetPlan.from_json(self.plan_path)
        if configuration == "auto":
            if self._compiled_trainer is not None:
                self.configuration = self._compiled_trainer.configuration
            else:
                self._sync_configuration(plan)
        else:
            if configuration not in plan.configurations:
                raise ValueError(f"Configuration {configuration!r} is not in the saved plan.")
            self.configuration = configuration
        model = build_unet_from_plan(plan, configuration=self.configuration)

        if not model_weights_path:
            model_weights_path = (
                self.output_path
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
        )
        spatial_dims = infer_spatial_dims(image, spacing=spacing)
        image = normalize_layout(
            image,
            spatial_dims=spatial_dims,
            layout=image_layout,
        )
        image = collapse_single_channel(
            image,
            spatial_dims=spatial_dims,
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

        regions_class_order = plan.regions_class_order
        if regions_class_order is None and manifest is not None:
            regions_class_order = manifest.task.regions_class_order
        pred = self._postprocess_prediction(
            pred_probs[0], plan, regions_class_order=regions_class_order
        )

        save_medical_image(
            pred.astype(np.int16),
            affine,
            Path(output_path),
            header=header,
            dtype=np.int16,
        )

    def _postprocess_prediction(self, pred_probs, plan, regions_class_order=None):
        if plan.task_type == "region_based":
            if regions_class_order is None or len(regions_class_order) != pred_probs.shape[-1]:
                raise ValueError(
                    "Region-based prediction requires one regions_class_order value "
                    "for every model output channel."
                )
            segmentation = np.zeros(pred_probs.shape[:-1], dtype=np.int16)
            for region_idx, class_id in enumerate(regions_class_order):
                segmentation[pred_probs[..., region_idx] > 0.5] = class_id
            return segmentation

        if plan.task_type == "binary":
            pred = (pred_probs[..., 0] > 0.5).astype(np.int16)
            if plan.target_class_ids:
                pred[pred > 0] = int(plan.target_class_ids[0])
            return pred

        return np.argmax(pred_probs, axis=-1).astype(np.int16)
