import json
import copy
from dataclasses import dataclass
from pathlib import Path

import keras

from medicai.losses import BinaryDiceCELoss, SparseDiceCELoss
from medicai.metrics.dice import BinaryDiceMetric, SparseDiceMetric
from medicai.trainers.nnunet.models.dynamic_unet import build_unet_from_config
from medicai.trainers.nnunet.training.specs import OutputSpec


@dataclass(frozen=True)
class NetworkContext:
    """Immutable planner context passed to trainer network hooks."""

    plan: object
    plan_config: object
    configuration: str


class _EpochCheckpoint(keras.callbacks.Callback):
    """Save weights at a fixed epoch interval (not a batch interval)."""

    def __init__(self, filepath, every_n_epochs):
        super().__init__()
        self.filepath = str(filepath)
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            self.model.save_weights(self.filepath)

# Trainer


class nnUNetTrainer:
    """Manage nnU-Net compilation and the Keras training loop.

    Subclass this trainer to customize the defaults used by
    :meth:`nnUNetPipeline.compile`. The supported extension methods are
    :meth:`create_loss` and :meth:`create_metrics`. Call the parent method
    when extending a default; direct ``loss``, ``metrics``, or ``optimizer``
    arguments passed to ``pipeline.compile`` replace the corresponding
    trainer result. The default optimizer is internal and is replaced directly
    through ``pipeline.compile(optimizer=...)``.

    A trainer can be constructed without ``model`` as a reusable recipe. The
    pipeline clones that recipe into a fresh runtime trainer for each fold.

    Parameters
    ----------
    model         : compiled or uncompiled Keras model; omit for a recipe
    train_dataset : iterable yielding (image, label) batches
                    image: [B, *spatial, C], label: [B, *spatial]
    val_dataset   : same format, used for validation
    plan          : nnUNetPlan (provides n_classes, patch_size, …)
    train_config  : TrainingConfig (hyperparameters)
    fold          : fold index for cross-validation (0–4)
    configuration : which plan entry to use ('3d_fullres', '2d', …)
    loss          : custom loss function or list of losses (will be summed)
    metrics       : list of custom metric functions/objects
    """

    def __init__(
        self,
        model=None,
        train_dataset=None,
        val_dataset=None,
        plan=None,
        train_config=None,
        fold=0,
        configuration="3d_fullres",
        loss=None,
        metrics=None,
        run_metadata=None,
        compile_kwargs=None,
        auto_compile=True,
    ):
        # A trainer constructed without runtime objects is a reusable recipe.
        # Subclasses can attach architecture options before the pipeline clones
        # the recipe into a fold-specific runtime trainer.
        if model is None:
            self._recipe_mode = True
            return

        self._recipe_mode = False
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.plan = plan
        self.cfg = train_config
        self.fold = fold
        self.configuration = configuration
        self.run_metadata = run_metadata or {}

        # Resolve network config for this configuration
        net_cfg_map = {
            "3d_fullres": plan.plan_3d_fullres,
            "3d_lowres": plan.plan_3d_lowres,
            "2d": plan.plan_2d,
        }
        self.net_cfg = net_cfg_map.get(configuration)
        self.n_classes = self.net_cfg.n_classes if self.net_cfg else 2
        self.task_type = getattr(plan, "task_type", "multi-class")

        # Output directory
        self.output_dir = (
            Path(train_config.checkpoint_dir)
            / plan.dataset_name
            / plan.network_type
            / configuration
            / f"fold_{fold}"
        )
        self.custom_loss_fn = loss
        self.custom_metrics = metrics
        self.compile_kwargs = dict(compile_kwargs or {})
        self.use_default_lr_schedule = "optimizer" not in self.compile_kwargs

        if auto_compile:
            self.compile()

        # Training state
        self.best_val_dice = -1.0
        self.epochs_no_improve = 0

    def create_runtime(
        self,
        *,
        model,
        train_dataset,
        val_dataset,
        plan,
        train_config,
        fold=0,
        configuration="3d_fullres",
    ):
        """Clone this recipe into an uncompiled fold-specific trainer.

        Recipe attributes, including subclass constructor options, are copied
        before runtime state is attached. The returned trainer has no shared
        model, optimizer, loss, metric, or training state with another fold.
        """
        if not getattr(self, "_recipe_mode", False):
            raise ValueError(
                "create_runtime() requires a recipe trainer created without a model."
            )
        runtime = copy.deepcopy(self)
        runtime._recipe_mode = False
        runtime.model = model
        runtime.train_dataset = train_dataset
        runtime.val_dataset = val_dataset
        runtime.plan = plan
        runtime.cfg = train_config
        runtime.fold = fold
        runtime.configuration = configuration
        runtime.run_metadata = {}
        runtime.net_cfg = {
            "3d_fullres": plan.plan_3d_fullres,
            "3d_lowres": plan.plan_3d_lowres,
            "2d": plan.plan_2d,
        }.get(configuration)
        runtime.n_classes = runtime.net_cfg.n_classes if runtime.net_cfg else 2
        runtime.task_type = getattr(plan, "task_type", "multi-class")
        runtime.output_dir = (
            Path(train_config.checkpoint_dir)
            / plan.dataset_name
            / plan.network_type
            / configuration
            / f"fold_{fold}"
        )
        runtime.custom_loss_fn = None
        runtime.custom_metrics = None
        runtime.compile_kwargs = {}
        runtime.use_default_lr_schedule = True
        runtime.best_val_dice = -1.0
        runtime.epochs_no_improve = 0
        return runtime

    def compile(self, **kwargs):
        """Compile the trainer's model with Keras-compatible options.

        Omitted ``optimizer``, ``loss``, and ``metrics`` values use the
        trainer defaults. Supplied values replace those defaults. Composition
        with the defaults belongs in :meth:`create_loss` and
        :meth:`create_metrics` overrides.
        """
        self.compile_kwargs = dict(kwargs)
        self.use_default_lr_schedule = "optimizer" not in self.compile_kwargs
        self._compile_model()

    def create_network(self, network_context: NetworkContext):
        """Create the plan-compatible network for a pipeline configuration.

        Subclasses may replace this method to customize the network while
        retaining the planner-derived context. The default implementation
        builds MedicAI's dynamic U-Net from the selected ``NetworkConfig``.
        Network creation is fold-independent; fold-specific initialization
        belongs to a later trainer lifecycle hook.
        """
        return build_unet_from_config(network_context.plan_config)

    def architecture_config(self, network_context: NetworkContext):
        """Return serializable architecture provenance for the network hook."""
        return {
            "family": "medicai_dynamic_unet",
            "configuration": network_context.configuration,
            "network_config": network_context.plan_config.to_dict(),
        }

    def derive_output_spec(self) -> OutputSpec:
        """Run a probe forward pass and validate the planned model contract."""
        if self.net_cfg is None:
            raise ValueError(f"No network configuration for {self.configuration!r}.")
        dummy = keras.ops.zeros(
            (1, *self.net_cfg.patch_size, self.net_cfg.n_modalities),
            dtype="float32",
        )
        outputs = self.model(dummy, training=True)
        if isinstance(outputs, dict):
            output_map = outputs
        elif isinstance(outputs, (list, tuple)):
            names = getattr(self.model, "output_names", None) or (
                "final",
                *(f"aux_{index}" for index in range(len(outputs) - 1)),
            )
            output_map = dict(zip(names, outputs, strict=True))
        else:
            output_map = {"final": outputs}

        if "final" not in output_map:
            raise ValueError("The network output contract must contain a 'final' output.")
        expected_channels = self.net_cfg.n_classes
        expected_output_rank = len(self.net_cfg.patch_size) + 1
        final_shape = tuple(int(value) for value in output_map["final"].shape[1:])
        if len(final_shape) != expected_output_rank:
            raise ValueError(
                f"Network output rank {len(final_shape) + 1} is incompatible with "
                f"{self.configuration!r} input rank {len(self.net_cfg.patch_size) + 2}."
            )
        if final_shape[:-1] != tuple(self.net_cfg.patch_size):
            raise ValueError(
                "The network final output spatial shape "
                f"{final_shape[:-1]} does not match the planned patch size "
                f"{tuple(self.net_cfg.patch_size)}."
            )
        final_channels = final_shape[-1]
        if final_channels != expected_channels:
            raise ValueError(
                f"The network final output has {final_channels} channels; "
                f"the plan requires {expected_channels}."
            )

        final_spatial = final_shape[:-1]
        scales = {}
        channels = {}
        for name, output in output_map.items():
            shape = tuple(int(value) for value in output.shape[1:])
            if len(shape) != expected_output_rank or shape[-1] != expected_channels:
                raise ValueError(
                    f"Output {name!r} has incompatible shape {shape}; expected "
                    f"{expected_channels} channels and {len(final_spatial)} spatial dimensions."
                )
            scales[name] = tuple(
                dimension / final_dimension
                for dimension, final_dimension in zip(shape[:-1], final_spatial, strict=True)
            )
            channels[name] = shape[-1]

        target_encoding = (
            "region"
            if self.task_type == "region_based"
            else "binary"
            if self.task_type == "binary"
            else "categorical"
        )
        return OutputSpec(
            names=tuple(output_map),
            scales=scales,
            channels=channels,
            activation=self.net_cfg.output_activation,
            target_encoding=target_encoding,
        )

    def _metric_monitor_name(self):
        """Return the correct monitor name depending on deep supervision."""
        metrics = self.metrics
        if isinstance(metrics, dict):
            metrics = [
                metric
                for values in metrics.values()
                for metric in (values if isinstance(values, (list, tuple)) else [values])
            ]
        elif not isinstance(metrics, (list, tuple)):
            metrics = [metrics] if metrics else []
        if metrics and hasattr(metrics[0], "name"):
            metric_name = metrics[0].name
        else:
            metric_name = "loss"
        use_ds = (
            self.cfg.deep_supervision and self.net_cfg is not None and self.net_cfg.deep_supervision
        )
        if use_ds:
            return f"val_final_{metric_name}"
        return f"val_{metric_name}"

    def _default_optimizer(self):
        """Create the internal nnU-Net default Keras optimizer."""
        cfg = self.cfg
        optimizer_kwargs = {
            "learning_rate": cfg.lr,
            "momentum": cfg.momentum,
            "nesterov": cfg.nesterov,
            "weight_decay": cfg.weight_decay,
            "global_clipnorm": 12.0,
            "use_ema": cfg.use_ema,
            "ema_momentum": cfg.ema_momentum,
        }
        # Keras treats gradient accumulation as disabled by default; when set,
        # gradient_accumulation_steps must be at least 2.
        if cfg.gradient_accumulation_steps > 1:
            optimizer_kwargs["gradient_accumulation_steps"] = cfg.gradient_accumulation_steps
        elif cfg.gradient_accumulation_steps != 1:
            raise ValueError("gradient_accumulation_steps must be 1 (disabled) or an integer >= 2.")
        return keras.optimizers.SGD(**optimizer_kwargs)

    def create_loss(self):
        """Create default loss(s) and weights; override to compose them."""
        cfg = self.cfg

        # Custom loss
        if self.custom_loss_fn is not None:
            base_loss = self.custom_loss_fn
        else:
            if self.task_type in {"binary", "multi-label", "region_based"}:
                loss_ignore_ids = self.plan.ignore_class_ids if self.task_type == "binary" else None
                base_loss = BinaryDiceCELoss(
                    from_logits=False,
                    num_classes=self.n_classes,
                    target_class_ids=(
                        None
                        if self.task_type == "region_based"
                        else self.plan.target_class_ids or None
                    ),
                    ignore_class_ids=loss_ignore_ids,
                )
            else:
                base_loss = SparseDiceCELoss(
                    from_logits=False,
                    num_classes=self.n_classes,
                    target_class_ids=self.plan.target_class_ids or None,
                    ignore_class_ids=self.plan.ignore_class_ids or None,
                )

        # Deep supervision setup
        if cfg.deep_supervision and self.net_cfg and self.net_cfg.deep_supervision:
            n_scales = self.net_cfg.n_pooling
            raw_weights = [0.5**i for i in range(n_scales)]
            total = sum(raw_weights)
            normalized_weights = [w / total for w in raw_weights]

            # If base_loss is already a mapper dict, we use it directly
            if isinstance(base_loss, dict):
                return base_loss, normalized_weights

            # Create the multi-output loss dictionary
            losses = {"final": base_loss}
            loss_weights = {"final": normalized_weights[0]}

            for i in range(1, n_scales):
                key = f"aux_{i-1}"
                losses[key] = base_loss
                loss_weights[key] = normalized_weights[i]

            return losses, loss_weights

        return base_loss, None

    def create_metrics(self):
        """Create default metrics; override to customize or extend them."""
        if self.custom_metrics is not None:
            if isinstance(self.custom_metrics, (list, tuple)):
                return list(self.custom_metrics)
            return [self.custom_metrics]

        if self.task_type in {"binary", "multi-label", "region_based"}:
            metric_ignore_ids = self.plan.ignore_class_ids if self.task_type == "binary" else None
            return [
                BinaryDiceMetric(
                    from_logits=False,
                    num_classes=self.n_classes,
                    target_class_ids=(
                        None
                        if self.task_type == "region_based"
                        else self.plan.target_class_ids or None
                    ),
                    ignore_class_ids=metric_ignore_ids,
                )
            ]

        return [
            SparseDiceMetric(
                from_logits=False,
                num_classes=self.n_classes,
                target_class_ids=self.plan.target_class_ids or None,
                ignore_class_ids=self.plan.ignore_class_ids or None,
            )
        ]

    def _metrics_for_model_outputs(self, metrics):
        """Route flat metric lists to the final deep-supervision output only."""
        use_deep_supervision = (
            self.cfg.deep_supervision
            and self.net_cfg is not None
            and self.net_cfg.deep_supervision
        )
        if not use_deep_supervision or metrics is None:
            return metrics

        output_names = [
            "final",
            *(f"aux_{index}" for index in range(self.net_cfg.n_pooling - 1)),
        ]
        if isinstance(metrics, dict):
            unexpected_outputs = set(metrics) - set(output_names)
            if unexpected_outputs:
                raise ValueError(
                    "Metric mapping contains unknown model output(s): "
                    f"{sorted(unexpected_outputs)}. Expected keys from {output_names}."
                )
            return {name: metrics.get(name, []) for name in output_names}

        metric_list = list(metrics) if isinstance(metrics, (list, tuple)) else [metrics]
        return {name: metric_list if name == "final" else [] for name in output_names}

    def _compile_model(self):
        """Compile the Keras model using resolved components."""
        default_loss, default_loss_weights = self.create_loss()
        default_metrics = self.create_metrics()
        compile_args = {
            "optimizer": self._default_optimizer(),
            "loss": default_loss,
            "loss_weights": default_loss_weights,
            "metrics": default_metrics,
        }
        compile_args.update(self.compile_kwargs)
        compile_args["metrics"] = self._metrics_for_model_outputs(
            compile_args.get("metrics")
        )
        self.model.compile(**compile_args)
        self.optimizer = self.model.optimizer
        self.losses = compile_args["loss"]
        self.loss_weights = compile_args.get("loss_weights")
        self.metrics = compile_args.get("metrics") or []

    @staticmethod
    def _json_safe(value):
        """Convert common configuration values to stable JSON-compatible data."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [nnUNetTrainer._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): nnUNetTrainer._json_safe(item) for key, item in value.items()}
        if hasattr(value, "get_config"):
            return {
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "config": nnUNetTrainer._json_safe(value.get_config()),
            }
        return {"class": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}

    def _run_signature(self):
        cfg = self.cfg.to_dict()
        cfg.pop("checkpoint_dir", None)
        return self._json_safe(
            {
                "dataset": self.plan.dataset_name,
                "network": self.plan.network_type,
                "trainer": f"{type(self).__module__}.{type(self).__qualname__}",
                "configuration": self.configuration,
                "fold": self.fold,
                "network_config": self.net_cfg.to_dict() if self.net_cfg else None,
                "architecture": self._json_safe(getattr(self, "architecture", None)),
                "output_spec": self._json_safe(
                    getattr(self, "output_spec", None).to_dict()
                    if getattr(self, "output_spec", None) is not None
                    else None
                ),
                "task_type": self.task_type,
                "target_class_ids": self.plan.target_class_ids,
                "ignore_class_ids": self.plan.ignore_class_ids,
                "training_config": cfg,
                "run_metadata": self.run_metadata,
                "compile_config": self._json_safe(
                    {
                        "optimizer": self.optimizer,
                        "loss": self.losses,
                        "metrics": self.metrics,
                        "compile_kwargs": self.compile_kwargs,
                    }
                ),
            }
        )

    def run(self, callbacks=None, resume=False, fit_kwargs=None):
        """
        Execute the full training loop using model.fit().

        Returns
        -------
        dict with training history
        """
        cfg = self.cfg
        if callbacks is None:
            callbacks = []
        else:
            callbacks = list(callbacks)
        fit_kwargs = dict(fit_kwargs or {})
        callback_signature = []
        for callback in callbacks:
            try:
                config = callback.get_config()
            except (AttributeError, NotImplementedError):
                config = None
            callback_signature.append(
                {
                    "class": f"{type(callback).__module__}.{type(callback).__qualname__}",
                    "config": self._json_safe(config),
                }
            )
        self.run_metadata["fit_kwargs"] = self._json_safe(fit_kwargs)
        self.run_metadata["callbacks"] = callback_signature
        self.output_dir.mkdir(parents=True, exist_ok=True)

        monitor_name = self._metric_monitor_name()
        backup_dir = self.output_dir / "training_backup"
        run_config_path = self.output_dir / "training_run.json"
        run_signature = self._run_signature()

        if not resume and backup_dir.exists() and any(backup_dir.iterdir()):
            raise ValueError(
                "An interrupted training backup already exists. Pass resume=True to restore it, "
                "or choose a new output directory for a fresh run."
            )
        if resume and run_config_path.exists():
            with run_config_path.open(encoding="utf-8") as stream:
                previous_signature = json.load(stream)
            if previous_signature != run_signature:
                raise ValueError(
                    "Cannot resume because training settings differ from the interrupted run. "
                    "Keep epochs, steps_per_epoch, batch size, patch size, optimizer, learning "
                    "rate schedule, and dataset split unchanged."
                )
        elif resume and backup_dir.exists() and any(backup_dir.iterdir()):
            raise ValueError(
                "A training backup exists without its run configuration; refusing to resume "
                "with an unverifiable schedule."
            )
        with run_config_path.open("w", encoding="utf-8") as stream:
            json.dump(run_signature, stream, indent=2, sort_keys=True)

        # Internally required callbacks
        internal_callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / "best_model.weights.h5"),
                monitor=monitor_name,
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            _EpochCheckpoint(
                filepath=self.output_dir / "checkpoint_latest.weights.h5",
                every_n_epochs=cfg.save_every_n_epochs,
            ),
            keras.callbacks.CSVLogger(
                str(self.output_dir / "training_log.csv"), append=resume
            ),
        ]

        if resume:
            internal_callbacks.append(
                keras.callbacks.BackupAndRestore(
                    backup_dir=str(backup_dir),
                    save_freq="epoch",
                )
            )

        if cfg.use_ema:
            internal_callbacks.append(keras.callbacks.SwapEMAWeights(swap_on_epoch=True))

        custom_lr_schedule = any(
            isinstance(
                callback,
                (keras.callbacks.LearningRateScheduler, keras.callbacks.ReduceLROnPlateau),
            )
            for callback in callbacks
        )
        if self.use_default_lr_schedule and cfg.lr_schedule == "poly" and not custom_lr_schedule:
            internal_callbacks.append(
                keras.callbacks.LearningRateScheduler(
                    lambda epoch, _lr: cfg.lr
                    * max(0.0, 1.0 - epoch / max(cfg.n_epochs, 1)) ** cfg.poly_exp
                )
            )

        all_callbacks = internal_callbacks + callbacks

        # Execute training
        fit_history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=cfg.n_epochs,
            steps_per_epoch=cfg.iters_per_epoch,
            callbacks=all_callbacks,
            verbose=fit_kwargs.pop("verbose", 1),
            **fit_kwargs,
        )

        self.history = fit_history.history
        self._save_checkpoint("final_model.weights.h5")
        return self.history

    def _save_checkpoint(self, filename):
        path = self.output_dir / filename
        self.model.save_weights(str(path))

    def load_best_checkpoint(self):
        """Load the best stored weights into the model."""
        path = self.output_dir / "best_model.weights.h5"
        if path.exists():
            self.model.load_weights(str(path))
