"""
medicai/trainer/nnunet/training/trainer.py
==================================
Full nnU-Net training pipeline.

Features
--------
  - Iteration-based training (1000 epochs × 250 iter/epoch) defaults
  - Easy integration with standard Keras 3 optimizers and losses
  - Deep supervision with decaying weights
  - 5-fold cross-validation compatible (fold index parameter)

Usage
-----
::

    trainer = nnUNetTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        plan=plan,
        train_config=config,
        fold=0,
    )
    trainer.run(callbacks=[...])
"""

import json
from pathlib import Path

import keras

from medicai.losses import BinaryDiceCELoss, SparseDiceCELoss
from medicai.metrics.dice import BinaryDiceMetric, SparseDiceMetric

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class nnUNetTrainer:
    """
    Manages the full nnU-Net training loop.

    Parameters
    ----------
    model         : compiled or uncompiled Keras model
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
        model,
        train_dataset,
        val_dataset,
        plan,
        train_config,
        fold=0,
        configuration="3d_fullres",
        loss=None,
        metrics=None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.plan = plan
        self.cfg = train_config
        self.fold = fold
        self.configuration = configuration

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
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.custom_loss_fn = loss
        self.custom_metrics = metrics

        # Compile model (builds optimizer, loss, and metrics internally)
        self._compile_model()

        # Training state
        self.best_val_dice = -1.0
        self.epochs_no_improve = 0

    # ------------------------------------------------------------------
    def _metric_monitor_name(self):
        """Return the correct monitor name depending on deep supervision."""
        if self.metrics and hasattr(self.metrics[0], "name"):
            metric_name = self.metrics[0].name
        else:
            metric_name = "loss"
        use_ds = (
            self.cfg.deep_supervision and self.net_cfg is not None and self.net_cfg.deep_supervision
        )
        if use_ds:
            return f"val_final_{metric_name}"
        return f"val_{metric_name}"

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _build_optimizer(self):
        """Resolve and returns the Keras optimizer."""
        cfg = self.cfg
        total_steps = cfg.n_epochs * cfg.iters_per_epoch

        # Resolve Learning Rate Schedule
        if isinstance(cfg.lr_schedule, str):
            if cfg.lr_schedule == "poly":
                lr = keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=cfg.lr,
                    decay_steps=total_steps,
                    end_learning_rate=0.0,
                    power=cfg.poly_exp,
                )
            else:
                lr = cfg.lr
        else:
            lr = cfg.lr_schedule

        # Resolve Optimizer
        if isinstance(cfg.optimizer, str):
            if cfg.optimizer == "sgd":
                return keras.optimizers.SGD(
                    learning_rate=lr,
                    momentum=cfg.momentum,
                    nesterov=cfg.nesterov,
                    weight_decay=cfg.weight_decay,
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    global_clipnorm=12.0,
                    use_ema=cfg.use_ema,
                    ema_momentum=cfg.ema_momentum,
                )
        return cfg.optimizer

    # ------------------------------------------------------------------
    def _build_loss(self):
        """Resolve and returns the loss and optional loss_weights."""
        cfg = self.cfg

        # Custom loss
        if self.custom_loss_fn is not None:
            base_loss = self.custom_loss_fn
        else:
            if self.task_type in {"binary", "multi-label"}:
                loss_ignore_ids = self.plan.ignore_class_ids if self.task_type == "binary" else None
                base_loss = BinaryDiceCELoss(
                    from_logits=False,
                    num_classes=self.n_classes,
                    target_class_ids=self.plan.target_class_ids or None,
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

    # ------------------------------------------------------------------
    def _build_metrics(self):
        """Resolve and returns the list of metrics."""
        if self.custom_metrics is not None:
            if isinstance(self.custom_metrics, (list, tuple)):
                return list(self.custom_metrics)
            return [self.custom_metrics]

        if self.task_type in {"binary", "multi-label"}:
            metric_ignore_ids = self.plan.ignore_class_ids if self.task_type == "binary" else None
            return [
                BinaryDiceMetric(
                    from_logits=False,
                    num_classes=self.n_classes,
                    target_class_ids=self.plan.target_class_ids or None,
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

    # ------------------------------------------------------------------
    def _compile_model(self):
        """Compile the Keras model using resolved components."""
        self.optimizer = self._build_optimizer()
        self.losses, self.loss_weights = self._build_loss()
        self.metrics = self._build_metrics()

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.losses,
            loss_weights=self.loss_weights,
            metrics=self.metrics,
        )

    # ------------------------------------------------------------------
    def run(self, callbacks=None):
        """
        Execute the full training loop using model.fit().

        Returns
        -------
        dict with training history
        """
        cfg = self.cfg
        if callbacks is None:
            callbacks = []

        # Enforce no duplicate checkpoints passed by user to prevent IO issues
        has_checkpoint = any(isinstance(c, keras.callbacks.ModelCheckpoint) for c in callbacks)
        if has_checkpoint:
            raise ValueError(
                "A ModelCheckpoint callback was found in your custom callbacks list. "
                "nnUNetTrainer injects its own checkpoint logic. Please remove it from your list."
            )

        monitor_name = self._metric_monitor_name()

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
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / "checkpoint_latest.weights.h5"),
                save_freq=cfg.save_every_n_epochs,
                save_weights_only=True,
                verbose=0,
            ),
            keras.callbacks.CSVLogger(str(self.output_dir / "training_log.csv")),
        ]

        if cfg.use_ema:
            internal_callbacks.append(keras.callbacks.SwapEMAWeights(swap_on_epoch=True))

        all_callbacks = internal_callbacks + callbacks

        # Execute training
        fit_history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=cfg.n_epochs,
            steps_per_epoch=cfg.iters_per_epoch,
            callbacks=all_callbacks,
            verbose=1,
        )

        self.history = fit_history.history
        self._save_checkpoint("final_model.weights.h5")
        return self.history

    # ------------------------------------------------------------------
    def _save_checkpoint(self, filename):
        path = self.output_dir / filename
        self.model.save_weights(str(path))

    def load_best_checkpoint(self):
        """Load the best stored weights into the model."""
        path = self.output_dir / "best_model.weights.h5"
        if path.exists():
            self.model.load_weights(str(path))
