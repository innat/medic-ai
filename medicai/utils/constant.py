class KerasConstants:
    ACTIVATION_EXCLUDES = {"get", "serialize", "deserialize"}
    INITIALIZER_EXCLUDES = {"get", "serialize", "deserialize", "Initializer"}
    OPTIMIZER_EXCLUDES = {"get", "serialize", "deserialize", "Optimizer", "legacy", "schedules"}
    REGULARIZER_EXCLUDES = {"get", "serialize", "deserialize", "Regularizer"}
    LOSS_EXCLUDES = {"get", "serialize", "deserialize", "Loss"}
    METRICS_EXCLUDES = {"get", "serialize", "deserialize", "Metric"}
    VALID_DECODER_NORMS = (False, "batch", "layer", "unit", "group", "instance", "sync_batch")
    VALID_DECODER_BLOCK_TYPE = ("upsampling", "transpose")

    keras_loaded = False
    keras_module = None

    @classmethod
    def ensure_keras_loaded(cls):
        """Lazily import keras only when needed."""
        if not cls.keras_loaded:
            import keras

            cls.keras_module = keras
            cls.keras_loaded = True

    @classmethod
    def get_valid_losses(cls):
        """Get tuple of valid Keras loss functions."""
        cls.ensure_keras_loaded()
        losses = [
            m
            for m in dir(cls.keras_module.losses)
            if (
                not m.startswith("_")
                and
                # not m[0].isupper() and
                m not in cls.LOSS_EXCLUDES
            )
        ]
        return tuple(sorted(losses))

    @classmethod
    def get_valid_metrics(cls):
        """Get tuple of valid Keras metrics."""
        cls.ensure_keras_loaded()
        metrics = [
            m
            for m in dir(cls.keras_module.metrics)
            if (not m.startswith("_") and not m[0].isupper() and m not in cls.METRICS_EXCLUDES)
        ]
        return tuple(sorted(metrics))

    @classmethod
    def get_valid_activations(cls):
        """Get tuple of valid Keras activation functions."""
        cls.ensure_keras_loaded()
        activations = [
            m
            for m in dir(cls.keras_module.activations)
            if (not m.startswith("_") and not m[0].isupper() and m not in cls.ACTIVATION_EXCLUDES)
        ]
        return tuple(sorted(activations))

    @classmethod
    def get_valid_initializers(cls):
        """Get tuple of valid Keras initializers (names are mostly capitalized here,
        but we want the *functions* or *instances* which may be lowercase factory functions)."""
        cls.ensure_keras_loaded()
        initializers = [
            m
            for m in dir(cls.keras_module.initializers)
            if (
                not m.startswith("_")
                and
                # For initializers, we might want to include capitalized classes like 'GlorotNormal'
                # but the user's original logic was to exclude them, so we stick to the original filter
                # to get the lowercase factory functions like 'glorot_normal'.
                not m[0].isupper()
                and m not in cls.INITIALIZER_EXCLUDES
            )
        ]
        return tuple(sorted(initializers))

    @classmethod
    def get_valid_optimizers(cls):
        """Get tuple of valid Keras optimizers (Classes/Factory Functions)."""
        cls.ensure_keras_loaded()
        optimizers = [
            m
            for m in dir(cls.keras_module.optimizers)
            if (not m.startswith("_") and m not in cls.OPTIMIZER_EXCLUDES)
        ]

        # We need a separate list for classes (capitalized) and factory functions (lowercase)
        # To get the modern class names (e.g., 'Adam', 'SGD'), we MUST include capitalized names.
        final_optimizers = []
        for m in optimizers:
            # Check for modern optimizer classes (e.g., 'Adam', 'SGD', 'Optimizer')
            # and exclude the base class 'Optimizer' and modules like 'schedules'
            if m[0].isupper() and m != "Optimizer":
                final_optimizers.append(m)
            # Check for factory functions (e.g., 'adam' in some versions)
            elif not m[0].isupper():
                final_optimizers.append(m)

        return tuple(sorted(final_optimizers))


keras_constants = KerasConstants()
