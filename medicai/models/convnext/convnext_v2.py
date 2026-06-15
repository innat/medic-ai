import keras
from keras import layers

from medicai.utils import (
    DescribeMixin,
    get_norm_layer,
    get_pooling_layer,
    registration,
    validate_activation,
)

from .convnext_backbone import ConvNeXtBackboneV2

MODEL_CONFIGS = {
    "atto": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [40, 80, 160, 320],
    },
    "femto": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [48, 96, 192, 384],
    },
    "pico": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [64, 128, 256, 512],
    },
    "nano": {
        "depths": [2, 2, 8, 2],
        "projection_dims": [80, 160, 320, 640],
    },
    "tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "small": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
    },
    "huge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [352, 704, 1408, 2816],
    },
}


@keras.saving.register_keras_serializable(package="convnext")
class ConvNeXtVariantsBaseV2(keras.Model):
    """
    Base class for all ConvNeXt V2 classification models (Atto, Femto, Nano, etc.).

    This class handles the creation of the ConvNeXtBackboneV2 and the optional
    classification head (Global Pooling, Layer Normalization, and Dense layer).

    Args:
        depths: A list or tuple of integers specifying the number of
            ConvNeXt V2 blocks in each of the 4 stages.
        projection_dims: A list or tuple of integers specifying the number
            of channels (filters) for the stem and each of the 4 stages.
        input_shape: A tuple specifying the input shape of the model,
            not including the batch size.
        include_rescaling: A boolean indicating whether to include an
            input preprocessing layer. Defaults to `False`.
        include_top: A boolean indicating whether to include the
            classification head. Defaults to `True`.
        num_classes: An integer specifying the number of classes.
            Only relevant if `include_top` is `True`. Defaults to 1000.
        pooling: (Optional) A string specifying the type of pooling
            (`"avg"` or `"max"`) if `include_top` is `False`.
        classifier_activation: A string specifying the activation function
            for the classification layer. Defaults to `"softmax"`.
        name: (Optional) The name of the model.
        **kwargs: Additional keyword arguments for the Keras Model constructor.
    """

    def __init__(
        self,
        *,
        depths,
        projection_dims,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1
        if name is None and self.__class__ is not ConvNeXtVariantsBaseV2:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        # number of classes must be positive.
        if include_top and num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        # verify input activation.
        classifier_activation = validate_activation(classifier_activation)

        backbone = ConvNeXtBackboneV2(
            input_shape=input_shape,
            depths=depths,
            projection_dims=projection_dims,
            include_rescaling=include_rescaling,
        )
        inputs = backbone.input
        x = backbone.output

        GlobalAvgPool = get_pooling_layer(
            spatial_dims=spatial_dims, layer_type="avg", global_pool=True
        )
        GlobalMaxPool = get_pooling_layer(
            spatial_dims=spatial_dims, layer_type="max", global_pool=True
        )
        GlobalNorm = get_norm_layer(layer_type="layer", epsilon=1e-6, name="head_norm")

        if include_top:
            x = GlobalAvgPool(x)
            x = GlobalNorm(x)
            x = layers.Dense(
                num_classes, activation=classifier_activation, dtype="float32", name="predictions"
            )(x)
        elif pooling is not None:
            if pooling == "avg":
                x = GlobalAvgPool(x)
            elif pooling == "max":
                x = GlobalMaxPool(x)
            x = GlobalNorm(x)
        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.pyramid_outputs = backbone.pyramid_outputs
        self.depths = depths
        self.projection_dims = projection_dims
        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.num_classes = num_classes
        self.pooling = pooling
        self.classifier_activation = classifier_activation
        self.name = name

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape[1:],
                "include_top": self.include_top,
                "include_rescaling": self.include_rescaling,
                "num_classes": self.num_classes,
                "pooling": self.pooling,
                "classifier_activation": self.classifier_activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_v2_atto", family="convnext")
class ConvNeXtV2Atto(ConvNeXtVariantsBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            depths=MODEL_CONFIGS["atto"]["depths"],
            projection_dims=MODEL_CONFIGS["atto"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_v2_femto", family="convnext")
class ConvNeXtV2Femto(ConvNeXtVariantsBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            depths=MODEL_CONFIGS["femto"]["depths"],
            projection_dims=MODEL_CONFIGS["femto"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_v2_pico", family="convnext")
class ConvNeXtV2Pico(ConvNeXtVariantsBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            depths=MODEL_CONFIGS["pico"]["depths"],
            projection_dims=MODEL_CONFIGS["pico"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_v2_nano", family="convnext")
class ConvNeXtV2Nano(ConvNeXtVariantsBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            depths=MODEL_CONFIGS["nano"]["depths"],
            projection_dims=MODEL_CONFIGS["nano"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_v2_tiny", family="convnext")
class ConvNeXtV2Tiny(ConvNeXtVariantsBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            depths=MODEL_CONFIGS["tiny"]["depths"],
            projection_dims=MODEL_CONFIGS["tiny"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_v2_small", family="convnext")
class ConvNeXtV2Small(ConvNeXtVariantsBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            depths=MODEL_CONFIGS["small"]["depths"],
            projection_dims=MODEL_CONFIGS["small"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_v2_base", family="convnext")
class ConvNeXtV2Base(ConvNeXtVariantsBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            depths=MODEL_CONFIGS["base"]["depths"],
            projection_dims=MODEL_CONFIGS["base"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_v2_large", family="convnext")
class ConvNeXtV2Large(ConvNeXtVariantsBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            depths=MODEL_CONFIGS["large"]["depths"],
            projection_dims=MODEL_CONFIGS["large"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_v2_huge", family="convnext")
class ConvNeXtV2Huge(ConvNeXtVariantsBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            depths=MODEL_CONFIGS["huge"]["depths"],
            projection_dims=MODEL_CONFIGS["huge"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


CONVNEXTV2_DOCSTRING = """
**{name}** classification model built on a ConvNeXt V2 backbone.

This variant combines a :class:`ConvNeXtBackboneV2` with an optional
classification head. It can be used for end-to-end classification or as a
feature extractor for downstream workflows that need pooled or unpooled
backbone outputs.

The model is built in three steps:

1. A ConvNeXt V2 backbone produces the final stage feature tensor and stores
   intermediate stage features in ``pyramid_outputs``.
2. If ``include_top=True``, global average pooling, layer normalization, and a
   dense prediction layer are applied.
3. If ``include_top=False``, the model returns either the unpooled backbone
   output or a pooled representation, depending on ``pooling``.

Args:
    input_shape: A tuple specifying the input shape of the model,
        not including the batch size. This can describe either 2D or 3D
        inputs.
    include_rescaling: A boolean indicating whether to include an
        input preprocessing layer before the backbone.
    include_top: A boolean indicating whether to include the
        classification head. If `False`, the model returns backbone features
        instead of class predictions.
    num_classes: An integer specifying the number of classes for the
        classification layer. This is only relevant if ``include_top`` is
        ``True``.
    pooling: (Optional) A string specifying the type of pooling to
        apply when ``include_top`` is ``False``. Supported values are
        ``"avg"`` and ``"max"``.
    classifier_activation: A string specifying the activation function
        used by the classification layer.
    name: (Optional) The name of the model.
    **kwargs: Additional keyword arguments.

Returns:
    A ``keras.Model`` whose output depends on the configuration:

    - If ``include_top=True``, the output is a classification tensor of shape
      ``(batch_size, num_classes)``.
    - If ``include_top=False`` and ``pooling`` is ``None``, the output is the
      final backbone feature tensor.
    - If ``include_top=False`` and ``pooling`` is ``"avg"`` or ``"max"``,
      the output is a pooled feature tensor with last dimension
      ``{projection_dim_last}``.

Examples:
    TensorFlow 2D classification::

        import tensorflow as tf
        from medicai.models import {name}

        model = {name}(
            input_shape=(224, 224, 3), num_classes=10
        )
        x = tf.random.normal((1, 224, 224, 3))
        y = model(x)
        print(y.shape) # (1, 10)

    PyTorch backend 2D classification::

        import torch
        from medicai.models import {name}

        model = {name}(
            input_shape=(224, 224, 3), num_classes=10
        )
        x = torch.randn((1, 224, 224, 3))
        y = model(x)
        print(y.shape) # (1, 10)

    Jax backend 2D classification::

        import jax
        import jax.numpy as jnp
        from medicai.models import {name}

        model = {name}(
            input_shape=(224, 224, 3), num_classes=10
        )
        x = jnp.random.normal((1, 224, 224, 3))
        y = model(x)
        print(y.shape) # (1, 10)

    TensorFlow 3D classification::

        import tensorflow as tf
        from medicai.models import {name}              

        model = {name}(
            input_shape=(64, 224, 224, 1), num_classes=10
        )
        x = tf.random.normal((1, 64, 224, 224, 1))
        y = model(x)                   
        print(y.shape) # (1, 10)   

    PyTorch backend 3D classification::

        import torch
        from medicai.models import {name}              

        model = {name}(
            input_shape=(64, 224, 224, 1), num_classes=10
        )
        x = torch.randn((1, 64, 224, 224, 1 ))
        y = model(x)            
        print(y.shape) # (1, 10)

    Jax backend 3D classification::

        import jax
        import jax.numpy as jnp     
        from medicai.models import {name}     
                 
        model = {name}(
            input_shape=(64, 224, 224, 1), num_classes=10
        )
        x = jnp.random.normal((1, 64, 224, 224, 1))
        y = model(x)    
        print(y.shape) # (1, 10)
"""

ConvNeXtV2Atto.__doc__ = CONVNEXTV2_DOCSTRING.format(
    name="ConvNeXtV2Atto",
    projection_dim_last=MODEL_CONFIGS["atto"]["projection_dims"][-1],
)
ConvNeXtV2Femto.__doc__ = CONVNEXTV2_DOCSTRING.format(
    name="ConvNeXtV2Femto",
    projection_dim_last=MODEL_CONFIGS["femto"]["projection_dims"][-1],
)
ConvNeXtV2Pico.__doc__ = CONVNEXTV2_DOCSTRING.format(
    name="ConvNeXtV2Pico",
    projection_dim_last=MODEL_CONFIGS["pico"]["projection_dims"][-1],
)
ConvNeXtV2Nano.__doc__ = CONVNEXTV2_DOCSTRING.format(
    name="ConvNeXtV2Nano",
    projection_dim_last=MODEL_CONFIGS["nano"]["projection_dims"][-1],
)
ConvNeXtV2Tiny.__doc__ = CONVNEXTV2_DOCSTRING.format(
    name="ConvNeXtV2Tiny",
    projection_dim_last=MODEL_CONFIGS["tiny"]["projection_dims"][-1],
)
ConvNeXtV2Small.__doc__ = CONVNEXTV2_DOCSTRING.format(
    name="ConvNeXtV2Small",
    projection_dim_last=MODEL_CONFIGS["small"]["projection_dims"][-1],
)
ConvNeXtV2Base.__doc__ = CONVNEXTV2_DOCSTRING.format(
    name="ConvNeXtV2Base",
    projection_dim_last=MODEL_CONFIGS["base"]["projection_dims"][-1],
)
ConvNeXtV2Large.__doc__ = CONVNEXTV2_DOCSTRING.format(
    name="ConvNeXtV2Large",
    projection_dim_last=MODEL_CONFIGS["large"]["projection_dims"][-1],
)
ConvNeXtV2Huge.__doc__ = CONVNEXTV2_DOCSTRING.format(
    name="ConvNeXtV2Huge",
    projection_dim_last=MODEL_CONFIGS["huge"]["projection_dims"][-1],
)
