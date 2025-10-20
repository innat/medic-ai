import keras
from keras import layers

from medicai.utils import (
    DescribeMixin,
    get_norm_layer,
    get_pooling_layer,
    registration,
    validate_activation,
)

from .convnext_backbone import ConvNeXtBackbone

MODEL_CONFIGS = {
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
    "xlarge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
    },
}


@keras.saving.register_keras_serializable(package="convnext")
class ConvNeXtVariantsBase(keras.Model):
    """
    Base class for all ConvNeXt V1 classification models (Tiny, Small, Base, etc.).

    This class handles the creation of the ConvNeXtBackbone and the optional
    classification head (Global Pooling, Layer Normalization, and Dense layer).

    Args:
        depths: A list or tuple of integers specifying the number of
            ConvNeXt blocks in each of the 4 stages.
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
        if name is None and self.__class__ is not ConvNeXtVariantsBase:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        # number of classes must be positive.
        if num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        # verify input activation.
        classifier_activation = validate_activation(classifier_activation)

        backbone = ConvNeXtBackbone(
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
        config = {
            "input_shape": self.input_shape[1:],
            "include_top": self.include_top,
            "include_rescaling": self.include_rescaling,
            "num_classes": self.num_classes,
            "pooling": self.pooling,
            "classifier_activation": self.classifier_activation,
            "name": self.name,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_tiny", family="convnext")
class ConvNeXtTiny(ConvNeXtVariantsBase, DescribeMixin):
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
@registration.register(name="convnext_small", family="convnext")
class ConvNeXtSmall(ConvNeXtVariantsBase, DescribeMixin):
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
@registration.register(name="convnext_base", family="convnext")
class ConvNeXtBase(ConvNeXtVariantsBase, DescribeMixin):
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
@registration.register(name="convnext_large", family="convnext")
class ConvNeXtLarge(ConvNeXtVariantsBase, DescribeMixin):
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
@registration.register(name="convnext_xlarge", family="convnext")
class ConvNeXtXLarge(ConvNeXtVariantsBase, DescribeMixin):
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
            depths=MODEL_CONFIGS["xlarge"]["depths"],
            projection_dims=MODEL_CONFIGS["xlarge"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


CONVNEXT_DOCSTRING = """
{name} model with a ConvNeXt V1 backbone, supporting both 2D and 3D inputs.

This class implements the full ConvNeXt architecture, including the feature
extraction backbone and the classification head (optional). The ConvNeXt
architecture is a modern, purely convolutional network designed to compete
with Vision Transformers (ViTs).

It can operate on 2D inputs (e.g., images of shape `(H, W, C)`) or 3D inputs
(e.g., volumetric data of shape `(D, H, W, C)`).

References:
    - "A ConvNet for the 2020s". CVPR 2022.
      [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)

Example:
    # TensorFlow / Keras - 2D cases.
    >>> import tensorflow as tf
    >>> from your_module import {name}
    >>> # Classification model
    >>> model = {name}(input_shape=(224, 224, 3), num_classes=1000)
    >>> x = tf.random.normal((1, 224, 224, 3))
    >>> y = model(x)
    >>> y.shape
    (1, 1000)

    >>> # Feature extractor (pooling)
    >>> model_fe = {name}(input_shape=(224, 224, 3), include_top=False, pooling='avg')
    >>> x = tf.random.normal((1, 224, 224, 3))
    >>> y = model_fe(x)
    >>> y.shape
    (1, {projection_dim_last})

Initializes the {name} model.

Args:
    input_shape: A tuple specifying the input shape of the model,
        not including the batch size. Can be `(height, width, channels)`
        for 2D or `(depth, height, width, channels)` for 3D.
    include_rescaling: A boolean indicating whether to include an
        input preprocessing layer. If `True`, inputs are rescaled
        from `[0, 255]` to `[0, 1]` and normalized if the input has
        3 channels. Defaults to `False`.
    include_top: A boolean indicating whether to include the
        classification head (GlobalAvgPool, LayerNorm, and Dense layer).
        If `False`, the model's output will be the features from the
        backbone's last stage. Defaults to `True`.
    num_classes: An integer specifying the number of classes for the
        classification layer. This is only relevant if `include_top`
        is `True`. Defaults to 1000.
    pooling: (Optional) A string specifying the type of pooling to
        apply to the output of the backbone. Can be `"avg"` for global
        average pooling or `"max"` for global max pooling. This is only
        relevant if `include_top` is `False`.
    classifier_activation: A string specifying the activation function
        to use for the classification layer. Defaults to `"softmax"`.
    name: (Optional) The name of the model.
    **kwargs: Additional keyword arguments.
"""

ConvNeXtTiny.__doc__ = CONVNEXT_DOCSTRING.format(
    name="ConvNeXtTiny",
    projection_dim_last=MODEL_CONFIGS["tiny"]["projection_dims"][-1],
)
ConvNeXtSmall.__doc__ = CONVNEXT_DOCSTRING.format(
    name="ConvNeXtSmall",
    projection_dim_last=MODEL_CONFIGS["small"]["projection_dims"][-1],
)
ConvNeXtBase.__doc__ = CONVNEXT_DOCSTRING.format(
    name="ConvNeXtBase",
    projection_dim_last=MODEL_CONFIGS["base"]["projection_dims"][-1],
)
ConvNeXtLarge.__doc__ = CONVNEXT_DOCSTRING.format(
    name="ConvNeXtLarge",
    projection_dim_last=MODEL_CONFIGS["large"]["projection_dims"][-1],
)
ConvNeXtXLarge.__doc__ = CONVNEXT_DOCSTRING.format(
    name="ConvNeXtXLarge",
    projection_dim_last=MODEL_CONFIGS["xlarge"]["projection_dims"][-1],
)
