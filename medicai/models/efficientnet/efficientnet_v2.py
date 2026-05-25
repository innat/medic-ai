import keras
from keras import layers

from medicai.utils import DescribeMixin, get_pooling_layer, registration, validate_activation

from .efficientnet_backbone import EfficientNetBackboneV2
from .efficientnet_layers import (
    DENSE_KERNEL_INITIALIZER,
)


@keras.saving.register_keras_serializable(package="efficientnet")
class EfficientNetBaseV2(keras.Model):
    def __init__(
        self,
        *,
        blocks_args,
        width_coefficient,
        depth_coefficient,
        drop_connect_rate,
        input_shape,
        dropout_rate=0.2,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1
        if name is None and self.__class__ is not EfficientNetBaseV2:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        # number of classes must be positive.
        if include_top and num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        # verify input activation.
        classifier_activation = validate_activation(classifier_activation)

        backbone = EfficientNetBackboneV2(
            blocks_args=blocks_args,
            input_shape=input_shape,
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            drop_connect_rate=drop_connect_rate,
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
        if include_top:
            x = GlobalAvgPool(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name="top_dropout")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                kernel_initializer=DENSE_KERNEL_INITIALIZER,
                dtype="float32",
                name="predictions",
            )(x)
        elif pooling == "avg":
            x = GlobalAvgPool(x)
        elif pooling == "max":
            x = GlobalMaxPool(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.pyramid_outputs = backbone.pyramid_outputs
        self.blocks_args = blocks_args
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.drop_connect_rate = drop_connect_rate
        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.num_classes = num_classes
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.classifier_activation = classifier_activation
        self.name = name

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "blocks_args": self.blocks_args,
            "include_top": self.include_top,
            "include_rescaling": self.include_rescaling,
            "num_classes": self.num_classes,
            "pooling": self.pooling,
            "dropout_rate": self.dropout_rate,
            "classifier_activation": self.classifier_activation,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="efficientnet")
@registration.register(name="efficientnet_v2_b0", family="efficientnet")
class EfficientNetV2B0(EfficientNetBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            blocks_args="efficientnetv2-b0",
            width_coefficient=1.0,
            depth_coefficient=1.0,
            drop_connect_rate=0.2,
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            dropout_rate=dropout_rate,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="efficientnet")
@registration.register(name="efficientnet_v2_b1", family="efficientnet")
class EfficientNetV2B1(EfficientNetBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            blocks_args="efficientnetv2-b1",
            width_coefficient=1.0,
            depth_coefficient=1.1,
            drop_connect_rate=0.2,
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            dropout_rate=dropout_rate,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="efficientnet")
@registration.register(name="efficientnet_v2_b2", family="efficientnet")
class EfficientNetV2B2(EfficientNetBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            blocks_args="efficientnetv2-b2",
            width_coefficient=1.1,
            depth_coefficient=1.2,
            drop_connect_rate=0.2,
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            dropout_rate=dropout_rate,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="efficientnet")
@registration.register(name="efficientnet_v2_b3", family="efficientnet")
class EfficientNetV2B3(EfficientNetBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            blocks_args="efficientnetv2-b3",
            width_coefficient=1.2,
            depth_coefficient=1.4,
            drop_connect_rate=0.2,
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            dropout_rate=dropout_rate,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="efficientnet")
@registration.register(name="efficientnet_v2_s", family="efficientnet")
class EfficientNetV2S(EfficientNetBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            blocks_args="efficientnetv2-s",
            width_coefficient=1.0,
            depth_coefficient=1.0,
            drop_connect_rate=0.2,
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            dropout_rate=dropout_rate,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="efficientnet")
@registration.register(name="efficientnet_v2_m", family="efficientnet")
class EfficientNetV2M(EfficientNetBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            blocks_args="efficientnetv2-m",
            width_coefficient=1.0,
            depth_coefficient=1.0,
            drop_connect_rate=0.2,
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            dropout_rate=dropout_rate,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="efficientnet")
@registration.register(name="efficientnet_v2_l", family="efficientnet")
class EfficientNetV2L(EfficientNetBaseV2, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            blocks_args="efficientnetv2-l",
            width_coefficient=1.0,
            depth_coefficient=1.0,
            drop_connect_rate=0.2,
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            dropout_rate=dropout_rate,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


EfficientNet_DOCSTRING = """
This class provides a complete {name} model, including the convolutional
backbone and the classification head (the "top"). The backbone follows the
EfficientNet V2 design, which combines scaled MBConv-style stages with the
EfficientNet V2 block configuration, and the full model can be used either
for end-to-end classification or as a feature extractor.

Args:
    input_shape: A tuple specifying the input shape of the model,
        not including the batch size. Can be `(height, width, channels)` for
        2D or `(depth, height, width, channels)` for 3D.
    include_rescaling: A boolean indicating whether to include a
        ``Rescaling`` and normalization step at the beginning of the model.
    include_top: A boolean indicating whether to include the fully
        connected classification layer at the top of the network. If
        `False`, the model's output will be the features from the
        backbone, without the final classifier.
    num_classes: An integer specifying the number of classes for the
        classification layer. This is only relevant if `include_top`
        is `True`.
    dropout_rate: A float specifying the dropout rate applied before the
        classification layer when ``include_top`` is ``True``.
    pooling: (Optional) A string specifying the type of pooling to
        apply to the output of the backbone. Can be `"avg"` for global
        average pooling or `"max"` for global max pooling. This is only
        relevant if `include_top` is `False`.
    classifier_activation: A string specifying the activation function
        to use for the classification layer.
    name: (Optional) The name of the model.

Returns:
    A ``keras.Model`` whose output depends on the configuration:
        - If ``include_top=True``, the output is a classification tensor of shape
        ``(batch_size, num_classes)``.
        - If ``include_top=False`` and ``pooling`` is ``None``, the output is the
        final backbone feature tensor.
        - If ``include_top=False`` and ``pooling`` is ``"avg"`` or ``"max"``,
        the output is a pooled feature tensor.

Examples:
    .. code-block:: python
    
        import jax
        import jax.numpy as jnp 
        from medicai.models.efficientnet import {name}

        model = {name}(
            input_shape=(224, 224, 3), include_top=True, num_classes=2
        )
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1, 224, 224, 3))
        y = model(x)
        print(y.shape) # (1, 2)

References:
    - EfficientNetV2: Smaller Models and Faster Training. ICML 2021.
        `arXiv:2104.00298 <https://arxiv.org/abs/2104.00298>`_
"""


EfficientNetV2B0.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetV2B0")
EfficientNetV2B1.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetV2B1")
EfficientNetV2B2.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetV2B2")
EfficientNetV2B3.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetV2B3")
EfficientNetV2S.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetV2S")
EfficientNetV2M.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetV2M")
EfficientNetV2L.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetV2L")
