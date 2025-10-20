import keras
from keras import layers

from medicai.utils import DescribeMixin, get_pooling_layer, keras_constants, registration

from .efficientnet_backbone import EfficientNetBackbone
from .efficientnet_layers import (
    DENSE_KERNEL_INITIALIZER,
)


@keras.saving.register_keras_serializable(package="efficientnetbase")
class EfficientNetBase(keras.Model):
    def __init__(
        self,
        *,
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
        if name is None and self.__class__ is not EfficientNetBase:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        backbone = EfficientNetBackbone(
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

            if classifier_activation is not None:
                if isinstance(classifier_activation, str):
                    classifier_activation = classifier_activation.lower()
                VALID_ACTIVATION_LIST = keras_constants.get_valid_activations()
                if classifier_activation not in VALID_ACTIVATION_LIST:
                    raise ValueError(
                        f"Invalid value for `classifier_activation`: {classifier_activation!r}. "
                        f"Supported values are: {VALID_ACTIVATION_LIST}"
                    )

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
@registration.register(name="efficientnet_b0", family="efficientnet")
class EfficientNetB0(EfficientNetBase, DescribeMixin):
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
@registration.register(name="efficientnet_b1", family="efficientnet")
class EfficientNetB1(EfficientNetBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        dropout_rate=0.2,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            width_coefficient=1.0,
            depth_coefficient=1.1,
            drop_connect_rate=0.2,
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="efficientnet")
@registration.register(name="efficientnet_b2", family="efficientnet")
class EfficientNetB2(EfficientNetBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.3,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            width_coefficient=1.1,
            depth_coefficient=1.2,
            drop_connect_rate=0.3,
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="efficientnet")
@registration.register(name="efficientnet_b3", family="efficientnet")
class EfficientNetB3(EfficientNetBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.3,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            width_coefficient=1.2,
            depth_coefficient=1.4,
            drop_connect_rate=0.3,
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
@registration.register(name="efficientnet_b4", family="efficientnet")
class EfficientNetB4(EfficientNetBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.4,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            width_coefficient=1.4,
            depth_coefficient=1.8,
            drop_connect_rate=0.4,
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
@registration.register(name="efficientnet_b5", family="efficientnet")
class EfficientNetB5(EfficientNetBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.4,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            width_coefficient=1.6,
            depth_coefficient=2.2,
            drop_connect_rate=0.4,
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
@registration.register(name="efficientnet_b6", family="efficientnet")
class EfficientNetB6(EfficientNetBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.5,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            width_coefficient=1.8,
            depth_coefficient=2.6,
            drop_connect_rate=0.5,
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
@registration.register(name="efficientnet_b7", family="efficientnet")
class EfficientNetB7(EfficientNetBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.5,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            width_coefficient=2.0,
            depth_coefficient=3.1,
            drop_connect_rate=0.5,
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
@registration.register(name="efficientnet_b8", family="efficientnet")
class EfficientNetB8(EfficientNetBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.5,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            width_coefficient=2.2,
            depth_coefficient=3.6,
            drop_connect_rate=0.5,
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
@registration.register(name="efficientnet_l2", family="efficientnet")
class EfficientNetL2(EfficientNetBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        dropout_rate=0.5,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        super().__init__(
            width_coefficient=4.3,
            depth_coefficient=5.3,
            drop_connect_rate=0.5,
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
{name} backbone model supporting both 2D and 3D inputs.

This class implements the feature extraction (backbone) part of the EfficientNet architecture,
which scales width, depth, and resolution uniformly using compound scaling.
It can operate on 2D inputs (e.g., images of shape `(H, W, C)`) or 3D inputs
(e.g., volumetric data of shape `(D, H, W, C)`).

The backbone produces multi-scale feature maps that can be used for downstream
tasks such as classification, detection, or segmentation.

References:
    - "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
      ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

Example:
    # TensorFlow / Keras - 2D cases.
    >>> import tensorflow as tf
    >>> from medicai.models import {name}
    >>> model = {name}(input_shape=(224, 224, 3), include_top=False)
    >>> x = tf.random.normal((1, 224, 224, 3))
    >>> y = model(x)
    >>> y.shape

    # PyTorch - 2D cases.
    >>> import torch
    >>> from medicai.models import {name}
    >>> model = {name}(input_shape=(224, 224, 3), include_top=False)
    >>> x = torch.randn(1, 224, 224, 3)
    >>> y = model(x)
    >>> y.shape

    # PyTorch - 3D cases
    >>> import torch
    >>> from medicai.models import {name}
    >>> model = {name}(input_shape=(96, 96, 96, 1), include_top=False)
    >>> x = torch.randn(1, 96, 96, 96, 1)
    >>> y = model(x)
    >>> y.shape


Initializes the {name} model.

Args:
    input_shape: A tuple specifying the input shape of the model,
        not including the batch size. Can be `(height, width, channels)`
        for 2D or `(depth, height, width, channels)` for 3D.
    include_rescaling: A boolean indicating whether to include a
        `Rescaling` layer at the beginning of the model. If `True`,
        the input pixels will be scaled from `[0, 255]` to `[0, 1]`.
    include_top: A boolean indicating whether to include the fully
        connected classification layer at the top of the network. If
        `False`, the model's output will be the features from the
        backbone, without the final classifier.
    num_classes: An integer specifying the number of classes for the
        classification layer. This is only relevant if `include_top`
        is `True`.
    dropout_rate: Apply dropout after pooling='avg' if `include_top`
        is `True`.
    pooling: (Optional) A string specifying the type of pooling to
        apply to the output of the backbone. Can be `"avg"` for global
        average pooling or `"max"` for global max pooling. This is only
        relevant if `include_top` is `False`.
    classifier_activation: A string specifying the activation function
        to use for the classification layer.
    name: (Optional) The name of the model.
    **kwargs: Additional keyword arguments.
"""


EfficientNetB0.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetB0")
EfficientNetB1.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetB1")
EfficientNetB2.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetB2")
EfficientNetB3.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetB3")
EfficientNetB4.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetB4")
EfficientNetB5.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetB5")
EfficientNetB6.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetB6")
EfficientNetB7.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetB7")
EfficientNetB8.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetB8")
EfficientNetL2.__doc__ = EfficientNet_DOCSTRING.format(name="EfficientNetL2")
