import keras
from keras import layers

from medicai.utils import DescribeMixin, registration, validate_activation
from medicai.utils.model_utils import get_pooling_layer

from .densenet_backbone import DenseNetBackbone


@keras.saving.register_keras_serializable(package="densenet")
class DenseNetBase(keras.Model):
    """
    A full DenseNet model for classification.

    This class provides a complete DenseNet model, including both the
    convolutional backbone and the classification head (the "top"). It is
    capable of handling both 2D and 3D inputs. The model's architecture is
    defined by the `DenseNetBackbone` and a final classification layer.

    The model can be used for a variety of tasks, including image
    classification on 2D images (with input shape `(height, width, channels)`)
    and volumetric data classification on 3D images (with input shape
    `(depth, height, width, channels)`).
    """

    def __init__(
        self,
        *,
        blocks,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        """
        Initializes the DenseNetBase model.

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
            pooling: (Optional) A string specifying the type of pooling to
                apply to the output of the backbone. Can be `"avg"` for global
                average pooling or `"max"` for global max pooling. This is only
                relevant if `include_top` is `False`.
            classifier_activation: A string specifying the activation function
                to use for the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
        spatial_dims = len(input_shape) - 1
        if name is None and self.__class__ is not DenseNetBase:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        # number of classes must be positive.
        if num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        # verify input activation.
        classifier_activation = validate_activation(classifier_activation)

        backbone = DenseNetBackbone(
            input_shape=input_shape, blocks=blocks, include_rescaling=include_rescaling
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
            x = layers.Dense(
                num_classes, activation=classifier_activation, dtype="float32", name="predictions"
            )(x)
        elif pooling == "avg":
            x = GlobalAvgPool(x)
        elif pooling == "max":
            x = GlobalMaxPool(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.pyramid_outputs = backbone.pyramid_outputs
        self.blocks = blocks
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
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="densenet")
@registration.register(family="densenet")
class DenseNet121(DenseNetBase, DescribeMixin):
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
            blocks=[6, 12, 24, 16],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="densenet")
@registration.register(family="densenet")
class DenseNet169(DenseNetBase, DescribeMixin):
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
            blocks=[6, 12, 32, 32],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="densenet")
@registration.register(family="densenet")
class DenseNet201(DenseNetBase, DescribeMixin):
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
            blocks=[6, 12, 48, 32],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="densenet")
@registration.register(family="densenet")
class DenseNet264(DenseNetBase, DescribeMixin):
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
            blocks=[6, 12, 64, 48],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


DENSENET_DOCSTRING = """
{name} model for classification, supporting both 2D and 3D inputs.

This class provides a complete **{name}** model, including the
convolutional backbone and the classification head (the "top"). DenseNet is
characterized by its **Dense Blocks** where each layer receives feature maps from
all preceding layers in the block, and the "bottleneck" (1x1 convolution) and
"compression" (1x1 convolution + 2x2 average pooling) layers in between blocks.

It can operate on **2D inputs** (e.g., images of shape `(H, W, C)`) or **3D inputs**
(e.g., volumetric data of shape `(D, H, W, C)`).

References:
    - "Densely Connected Convolutional Networks". CVPR 2017.
      [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)

Example:
    # TensorFlow / Keras - 2D cases.
    >>> import tensorflow as tf
    >>> from medicai.models import {name}
    >>> # Classification model
    >>> model = {name}(input_shape=(224, 224, 3), num_classes=10)
    >>> x = tf.random.normal((1, 224, 224, 3))
    >>> y = model(x)
    >>> y.shape
    (1, 10)

    # PyTorch - 2D cases.
    >>> import torch
    >>> from medicai.models import {name}
    >>> # Classification model
    >>> model = {name}(input_shape=(224, 224, 3), num_classes=10) 
    >>> x = torch.randn((1, 224, 224, 3))
    >>> y = model(x)
    >>> y.shape
    torch.Size([1, 10])


Initializes the {name} model.

Args:
    input_shape: A tuple specifying the input shape of the model,
        not including the batch size. Can be `(height, width, channels)`
        for 2D or `(depth, height, width, channels)` for 3D.
    include_rescaling: A boolean indicating whether to include a
        `Rescaling` layer at the beginning of the model. If `True`,
        the input pixels will be scaled from `[0, 255]` to `[0, 1]`.
        Defaults to `False`.
    include_top: A boolean indicating whether to include the fully
        connected classification layer at the top of the network. If
        `False`, the model's output will be the features from the
        backbone, without the final classifier. Defaults to `True`.
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

DenseNet121.__doc__ = DENSENET_DOCSTRING.format(
    name="DenseNet121",
)
DenseNet169.__doc__ = DENSENET_DOCSTRING.format(
    name="DenseNet169",
)
DenseNet201.__doc__ = DENSENET_DOCSTRING.format(
    name="DenseNet201",
)
DenseNet264.__doc__ = DENSENET_DOCSTRING.format(
    name="DenseNet264",
)
