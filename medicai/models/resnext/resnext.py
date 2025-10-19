import keras
from keras import layers

from medicai.utils import DescribeMixin, keras_constants, registration
from medicai.utils.model_utils import get_pooling_layer

from .resnext_backbone import ResNeXtBackbone


@keras.saving.register_keras_serializable(package="resnextbase")
class ResNeXtBase(keras.Model):
    def __init__(
        self,
        *,
        blocks,
        input_shape,
        groups,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1
        if name is None and self.__class__ is not ResNeXtBase:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        backbone = ResNeXtBackbone(
            input_shape=input_shape,
            blocks=blocks,
            include_rescaling=include_rescaling,
            groups=groups,
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

            VALID_ACTIVATION_LIST = keras_constants.get_valid_activations()
            if classifier_activation not in VALID_ACTIVATION_LIST:
                raise ValueError(
                    f"Invalid value for `classifier_activation`: {classifier_activation!r}. "
                    f"Supported values are: {VALID_ACTIVATION_LIST}"
                )

            x = layers.Dense(
                num_classes, activation=classifier_activation, dtype="float32", name="predictions"
            )(x)
        elif pooling == "avg":
            x = GlobalAvgPool(x)
        elif pooling == "max":
            x = GlobalMaxPool(x)
        elif pooling is not None:
            raise ValueError("pooling must be one of: None, 'avg', 'max'")

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.pyramid_outputs = backbone.pyramid_outputs
        self.blocks = blocks
        self.groups = groups
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
            "groups": self.groups,
            "classifier_activation": self.classifier_activation,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="resnext")
@registration.register(name="resnext50", family="resnext")
class ResNeXt50(ResNeXtBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        groups=32,
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
            blocks=[3, 4, 6, 3],
            groups=groups,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="resnext")
@registration.register(name="resnext101", family="resnext")
class ResNeXt101(ResNeXtBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        groups=32,
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
            blocks=[3, 4, 23, 3],
            groups=groups,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


RESNEXT_DOCSTRING = """
{name} model for classification, supporting both 2D and 3D inputs.

This class provides a complete **{name}** model, including the
convolutional backbone and the classification head (the "top"). ResNeXt
is characterized by its **grouped convolutions** and **cardinality** parameter,
which introduces a new dimension called "cardinality" (the size of the set of
transformations) in addition to width and depth. This architecture improves
accuracy without increasing computational complexity significantly.

It can operate on **2D inputs** (e.g., images of shape `(H, W, C)`) or **3D inputs**
(e.g., volumetric data of shape `(D, H, W, C)`).

References:
    - "Aggregated Residual Transformations for Deep Neural Networks". CVPR 2017.
      [arXiv:1611.05431](https://arxiv.org/abs/1611.05431)

Architecture:
    - Input → Stem (7x7 Conv, BN, ReLU, MaxPool) → Stage1-4 → Classification Head
    - Each stage contains multiple residual blocks with grouped convolutions
    - Cardinality (groups): 32 by default
    - Bottleneck ratio: 4x (filters expand by 4x in each block)

Example:
    ```python
    # TensorFlow / Keras / Torch / Jax - 2D classification
    >>> import tensorflow as tf
    >>> from medicai.models import {name}
    >>> # Create model for 10-class classification
    >>> model = {name}(
    ...     input_shape=(224, 224, 3), 
    ...     num_classes=10,
    ...     include_top=True
    ... )
    >>> x = tf.random.normal((1, 224, 224, 3))
    >>> y = model(x)
    >>> y.shape
    (1, 10)

    # Feature extraction without classification head
    >>> model = {name}(
    ...     input_shape=(224, 224, 3),
    ...     include_top=False,
    ...     pooling='avg'
    ... )
    >>> features = model(x)
    >>> features.shape  # Global average pooled features
    (1, 2048)

    # 3D input for volumetric data
    >>> model_3d = {name}(input_shape=(128, 128, 128, 1), num_classes=2)
    >>> x_3d = tf.random.normal((1, 128, 128, 128, 1))
    >>> y_3d = model_3d(x_3d)
    >>> y_3d.shape
    (1, 2)
    ```

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
        relevant if `include_top` is `False`. If `None`, the model will
        output the 4D (2D) or 5D (3D) tensor from the last convolutional
        layer. Defaults to `None`.
    classifier_activation: A string specifying the activation function
        to use for the classification layer. Common values include
        `"softmax"` for multi-class classification and `"sigmoid"` for
        multi-label classification. Defaults to `"softmax"`.
    name: (Optional) A string specifying the name of the model.
        Defaults to `None`.
    **kwargs: Additional keyword arguments passed to the base class.

Raises:
    ValueError: If `input_shape` is invalid or `pooling` is not one of
        `None`, `"avg"`, or `"max"`.

Returns:
    A `keras.Model` instance implementing the {name} architecture.

Note:
    - The model uses batch normalization after each convolutional layer.
    - ReLU activation is used throughout the network.
    - The backbone produces feature pyramids at multiple scales (P1-P5)
      accessible via `model.pyramid_outputs`.
    - For 2D inputs: output shapes are based on 224x224 input resolution.
    - For 3D inputs: output shapes are based on 128x128x128 input resolution.
"""

ResNeXt50.__doc__ = RESNEXT_DOCSTRING.format(name="ResNeXt50")
ResNeXt101.__doc__ = RESNEXT_DOCSTRING.format(name="ResNeXt101")
