import keras
from keras import layers

from medicai.utils import get_pooling_layer
from medicai.utils.model_utils import BACKBONE_ZOO

from .resnet_backbone import ResNetBackbone


@keras.saving.register_keras_serializable(package="resnetbase")
class ResNetBase(keras.Model):
    """
    A full ResNetBase model for classification.

    This class provides a complete ResNetBase model, including both the
    convolutional backbone and the classification head (the "top"). It is
    capable of handling both 2D and 3D inputs. The model's architecture is
    defined by the `ResNetBackbone` and a final classification layer.

    The model can be used for a variety of tasks, including image
    classification on 2D images (with input shape `(height, width, channels)`)
    and volumetric data classification on 3D images (with input shape
    `(depth, height, width, channels)`).
    """

    def __init__(
        self,
        *,
        block_type,
        num_blocks,
        conv_filters,
        conv_kernel_sizes,
        num_filters,
        num_strides,
        use_pre_activation,
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
        Initializes the ResNetBase model.

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
        backbone = ResNetBackbone(
            input_shape=input_shape,
            block_type=block_type,
            num_blocks=num_blocks,
            conv_filters=conv_filters,
            conv_kernel_sizes=conv_kernel_sizes,
            num_filters=num_filters,
            num_strides=num_strides,
            use_pre_activation=use_pre_activation,
        )
        inputs = backbone.inputs
        x = backbone.output

        GlobalAvgPool = get_pooling_layer(
            spatial_dims=spatial_dims, layer_type="avg", global_pool=True
        )
        GlobalMaxPool = get_pooling_layer(
            spatial_dims=spatial_dims, layer_type="max", global_pool=True
        )
        if include_top:
            x = GlobalAvgPool(x)
            x = layers.Dense(num_classes, activation=classifier_activation, name="predictions")(x)
        elif pooling == "avg":
            x = GlobalAvgPool(x)
        elif pooling == "max":
            x = GlobalMaxPool(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.pyramid_outputs = backbone.pyramid_outputs
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.conv_filters = conv_filters
        self.conv_kernel_sizes = conv_kernel_sizes
        self.num_filters = num_filters
        self.num_strides = num_strides
        self.use_pre_activation = use_pre_activation
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
            "block_type": self.block_type,
            "num_blocks": self.num_blocks,
            "conv_filters": self.conv_filters,
            "conv_kernel_sizes": self.conv_kernel_sizes,
            "num_filters": self.num_filters,
            "num_strides": self.num_strides,
            "use_pre_activation": self.use_pre_activation,
            "name": self.name,
        }
        return config


class ResNet18(ResNetBase):
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
        spatial_dims = len(input_shape) - 1
        name = name or f"{self.__class__.__name__}{spatial_dims}D"
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            block_type="basic_block",
            num_blocks=[2, 2, 2, 2],
            conv_filters=[64],
            conv_kernel_sizes=[7],
            num_filters=[64, 128, 256, 512],
            num_strides=[1, 2, 2, 2],
            use_pre_activation=False,
            **kwargs,
        )


class ResNet34(ResNetBase):
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
        spatial_dims = len(input_shape) - 1
        name = name or f"{self.__class__.__name__}{spatial_dims}D"
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            block_type="basic_block",
            num_blocks=[3, 4, 6, 3],
            conv_filters=[64],
            conv_kernel_sizes=[7],
            num_filters=[64, 128, 256, 512],
            num_strides=[1, 2, 2, 2],
            use_pre_activation=False,
            **kwargs,
        )


class ResNet50(ResNetBase):
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
        spatial_dims = len(input_shape) - 1
        name = name or f"{self.__class__.__name__}{spatial_dims}D"
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            block_type="bottleneck_block",
            num_blocks=[3, 4, 6, 3],
            conv_filters=[64],
            conv_kernel_sizes=[7],
            num_filters=[64, 128, 256, 512],
            num_strides=[1, 2, 2, 2],
            use_pre_activation=False,
            **kwargs,
        )


class ResNet101(ResNetBase):
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
        spatial_dims = len(input_shape) - 1
        name = name or f"{self.__class__.__name__}{spatial_dims}D"
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            block_type="bottleneck_block",
            num_blocks=[3, 4, 23, 3],
            conv_filters=[64],
            conv_kernel_sizes=[7],
            num_filters=[64, 128, 256, 512],
            num_strides=[1, 2, 2, 2],
            use_pre_activation=False,
            **kwargs,
        )


class ResNet152(ResNetBase):
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
        spatial_dims = len(input_shape) - 1
        name = name or f"{self.__class__.__name__}{spatial_dims}D"
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            block_type="bottleneck_block",
            num_blocks=[3, 8, 36, 3],
            conv_filters=[64],
            conv_kernel_sizes=[7],
            num_filters=[64, 128, 256, 512],
            num_strides=[1, 2, 2, 2],
            use_pre_activation=False,
            **kwargs,
        )


class ResNet50v2(ResNetBase):
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
        spatial_dims = len(input_shape) - 1
        name = name or f"{self.__class__.__name__}{spatial_dims}D"
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            block_type="bottleneck_block",
            num_blocks=[3, 4, 6, 3],
            conv_filters=[64],
            conv_kernel_sizes=[7],
            num_filters=[64, 128, 256, 512],
            num_strides=[1, 2, 2, 2],
            use_pre_activation=True,
            **kwargs,
        )


class ResNet101v2(ResNetBase):
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
        spatial_dims = len(input_shape) - 1
        name = name or f"{self.__class__.__name__}{spatial_dims}D"
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            block_type="bottleneck_block",
            num_blocks=[3, 4, 23, 3],
            conv_filters=[64],
            conv_kernel_sizes=[7],
            num_filters=[64, 128, 256, 512],
            num_strides=[1, 2, 2, 2],
            use_pre_activation=True,
            **kwargs,
        )


class ResNet152v2(ResNetBase):
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
        spatial_dims = len(input_shape) - 1
        name = name or f"{self.__class__.__name__}{spatial_dims}D"
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            block_type="bottleneck_block",
            num_blocks=[3, 8, 36, 3],
            conv_filters=[64],
            conv_kernel_sizes=[7],
            num_filters=[64, 128, 256, 512],
            num_strides=[1, 2, 2, 2],
            use_pre_activation=True,
            **kwargs,
        )


BACKBONE_ZOO["resnet18"] = ResNet18
BACKBONE_ZOO["resnet34"] = ResNet34
BACKBONE_ZOO["resnet50"] = ResNet50
BACKBONE_ZOO["resnet101"] = ResNet101
BACKBONE_ZOO["resnet152"] = ResNet152
BACKBONE_ZOO["resnet50v2"] = ResNet50v2
BACKBONE_ZOO["resnet101v2"] = ResNet101v2
BACKBONE_ZOO["resnet152v2"] = ResNet152v2
