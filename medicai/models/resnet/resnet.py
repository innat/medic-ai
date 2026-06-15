import keras
from keras import layers

from medicai.utils import DescribeMixin, get_pooling_layer, registration, validate_activation

from .resnet_backbone import ResNetBackbone


@keras.saving.register_keras_serializable(package="resnet")
class ResNetBase(keras.Model):
    """
    A full ResNetBase model for classification.

    This class provides a complete ResNet model, including both the
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
        groups=1,
        width_per_group=64,
        se_block=False,
        se_ratio=16,
        se_activation="relu",
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        """
        Initializes the ResNetBase model.

        Args:
            block_type: A string specifying the type of residual block to use.
                Can be "basic_block" (for ResNet-18/34) or "bottleneck" (for ResNet-50/101/152).
            num_blocks: A list of integers specifying the number of residual blocks in each
                stage of the model.
            conv_filters: A list of integers specifying the number of filters in the initial
                convolutional stem.
            conv_kernel_sizes: A list of integers specifying the kernel size for the initial
                convolutional stem.
            num_filters: A list of integers specifying the number of filters for each
                stage of the residual blocks.
            num_strides: A list of integers specifying the stride for each stage of
                the residual blocks.
            use_pre_activation: A boolean indicating whether to use the pre-activation
                version of the ResNet block as described in ResNet v2.
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
            groups: int. Number of groups for grouped convolution. Defaults to 1.
            width_per_group: int. Bottleneck width for ResNeXt. Defaults to 64.
            se_block: bool. If `True`, apply Squeeze-and-Excitation block.
                Defaults to `False`.
            se_ratio: int. Reduction ratio for SE block. Defaults to 16.
            se_activation: str. Activation function for SE block. Defaults to "relu"
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
        if name is None and self.__class__ is not ResNetBase:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        # number of classes must be positive.
        if include_top and num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        # verify input activation.
        classifier_activation = validate_activation(classifier_activation)

        backbone = ResNetBackbone(
            input_shape=input_shape,
            block_type=block_type,
            num_blocks=num_blocks,
            conv_filters=conv_filters,
            conv_kernel_sizes=conv_kernel_sizes,
            num_filters=num_filters,
            num_strides=num_strides,
            use_pre_activation=use_pre_activation,
            include_rescaling=include_rescaling,
            groups=groups,
            width_per_group=width_per_group,
            se_block=se_block,
            se_ratio=se_ratio,
            se_activation=se_activation,
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
        self.se_block = se_block
        self.se_ratio = se_ratio
        self.se_activation = se_activation
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


@keras.saving.register_keras_serializable(package="resnet")
@registration.register(family="resnet")
class ResNet18(ResNetBase, DescribeMixin):
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
        """
        Initializes the ResNet-18 model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            include_rescaling: A boolean indicating whether to include a `Rescaling`
                layer at the beginning of the model.
            include_top: A boolean indicating whether to include the fully connected
                classification layer at the top of the network.
            num_classes: An integer specifying the number of classes for the
                classification layer.
            pooling: (Optional) A string specifying the type of pooling to apply
                to the output of the backbone.
            classifier_activation: A string specifying the activation function for
                the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
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


@keras.saving.register_keras_serializable(package="resnet")
@registration.register(family="resnet")
class ResNet34(ResNetBase, DescribeMixin):
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
        """
        Initializes the ResNet-34 model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            include_rescaling: A boolean indicating whether to include a `Rescaling`
                layer at the beginning of the model.
            include_top: A boolean indicating whether to include the fully connected
                classification layer at the top of the network.
            num_classes: An integer specifying the number of classes for the
                classification layer.
            pooling: (Optional) A string specifying the type of pooling to apply
                to the output of the backbone.
            classifier_activation: A string specifying the activation function for
                the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
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


@keras.saving.register_keras_serializable(package="resnet")
@registration.register(family="resnet")
class ResNet50(ResNetBase, DescribeMixin):
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
        """
        Initializes the ResNet-50 model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            include_rescaling: A boolean indicating whether to include a `Rescaling`
                layer at the beginning of the model.
            include_top: A boolean indicating whether to include the fully connected
                classification layer at the top of the network.
            num_classes: An integer specifying the number of classes for the
                classification layer.
            pooling: (Optional) A string specifying the type of pooling to apply
                to the output of the backbone.
            classifier_activation: A string specifying the activation function for
                the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
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


@keras.saving.register_keras_serializable(package="resnet")
@registration.register(family="resnet")
class ResNet101(ResNetBase, DescribeMixin):
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
        """
        Initializes the ResNet-101 model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            include_rescaling: A boolean indicating whether to include a `Rescaling`
                layer at the beginning of the model.
            include_top: A boolean indicating whether to include the fully connected
                classification layer at the top of the network.
            num_classes: An integer specifying the number of classes for the
                classification layer.
            pooling: (Optional) A string specifying the type of pooling to apply
                to the output of the backbone.
            classifier_activation: A string specifying the activation function for
                the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
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


@keras.saving.register_keras_serializable(package="resnet")
@registration.register(family="resnet")
class ResNet152(ResNetBase, DescribeMixin):
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
        """
        Initializes the ResNet-152 model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            include_rescaling: A boolean indicating whether to include a `Rescaling`
                layer at the beginning of the model.
            include_top: A boolean indicating whether to include the fully connected
                classification layer at the top of the network.
            num_classes: An integer specifying the number of classes for the
                classification layer.
            pooling: (Optional) A string specifying the type of pooling to apply
                to the output of the backbone.
            classifier_activation: A string specifying the activation function for
                the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
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


@keras.saving.register_keras_serializable(package="resnet")
@registration.register(name="resnet50_v2", family="resnet")
class ResNet50v2(ResNetBase, DescribeMixin):
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
        """
        Initializes the ResNet-50 v2 model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            include_rescaling: A boolean indicating whether to include a `Rescaling`
                layer at the beginning of the model.
            include_top: A boolean indicating whether to include the fully connected
                classification layer at the top of the network.
            num_classes: An integer specifying the number of classes for the
                classification layer.
            pooling: (Optional) A string specifying the type of pooling to apply
                to the output of the backbone.
            classifier_activation: A string specifying the activation function for
                the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
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


@keras.saving.register_keras_serializable(package="resnet")
@registration.register(name="resnet101_v2", family="resnet")
class ResNet101v2(ResNetBase, DescribeMixin):
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
        """
        Initializes the ResNet-101 v2 model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            include_rescaling: A boolean indicating whether to include a `Rescaling`
                layer at the beginning of the model.
            include_top: A boolean indicating whether to include the fully connected
                classification layer at the top of the network.
            num_classes: An integer specifying the number of classes for the
                classification layer.
            pooling: (Optional) A string specifying the type of pooling to apply
                to the output of the backbone.
            classifier_activation: A string specifying the activation function for
                the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
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


@keras.saving.register_keras_serializable(package="resnet")
@registration.register(name="resnet152_v2", family="resnet")
class ResNet152v2(ResNetBase, DescribeMixin):
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
        """
        Initializes the ResNet-152 v2 model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            include_rescaling: A boolean indicating whether to include a `Rescaling`
                layer at the beginning of the model.
            include_top: A boolean indicating whether to include the fully connected
                classification layer at the top of the network.
            num_classes: An integer specifying the number of classes for the
                classification layer.
            pooling: (Optional) A string specifying the type of pooling to apply
                to the output of the backbone.
            classifier_activation: A string specifying the activation function for
                the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
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


@keras.saving.register_keras_serializable(package="resnet")
@registration.register(name="resnet50_vd", family="resnet")
class ResNet50vd(ResNetBase, DescribeMixin):
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
        """
        Initializes the ResNet-50 vd model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            include_rescaling: A boolean indicating whether to include a `Rescaling`
                layer at the beginning of the model.
            include_top: A boolean indicating whether to include the fully connected
                classification layer at the top of the network.
            num_classes: An integer specifying the number of classes for the
                classification layer.
            pooling: (Optional) A string specifying the type of pooling to apply
                to the output of the backbone.
            classifier_activation: A string specifying the activation function for
                the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            block_type="bottleneck_block_vd",
            num_blocks=[3, 4, 6, 3],
            conv_filters=[32, 32, 64],
            conv_kernel_sizes=[3, 3, 3],
            num_filters=[64, 128, 256, 512],
            num_strides=[1, 2, 2, 2],
            use_pre_activation=False,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="resnet")
@registration.register(name="resnet200_vd", family="resnet")
class ResNet200vd(ResNetBase, DescribeMixin):
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
        """
        Initializes the ResNet-200 vd model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            include_rescaling: A boolean indicating whether to include a `Rescaling`
                layer at the beginning of the model.
            include_top: A boolean indicating whether to include the fully connected
                classification layer at the top of the network.
            num_classes: An integer specifying the number of classes for the
                classification layer.
            pooling: (Optional) A string specifying the type of pooling to apply
                to the output of the backbone.
            classifier_activation: A string specifying the activation function for
                the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            block_type="bottleneck_block_vd",
            num_blocks=[3, 12, 48, 3],
            conv_filters=[32, 32, 64],
            conv_kernel_sizes=[3, 3, 3],
            num_filters=[64, 128, 256, 512],
            num_strides=[1, 2, 2, 2],
            use_pre_activation=False,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="resnet")
@registration.register(family="resnet")
class ResNeXt50(ResNetBase, DescribeMixin):
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
        """
        ResNeXt-50 32x4d model for classification.

        This model extends ResNet-50 with **grouped convolutions** and follows the
        ResNeXt architecture described in "Aggregated Residual Transformations for
        Deep Neural Networks". It uses **bottleneck blocks** with **cardinality=32**
        and **width_per_group=4**, resulting in a total of **50 layers**.

        ResNeXt introduces a new dimension called "cardinality" (the size of the set
        of transformations) in addition to width and depth, which improves accuracy
        without significantly increasing computational complexity.
    
        Initializes the ResNeXt-50 32x4 model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            include_rescaling: A boolean indicating whether to include a `Rescaling`
                layer at the beginning of the model.
            include_top: A boolean indicating whether to include the fully connected
                classification layer at the top of the network.
            num_classes: An integer specifying the number of classes for the
                classification layer.
            pooling: (Optional) A string specifying the type of pooling to apply
                to the output of the backbone.
            classifier_activation: A string specifying the activation function for
                the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
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
            groups=32,
            width_per_group=4,
            use_pre_activation=False,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="resnet")
@registration.register(family="resnet")
class ResNeXt101(ResNetBase, DescribeMixin):
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
        """
        ResNeXt-101 32x8d model for classification.

        This model extends ResNet-101 with **grouped convolutions** and follows the
        ResNeXt architecture described in "Aggregated Residual Transformations for
        Deep Neural Networks". It uses **bottleneck blocks** with **cardinality=32**
        and **width_per_group=8**, resulting in a total of **101 layers**.

        As a deeper variant of ResNeXt, it can capture more complex features from
        the input data, often leading to higher accuracy on challenging datasets,
        though at a higher computational cost compared to ResNeXt-50.

        ResNeXt introduces a new dimension called "cardinality" (the size of the set
        of transformations) in addition to width and depth, which improves accuracy
        without significantly increasing computational complexity.

        Initializes the ResNeXt-101 32x8 model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            include_rescaling: A boolean indicating whether to include a `Rescaling`
                layer at the beginning of the model.
            include_top: A boolean indicating whether to include the fully connected
                classification layer at the top of the network.
            num_classes: An integer specifying the number of classes for the
                classification layer.
            pooling: (Optional) A string specifying the type of pooling to apply
                to the output of the backbone.
            classifier_activation: A string specifying the activation function for
                the classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
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
            groups=32,
            width_per_group=8,
            use_pre_activation=False,
            **kwargs,
        )


RESNET_DOCSTRING = """
This class provides a complete {name} model, including the convolutional
backbone and the classification head (the "top"). The backbone follows the
ResNet-family design and may use standard residual blocks, pre-activation
blocks, deep stems, or grouped convolutions depending on the variant.

Args:
    input_shape: A tuple specifying the input shape of the model,
        not including the batch size. Can be `(height, width, channels)` for
        2D or `(depth, height, width, channels)` for 3D.
    include_rescaling: A boolean indicating whether to include a
        ``Rescaling`` layer at the beginning of the model.
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

Returns:
    A ``keras.Model`` whose output depends on the configuration:

        - If ``include_top=True``, the output is a classification tensor of shape
        ``(batch_size, num_classes)``.
        - If ``include_top=False`` and ``pooling`` is ``None``, the output is the
        final backbone feature tensor.
        - If ``include_top=False`` and ``pooling`` is ``"avg"`` or ``"max"``,
        the output is a pooled feature tensor.

Examples:

    PyTorch backend 2D classification::

        import torch
        from medicai.models.resnet import {name}

        model = {name}(
            input_shape=(224, 224, 3), include_top=True, num_classes=2
        )
        x = torch.randn((1, 224, 224, 3))
        y = model(x)
        print(y.shape) # torch.Size([1, 2])

    Jax backend 3D classification::

        import jax
        import jax.numpy as jnp
        from medicai.models.resnet import {name}

        model = {name}(
            input_shape=(64, 224, 224, 1), num_classes=10
        )
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1, 64, 224, 224, 1))
        y = model(x)
        print(y.shape) # (1, 10)

References:
    - Deep Residual Learning for Image Recognition. CVPR 2016.
        `arXiv:1512.03385 <https://arxiv.org/abs/1512.03385>`_
    - Identity Mappings in Deep Residual Networks. ECCV 2016.
        `arXiv:1603.05027 <https://arxiv.org/abs/1603.05027>`_
    - Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.
        `arXiv:1611.05431 <https://arxiv.org/abs/1611.05431>`_
"""

ResNet18.__doc__ = RESNET_DOCSTRING.format(name="ResNet18")
ResNet34.__doc__ = RESNET_DOCSTRING.format(name="ResNet34")
ResNet50.__doc__ = RESNET_DOCSTRING.format(name="ResNet50")
ResNet101.__doc__ = RESNET_DOCSTRING.format(name="ResNet101")
ResNet152.__doc__ = RESNET_DOCSTRING.format(name="ResNet152")
ResNet50v2.__doc__ = RESNET_DOCSTRING.format(name="ResNet50v2")
ResNet101v2.__doc__ = RESNET_DOCSTRING.format(name="ResNet101v2")
ResNet152v2.__doc__ = RESNET_DOCSTRING.format(name="ResNet152v2")
ResNet50vd.__doc__ = RESNET_DOCSTRING.format(name="ResNet50vd")
ResNet200vd.__doc__ = RESNET_DOCSTRING.format(name="ResNet200vd")
ResNeXt50.__doc__ = RESNET_DOCSTRING.format(name="ResNeXt50")
ResNeXt101.__doc__ = RESNET_DOCSTRING.format(name="ResNeXt101")