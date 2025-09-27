import keras
from keras import layers

from medicai.utils import DescribeMixin, get_pooling_layer, registration

from .resnet_backbone import ResNetBackbone


@keras.saving.register_keras_serializable(package="resnetbase")
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


@keras.saving.register_keras_serializable(package="resnet18")
@registration.register(family="resnet")
class ResNet18(ResNetBase, DescribeMixin):
    """
    ResNet-18 model for classification.

    This model uses **basic blocks** with a total of **18 layers** and
    is a popular choice for various computer vision tasks. It is lighter and
    faster to train compared to deeper ResNet models. It uses the original
    ResNet v1 architecture, which applies activation after the addition
    of the shortcut.
    """

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


@keras.saving.register_keras_serializable(package="resnet34")
@registration.register(family="resnet")
class ResNet34(ResNetBase, DescribeMixin):
    """
    ResNet-34 model for classification.

    This model uses **basic blocks** with a total of **34 layers** and
    is a popular choice for various computer vision tasks. It's a deeper
    version of ResNet-18, offering increased representational power while
    still being computationally efficient. It uses the original ResNet v1
    architecture, which applies activation after the addition of the
    shortcut.
    """

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


@keras.saving.register_keras_serializable(package="resnet50")
@registration.register(family="resnet")
class ResNet50(ResNetBase, DescribeMixin):
    """
    ResNet-50 model for classification.

    This model uses **bottleneck blocks** with a total of **50 layers**.
    It's a very popular and well-balanced choice, widely used for transfer
    learning and general-purpose image classification due to its depth and
    computational efficiency. It uses the original ResNet v1 architecture.
    """

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


@keras.saving.register_keras_serializable(package="resnet101")
@registration.register(family="resnet")
class ResNet101(ResNetBase, DescribeMixin):
    """
    ResNet-101 model for classification.

    This model uses **bottleneck blocks** with a total of **101 layers**.
    As a deeper variant of ResNet, it can capture more complex features
    from the input data, often leading to higher accuracy on challenging
    datasets, though at a higher computational cost. It uses the original
    ResNet v1 architecture.
    """

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


@keras.saving.register_keras_serializable(package="resnet152")
@registration.register(family="resnet")
class ResNet152(ResNetBase, DescribeMixin):
    """
    ResNet-152 model for classification.

    This model uses **bottleneck blocks** with a total of **152 layers**.
    As the deepest variant of the original ResNet v1 series, it is designed
    to handle highly complex visual recognition tasks. It uses the original
    ResNet v1 architecture, which applies activation after the addition of the
    shortcut.
    """

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


@keras.saving.register_keras_serializable(package="resnet50v2")
@registration.register(family="resnet")
class ResNet50v2(ResNetBase, DescribeMixin):
    """
    ResNet-50 v2 model for classification.

    This model uses **bottleneck blocks** with a total of **50 layers**.
    Unlike the original ResNet v1, it implements the **pre-activation**
    design, where Batch Normalization and ReLU are applied *before* the
    convolutional layers. This modification helps with training deep networks
    by preventing vanishing gradients and improving information flow.
    """

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


@keras.saving.register_keras_serializable(package="resnet101v2")
@registration.register(family="resnet")
class ResNet101v2(ResNetBase, DescribeMixin):
    """
    ResNet-101 v2 model for classification.

    This model uses **bottleneck blocks** with a total of **101 layers**.
    Similar to ResNet-50 v2, it employs the **pre-activation** design,
    which helps to mitigate the vanishing gradient problem and enhances
    the training of this very deep network.
    """

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


@keras.saving.register_keras_serializable(package="resnet152v2")
@registration.register(family="resnet")
class ResNet152v2(ResNetBase, DescribeMixin):
    """
    ResNet-152 v2 model for classification.

    This model uses **bottleneck blocks** with a total of **152 layers**.
    It employs the **pre-activation** design, which places Batch Normalization
    and ReLU activations before the convolutional layers. This design helps
    to stabilize the training of very deep networks and improves performance
    by mitigating the vanishing gradient problem.
    """

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


@keras.saving.register_keras_serializable(package="resnet50vd")
@registration.register(family="resnet")
class ResNet50vd(ResNetBase, DescribeMixin):
    """
    ResNet-50 "very deep" (vd) model for classification.

    This model uses a "very deep" convolutional stem, which replaces the
    initial 7x7 convolutional layer with three smaller 3x3 layers. This
    modification increases the model's capacity and has been shown to
    improve accuracy. The model uses **bottleneck block vd** with a total
    of **50 layers**.
    """

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


@keras.saving.register_keras_serializable(package="resnet200vd")
@registration.register(family="resnet")
class ResNet200vd(ResNetBase, DescribeMixin):
    """
    ResNet-200 "very deep" (vd) model for classification.

    This model uses a "very deep" convolutional stem with three 3x3 layers,
    and **bottleneck block vd** with a total of **200 layers**. This variant
    is a powerful and computationally intensive model designed to achieve
    state-of-the-art performance on large-scale classification tasks.
    """

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
