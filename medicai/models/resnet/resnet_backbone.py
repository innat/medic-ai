import keras
from keras import layers, ops

from medicai.utils import (
    DescribeMixin,
    get_conv_layer,
    get_pooling_layer,
    get_reshaping_layer,
    parse_model_inputs,
)

from .resnet_layers import apply_resnet_block


@keras.saving.register_keras_serializable(package="resnet.backbone")
class ResNetBackbone(keras.Model, DescribeMixin):
    """ResNet and ResNetV2 core network with hyperparameters.

    This class implements a ResNet backbone as described in [Deep Residual
    Learning for Image Recognition](https://arxiv.org/abs/1512.03385)(
    CVPR 2016), [Identity Mappings in Deep Residual Networks](
    https://arxiv.org/abs/1603.05027)(ECCV 2016), [ResNet strikes back: An
    improved training procedure in timm](https://arxiv.org/abs/2110.00476)(
    NeurIPS 2021 Workshop) and [Bag of Tricks for Image Classification with
    Convolutional Neural Networks](https://arxiv.org/abs/1812.01187).

    The difference in ResNet and ResNetV2 rests in the structure of their
    individual building blocks. In **ResNetV2**, the batch normalization and
    ReLU activation precede the convolution layers, as opposed to **ResNet** where
    the batch normalization and ReLU activation are applied after the
    convolution layers. The `use_pre_activation` argument controls this behavior.

    **ResNetVd** introduces two key modifications to the standard ResNet:
    1. The initial convolutional layer is replaced by a series of three
    smaller successive convolutional layers (a "deep stem").
    2. Shortcut connections in downsampling stages use an average pooling
    operation rather than performing downsampling within the convolutional
    layers themselves.

    Reference:
        https://github.com/keras-team/keras-hub/tree/master
    """

    def __init__(
        self,
        *,
        input_shape,
        num_blocks,
        block_type,
        input_tensor=None,
        conv_filters=[64],
        conv_kernel_sizes=[7],
        num_filters=[64, 128, 256, 512],
        num_strides=[1, 2, 2, 2],
        use_pre_activation=False,
        groups=1,
        width_per_group=64,
        include_rescaling=False,
        **kwargs,
    ):
        """
        Initializes the ResNetBackbone model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size. Can be `(height, width, channels)`
                for 2D or `(depth, height, width, channels)` for 3D.
            input_tensor: (Optional) Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            conv_filters: A list of integers for the number of filters of the initial
                convolution(s). Use `[64]` for ResNet-50/101/152 (v1/v2) and `[32, 32, 64]`
                for ResNet50-vd and ResNet200-vd.
            conv_kernel_sizes: A list of integers for the kernel sizes of the initial
                convolution(s). Use `[7]` for ResNet-50/101/152 (v1/v2) and `[3, 3, 3]`
                for ResNet50-vd and ResNet200-vd.
            num_filters: A list of integers for the number of filters for each
                stack.
            num_blocks: A list of integers for the number of blocks for each stack.
            num_strides: A list of integers for the strides for each stack.
            block_type: A string for the block type to stack. One of `"basic_block"`,
                `"bottleneck_block"`, or `"bottleneck_block_vd"`. Use `"basic_block"`
                for ResNet18 and ResNet34. Use `"bottleneck_block"` for ResNet50,
                ResNet101 and ResNet152. Use `"bottleneck_block_vd"` for the vd models.
            include_rescaling: A boolean indicating whether to include a
                `Rescaling` layer at the beginning of the model. If `True`,
                the input pixels will be scaled from `[0, 255]` to `[0, 1]`.
            use_pre_activation: A boolean indicating whether to use pre-activation or not.
                `True` for ResNetV2, `False` for ResNetVd.
            groups: int. Number of groups for grouped convolution. Defaults to 1.
            width_per_group: int. Bottleneck width for ResNeXt. Defaults to 64.

        Examples:
        ```python
        import numpy as np
        import tensorflow as tf
        from medicai.models.backbones import ResNetBackbone

        input_data = np.random.uniform(0, 1, size=(2, 224, 224, 3))

        # ResNet-50 (v1) backbone with a custom config.
        model = ResNetBackbone(
            input_shape=(224, 224, 3),
            conv_filters=[64],
            conv_kernel_sizes=[7],
            num_filters=[64, 128, 256, 512],
            num_blocks=[3, 4, 6, 3],
            num_strides=[1, 2, 2, 2],
            block_type="bottleneck_block",
            use_pre_activation=False,
        )
        model(input_data)
        ```
        """

        if len(conv_filters) != len(conv_kernel_sizes):
            raise ValueError(
                "The length of `conv_filters` and"
                "`conv_kernel_sizes` must be the same. "
                f"Received: conv_filters={conv_filters}, "
                f"conv_kernel_sizes={conv_kernel_sizes}."
            )
        if len(num_filters) != len(num_blocks) or len(num_filters) != len(num_strides):
            raise ValueError(
                "The length of `num_filters`, `num_blocks` "
                "and `num_strides` must be the same. Received: "
                f"num_filters={num_filters}, "
                f"num_blocks={num_blocks}, "
                f"num_strides={num_strides}"
            )
        if num_filters[0] != 64:
            raise ValueError(
                "The first element of `num_filters` must be 64. "
                f"Received: num_filters={num_filters}"
            )
        if block_type not in ("basic_block", "bottleneck_block", "bottleneck_block_vd"):
            raise ValueError(
                '`block_type` must be one of `"basic_block"`, `"bottleneck_block"`, '
                f'or `"bottleneck_block_vd"`. Received block_type={block_type}.'
            )

        # Validate block type and ResNeXt parameters
        # TODO: Should we keep it or make it flexible!
        if block_type != "bottleneck_block" and (groups != 1 or width_per_group != 64):
            raise ValueError(
                f"Invalid configuration: only `bottleneck_block` supports grouped convolutions "
                f"(groups={groups}, width_per_group={width_per_group}). "
                f"Set groups=1 and width_per_group=64 when using {block_type}."
            )

        num_input_convs = len(conv_filters)
        num_stacks = len(num_filters)

        # === Functional Model ===
        spatial_dims = len(input_shape) - 1
        input = parse_model_inputs(input_shape, input_tensor)
        x = input

        if include_rescaling:
            x = layers.Rescaling(1.0 / 255)(x)

        # The padding between torch and tensorflow/jax differs when `strides>1`.
        # Therefore, we need to manually pad the tensor.
        x = get_reshaping_layer(
            spatial_dims=spatial_dims,
            layer_type="padding",
            padding=(conv_kernel_sizes[0] - 1) // 2,
            name="conv1_pad",
        )(x)

        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=conv_filters[0],
            kernel_size=conv_kernel_sizes[0],
            strides=2,
            use_bias=False,
            padding="valid",
            name="conv1_conv",
        )(x)

        for conv_index in range(1, num_input_convs):
            x = layers.BatchNormalization(
                axis=-1,
                epsilon=1e-5,
                momentum=0.9,
                name=f"conv{conv_index}_bn",
            )(x)
            x = layers.Activation("relu", name=f"conv{conv_index}_relu")(x)

            x = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=conv_filters[conv_index],
                kernel_size=conv_kernel_sizes[conv_index],
                strides=1,
                use_bias=False,
                padding="same",
                name=f"conv{conv_index + 1}_conv",
            )(x)

        if not use_pre_activation:
            x = layers.BatchNormalization(
                axis=-1,
                epsilon=1e-5,
                momentum=0.9,
                name=f"conv{num_input_convs}_bn",
            )(x)
            x = layers.Activation(
                "relu",
                name=f"conv{num_input_convs}_relu",
            )(x)

        pyramid_outputs = {}
        pyramid_outputs["P1"] = x

        if use_pre_activation:
            # A workaround for ResNetV2: we need -inf padding to prevent zeros
            # from being the max values in the following `MaxPooling2D`.

            pad_width = [[0, 0]]  # batch
            pad_width += [[1, 1]] * spatial_dims  # spatial dims
            pad_width += [[0, 0]]  # channels
            x = ops.pad(x, pad_width=pad_width, constant_values=float("-inf"))
        else:
            x = get_reshaping_layer(
                spatial_dims=spatial_dims, layer_type="padding", padding=1, name="pool1_pad"
            )(x)

        x = get_pooling_layer(
            spatial_dims=spatial_dims,
            layer_type="max",
            pool_size=3,
            strides=2,
            name="pool1_pool",
        )(x)

        for stack_index in range(num_stacks):
            x = apply_resnet_block(
                x,
                filters=num_filters[stack_index],
                blocks=num_blocks[stack_index],
                stride=num_strides[stack_index],
                block_type=block_type,
                use_pre_activation=use_pre_activation,
                first_shortcut=(block_type != "basic_block" or stack_index > 0),
                groups=groups,
                width_per_group=width_per_group,
                name=f"stack{stack_index}",
            )
            pyramid_outputs[f"P{stack_index + 2}"] = x

        if use_pre_activation:
            x = layers.BatchNormalization(
                axis=-1,
                epsilon=1e-5,
                momentum=0.9,
                name="post_bn",
            )(x)
            x = layers.Activation("relu", name="post_relu")(x)

        super().__init__(
            inputs=input,
            outputs=x,
            name="ResNetBackbone",
            **kwargs,
        )

        # === Config ===
        self.conv_filters = conv_filters
        self.conv_kernel_sizes = conv_kernel_sizes
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.num_strides = num_strides
        self.block_type = block_type
        self.groups = groups
        self.width_per_group = width_per_group
        self.use_pre_activation = use_pre_activation
        self.pyramid_outputs = pyramid_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "conv_filters": self.conv_filters,
                "conv_kernel_sizes": self.conv_kernel_sizes,
                "num_filters": self.num_filters,
                "num_blocks": self.num_blocks,
                "num_strides": self.num_strides,
                "block_type": self.block_type,
                "use_pre_activation": self.use_pre_activation,
                "input_shape": self.input_shape[1:],
            }
        )
        return config
