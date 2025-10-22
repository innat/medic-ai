import keras
from keras import constraints, initializers, layers, ops, regularizers

from medicai.utils import get_act_layer, get_conv_layer, get_norm_layer, get_pooling_layer

from .resize import ResizingND


def ConvBnAct(
    filters,
    kernel_size=3,
    strides=1,
    padding="same",
    normalization="batch",
    activation="relu",
    name="",
):
    """
    Generic convolution → normalization → activation block.
    Works for both 2D and 3D input tensors (auto-detects spatial dims).

    Args:
        filters (int): Number of output filters.
        kernel_size (int | tuple): Size of the convolution kernel.
        strides (int | tuple): Stride value for convolution.
        padding (str): Padding type, e.g. 'same' or 'valid'.
        normalization (str | None): Type of normalization ('batch', 'layer', etc.) or None.
        activation (str | None): Type of activation (e.g. 'relu', 'gelu', etc.) or None.
        name (str): Optional prefix for layer naming.

    Returns:
        function: A function that applies the block to an input tensor.
    """

    def apply(x):
        # Determine 2D or 3D dynamically
        spatial_dims = len(x.shape) - 2
        dim_str = f"{spatial_dims}D"

        conv_name = f"{name}_conv_{dim_str}"
        norm_name = f"{name}_norm_{dim_str}"
        act_name = f"{name}_activation"

        # Apply convolution
        x = get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=not normalization,
            name=conv_name,
        )(x)

        # Apply normalization if specified
        if normalization:
            x = get_norm_layer(layer_type=normalization, name=norm_name)(x)

        # Apply activation if specified
        if activation:
            x = get_act_layer(layer_type=activation, name=act_name)(x)

        return x

    return apply


class DepthwiseConv3D(layers.Conv3D):
    """
    Depthwise 3D convolution layer.

    This layer performs a depthwise convolution: it applies a single
    convolutional filter per input channel (channel-wise convolution).
    Unlike `Conv3D`, which mixes information across channels,
    `DepthwiseConv3D` processes each channel independently, making it
    computationally cheaper.

    Args:
        kernel_size (int or tuple of 3 ints):
            Size of the depthwise convolution window.
        strides (int or tuple of 3 ints, default=1):
            Stride length of the convolution.
        padding (str, default="valid"):
            One of `"valid"` or `"same"`. Padding method.
        depth_multiplier (int, default=1):
            Number of depthwise convolution output channels for each input channel.
        dilation_rate (int or tuple of 3 ints, default=1):
            Dilation rate for dilated convolution.
        depthwise_initializer (str or keras.initializers.Initializer, default="glorot_uniform"):
            Initializer for the depthwise kernel matrix.
        bias_initializer (str or keras.initializers.Initializer, default="zeros"):
            Initializer for the bias vector.
        depthwise_regularizer (str or keras.regularizers.Regularizer, optional):
            Regularizer function applied to the depthwise kernel.
        bias_regularizer (str or keras.regularizers.Regularizer, optional):
            Regularizer function applied to the bias vector.
        activity_regularizer (str or keras.regularizers.Regularizer, optional):
            Regularizer function applied to the output.
        depthwise_constraint (str or keras.constraints.Constraint, optional):
            Constraint function applied to the depthwise kernel.
        bias_constraint (str or keras.constraints.Constraint, optional):
            Constraint function applied to the bias vector.

    Input shape:
        5D tensor with shape `(batch, depth, height, width, channels)`.

    Output shape:
        5D tensor with shape `(batch, new_depth, new_height, new_width, channels * depth_multiplier)`.

    Example:
    ```python
    x = keras.Input((16, 64, 64, 3))
    y = DepthwiseConv3D(kernel_size=3, padding="same", depth_multiplier=2)(x)
    model = keras.Model(x, y)
    model.summary()
    ```
    """

    def __init__(
        self,
        kernel_size,
        strides=(1, 1, 1),
        padding="valid",
        depth_multiplier=1,
        dilation_rate=(1, 1, 1),
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super().__init__(
            filters=None,  # computed later in build()
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            kernel_initializer=self.depthwise_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.depthwise_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=self.depthwise_constraint,
            bias_constraint=self.bias_constraint,
            **kwargs,
        )

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError(
                "The channel dimension of the inputs must be defined. "
                f"Found `None` in input shape: {input_shape}"
            )
        input_dim = int(input_shape[-1])
        self.filters = input_dim * self.depth_multiplier
        self.groups = input_dim  # enforce depthwise behavior
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth_multiplier": self.depth_multiplier,
                "depthwise_initializer": initializers.serialize(self.depthwise_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "depthwise_regularizer": regularizers.serialize(self.depthwise_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": regularizers.serialize(self.activity_regularizer),
                "depthwise_constraint": constraints.serialize(self.depthwise_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        config.pop("filters", None)  # derived
        config.pop("kernel_initializer", None)
        config.pop("kernel_regularizer", None)
        config.pop("kernel_constraint", None)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SeparableConv3D(layers.Layer):
    """
    3D Separable convolution layer.

    This layer performs a depthwise convolution that acts separately on
    channels, followed by a pointwise convolution that mixes channels.
    This decomposition significantly reduces the computational complexity
    and the number of parameters compared to a standard `Conv3D` layer,
    making it suitable for deep 3D models (e.g., video or volumetric data).

    The layer is implemented as a sequence of:
    1. `DepthwiseConv3D` (kernel_size=KxKxK, groups=C_in)
    2. `Conv3D` (kernel_size=1x1x1)

    Args:
        filters (int): The number of output filters in the pointwise convolution.
        kernel_size (int or tuple of 3 ints): The depth, height, and width of the
            depthwise convolution window.
        strides (int or tuple of 3 ints, optional): The strides of the
            depthwise convolution along the depth, height, and width axes.
            Defaults to (1, 1, 1).
        padding (str, optional): One of `"valid"` or `"same"`. Defaults to `"valid"`.
        data_format (str, optional): The ordering of the dimensions in the inputs.
            Only `"channels_last"` is typically supported for 3D Keras implementations.
            Defaults to `"channels_last"`.
        dilation_rate (int or tuple of 3 ints, optional): The dilation rate for the
            depthwise convolution. Defaults to (1, 1, 1).
        depth_multiplier (int, optional): The number of depthwise convolution output
            channels for each input channel. Defaults to 1.
        activation (str or callable, optional): Activation function to use after the
            pointwise convolution. If `None`, no activation is applied (linear activation).
            Defaults to `None`.
        use_bias (bool, optional): Whether the layer uses a bias vector. Defaults to `True`.
        depthwise_initializer (str or keras.initializers.Initializer, optional):
            Initializer for the depthwise kernel. Defaults to `"glorot_uniform"`.
        pointwise_initializer (str or keras.initializers.Initializer, optional):
            Initializer for the pointwise kernel. Defaults to `"glorot_uniform"`.
        bias_initializer (str or keras.initializers.Initializer, optional):
            Initializer for the bias vector. Defaults to `"zeros"`.
        depthwise_regularizer (str or keras.regularizers.Regularizer, optional):
            Regularizer function applied to the depthwise kernel. Defaults to `None`.
        pointwise_regularizer (str or keras.regularizers.Regularizer, optional):
            Regularizer function applied to the pointwise kernel. Defaults to `None`.
        bias_regularizer (str or keras.regularizers.Regularizer, optional):
            Regularizer function applied to the bias vector. Defaults to `None`.
        activity_regularizer (str or keras.regularizers.Regularizer, optional):
            Regularizer function applied to the output of the layer. Defaults to `None`.
        depthwise_constraint (str or keras.constraints.Constraint, optional):
            Constraint function applied to the depthwise kernel. Defaults to `None`.
        pointwise_constraint (str or keras.constraints.Constraint, optional):
            Constraint function applied to the pointwise kernel. Defaults to `None`.
        bias_constraint (str or keras.constraints.Constraint, optional):
            Constraint function applied to the bias vector. Defaults to `None`.

    Input shape:
        5D tensor with shape:
        `(batch_size, spatial_dim_1, spatial_dim_2, spatial_dim_3, channels)`
        if `data_format='channels_last'`.

    Output shape:
        5D tensor with shape:
        `(batch_size, new_d, new_h, new_w, filters)`
        if `data_format='channels_last'`.

    Example:
    ```python
    # Input is a 3D volume (16 frames, 64x64 spatial, 3 channels)
    x = keras.Input((16, 64, 64, 3))

    # Apply SeparableConv3D with 128 output filters and 5x5x5 kernel
    y = SeparableConv3D(
        filters=128,
        kernel_size=5,
        padding="same",
        depth_multiplier=2,
        activation='relu'
    )(x)

    model = keras.Model(x, y)
    ```
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding="valid",
        data_format="channels_last",
        dilation_rate=(1, 1, 1),
        depth_multiplier=1,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        pointwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.depth_multiplier = depth_multiplier
        self.activation = layers.Activation(activation)
        self.use_bias = use_bias

        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.pointwise_initializer = initializers.get(pointwise_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.pointwise_regularizer = regularizers.get(pointwise_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.pointwise_constraint = constraints.get(pointwise_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Depthwise Convolution
        self.depthwise_conv = DepthwiseConv3D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            depth_multiplier=depth_multiplier,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
        )

        # Pointwise Convolution (1x1x1 Conv3D)
        self.pointwise_conv = layers.Conv3D(
            filters=filters,
            kernel_size=1,  # pointwise
            strides=1,
            padding="valid",
            data_format=data_format,
            dilation_rate=1,
            use_bias=use_bias,
            kernel_initializer=pointwise_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=pointwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=pointwise_constraint,
            bias_constraint=bias_constraint,
        )

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.pointwise_conv(x)
        return self.activation(x)

    def compute_output_shape(self, input_shape):
        output_shape = self.depthwise_conv.compute_output_shape(input_shape)
        output_shape = self.pointwise_conv.compute_output_shape(output_shape)
        return output_shape

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "depth_multiplier": self.depth_multiplier,
            "activation": layers.serialize(self.activation),
            "use_bias": self.use_bias,
            "depthwise_initializer": initializers.serialize(self.depthwise_initializer),
            "pointwise_initializer": initializers.serialize(self.pointwise_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "depthwise_regularizer": regularizers.serialize(self.depthwise_regularizer),
            "pointwise_regularizer": regularizers.serialize(self.pointwise_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "depthwise_constraint": constraints.serialize(self.depthwise_constraint),
            "pointwise_constraint": constraints.serialize(self.pointwise_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AtrousSpatialPyramidPooling(layers.Layer):
    """Implements the Atrous Spatial Pyramid Pooling support for 2D and 3D.
    For 2D input shape: `height, width, channel`.
    For 3D input shape: `depth, height, width, channel`.

    Reference for Atrous Spatial Pyramid Pooling [Rethinking Atrous Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf) and
    [Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation](https://arxiv.org/pdf/1802.02611.pdf)

    Args:
    dilation_rates: list of ints. The dilation rate for parallel dilated conv.
        Usually a sample choice of rates are `[6, 12, 18]`.
    num_channels: int. The number of output channels, defaults to `256`.
    activation: str. Activation to be used, defaults to `relu`.
    dropout: float. The dropout rate of the final projection output after the
        activations and batch norm, defaults to `0.0`, which means no dropout is
        applied to the output.

    Example:
    ```python
    input_tensor = keras.layers.Input((224, 224, 3))
    backbone = medicai.models.EfficientNetB0(
        input_tensor=input_tensor,
        include_top=False
    )
    output = backbone(inp)
    output = SpatialPyramidPooling(
        dilation_rates=[6, 12, 18]
    )(output)
    ```

    Reference:
        - https://github.com/keras-team/keras-hub
    """

    def __init__(
        self,
        dilation_rates,
        num_channels=256,
        activation="relu",
        separable=False,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dilation_rates = dilation_rates
        self.num_channels = num_channels
        self.activation = activation
        self.dropout = dropout
        self.separable = separable

    def build(self, input_shape):
        spatial_dims = len(input_shape) - 2

        if spatial_dims not in [2, 3]:
            raise ValueError(
                f"Unsupported spatial dimensions: {spatial_dims}. "
                f"Only 2D (height, width, channel) and "
                f"3D (depth, height, width, channel) are supported."
            )

        # This is the parallel networks that process the input features with
        # different dilation rates. The output from each channel will be merged
        # together and feed to the output.
        self.aspp_parallel_channels = []

        # Channel1 with Conv2D and 1x1 kernel size.
        conv_sequential = keras.Sequential(
            [
                get_conv_layer(
                    spatial_dims=spatial_dims,
                    layer_type="conv",
                    filters=self.num_channels,
                    kernel_size=1,
                    use_bias=False,
                    name="aspp_conv_1",
                ),
                get_norm_layer(layer_type="batch", name="aspp_bn_1"),
                get_act_layer(layer_type=self.activation, name="aspp_activation_1"),
            ]
        )
        conv_sequential.build(input_shape)
        self.aspp_parallel_channels.append(conv_sequential)

        # Channel 2 and afterwards are based on self.dilation_rates, and each of
        # them will have conv2D with 3x3 kernel size.
        for i, dilation_rate in enumerate(self.dilation_rates):
            conv_sequential = keras.Sequential(
                [
                    get_conv_layer(
                        spatial_dims=spatial_dims,
                        layer_type="separable_conv" if self.separable else "conv",
                        filters=self.num_channels,
                        kernel_size=3,
                        padding="same",
                        dilation_rate=dilation_rate,
                        use_bias=False,
                        name=f"aspp_conv_{i + 2}",
                    ),
                    get_norm_layer(layer_type="batch", name=f"aspp_bn_{i + 2}"),
                    get_act_layer(layer_type=self.activation, name=f"aspp_activation_{i + 2}"),
                ]
            )
            conv_sequential.build(input_shape)
            self.aspp_parallel_channels.append(conv_sequential)

        pool_sequential = keras.Sequential(
            [
                get_pooling_layer(
                    spatial_dims=spatial_dims,
                    layer_type="avg",
                    global_pool=True,
                    keepdims=True,
                    name="average_pooling",
                ),
                get_conv_layer(
                    spatial_dims=spatial_dims,
                    layer_type="conv",
                    filters=self.num_channels,
                    kernel_size=1,
                    use_bias=False,
                    name="conv_pooling",
                ),
                get_norm_layer(layer_type="batch", name="bn_pooling"),
                get_act_layer(layer_type=self.activation, name="activation_pooling"),
            ]
        )
        pool_sequential.build(input_shape)
        self.aspp_parallel_channels.append(pool_sequential)

        # Final projection layers
        projection = keras.Sequential(
            [
                get_conv_layer(
                    spatial_dims=spatial_dims,
                    layer_type="conv",
                    filters=self.num_channels,
                    kernel_size=1,
                    use_bias=False,
                    name="conv_projection",
                ),
                get_norm_layer(layer_type="batch", name="bn_projection"),
                get_act_layer(layer_type=self.activation, name="activation_projection"),
                keras.layers.Dropout(rate=self.dropout, name="dropout"),
            ],
        )
        projection_input_channels = (2 + len(self.dilation_rates)) * self.num_channels
        projection.build(input_shape[:-1] + (projection_input_channels,))
        self.projection = projection
        self.spatial_dims = spatial_dims
        self.built = True

    def call(self, inputs):
        result = []
        for channel in self.aspp_parallel_channels:
            temp = ops.cast(channel(inputs), inputs.dtype)
            result.append(temp)

        input_shape = ops.shape(inputs)
        target_size = tuple(input_shape[i] for i in range(1, 1 + self.spatial_dims))

        result[-1] = ResizingND(
            target_size, interpolation="bilinear" if self.spatial_dims == 2 else "trilinear"
        )(result[-1])

        result = ops.concatenate(result, axis=-1)
        return self.projection(result)

    def compute_output_shape(self, inputs_shape):
        return tuple((inputs_shape[:-1]) + (self.num_channels,))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dilation_rates": self.dilation_rates,
                "num_channels": self.num_channels,
                "activation": self.activation,
                "dropout": self.dropout,
            }
        )
        return config
