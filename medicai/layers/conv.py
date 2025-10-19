from keras import constraints, initializers, layers, regularizers

from medicai.utils import get_act_layer, get_conv_layer, get_norm_layer


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
