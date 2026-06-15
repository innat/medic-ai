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
    This helper creates a lightweight configurable convolutional block commonly used in CNN architectures. 
    The block dynamically supports both ``2D`` and ``3D`` inputs by automatically detecting the 
    spatial dimensionality from the input tensor. The operation order is:

    1. Convolution 
    2. Normalization (**optional**) 
    3. Activation (**optional**)

    Bias is automatically disabled in the convolution layer when a normalization layer is used, 
    since normalization already introduces learnable affine parameters.

    Args: 
        filters (int): Number of convolution output channels. kernel_size (int or tuple, 
            optional): Size of the convolution kernel. Defaults to ``3``. 
        strides (int or tuple, optional): Convolution stride value. Defaults to ``1``. 
        padding (str, optional): Padding mode for convolution. Common options are ``"same"`` 
            and ``"valid"``. Defaults to ``"same"``. 
        normalization (str or None, optional): Type of normalization layer to apply after 
            convolution. Examples include ``"batch"``, ``"layer"``, etc. If ``None``, 
            normalization is skipped. Defaults to ``"batch"``. 
        activation (str or None, optional): Activation function applied after normalization. 
            Examples include ``"relu"``, ``"gelu"``, ``"swish"``, etc. If ``None``, 
            activation is skipped. Defaults to ``"relu"``. 
        name (str, optional): Prefix used for internal layer naming. Defaults to ``""``.

    Example:
        .. code-block:: python

            import numpy as np
            from medicai.layers import ConvBnAct

            x = np.random.randn(1, 128, 128, 3).astype(np.float32)
            y = ConvBnAct(
                filters=64, 
                kernel_size=3, 
                strides=2, 
                activation="relu"
            )(x)
            print(y.shape) # (1, 64, 64, 64)

        .. code-block:: python

            import numpy as np
            from medicai.layers import ConvBnAct

            x = np.random.randn(1, 64, 128, 128, 3).astype(np.float32)
            y = ConvBnAct(
                filters=64, 
                kernel_size=3, 
                strides=2,
                activation=None,
                normalization=None
            )(x)
            print(y.shape) # (1, 32, 64, 64, 64)

    Returns:
        Callable: A function ``apply(x)`` that accepts a ``keras.KerasTensor`` and returns 
        a ``keras.KerasTensor`` of shape ``(batch, *spatial_dims, filters)``, where 
        ``spatial_dims`` are determined by the stride and padding applied to the input 
        spatial dimensions.

    Raises:
        ValueError: If the input tensor ``x`` does not have a rank of ``4`` (2D) or 
            ``5`` (3D), i.e., spatial dimensionality is neither ``2`` nor ``3``.
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
    This layer implements a depthwise convolution for ``3D`` inputs, where
    each input channel is convolved independently using its own spatial
    kernel. Unlike a standard ``Conv3D`` layer, which mixes information
    across channels, this layer preserves channel-wise separation and
    only increases channels via ``depth_multiplier``. This design significantly 
    reduces computational cost while retaining strong spatial feature extraction 
    capability, making it suitable for lightweight 3D CNN architectures. The output 
    channels are computed as: ``output_channels = input_channels × depth_multiplier``

    Args:
        kernel_size (int or tuple of 3 ints):
            Size of the depthwise convolution window.
        strides (int or tuple of 3 ints, default=1):
            Stride length of the convolution.
        padding (str, default="valid"):
            One of ``"valid"`` or ``"same"``. Padding method.
        depth_multiplier (int, default=1):
            Number of depthwise convolution output channels for each input channel.
        dilation_rate (int or tuple of 3 ints, default=1):
            Dilation rate for dilated convolution.
        depthwise_initializer (str or ``keras.initializers.Initializer``, default="glorot_uniform"):
            Initializer for the depthwise kernel matrix.
        bias_initializer (str or ``keras.initializers.Initializer``, default="zeros"):
            Initializer for the bias vector.
        depthwise_regularizer (str or ``keras.regularizers.Regularizer``, optional):
            Regularizer function applied to the depthwise kernel.
        bias_regularizer (str or ``keras.regularizers.Regularizer``, optional):
            Regularizer function applied to the bias vector.
        activity_regularizer (str or ``keras.regularizers.Regularizer``, optional):
            Regularizer function applied to the output.
        depthwise_constraint (str or ``keras.constraints.Constraint``, optional):
            Constraint function applied to the depthwise kernel.
        bias_constraint (str or ``keras.constraints.Constraint``, optional):
            Constraint function applied to the bias vector.

    Example:
        .. code-block:: python

            import numpy as np
            from medicai.layers import DepthwiseConv3D

            x = np.random.randn(1, 16, 64, 64, 3).astype(np.float32)
            y = DepthwiseConv3D(
                kernel_size=3,
                padding="same",
                depth_multiplier=2
            )(x)
            print(y.shape) # (1, 16, 64, 64, 6)
    
    Returns:
        keras.KerasTensor: Output tensor of shape 
        ``(batch, d_out, h_out, w_out, input_channels × depth_multiplier)``,
        where the spatial dimensions ``d_out``, ``h_out``, and ``w_out`` depend 
        on the input shape, ``kernel_size``, ``strides``, ``padding``, and 
        ``dilation_rate``.

    Raises:
        ValueError: If the channel dimension of the input tensor is ``None`` 
            (i.e., undefined at build time), since the number of depthwise 
            filters cannot be inferred.
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
    This layer decomposes a standard ``Conv3D`` operation into two efficient steps:

    1. **Depthwise 3D convolution**: Applies a spatial convolution independently on each input channel (no channel mixing).
    2. **Pointwise 3D convolution**: Combines information across channels to produce the final feature representation.

    This factorization significantly reduces the number of parameters
    and computational cost compared to a full ``3D`` convolution, making it
    suitable for volumetric data, medical imaging, and video models.

    Args:
        filters (int): The number of output filters in the pointwise convolution.
        kernel_size (int or tuple of ``3`` ints): The depth, height, and width of the
            depthwise convolution window.
        strides (int or tuple of ``3`` ints, optional): The strides of the
            depthwise convolution along the depth, height, and width axes.
            Defaults to ``(1, 1, 1)``.
        padding (str, optional): One of ``"valid"`` or ``"same"``. Defaults to ``"valid"``.
        data_format (str, optional): The ordering of the dimensions in the inputs.
            Only ``"channels_last"`` is typically supported for 3D Keras implementations.
            Defaults to ``"channels_last"``.
        dilation_rate (int or tuple of ``3`` ints, optional): The dilation rate for the
            depthwise convolution. Defaults to ``(1, 1, 1)``.
        depth_multiplier (int, optional): The number of depthwise convolution output
            channels for each input channel. Defaults to 1.
        activation (str or callable, optional): Activation function to use after the
            pointwise convolution. If ``None``, no activation is applied (linear activation).
            Defaults to ``None``.
        use_bias (bool, optional): Whether the layer uses a bias vector. Defaults to ``True``.
        depthwise_initializer (str or ``keras.initializers.Initializer``, optional):
            Initializer for the depthwise kernel. Defaults to ``"glorot_uniform"``.
        pointwise_initializer (str or ``keras.initializers.Initializer``, optional):
            Initializer for the pointwise kernel. Defaults to ``"glorot_uniform"``.
        bias_initializer (str or ``keras.initializers.Initializer``, optional):
            Initializer for the bias vector. Defaults to ``"zeros"``.
        depthwise_regularizer (str or ``keras.regularizers.Regularizer``, optional):
            Regularizer function applied to the depthwise kernel. Defaults to ``None``.
        pointwise_regularizer (str or ``keras.regularizers.Regularizer``, optional):
            Regularizer function applied to the pointwise kernel. Defaults to ``None``.
        bias_regularizer (str or ``keras.regularizers.Regularizer``, optional):
            Regularizer function applied to the bias vector. Defaults to ``None``.
        activity_regularizer (str or ``keras.regularizers.Regularizer``, optional):
            Regularizer function applied to the output of the layer. Defaults to ``None``.
        depthwise_constraint (str or ``keras.constraints.Constraint``, optional):
            Constraint function applied to the depthwise kernel. Defaults to ``None``.
        pointwise_constraint (str or ``keras.constraints.Constraint``, optional):
            Constraint function applied to the pointwise kernel. Defaults to ``None``.
        bias_constraint (str or ``keras.constraints.Constraint``, optional):
            Constraint function applied to the bias vector. Defaults to ``None``.

    Example:
        .. code-block:: python
    
            import numpy as np
            from medicai.layers import SeparableConv3D

            x = np.random.randn(1, 16, 64, 64, 3).astype(np.float32)
            y = SeparableConv3D(
                filters=128,
                kernel_size=5,
                padding="same",
                depth_multiplier=2,
                activation="relu",
            )(x)
            print(y.shape) # (1, 16, 64, 64, 128)

    Returns:
        keras.KerasTensor: Output tensor of shape
        ``(batch, d_out, h_out, w_out, filters)``, where the spatial dimensions
        ``d_out``, ``h_out``, and ``w_out`` are determined by the depthwise
        convolution's ``kernel_size``, ``strides``, ``padding``, and
        ``dilation_rate``. The pointwise ``1×1×1`` convolution then projects
        the intermediate ``input_channels × depth_multiplier`` channels to
        ``filters`` output channels.

    Raises:
        ValueError: If the channel dimension of the input tensor is ``None``
            at build time, raised internally by ``DepthwiseConv3D`` since the
            depthwise filter count cannot be inferred.
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
    """
    Atrous Spatial Pyramid Pooling (ASPP) layer supporting both ``2D`` and ``3D`` inputs. This layer implements multi-scale context aggregation using parallel
    atrous (dilated) convolutions, combined with global average pooling.

    ASPP is designed to capture features at multiple receptive field scales
    without reducing spatial resolution, making it especially effective for
    dense prediction tasks such as semantic segmentation. The module consists of:

    1. A ``1x1`` convolution branch for local feature encoding.
    2. Multiple parallel dilated convolution branches with different
       dilation rates (multi-scale context).
    3. A global average pooling branch for image-level context.
    4. A final projection layer that fuses all features.

    The outputs of all branches are concatenated along the channel dimension
    and passed through a final convolutional projection.

    Args:
        dilation_rates (list[int]): List of dilation rates for parallel atrous convolutions.
            Typical values: ``[6, 12, 18]``.
        num_channels (int, optional): Number of output channels per branch.
            Defaults to ``256``.
        activation (str, optional): Activation function used in all ASPP branches.
            Defaults to ``"relu"``.
        separable (bool, optional): If ``True``, uses depthwise separable convolutions instead of
            standard convolutions in atrous branches. Defaults to ``False``.
        dropout (float, optional): Dropout rate applied after the final projection layer.
            Defaults to ``0.0``.
        **kwargs: Additional keyword arguments passed to ``keras.layers.Layer``.

    Example:
        .. code-block:: python

            import numpy as np
            from medicai.layers import AtrousSpatialPyramidPooling

            x = np.random.randn(1, 16, 64, 64, 3).astype(np.float32)
            y = AtrousSpatialPyramidPooling(
                dilation_rates=[6, 12, 18],
                num_channels=256,
            )(x)
            print(y.shape) # (1, 16, 64, 64, 256)

        .. code-block:: python

            import numpy as np
            from medicai.layers import AtrousSpatialPyramidPooling

            x = np.random.randn(1, 224, 224, 3).astype(np.float32)
            y = AtrousSpatialPyramidPooling(
                dilation_rates=[6, 12, 18],
                num_channels=256,
            )(x)
            print(y.shape) # (1, 224, 224, 256)
 
    Returns:
        keras.KerasTensor: Output tensor of shape
        ``(batch, *spatial_dims, num_channels)``, where ``spatial_dims``
        matches the input spatial dimensions exactly (preserved via ``"same"``
        padding and the global pooling branch's bilinear/trilinear resize).
        The final projection fuses ``(2 + len(dilation_rates)) × num_channels``
        concatenated branch features down to ``num_channels``.

    Raises:
        ValueError: If the spatial dimensionality of the input tensor is
            neither ``2`` nor ``3`` (i.e., input rank is not ``4`` or ``5``).
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

        pool_target = tuple(input_shape[i] for i in range(1, 1 + spatial_dims))
        self._pool_resize = ResizingND(
            target_shape=pool_target,
            interpolation="bilinear" if spatial_dims == 2 else "trilinear",
            name="pool_resize",
        )
        pool_resize_input = (
            input_shape[:1] + (1,) * spatial_dims + (self.num_channels,)
        )
        self._pool_resize.build(pool_resize_input)

        self.built = True

    def call(self, inputs):
        result = []
        for channel in self.aspp_parallel_channels:
            temp = ops.cast(channel(inputs), inputs.dtype)
            result.append(temp)

        result[-1] = self._pool_resize(result[-1])

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
