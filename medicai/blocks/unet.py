from keras import layers

from medicai.utils import get_act_layer, get_conv_layer, get_norm_layer


class UNetBasicBlock(layers.Layer):
    """
    Basic convolutional block used in UNet-style architectures. This block consists of two sequential convolution layers, where each
    convolution is followed by:

    1. Normalization
    2. LeakyReLU activation

    An optional ``dropout`` layer can be inserted between the two convolution
    stages for regularization. The block automatically supports both ``2D`` and ``3D`` inputs
    by inferring the number of spatial dimensions from the input tensor shape. Operation order:

    1. Convolution
    2. Normalization
    3. LeakyReLU
    4. Optional Dropout
    5. Convolution
    6. Normalization
    7. LeakyReLU

    Args:
        filters (int): Number of output channels for both convolution layers.
        kernel_size (int or tuple, optional): Size of the convolution kernel.
            Defaults to ``3``.
        stride (int or tuple, optional): Stride applied in the first convolution layer.
            The second convolution always uses stride ``1``.
            Defaults to ``1``.
        norm_name (str or None, optional):
            Type of normalization layer applied after convolutions.

            Common options include:

            - ``"batch"``
            - ``"instance"``
            - ``"layer"``
            - ``None`` (disable normalization)

            Defaults to ``"instance"``.

        dropout_rate (float or None, optional): Dropout probability applied between the
            two convolution blocks. If ``None``, dropout is disabled.
            Defaults to ``None``.
        name (str, optional): Name prefix used for internal layers.
            Defaults to ``"unet_basic_block"``.
        **kwargs: Additional keyword arguments passed to ``keras.layers.Layer``.

    Example:
        .. code-block:: python

            import numpy as np
            from medicai.blocks import UNetBasicBlock

            x = np.random.randn(1, 128, 128, 32).astype(np.float32)
            block = UNetBasicBlock(
                filters=64,
                kernel_size=3,
                stride=1,
                norm_name="batch",
                dropout_rate=0.1,
            )
            y = block(x)
            print(y.shape) # (1, 124, 124, 64)

    Returns:
        ``keras.KerasTensor``: Output tensor of shape
        ``(batch, H_out, W_out, filters)`` for 2D inputs or
        ``(batch, D_out, H_out, W_out, filters)`` for 3D inputs,
        where the spatial dimensions depend on ``kernel_size``,
        ``stride``, and ``padding`` of the two convolution layers.

    Raises:
        ValueError: If ``norm_name`` is not a recognized normalization
            type supported by ``get_norm_layer`` (e.g., an unregistered
            string other than ``"batch"``, ``"instance"``, ``"layer"``,
            or ``None``).
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        norm_name="instance",
        dropout_rate=None,
        name="unet_basic_block",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_name = norm_name
        self.dropout_rate = dropout_rate
        self.use_dropout = dropout_rate is not None
        self.prefix = name

    def build(self, input_shape):
        spatial_dims = len(input_shape) - 2

        self.conv1 = get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.stride,
            use_bias=False,
            name=f"{self.prefix}_conv1",
        )
        self.conv1.build(input_shape)

        self.norm1 = get_norm_layer(self.norm_name, name=f"{self.prefix}_norm1")
        out1_shape = self.conv1.compute_output_shape(input_shape)
        self.norm1.build(out1_shape)

        self.act1 = get_act_layer("leaky_relu", negative_slope=0.01, name=f"{self.prefix}_act1")

        if self.use_dropout:
            self.dropout = layers.Dropout(self.dropout_rate, name=f"{self.prefix}_dropout")

        self.conv2 = get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            use_bias=False,
            name=f"{self.prefix}_conv2",
        )
        self.conv2.build(out1_shape)

        out2_shape = self.conv2.compute_output_shape(out1_shape)
        self.norm2 = get_norm_layer(self.norm_name, name=f"{self.prefix}_norm2")
        self.norm2.build(out2_shape)

        self.act2 = get_act_layer("leaky_relu", negative_slope=0.01, name=f"{self.prefix}_act2")

        self.built = True

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        x = self.act1(x)

        if self.use_dropout:
            x = self.dropout(x, training=training)

        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.act2(x)

        return x

    def compute_output_shape(self, input_shape):
        shape = self.conv1.compute_output_shape(input_shape)
        shape = self.conv2.compute_output_shape(shape)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "norm_name": self.norm_name,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class UNetOutBlock(layers.Layer):
    """
    Output projection block used in UNet-style segmentation architectures. This block
    applies a final ``1×1`` (or ``1×1×1`` for ``3D``) convolution to project feature maps
    into the desired number of output classes. The layer optionally supports:

    1. Dropout for regularization before prediction
    2. Final activation function such as ``softmax`` or ``sigmoid``

    The block automatically supports both ``2D`` and ``3D`` inputs by detecting
    the number of spatial dimensions from the input tensor shape. Operation order:

    1. Optional Dropout
    2. ``1×1`` / ``1×1×1`` Convolution
    3. Optional Activation

    Args:
        num_classes (int):
            Number of output classes or prediction channels. Examples:

            - Binary segmentation → ``1``
            - Multi-class segmentation → ``N`` classes

        dropout_rate (float or None, optional):
            Dropout probability applied before the output projection.
            If ``None``, dropout is disabled.
            Defaults to ``None``.
        activation (str or callable or None, optional):
            Activation function applied after the output convolution.
            Common choices:

            - ``"sigmoid"`` for binary segmentation
            - ``"softmax"`` for multi-class segmentation
            - ``None`` for raw logits
            Defaults to ``None``.

        name (str, optional):
            Name prefix used for internal layers.
            Defaults to ``"unet_out_block"``.
        **kwargs:
            Additional keyword arguments passed to ``keras.layers.Layer``.

    Example:
        .. code-block:: python

            import numpy as np
            from medicai.blocks import UNetOutBlock

            x = np.random.randn(1, 128, 128, 64).astype(np.float32)
            out_block = UNetOutBlock(
                num_classes=5,
                activation="sigmoid"
            )
            y = out_block(x)
            print(y.shape)  # (1, 128, 128, 5)

    Returns:
        ``keras.KerasTensor``: Output prediction tensor of shape
        ``(batch, H, W, num_classes)`` for 2D inputs or
        ``(batch, D, H, W, num_classes)`` for 3D inputs,
        where all spatial dimensions are fully preserved by the
        ``1×1`` / ``1×1×1`` convolution.

    Raises:
        ValueError: If ``activation`` is not a valid Keras activation
            identifier or callable, raised internally by ``get_conv_layer``
            when building the output convolution.
    """

    def __init__(
        self, num_classes, dropout_rate=None, activation=None, name="unet_out_block", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.prefix = name
        self.use_dropout = dropout_rate is not None

    def build(self, input_shape):
        spatial_dims = len(input_shape) - 2

        self.conv = get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=self.num_classes,
            kernel_size=1,
            strides=1,
            use_bias=True,
            activation=self.activation,
            dtype="float32",
            name=f"{self.prefix}_conv",
        )
        self.conv.build(input_shape)

        if self.use_dropout:
            self.dropout = layers.Dropout(self.dropout_rate, name=f"{self.prefix}_dropout")

        self.built = True

    def call(self, inputs, training=None):
        x = inputs

        if self.use_dropout:
            x = self.dropout(x, training=training)

        x = self.conv(x)
        return x

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class UNetResBlock(layers.Layer):
    """
    Residual convolutional block used in UNet-style architectures. This block extends
    the standard UNet convolution block by introducing a residual skip connection
    between the input and output features, similar to residual learning used in ResNet
    architectures. The block consists of:

    1. Convolution
    2. Normalization
    3. LeakyReLU activation
    4. Optional Dropout
    5. Convolution
    6. Normalization
    7. Residual addition
    8. Final LeakyReLU activation

    If the input and output feature dimensions do not match, or if
    spatial downsampling is applied via ``stride > 1``, a ``1×1``
    projection convolution is automatically applied to the residual path
    to align tensor shapes before addition. The block automatically supports
    both ``2D`` and ``3D`` inputs by inferring the number of spatial dimensions from
    the input tensor shape.

    Args:
        filters (int): Number of output feature channels.
        kernel_size (int or tuple, optional): Size of the convolution kernel.
            Defaults to ``3``.
        stride (int or tuple, optional): Stride applied in the first convolution layer.
            If ``stride > 1``, spatial downsampling is performed and the
            residual branch is projected accordingly. Defaults to ``1``.
        norm_name (str or None, optional): Type of normalization layer applied after
            convolutions. Common options include:

            - ``"batch"``
            - ``"instance"``
            - ``"layer"``
            - ``None`` (disable normalization)
            Defaults to ``"instance"``.

        dropout_rate (float or None, optional): Dropout probability applied between
            convolution layers. If ``None``, dropout is disabled. Defaults to ``None``.
        name (str, optional): Name prefix used for internal layers.
            Defaults to ``"unet_residual_block"``.
        **kwargs: Additional keyword arguments passed to ``keras.layers.Layer``.

    Example:
        .. code-block:: python

            import numpy as np
            from medicai.blocks import UNetResBlock

            x = np.random.randn(1, 128, 128, 32).astype(np.float32)
            block = UNetResBlock(
                filters=64,
                stride=2,
                norm_name="batch",
                dropout_rate=0.1,
            )
            y = block(x)
            print(y.shape) # (1, 64, 64, 64)

    Returns:
        ``keras.KerasTensor``: Output tensor of shape
        ``(batch, H_out, W_out, filters)`` for 2D inputs or
        ``(batch, D_out, H_out, W_out, filters)`` for 3D inputs.
        Spatial dimensions are reduced when ``stride > 1``. If input
        channels differ from ``filters`` or ``stride > 1``, a ``1×1``
        projection convolution is automatically applied to the residual
        path to align shapes before addition.

    Raises:
        ValueError: If ``norm_name`` is not a recognized normalization
            type supported by ``get_norm_layer`` (e.g., an unregistered
            string other than ``"batch"``, ``"instance"``, ``"layer"``,
            or ``None``).
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        norm_name="instance",
        dropout_rate=None,
        name="unet_residual_block",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_name = norm_name
        self.dropout_rate = dropout_rate
        self.use_dropout = dropout_rate is not None
        self.prefix = name

    def build(self, input_shape):
        spatial_dims = len(input_shape) - 2
        in_channels = input_shape[-1]

        self.conv1 = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="same",
            use_bias=False,
            name=f"{self.prefix}_conv1",
        )
        self.conv1.build(input_shape)

        self.norm1 = get_norm_layer(layer_type=self.norm_name, name=f"{self.prefix}_norm1")
        self.norm1.build(self.conv1.compute_output_shape(input_shape))
        self.act1 = get_act_layer(
            layer_type="leaky_relu", negative_slope=0.01, name=f"{self.prefix}_act1"
        )

        if self.use_dropout:
            self.dropout = layers.Dropout(self.dropout_rate, name=f"{self.prefix}_dropout")

        conv1_out_shape = self.norm1.compute_output_shape(
            self.conv1.compute_output_shape(input_shape)
        )
        self.conv2 = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"{self.prefix}_conv2",
        )
        self.conv2.build(conv1_out_shape)

        self.norm2 = get_norm_layer(layer_type=self.norm_name, name=f"{self.prefix}_norm2")
        self.norm2.build(self.conv2.compute_output_shape(conv1_out_shape))

        needs_res_conv = (in_channels != self.filters) or (self.stride != 1)
        if needs_res_conv:
            self.res_conv = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=self.filters,
                kernel_size=1,
                strides=self.stride,
                padding="same",
                use_bias=False,
                name=f"{self.prefix}_res_conv",
            )
            self.res_conv.build(input_shape)
            self.res_norm = get_norm_layer(
                layer_type=self.norm_name, name=f"{self.prefix}_res_norm"
            )
            self.res_norm.build(self.res_conv.compute_output_shape(input_shape))
        else:
            self.res_conv = None
            self.res_norm = None

        self.act2 = get_act_layer(
            layer_type="leaky_relu", negative_slope=0.01, name=f"{self.prefix}_act2"
        )

        self.built = True

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out, training=training)
        out = self.act1(out)

        if self.use_dropout:
            out = self.dropout(out, training=training)

        out = self.conv2(out)
        out = self.norm2(out, training=training)

        # Handle residual connection
        if self.res_conv is not None:
            skip = self.res_conv(identity)
            skip = self.res_norm(skip, training=training)
        else:
            skip = identity

        out = layers.add([out, skip])
        out = self.act2(out)

        return out

    def compute_output_shape(self, input_shape):
        # spatial_dims = len(input_shape) - 2
        # batch_size = input_shape[0]

        # spatial_shape = []
        # for i in range(1, spatial_dims + 1):
        #     spatial_dim = input_shape[i]
        #     if self.stride > 1:
        #         # For 'same' padding with stride, output size is ceil(input_size / stride)
        #         spatial_dim = (spatial_dim + self.stride - 1) // self.stride
        #     spatial_shape.append(spatial_dim)

        # output_shape = [batch_size] + spatial_shape + [self.filters]
        # return tuple(output_shape)

        shape = self.conv1.compute_output_shape(input_shape)
        shape = self.conv2.compute_output_shape(shape)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "norm_name": self.norm_name,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
