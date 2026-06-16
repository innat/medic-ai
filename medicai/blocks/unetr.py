from keras import layers

from medicai.blocks import UNetBasicBlock, UNetResBlock
from medicai.utils import get_conv_layer


class UNETRBasicBlock(layers.Layer):
    """
    A basic convolutional block used in UNETR-style architectures, supporting
    both ``2D`` and ``3D`` inputs. The spatial dimensionality is automatically
    detected from the incoming tensor shape, allowing the same block to be used
    in both image and volumetric segmentation pipelines without any configuration
    change. The block consists of two convolutional layers with normalization and
    LeakyReLU activation, and optional dropout. Depending on ``res_block``, the
    block either includes a residual skip connection (``UNetResBlock``) or applies
    the two convolutions sequentially without a shortcut (``UNetBasicBlock``).

    Args:
        filters (int): Number of output channels for both convolutional layers.
        kernel_size (int, optional): Size of the convolutional kernel in all
            spatial dimensions. Defaults to ``3``.
        stride (int, optional): Stride of the first convolutional layer in all
            spatial dimensions. Defaults to ``1``.
        norm_name (str or None, optional): Normalization layer to use. Options
            are ``"instance"``, ``"batch"``, or ``None`` for no normalization.
            Defaults to ``"instance"``.
        res_block (bool, optional): If ``True``, uses ``UNetResBlock`` (residual
            connection); otherwise uses ``UNetBasicBlock``. Defaults to ``True``.
        **kwargs: Additional keyword arguments passed to ``keras.layers.Layer``.

    Example:
        .. code-block:: python

            import numpy as np
            from medicai.blocks import UNETRBasicBlock

            x = np.random.randn(1, 128, 128, 32).astype(np.float32)
            block = UNETRBasicBlock(
                filters=64,
                kernel_size=3,
                stride=1,
                norm_name="batch",
            )
            y = block(x)
            print(y.shape) # (1, 128, 128, 64)

    Returns:
        ``keras.KerasTensor``: Output tensor of shape
        ``(batch, *spatial_dims, filters)``, where spatial dimensions
        are determined by the input shape, ``stride``, ``kernel_size``,
        and ``padding`` of the underlying block.

    Raises:
        ValueError: If ``norm_name`` is not one of ``"instance"``,
            ``"batch"``, or ``None``, depending on what the underlying
            ``UNetResBlock`` or ``UNetBasicBlock`` validates.
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        norm_name="instance",
        res_block=True,
        dropout_rate=None,
        name="unetr_basic_block",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_name = norm_name
        self.res_block = res_block
        self.dropout_rate = dropout_rate

        # child block
        if res_block:
            self.block = UNetResBlock(
                filters=filters,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
                dropout_rate=dropout_rate,
            )
        else:
            self.block = UNetBasicBlock(
                filters=filters,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
                dropout_rate=dropout_rate,
            )

    def build(self, input_shape):
        self.block.build(input_shape)
        self.built = True

    def call(self, inputs, training=None):
        return self.block(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "norm_name": self.norm_name,
                "res_block": self.res_block,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class UNETRUpsamplingBlock(layers.Layer):
    """
    UNETR decoder block that performs transposed convolution upsampling,
    skip connection concatenation, and subsequent convolutional feature
    processing. It supports both ``2D`` and ``3D`` inputs, with spatial
    dimensionality automatically detected from the incoming tensor shapes.

    The block expects a list of two inputs:

    1. ``x_in``: Feature map from the previous decoder stage, to be upsampled.
    2. ``x_skip``: Encoder skip connection concatenated after upsampling.

    The operation sequence is:

    1. Transposed convolution upsamples ``x_in`` by ``upsample_kernel_size``.
    2. The upsampled tensor is concatenated with ``x_skip`` along the channel axis.
    3. The concatenated features are processed by either a residual block
       (``UNetResBlock``) or a sequential block (``UNetBasicBlock``).

    Args:
        filters (int): Number of output channels after feature processing.
        kernel_size (int or tuple, optional): Kernel size for the internal
            convolutional block. Defaults to ``3``.
        stride (int or tuple, optional): Stride for the internal convolutional
            block. Defaults to ``1``.
        upsample_kernel_size (int or tuple, optional): Kernel size and stride
            for the transposed convolution upsampling layer. Defaults to ``2``.
        norm_name (str or None, optional): Normalization layer used in the
            internal processing block. Options are ``"instance"``, ``"batch"``,
            or ``None``. Defaults to ``"instance"``.
        res_block (bool, optional): If ``True``, uses ``UNetResBlock`` (with
            residual connection); otherwise uses ``UNetBasicBlock`` (sequential).
            Defaults to ``True``.
        **kwargs: Additional keyword arguments passed to ``keras.layers.Layer``.

    Example:
        .. code-block:: python

            import numpy as np
            from medicai.blocks import UNETRUpsamplingBlock

            x_in   = np.random.randn(1, 16, 16, 256).astype(np.float32)
            x_skip = np.random.randn(1, 32, 32, 128).astype(np.float32)
            block = UNETRUpsamplingBlock(filters=128, norm_name="batch")
            y = block([x_in, x_skip])
            print(y.shape)  # (1, 32, 32, 128)

    Returns:
        ``keras.KerasTensor``: Output tensor of shape
        ``(batch, *upsampled_spatial_dims, filters)``, where spatial
        dimensions are scaled by ``upsample_kernel_size`` relative to
        ``x_in`` and must match the spatial dimensions of ``x_skip``
        after upsampling.

    Raises:
        ValueError: If ``norm_name`` is not one of ``"instance"``,
            ``"batch"``, or ``None``, depending on what the underlying
            ``UNetResBlock`` or ``UNetBasicBlock`` validates.
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        upsample_kernel_size=2,
        norm_name="instance",
        res_block=True,
        name="unetr_upsampling_block",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.res_block = res_block

    def build(self, input_shape):
        x_shape, skip_shape = input_shape
        spatial_dims = len(x_shape) - 2

        # (upsample)
        self.up = get_conv_layer(
            spatial_dims,
            layer_type="conv_transpose",
            filters=self.filters,
            kernel_size=self.upsample_kernel_size,
            strides=self.upsample_kernel_size,
            use_bias=False,
            name="unetr_up_conv_transpose",
        )

        # concat layer
        self.concat = layers.Concatenate(axis=-1)

        # conv block (residual or basic)
        if self.res_block:
            self.block = UNetResBlock(
                filters=self.filters,
                kernel_size=self.kernel_size,
                stride=1,
                norm_name=self.norm_name,
            )
        else:
            self.block = UNetBasicBlock(
                filters=self.filters,
                kernel_size=self.kernel_size,
                stride=1,
                norm_name=self.norm_name,
            )

        self.up.build(x_shape)
        concat_channels = self.filters + skip_shape[-1]
        up_output_shape = self.up.compute_output_shape(x_shape)
        block_input_shape = (*up_output_shape[:-1], concat_channels)
        self.block.build(block_input_shape)
        self.built = True

    def call(self, inputs, training=None):
        x, skip = inputs
        x = self.up(x, training=training)
        x = self.concat([x, skip])
        x = self.block(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "upsample_kernel_size": self.upsample_kernel_size,
                "norm_name": self.norm_name,
                "res_block": self.res_block,
            }
        )
        return config


class UNETRPreUpsamplingBlock(layers.Layer):
    """
    UNETR projection upsampling block that progressively upsamples encoder
    features before they are merged into the decoder pathway. It supports
    both ``2D`` and ``3D`` inputs, with spatial dimensionality automatically
    detected from the incoming tensor shape.

    The block applies ``1 + num_layer`` transposed convolution upsampling
    steps in total:

    1. An initial transposed convolution to project and upsample the input.
    2. ``num_layer`` additional stages, each consisting of a transposed
       convolution followed by an optional convolutional sub-block
       (``UNetResBlock`` or ``UNetBasicBlock``) controlled by ``conv_block``
       and ``res_block``.

    Args:
        filters (int): Number of output channels for all transposed convolution
            and convolutional sub-block layers.
        num_layer (int): Number of repeated upsample stages after the initial
            transposed convolution.
        kernel_size (int or tuple): Kernel size for the optional convolutional
            sub-blocks.
        stride (int or tuple): Stride for the optional convolutional sub-blocks.
        upsample_kernel_size (int or tuple): Kernel size and stride used for all
            transposed convolution upsampling layers.
        conv_block (bool, optional): If ``True``, applies a convolutional
            sub-block after each transposed convolution stage. Defaults to
            ``False``.
        res_block (bool, optional): If ``True`` and ``conv_block=True``, uses
            ``UNetResBlock``; otherwise uses ``UNetBasicBlock``. Defaults to
            ``False``.
        **kwargs: Additional keyword arguments passed to ``keras.layers.Layer``.

    Example:
        .. code-block:: python

            import numpy as np
            from medicai.blocks import UNETRPreUpsamplingBlock

            x = np.random.randn(1, 16, 16, 16, 32).astype(np.float32)
            block = UNETRPreUpsamplingBlock(
                filters=128,
                num_layer=2,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                conv_block=True,
                res_block=True,
            )
            y = block(x)
            print(y.shape)  # (1, 128, 128, 128, 128)

    Returns:
        ``keras.KerasTensor``: Output tensor of shape
        ``(batch, *upsampled_spatial_dims, filters)``, where each spatial
        dimension is scaled by ``upsample_kernel_size`` a total of
        ``1 + num_layer`` times.

    Raises:
        ValueError: If ``norm_name`` passed to the internal ``UNetResBlock``
            or ``UNetBasicBlock`` is invalid, depending on what those blocks
            validate at build time.
    """

    def __init__(
        self,
        filters,
        num_layer,
        kernel_size,
        stride,
        upsample_kernel_size,
        conv_block=False,
        res_block=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsample_kernel_size = upsample_kernel_size
        self.conv_block = conv_block
        self.res_block = res_block

    def build(self, input_shape):
        spatial_dims = len(input_shape) - 2

        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            layer_type="conv_transpose",
            filters=self.filters,
            kernel_size=self.upsample_kernel_size,
            strides=self.upsample_kernel_size,
            padding="same",
        )
        self.transp_conv_init.build(input_shape)

        self.blocks = []
        current_shape = self.transp_conv_init.compute_output_shape(input_shape)

        for _ in range(self.num_layer):
            # Transpose Conv layer (instantiated inside the loop
            up_layer = get_conv_layer(
                spatial_dims,
                layer_type="conv_transpose",
                filters=self.filters,
                kernel_size=self.upsample_kernel_size,
                strides=self.upsample_kernel_size,
                padding="same",
            )
            up_layer.build(current_shape)
            current_shape = up_layer.compute_output_shape(current_shape)

            conv_layer = None
            if self.conv_block:
                # Convolutional/Residual layer
                if self.res_block:
                    conv_layer = UNetResBlock(
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        norm_name="instance",
                    )
                else:
                    conv_layer = UNetBasicBlock(
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        norm_name="instance",
                    )

                # Build the conv layer
                conv_layer.build(current_shape)
                current_shape = conv_layer.compute_output_shape(current_shape)

            self.blocks.append((up_layer, conv_layer))

        self.built = True

    def call(self, inputs, training=None):
        x = inputs

        # 1. Initial upsample
        x = self.transp_conv_init(x, training=training)

        # 2. Sequential blocks (Up+Conv/Res)
        for up_layer, conv_layer in self.blocks:
            x = up_layer(x, training=training)
            if conv_layer is not None:
                x = conv_layer(x, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "num_layer": self.num_layer,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "upsample_kernel_size": self.upsample_kernel_size,
                "conv_block": self.conv_block,
                "res_block": self.res_block,
            }
        )
        return config
