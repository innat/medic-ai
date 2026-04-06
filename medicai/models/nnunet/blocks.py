"""
medicai/models/nnunet/blocks.py
==============================
Backend-agnostic building blocks for the U-Net architecture.

All layers use keras.layers / keras.ops exclusively — no backend-specific ops.

Blocks
------
ConvNormAct      : Conv (2D or 3D) -> InstanceNorm -> LeakyReLU
DoubleConvBlock  : Two stacked ConvNormAct (the basic U-Net unit)
DownBlock        : Strided conv downsampling + DoubleConvBlock
UpBlock          : Transposed conv upsampling -> concat skip -> DoubleConvBlock
"""

import keras
from keras import layers
from medicai.utils.model_utils import get_conv_layer, get_norm_layer


# ---------------------------------------------------------------------------
# ConvNormAct
# ---------------------------------------------------------------------------

class ConvNormAct(keras.Layer):
    """
    Conv -> InstanceNorm -> LeakyReLU.

    Parameters
    ----------
    filters     : number of output feature maps
    kernel_size : int or list; per-axis kernel sizes for anisotropy support
    spatial_dims: 2 or 3
    stride      : int or list; stride for the convolution
    dilation    : dilation rate (int)
    negative_slope : LeakyReLU slope (0.01 matches nnU-Net default)
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        spatial_dims=3,
        stride=1,
        dilation=1,
        negative_slope=0.01,
        use_norm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.spatial_dims = spatial_dims
        self.stride = stride
        self.dilation = dilation
        self.negative_slope = negative_slope
        self.use_norm = use_norm

        self.conv = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            dilation_rate=dilation,
            use_bias=not use_norm,  # bias redundant when normalising
            kernel_initializer="he_normal",
        )
        self.norm = get_norm_layer("instance") if use_norm else None
        self.act = layers.LeakyReLU(negative_slope=negative_slope)

    def call(self, x, training=None):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x, training=training)
        x = self.act(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            filters=self.filters,
            kernel_size=self.kernel_size,
            spatial_dims=self.spatial_dims,
            stride=self.stride,
            dilation=self.dilation,
            negative_slope=self.negative_slope,
            use_norm=self.use_norm,
        )
        return config


# ---------------------------------------------------------------------------
# DoubleConvBlock
# ---------------------------------------------------------------------------

class DoubleConvBlock(keras.Layer):
    """Two consecutive ConvNormAct layers. Standard U-Net residual unit."""

    def __init__(
        self,
        filters,
        kernel_size=3,
        spatial_dims=3,
        negative_slope=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.spatial_dims = spatial_dims
        self.negative_slope = negative_slope
        self.conv1 = ConvNormAct(
            filters, kernel_size, spatial_dims, negative_slope=negative_slope
        )
        self.conv2 = ConvNormAct(
            filters, kernel_size, spatial_dims, negative_slope=negative_slope
        )

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            filters=self.filters,
            kernel_size=self.kernel_size,
            spatial_dims=self.spatial_dims,
            negative_slope=self.negative_slope,
        )
        return cfg


# ---------------------------------------------------------------------------
# DownBlock
# ---------------------------------------------------------------------------

class DownBlock(keras.Layer):
    """
    Encoder stage: strided convolution (downsampling) -> DoubleConvBlock.

    Using strided conv instead of max-pool gives the network more flexibility
    and is what nnU-Net v2 uses by default.

    Parameters
    ----------
    filters      : output feature maps
    kernel_size  : conv kernel size (3 for isotropic, [1,3,3] for anisotropic)
    pool_kernel  : downsampling stride per axis, e.g. [2,2,2] or [1,2,2]
    spatial_dims : 2 or 3
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        pool_kernel=2,
        spatial_dims=3,
        negative_slope=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_kernel = pool_kernel
        self.spatial_dims = spatial_dims
        self.negative_slope = negative_slope
        self.down_conv = ConvNormAct(
            filters,
            kernel_size=pool_kernel,   # pool_kernel as stride
            spatial_dims=spatial_dims,
            stride=pool_kernel,
            negative_slope=negative_slope,
        )
        self.double_conv = DoubleConvBlock(
            filters, kernel_size, spatial_dims, negative_slope
        )

    def call(self, x, training=None):
        x = self.down_conv(x, training=training)
        x = self.double_conv(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            filters=self.filters,
            kernel_size=self.kernel_size,
            pool_kernel=self.pool_kernel,
            spatial_dims=self.spatial_dims,
            negative_slope=self.negative_slope,
        )
        return cfg


# ---------------------------------------------------------------------------
# UpBlock
# ---------------------------------------------------------------------------

class UpBlock(keras.Layer):
    """
    Decoder stage: transposed conv (upsample) -> concat skip -> DoubleConvBlock.

    Parameters
    ----------
    filters      : output feature maps
    kernel_size  : conv kernel
    up_kernel    : upsampling stride per axis (inverse of pool_kernel)
    spatial_dims : 2 or 3
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        up_kernel=2,
        spatial_dims=3,
        negative_slope=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.up_kernel = up_kernel
        self.spatial_dims = spatial_dims
        self.negative_slope = negative_slope
        self.up_conv = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv_transpose",
            filters=filters,
            kernel_size=up_kernel,
            strides=up_kernel,
            padding="same",
            kernel_initializer="he_normal",
        )
        self.concat = layers.Concatenate(axis=-1)
        self.double_conv = DoubleConvBlock(
            filters, kernel_size, spatial_dims, negative_slope
        )

    def call(self, x, skip, training=None):
        x = self.up_conv(x)
        x = self.concat([x, skip])
        x = self.double_conv(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            filters=self.filters,
            kernel_size=self.kernel_size,
            up_kernel=self.up_kernel,
            spatial_dims=self.spatial_dims,
            negative_slope=self.negative_slope,
        )
        return cfg


# ---------------------------------------------------------------------------
# Segmentation Head
# ---------------------------------------------------------------------------

class SegmentationHead(keras.Layer):
    """1x1 (or 1x1x1) convolution -> softmax output."""

    def __init__(self, n_classes, spatial_dims=3, activation="softmax", **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.spatial_dims = spatial_dims
        self.activation = activation
        self.conv = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=n_classes,
            kernel_size=1,
            padding="same",
            activation=activation,
            kernel_initializer="he_normal",
        )

    def call(self, x, training=None):
        return self.conv(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            n_classes=self.n_classes,
            spatial_dims=self.spatial_dims,
            activation=self.activation,
        )
        return config
