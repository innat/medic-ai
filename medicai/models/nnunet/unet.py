"""
medicai/models/nnunet/unet.py
============================
Static 2D / 3D U-Net with deep supervision.

Architecture
------------
                Input [B, *spatial, C]
                  │
            Stem conv
                  │
        ┌────────────────────────────┐
        │  Encoder (n_pooling stages)│
        │    DownBlock×n             │
        └────────────────────────────┘
                  │
             Bottleneck
                  │
        ┌────────────────────────────┐
        │  Decoder (n_pooling stages)│
        │    UpBlock×n               │
        └────────────────────────────┘
                  │
             SegHead (per decoder level for deep supervision)

Deep supervision
----------------
During training  : returns list of softmax outputs [full, 1/2, 1/4, …]
During inference : returns only the full-resolution output
"""

import keras
from keras import layers, ops

from .blocks import (
    ConvNormAct,
    DoubleConvBlock,
    DownBlock,
    SegmentationHead,
    UpBlock,
)
from medicai.layers.resize import ResizingND


class UNet(keras.Model):
    """
    Generic U-Net for 2D or 3D medical image segmentation.

    Parameters
    ----------
    spatial_dims        : 2 or 3
    n_classes           : number of output classes (including background)
    n_input_channels    : number of input modalities / channels
    n_pooling           : number of encoder stages (depth of U-Net)
    base_filters        : feature maps in first encoder stage
    max_filters         : feature map cap (default 320, same as nnU-Net)
    pool_op_kernel_sizes: per-stage pooling strides [[kD,kH,kW],…]
                          If None, all stages use 2×2×2 (or 2×2 for 2D)
    conv_kernel_sizes   : per-stage conv kernel sizes; same length as n_pooling+1
                          If None, all stages use 3×3×3 (or 3×3)
    deep_supervision    : return multi-scale outputs during training
    negative_slope      : LeakyReLU slope
    """

    def __init__(
        self,
        spatial_dims=3,
        n_classes=2,
        n_input_channels=1,
        n_pooling=5,
        base_filters=32,
        max_filters=320,
        pool_op_kernel_sizes=None,
        conv_kernel_sizes=None,
        deep_supervision=True,
        negative_slope=0.01,
        output_activation="softmax",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spatial_dims = spatial_dims
        self.n_classes = n_classes
        self.n_input_channels = n_input_channels
        self.n_pooling = n_pooling
        self.base_filters = base_filters
        self.max_filters = max_filters
        self.deep_supervision = deep_supervision
        self.negative_slope = negative_slope
        self.output_activation = output_activation

        # Default pooling strides
        if pool_op_kernel_sizes is None:
            k = [2] * spatial_dims
            pool_op_kernel_sizes = [k] * n_pooling

        # Default conv kernels
        if conv_kernel_sizes is None:
            k = [3] * spatial_dims
            conv_kernel_sizes = [k] * (n_pooling + 1)

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        # ---- Build encoder feature map schedule
        filter_schedule = _make_filter_schedule(base_filters, max_filters, n_pooling)
        self._filter_schedule = filter_schedule  # [f0, f1, …, f_n_pooling]

        # ---- Stem (first conv before any downsampling)
        self.stem = DoubleConvBlock(
            filters=filter_schedule[0],
            kernel_size=conv_kernel_sizes[0],
            spatial_dims=spatial_dims,
            negative_slope=negative_slope,
            name="stem",
        )

        # ---- Encoder
        self.encoder_blocks = []
        for stage in range(n_pooling):
            block = DownBlock(
                filters=filter_schedule[stage + 1],
                kernel_size=conv_kernel_sizes[min(stage + 1, len(conv_kernel_sizes) - 1)],
                pool_kernel=pool_op_kernel_sizes[stage],
                spatial_dims=spatial_dims,
                negative_slope=negative_slope,
                name=f"enc_{stage}",
            )
            self.encoder_blocks.append(block)

        # ---- Decoder
        self.decoder_blocks = []
        for stage in range(n_pooling):
            dec_stage = n_pooling - 1 - stage  # mirror of encoder
            skip_filters = filter_schedule[dec_stage]
            out_filters = filter_schedule[dec_stage]
            up_kernel = pool_op_kernel_sizes[dec_stage]

            block = UpBlock(
                filters=out_filters,
                kernel_size=conv_kernel_sizes[min(dec_stage, len(conv_kernel_sizes) - 1)],
                up_kernel=up_kernel,
                spatial_dims=spatial_dims,
                negative_slope=negative_slope,
                name=f"dec_{stage}",
            )
            self.decoder_blocks.append(block)

        # ---- Segmentation heads (one per decoder level for deep supervision)
        self.seg_heads = []
        self.resizers = []
        for stage in range(n_pooling):
            head = SegmentationHead(
                n_classes=n_classes,
                spatial_dims=spatial_dims,
                activation=output_activation,
                name=f"seg_head_{stage}",
            )
            self.seg_heads.append(head)

            # Resizers for deep supervision (aux outputs → full resolution)
            if stage > 0:
                resizer = ResizingND(
                    scale_factor=1.0,  # placeholder; dynamic resize in call()
                    interpolation="bilinear" if spatial_dims == 2 else "trilinear",
                    name=f"aux_resizer_{stage}",
                )
                self.resizers.append(resizer)
            else:
                self.resizers.append(None)

    # ------------------------------------------------------------------
    def call(self, x, training=False):
        """
        Forward pass.

        Parameters
        ----------
        x : tensor [B, *spatial, C]  — channel-last convention

        Returns
        -------
        training=True  : list of [full_res, half_res, …] softmax maps
        training=False : single full-resolution softmax map
        """
        # Stem
        x = self.stem(x, training=training)
        encoder_skips = [x]

        # Encoder
        for block in self.encoder_blocks:
            x = block(x, training=training)
            encoder_skips.append(x)

        # Bottleneck is the last encoder output; pop it (no skip for it)
        x = encoder_skips.pop()  # bottleneck features

        # Decoder
        decoder_outputs = []
        for i, block in enumerate(self.decoder_blocks):
            skip = encoder_skips[-(i + 1)]
            x = block(x, skip, training=training)
            decoder_outputs.append(x)

        # Segmentation heads on decoder outputs
        seg_outputs = []
        for i, head in enumerate(self.seg_heads):
            out = head(decoder_outputs[i], training=training)
            seg_outputs.append(out)

        # Full resolution is first decoder output (index 0)
        if training and self.deep_supervision:
            # Return a dictionary of outputs for Keras 3 multi-output training
            # All outputs are resized to match segment_outputs[0] shape
            target_shape = ops.shape(seg_outputs[0])[1:-1]
            out_dict = {"final": seg_outputs[0]}
            for i in range(1, len(seg_outputs)):
                # Functional resize — avoids fragile layer mutation
                interp = "bilinear" if self.spatial_dims == 2 else "trilinear"
                resized = ops.image.resize(
                    seg_outputs[i],
                    size=target_shape,
                    interpolation=interp,
                )
                out_dict[f"aux_{i-1}"] = resized
            return out_dict
        else:
            return seg_outputs[0]  # single full-resolution output

    # ------------------------------------------------------------------
    def get_config(self):
        config = super().get_config()
        config.update(
            spatial_dims=self.spatial_dims,
            n_classes=self.n_classes,
            n_input_channels=self.n_input_channels,
            n_pooling=self.n_pooling,
            base_filters=self.base_filters,
            max_filters=self.max_filters,
            pool_op_kernel_sizes=self.pool_op_kernel_sizes,
            conv_kernel_sizes=self.conv_kernel_sizes,
            deep_supervision=self.deep_supervision,
            negative_slope=self.negative_slope,
            output_activation=self.output_activation,
        )
        return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_filter_schedule(
    base_filters,
    max_filters,
    n_pooling,
):
    """
    Feature map schedule: double at each encoder stage, capped at max_filters.

    Returns list of length n_pooling + 1:
      [base, base*2, base*4, …, min(base*2^n, max)]
    """
    schedule = []
    f = base_filters
    for _ in range(n_pooling + 1):
        schedule.append(min(f, max_filters))
        f *= 2
    return schedule
