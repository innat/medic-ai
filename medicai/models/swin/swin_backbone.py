from functools import partial

import keras
import numpy as np
from keras import layers

from medicai.blocks import UNETRBasicBlock
from medicai.utils import DescribeMixin, get_norm_layer, parse_model_inputs

from .swin_layers import (
    SwinBasicLayer,
    SwinBasicLayerV2,
    SwinPatchingAndEmbedding,
    SwinPatchMerging,
    SwinPatchMergingV2,
)


class SwinBackboneBase(keras.Model):
    """
    Base class for Swin Transformer backbone in 2D and 3D tasks. This class will be
    used for building SwinBackbone and SwinBackboneV2 class.

    This backbone implements the hierarchical Swin Transformer design, consisting of:
      - Patch embedding to convert the input volume/image into non-overlapping patches
      - Dropout applied to the embedded patches
      - Multiple Swin Transformer stages, each made of shifted window attention
        and MLP blocks
      - Optional patch merging (downsampling) between stages
    """

    patch_embedding = None
    patch_merging = None
    swin_basic_block = None

    def __init__(
        self,
        *,
        input_shape,
        input_tensor=None,
        include_rescaling=False,
        embed_dim=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4.0,
        patch_norm=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        qkv_bias=True,
        stage_wise_conv=False,
        downsampling_strategy="swin_transformer_like",
        **kwargs,
    ):
        # Check that the input is well specified.
        input_shape, patch_size, window_size, downsampling_strategy = resolve_input_params(
            input_shape, patch_size, window_size, downsampling_strategy
        )
        spatial_dims = len(input_shape) - 1
        input_spec = parse_model_inputs(input_shape, input_tensor, name="swin_input")

        if not (0 <= drop_rate <= 1):
            raise ValueError("drop_rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attn_drop_rate should be between 0 and 1.")

        if not (0 <= drop_path_rate <= 1):
            raise ValueError("drop_path_rate should be between 0 and 1.")

        pyramid_outputs = {}  # To store the swin-basic features
        dpr = np.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        num_layers = len(depths)
        norm_layer = partial(layers.LayerNormalization, epsilon=1e-05)

        x = input_spec

        # rescaling
        if include_rescaling:
            x = layers.Rescaling(1.0 / 255)(x)

        # patch embedding
        x = self.patch_embedding(
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            name="patching_and_embedding",
        )(x)
        # stem / early dropout
        x = layers.Dropout(drop_rate, name="pos_drop")(x)

        # patch embedding feature
        pyramid_outputs["P1"] = get_norm_layer(layer_type="layer", epsilon=1e-5)(x)

        # if True, apply residual convolution from SwinUNETR-V2
        if stage_wise_conv:
            x = UNETRBasicBlock(
                filters=embed_dim,
                kernel_size=3,
                stride=1,
                norm_name="instance",
                res_block=True,
            )(x)

        # Iterating over each stage of swin transformer
        for i in range(num_layers):

            # if True, apply residual convolution, used in SwinUNETR-V2
            if stage_wise_conv and i > 0:  # Skip for i=0 since we already processed P1
                x = UNETRBasicBlock(
                    filters=embed_dim * 2**i,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )(x)

            # swin-transformer and swin-unetr uses bit different downsampling strategy.
            if downsampling_strategy == "swin_unetr_like":
                downsampling_layer = self.patch_merging
            else:
                downsampling_layer = self.patch_merging if (i < num_layers - 1) else None

            # basic building block
            layer_kwargs = dict(
                input_dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsampling_layer=downsampling_layer,
                name=f"swin_feature{i + 1}",
            )
            layer_kwargs.update(self.extra_block_kwargs())
            layer = self.swin_basic_block(**layer_kwargs)
            x = layer(x)

            # swin features.
            # swin-v1 -> patch-merging use `norm -> reduction`
            # swin-v2 (`stage_wise_conv=True``) -> patch-merging use `reduction -> norm`
            pyramid_outputs[f"P{i + 2}"] = (
                x if stage_wise_conv else get_norm_layer(layer_type="layer", epsilon=1e-5)(x)
            )

        super().__init__(inputs=input_spec, outputs=x, **kwargs)

        self.input_tensor = input_tensor
        self.pyramid_outputs = pyramid_outputs
        self.include_rescaling = include_rescaling
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.patch_norm = patch_norm
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.depths = depths
        self.downsampling_strategy = downsampling_strategy

    def extra_block_kwargs(self):
        # Subclasses (V1/V2) override to pass extra args to SwinBasicBlock.
        return {}

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "include_rescaling": self.include_rescaling,
            "embed_dim": self.embed_dim,
            "patch_norm": self.patch_norm,
            "window_size": self.window_size,
            "patch_size": self.patch_size,
            "mlp_ratio": self.mlp_ratio,
            "drop_rate": self.drop_rate,
            "drop_path_rate": self.drop_path_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "qkv_bias": self.qkv_bias,
            "downsampling_strategy": self.downsampling_strategy,
        }
        return config


class SwinBackbone(SwinBackboneBase, DescribeMixin):
    """
    Swin Transformer V1 backbone in 2D and 3D tasks.

    This backbone implements the hierarchical Swin Transformer design, consisting of:
    - Patch embedding to convert the input volume/image into non-overlapping patches
    - Dropout applied to the embedded patches
    - Multiple Swin Transformer stages, each made of shifted window attention
        and MLP blocks
    - Optional patch merging (downsampling) between stages

    Features:
        - Supports both 2D and 3D inputs
        - Optional normalization after patch embedding (`patch_norm`)
        - Stochastic depth (`drop_path_rate`)
        - Attention dropout (`attn_drop_rate`)
        - Flexible downsampling strategy (`swin_transformer_like` or `swin_unetr_like`)
        - Optional stage-wise convolutional residual blocks (`stage_wise_conv=True`),
        following the SwinUNETR-V2 design

    Stage-wise Residual Convolution (SwinUNETR-V2):
        If `stage_wise_conv=True`, an additional convolutional residual block
        (`UnetrBasicBlock`) is inserted at the **beginning of each Swin stage**.
        This improves local feature extraction before self-attention.

        Data flow with `stage_wise_conv=True`:
            PatchEmbed → Dropout
            └─ Stage 0: [UnetrBasicBlock] → SwinBasicLayer(0)
            └─ Stage 1: [UnetrBasicBlock] → SwinBasicLayer(1)
            └─ Stage 2: [UnetrBasicBlock] → SwinBasicLayer(2)
            └─ Stage 3: [UnetrBasicBlock] → SwinBasicLayer(3)

    Reference:
        - https://arxiv.org/abs/2103.14030 (Swin Transformer V1)
        - https://github.com/Project-MONAI/MONAI (SwinUNETR-V2)
    """

    patch_embedding = SwinPatchingAndEmbedding
    patch_merging = SwinPatchMerging
    swin_basic_block = SwinBasicLayer

    def __init__(self, *args, qk_scale=None, **kwargs):
        """
        Initializes the SwinBackbone model.

        Args:
            input_shape (tuple): Shape of the input tensor.
                For 3D: (D, H, W, C), for 2D: (H, W, C).
            input_tensor (tf.Tensor, optional): Optional Keras tensor to use as
                model input. If None, a new input tensor is created. Default is None.
            include_rescaling (bool): Whether to include a Rescaling layer at the
                start to normalize inputs (1/255). Default: False.
            embed_dim (int): Number of linear projection output channels. Default: 96.
            patch_size (int | tuple): Patch size for embedding. Default: 4.
            window_size (int | tuple): Attention window size. Default: 7.
            mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default: 4.0.
            patch_norm (bool): If True, apply LayerNormalization after patch embedding.
                Default: False.
            drop_rate (float): Dropout rate. Default: 0.0.
            attn_drop_rate (float): Attention dropout rate. Default: 0.0.
            drop_path_rate (float): Stochastic depth rate per layer. Default: 0.2.
            depths (list): Number of Swin Transformer blocks in each stage.
                Default: [2, 2, 6, 2].
            num_heads (list): Number of attention heads per stage.
                Default: [3, 6, 12, 24].
            qkv_bias (bool): If True, add a learnable bias to query, key, and value.
                Default: True.
            qk_scale (float, optional): Scaling factor applied to query-key dot
                products in the attention computation. Defaults to ``None``, in
                which case the scale is computed as ``head_dim ** -0.5``.
            stage_wise_conv (bool): If True, use the SwinUNETR-V2 variant with
                convolutional residual blocks before each stage.
                Default: False.
            downsampling_strategy (str): Strategy for downsampling between stages.
                - "swin_transformer_like": standard Swin Transformer, mainly for
                    classification task.
                - "swin_unetr_like": Swin-UNETR-style patch merging, mainly for
                    segmentation task.
            **kwargs: Additional keyword arguments passed to `keras.Model`.
        """
        self.qk_scale = qk_scale
        super().__init__(*args, **kwargs)

    def extra_block_kwargs(self):
        return {"qk_scale": self.qk_scale}

    def get_config(self):
        config = super().get_config()
        config["qk_scale"] = self.qk_scale
        return config


class SwinBackboneV2(SwinBackboneBase, DescribeMixin):
    """
    Swin Transformer V2 backbone in 2D and 3D tasks.

    Key Difference from Swin V1:
        - Uses **SwinBasicLayerV2**, which incorporates scaled cosine attention,
          log-spaced continuous relative position bias, and improved numerical stability.
        - Post-Norm + Residual Scaling in SwinTransformerBlockV2.
        - Patch merging is performed with **SwinPatchMergingV2** for enhanced stability.

    This backbone implements the hierarchical Swin Transformer design, consisting of:
    - Patch embedding to convert the input volume/image into non-overlapping patches
    - Dropout applied to the embedded patches
    - Multiple Swin Transformer stages, each made of shifted window attention
        and MLP blocks
    - Optional patch merging (downsampling) between stages

    Features:
        - Supports both 2D and 3D inputs
        - Optional normalization after patch embedding (`patch_norm`)
        - Stochastic depth (`drop_path_rate`)
        - Attention dropout (`attn_drop_rate`)
        - Flexible downsampling strategy (`swin_transformer_like` or `swin_unetr_like`)
        - Optional stage-wise convolutional residual blocks (`stage_wise_conv=True`),
        following the SwinUNETR-V2 design

    Stage-wise Residual Convolution (SwinUNETR-V2):
        If `stage_wise_conv=True`, an additional convolutional residual block
        (`UnetrBasicBlock`) is inserted at the **beginning of each Swin stage**.
        This improves local feature extraction before self-attention.

        Data flow with `stage_wise_conv=True`:
            PatchEmbed → Dropout
            └─ Stage 0: [UnetrBasicBlock] → SwinBasicLayer(0)
            └─ Stage 1: [UnetrBasicBlock] → SwinBasicLayer(1)
            └─ Stage 2: [UnetrBasicBlock] → SwinBasicLayer(2)
            └─ Stage 3: [UnetrBasicBlock] → SwinBasicLayer(3)

    Reference:
        - https://arxiv.org/abs/2111.09883 (Swin Transformer V2)
        - https://github.com/Project-MONAI/MONAI (SwinUNETR-V2)
    """

    patch_embedding = SwinPatchingAndEmbedding
    patch_merging = SwinPatchMergingV2
    swin_basic_block = SwinBasicLayerV2

    def __init__(self, *args, pretrained_window_size=None, **kwargs):
        """
        Initializes the SwinBackbone V2 model.

        Args:
            input_shape (tuple): Shape of the input tensor.
                For 3D: (D, H, W, C), for 2D: (H, W, C).
            input_tensor (tf.Tensor, optional): Optional Keras tensor to use as
                model input. If None, a new input tensor is created. Default is None.
            include_rescaling (bool): Whether to include a Rescaling layer at the
                start to normalize inputs (1/255). Default: False.
            embed_dim (int): Number of linear projection output channels. Default: 96.
            patch_size (int | tuple): Patch size for embedding. Default: 4.
            window_size (int | tuple): Attention window size. Default: 7.
            mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default: 4.0.
            patch_norm (bool): If True, apply LayerNormalization after patch embedding.
                Default: False.
            drop_rate (float): Dropout rate. Default: 0.0.
            attn_drop_rate (float): Attention dropout rate. Default: 0.0.
            drop_path_rate (float): Stochastic depth rate per layer. Default: 0.2.
            depths (list): Number of Swin Transformer blocks in each stage.
                Default: [2, 2, 6, 2].
            num_heads (list): Number of attention heads per stage.
                Default: [3, 6, 12, 24].
            qkv_bias (bool): If True, add a learnable bias to query, key, and value.
                Default: True.
            pretrained_window_size (int | tuple | None): Pretrained window size(s)
                used for positional bias interpolation. Default: None.
            stage_wise_conv (bool): If True, use the SwinUNETR-V2 variant with
                convolutional residual blocks before each stage.
                Default: False.
            downsampling_strategy (str): Strategy for downsampling between stages.
                - "swin_transformer_like": standard Swin Transformer, mainly for
                    classification task.
                - "swin_unetr_like": Swin-UNETR-style patch merging, mainly for
                    segmentation task.
            **kwargs: Additional keyword arguments passed to `keras.Model`.
        """
        self.pretrained_window_size = pretrained_window_size
        super().__init__(*args, **kwargs)

    def extra_block_kwargs(self):
        return {"pretrained_window_size": self.pretrained_window_size}

    def get_config(self):
        cfg = super().get_config()
        cfg["pretrained_window_size"] = self.pretrained_window_size
        return cfg


def resolve_input_params(input_shape, patch_size, window_size, downsampling_strategy):
    # Input shape must be provided.
    if not input_shape:
        raise ValueError(
            "Argument `input_shape` must be provided. "
            "It should be a tuple of integers specifying the dimensions of the input "
            "data, not including the batch size. "
            "For 2D data, the format is `(height, width, channels)`. "
            "For 3D data, the format is `(depth, height, width, channels)`."
        )

    # Parse input specification.
    spatial_dims = len(input_shape) - 1

    # Check that the input video is well specified.
    if spatial_dims not in (2, 3):
        raise ValueError(
            f"Invalid `input_shape`: {input_shape}. "
            f"Expected 3D (H, W, C) for 2D data or 4D (D, H, W, C) for 3D data, "
            f"but got {len(input_shape)}D."
        )
    if any(dim is None for dim in input_shape[:-1]):
        raise ValueError(
            "Swin Transformer requires a fixed spatial input shape. "
            f"Got input_shape={input_shape}"
        )

    if isinstance(patch_size, int):
        patch_size = (patch_size,) * spatial_dims
    elif isinstance(patch_size, (list, tuple)) and len(patch_size) != spatial_dims:
        raise ValueError(
            f"patch_size must have length {spatial_dims} for {spatial_dims}D input. "
            f"Got {patch_size} with length {len(patch_size)}"
        )

    if isinstance(window_size, int):
        window_size = (window_size,) * spatial_dims
    elif isinstance(window_size, (list, tuple)) and len(window_size) != spatial_dims:
        raise ValueError(
            f"window_size must have length {spatial_dims} for {spatial_dims}D input. "
            f"Got {window_size} with length {len(window_size)}"
        )

    if downsampling_strategy not in ("swin_unetr_like", "swin_transformer_like"):
        raise ValueError(
            f"Invalid `downsampling_strategy`: {downsampling_strategy}. "
            "Expected one of: "
            "'swin_transformer_like' (for 2D/3D classification tasks) or "
            "'swin_unetr_like' (for 2D/3D segmentation tasks)."
        )

    return input_shape, patch_size, window_size, downsampling_strategy

SWIN_BACKBONE_DOCSTRING = """
{version} supporting both 2D and 3D inputs.

This class builds the hierarchical Swin Transformer backbone used for
classification, segmentation, and other dense prediction tasks. The model
exposes intermediate multi-scale feature maps through ``pyramid_outputs``.

The backbone is constructed in the following stages:

1. An input layer is created from ``input_shape`` or ``input_tensor``.
2. An optional ``Rescaling`` layer normalizes raw image intensities.
3. Patch embedding converts the input into non-overlapping patch tokens and a
   dropout layer is applied.
4. A sequence of hierarchical Swin stages is applied. Each stage uses
   shifted-window attention, MLP blocks, and optional patch merging between
   stages.
5. If ``stage_wise_conv=True``, residual convolutional blocks are inserted
   before Swin stages to strengthen local feature extraction. This is inspired
   by the **SwinUNETR-V2** design and can be beneficial for segmentation
   tasks.
6. Intermediate stage outputs are stored in ``pyramid_outputs``.

Args:
    input_shape (tuple):
        Shape of the input tensor excluding batch size. This can describe
        either 2D inputs ``(H, W, C)`` or 3D inputs ``(D, H, W, C)``.
    input_tensor (keras.Tensor, optional):
        Optional Keras tensor to use as model input.
    include_rescaling (bool, default=False):
        If ``True``, applies input rescaling by ``1/255``.
    embed_dim (int, default=96):
        Number of channels produced by the patch embedding layer.
    patch_size (int or tuple, default=4):
        Patch size used by the embedding layer.
    window_size (int or tuple, default=7):
        Local attention window size used by Swin blocks.
    mlp_ratio (float, default=4.0):
        Expansion ratio of the MLP hidden dimension relative to the embedding
        dimension.
    patch_norm (bool, default=False):
        Whether to apply normalization immediately after patch embedding.
    drop_rate (float, default=0.0):
        Dropout rate applied after patch embedding and inside Swin blocks.
    attn_drop_rate (float, default=0.0):
        Dropout rate applied to attention weights.
    drop_path_rate (float, default=0.2):
        Stochastic depth rate used across Swin blocks.
    depths (list, default=[2, 2, 6, 2]):
        Number of Swin blocks in each stage.
    num_heads (list, default=[3, 6, 12, 24]):
        Number of attention heads in each stage.
    qkv_bias (bool, default=True):
        Whether to use learnable bias terms in query, key, and value
        projections.
{extra_args}\
    stage_wise_conv (bool, default=False):
        If ``True``, inserts a residual convolutional block before each Swin
        stage following the SwinUNETR-V2 style.
    downsampling_strategy (str, default="swin_transformer_like"):
        Strategy for stage-to-stage downsampling.

        - ``"swin_transformer_like"`` keeps the standard Swin Transformer
          patch-merging behavior and is commonly used for classification.
        - ``"swin_unetr_like"`` keeps patch merging at every stage and is
          commonly used for segmentation-style backbones.
    **kwargs:
        Additional keyword arguments passed to ``keras.Model``.

.. rubric:: Feature pyramid
   :class: api-subheading

The ``pyramid_outputs`` dictionary stores feature tensors from the patch
embedding stage and each Swin stage. These tensors can be reused to build
decoder-style models such as segmentation or detection networks.

Example:
    Build the backbone and inspect the feature pyramid::

        import torch
        from medicai.models import {class_name}

        model = {class_name}(
            input_shape=(224, 224, 3),
            embed_dim=96,
            patch_size=4,
            window_size=7,
        )
        x = torch.randn((1, 224, 224, 3))
        y = model(x)
        print(y.shape) # torch.Size([1, 7, 7, 768])

Returns:
    A ``keras.Model`` whose forward pass returns the final backbone feature
    tensor. Intermediate multi-scale features are available in the
    ``pyramid_outputs`` attribute.

References:
    - {reference_title}.
      `arXiv:{reference_id} <https://arxiv.org/abs/{reference_id}>`_
"""

SwinBackbone.__doc__ = SWIN_BACKBONE_DOCSTRING.format(
    class_name="SwinBackbone",
    version="Swin Transformer V1 backbone",
    reference_title="Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
    reference_id="2103.14030",
    extra_args=(
        "    qk_scale (float, optional):\n"
        "        Optional scaling factor applied to query-key dot products.\n"
    ),
)
SwinBackboneV2.__doc__ = SWIN_BACKBONE_DOCSTRING.format(
    class_name="SwinBackboneV2",
    version="Swin Transformer V2 backbone",
    reference_title="Swin Transformer V2: Scaling Up Capacity and Resolution",
    reference_id="2111.09883",
    extra_args=(
        "    pretrained_window_size (int or tuple or None, default=None):\n"
        "        Optional pretrained window size used for relative position bias\n"
        "        interpolation in Swin V2 blocks.\n"
    ),
)
