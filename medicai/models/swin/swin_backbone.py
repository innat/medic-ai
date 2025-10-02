from functools import partial

from medicai.utils import hide_warnings

hide_warnings()

import keras
import numpy as np
from keras import layers

from medicai.blocks import UnetrBasicBlock
from medicai.utils import DescribeMixin, parse_model_inputs

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
        patch_norm=True,
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
        pyramid_outputs["P1"] = x

        # if True, apply residual convolution from SwinUNETR-V2
        if stage_wise_conv:
            x = UnetrBasicBlock(
                spatial_dims,
                out_channels=embed_dim * 2,
                kernel_size=3,
                stride=1,
                norm_name="instance",
                res_block=True,
            )(x)

        # Iterating over each stage of swin transformer
        for i in range(num_layers):

            # if True, apply residual convolution, used in SwinUNETR-V2
            if stage_wise_conv:
                x = UnetrBasicBlock(
                    spatial_dims,
                    out_channels=embed_dim * 2**i,
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

            # swin features
            pyramid_outputs[f"P{i + 2}"] = x

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
                Default: True.
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
                Default: True.
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
