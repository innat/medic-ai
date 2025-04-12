from functools import partial

from medicai.utils import hide_warnings

hide_warnings()

import keras
import numpy as np
from keras import layers

from ...layers.swin import SwinBasicLayer, SwinPatchingAndEmbedding, SwinPatchMerging


def parse_model_inputs(input_shape, input_tensor, **kwargs):
    if input_tensor is None:
        return keras.layers.Input(shape=input_shape, **kwargs)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return keras.layers.Input(tensor=input_tensor, shape=input_shape, **kwargs)
        else:
            return input_tensor


class SwinBackbone(keras.Model):
    """Swin Transformer backbone for 3D video.

    This class implements the Swin Transformer architecture as a backbone for
    video understanding tasks. It consists of patch embedding, multiple Swin
    Transformer layers with window-based multi-head self-attention and shifted
    window partitioning, and patch merging for downsampling.
    """

    def __init__(
        self,
        *,
        input_shape=(32, 224, 224, 3),
        input_tensor=None,
        embed_dim=96,
        patch_size=[2, 4, 4],
        window_size=[8, 7, 7],
        mlp_ratio=4.0,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        qkv_bias=True,
        qk_scale=None,
        **kwargs,
    ):
        """Initializes the SwinBackbone.

        Args:
            input_shape (tuple): Shape of the input video tensor (D, H, W, C).
                Default is (32, 224, 224, 3).
            input_tensor (tf.Tensor, optional): Optional Keras tensor to use as
                model input. If None, a new input tensor is created. Default is None.
            embed_dim (int): Number of linear projection output channels. Default is 96.
            patch_size (list): Size of the video patches (PD, PH, PW). Default is [2, 4, 4].
            window_size (list): Size of the attention windows (WD, WH, WW). Default is [8, 7, 7].
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default is 4.0.
            patch_norm (bool): If True, apply LayerNormalization after patch embedding.
                Default is True.
            drop_rate (float): Dropout rate. Default is 0.0.
            attn_drop_rate (float): Attention dropout rate. Default is 0.0.
            drop_path_rate (float): Stochastic depth rate per layer. Default is 0.2.
            depths (list): Number of Swin Transformer blocks in each of the stages.
                Default is [2, 2, 6, 2].
            num_heads (list): Number of attention heads in different layers.
                Default is [3, 6, 12, 24].
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
                Default is True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set.
                Default is None.
            **kwargs: Additional keyword arguments passed to the base Model class.
        """
        # Parse input specification.
        input_spec = parse_model_inputs(input_shape, input_tensor, name="videos")

        # Check that the input video is well specified.
        if (
            input_spec.shape[-4] is None
            or input_spec.shape[-3] is None
            or input_spec.shape[-2] is None
        ):
            raise ValueError(
                "Depth, height and width of the video must be specified" " in `input_shape`."
            )

        x = input_spec

        norm_layer = partial(layers.LayerNormalization, epsilon=1e-05)

        x = SwinPatchingAndEmbedding(
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            name="patching_and_embedding",
        )(x)
        x = layers.Dropout(drop_rate, name="pos_drop")(x)

        dpr = np.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        num_layers = len(depths)
        for i in range(num_layers):
            layer = SwinBasicLayer(
                input_dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsampling_layer=SwinPatchMerging,
                name=f"swin_feature{i + 1}",
            )
            x = layer(x)

        super().__init__(inputs=input_spec, outputs=x, **kwargs)

        self.input_tensor = input_tensor
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
        self.qk_scale = qk_scale
        self.depths = depths

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
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
            "qk_scale": self.qk_scale,
        }
        return config
