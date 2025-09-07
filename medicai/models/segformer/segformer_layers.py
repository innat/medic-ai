import keras
import numpy as np
from keras import ops

from medicai.layers import DropPath
from medicai.utils import (
    get_conv_layer,
    get_norm_layer,
    get_reshaping_layer,
    parse_model_inputs,
)


class OverlappingPatchingAndEmbedding(keras.layers.Layer):
    def __init__(self, project_dim, patch_size, stride, **kwargs):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.patch_size = patch_size
        self.stride = stride

    def build(self, input_shape):
        spatial_dims = len(input_shape) - 2
        padding_size = self.patch_size // 2
        self.padding = get_reshaping_layer(
            spatial_dims=spatial_dims, layer_type="padding", padding=padding_size
        )
        self.proj = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.project_dim,
            kernel_size=self.patch_size,
            strides=self.stride,
            padding="valid",
        )
        self.norm = get_norm_layer(norm_name="layer", epsilon=1e-5)

    def call(self, x):
        x = self.padding(x)
        x = self.proj(x)

        batch_size = ops.shape(x)[0]
        spatial_dims = ops.shape(x)[1:-1]
        channels = ops.shape(x)[-1]
        seq_len = ops.prod(spatial_dims)

        x = ops.reshape(x, (batch_size, seq_len, channels))
        x = self.norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "patch_size": self.patch_size,
                "stride": self.stride,
            }
        )
        return config


class SegFormerMultiheadAttention(keras.layers.Layer):
    def __init__(
        self,
        project_dim,
        num_heads,
        sr_ratio,
        spatial_dims,
        attention_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.scale = (project_dim // num_heads) ** -0.5
        self.project_dim = project_dim
        self.spatial_dims = spatial_dims
        self.q = keras.layers.Dense(project_dim)
        self.k = keras.layers.Dense(project_dim)
        self.v = keras.layers.Dense(project_dim)
        self.proj = keras.layers.Dense(project_dim)
        self.dropout = keras.layers.Dropout(attention_dropout)
        self.proj_drop = keras.layers.Dropout(proj_dropout)

        if self.sr_ratio > 1:
            self.sr = get_conv_layer(
                spatial_dims=self.spatial_dims,
                layer_type="conv",
                filters=self.project_dim,
                kernel_size=self.sr_ratio,
                strides=self.sr_ratio,
            )
            self.norm = get_norm_layer(norm_name="layer", epsilon=1e-5)

    def call(self, x):
        B, N, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]

        # Project queries
        q = self.q(x)
        q = ops.reshape(q, (B, N, self.num_heads, C // self.num_heads))
        q = ops.transpose(q, [0, 2, 1, 3])  # (B, heads, N, C//heads)

        if self.sr_ratio > 1:
            spatial_shape = ops.cast(
                ops.round(ops.power(ops.cast(N, "float32"), 1.0 / self.spatial_dims)), "int32"
            )
            spatial_shape = [spatial_shape] * self.spatial_dims
            x = ops.reshape(x, (B, *spatial_shape, C))

            x = self.sr(x)
            x = ops.reshape(x, (B, -1, C))
            x = self.norm(x)
        else:
            x = x

        k = self.k(x)
        v = self.v(x)

        k = ops.transpose(
            ops.reshape(k, (B, -1, self.num_heads, C // self.num_heads)),
            [0, 2, 1, 3],
        )
        v = ops.transpose(
            ops.reshape(v, (B, -1, self.num_heads, C // self.num_heads)),
            [0, 2, 1, 3],
        )

        attn = (q @ ops.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = ops.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = ops.transpose(out, [0, 2, 1, 3])
        out = ops.reshape(out, (B, N, C))

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MixFFN(keras.layers.Layer):
    def __init__(self, channels, mid_channels, spatial_dims, dropout=0.0):
        super().__init__()
        self.fc1 = keras.layers.Dense(mid_channels)
        self.fc2 = keras.layers.Dense(channels)
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)
        self.spatial_dims = spatial_dims
        self.mid_channels = mid_channels

    def build(self, input_shape):
        input_channels = input_shape[-1]

        if self.spatial_dims == 2:
            self.dwconv = keras.Sequential(
                [
                    get_conv_layer(
                        spatial_dims=self.spatial_dims,
                        layer_type="depthwise_conv",
                        kernel_size=3,
                        strides=1,
                        padding="same",
                    ),
                    get_norm_layer(norm_name="batch", name="dwconv_bn"),
                ]
            )
        elif self.spatial_dims == 3:
            self.dwconv = keras.Sequential(
                [
                    get_conv_layer(
                        spatial_dims=self.spatial_dims,
                        layer_type="conv",
                        filters=self.mid_channels,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        groups=input_channels,  # Crucial for depthwise behavior
                    ),
                    get_norm_layer(norm_name="batch", name="dwconv_bn"),
                ]
            )
        else:
            raise ValueError(
                "spatial_dims should be either 2 or 3. " f"Got spatial_dims: {self.spatial_dims}"
            )

    def call(self, x):
        x = self.fc1(x)

        B, N, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]

        L = ops.cast(
            ops.round(
                ops.power(ops.cast(N, "float32"), 1.0 / ops.cast(self.spatial_dims, "float32"))
            ),
            "int32",
        )
        spatial_shape = [L] * self.spatial_dims

        x = ops.reshape(x, (B, *spatial_shape, C))

        x = self.dwconv(x)
        x = ops.reshape(x, (B, -1, C))
        x = ops.nn.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class HierarchicalTransformerEncoder(keras.layers.Layer):
    def __init__(
        self,
        project_dim,
        num_heads,
        spatial_dims,
        sr_ratio=1,
        drop_prob=0.0,
        layer_norm_epsilon=1e-5,
        attention_dropout=0.0,
        proj_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.drop_prop = drop_prob
        self.sr_ratio = sr_ratio
        self.layer_norm_epsilon = layer_norm_epsilon
        self.drop_prob = drop_prob
        self.attention_dropout = attention_dropout
        self.proj_dropout = proj_dropout
        self.spatial_dims = spatial_dims

    def build(self, input_shape):
        # spatial_dims = len(input_shape) - 1
        self.norm1 = get_norm_layer(norm_name="layer", epsilon=self.layer_norm_epsilon)
        self.attn = SegFormerMultiheadAttention(
            project_dim=self.project_dim,
            num_heads=self.num_heads,
            sr_ratio=self.sr_ratio,
            spatial_dims=self.spatial_dims,
            attention_dropout=self.attention_dropout,
            proj_dropout=self.proj_dropout,
        )
        self.drop_path = DropPath(self.drop_prop)
        self.norm2 = get_norm_layer(norm_name="layer", epsilon=self.layer_norm_epsilon)
        self.mlp = MixFFN(
            channels=self.project_dim,
            mid_channels=int(self.project_dim * 4),
            spatial_dims=self.spatial_dims,
            dropout=0.0,
        )

    def call(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "num_heads": self.num_heads,
                "drop_prop": self.drop_prop,
                "sr_ratio": self.sr_ratio,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "attention_dropout": self.attention_dropout,
                "proj_dropout": self.proj_dropout,
            }
        )
        return config


class MixVisionTransformer(keras.Model):
    def __init__(
        self,
        input_shape,
        max_drop_path_rate=0.1,
        hidden_dims=[32, 64, 160, 256],
        layerwise_sr_ratios=[4, 2, 1, 1],
        layerwise_patch_sizes=[7, 3, 3, 3],
        layerwise_strides=[4, 2, 2, 2],
        layerwise_num_heads=[1, 2, 5, 8],
        layerwise_depths=[2, 2, 2, 2],
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1
        num_layers = len(layerwise_depths)

        dpr = [x for x in np.linspace(0.0, max_drop_path_rate, sum(layerwise_depths))]

        # --- Functional API Logic ---
        inputs = parse_model_inputs(input_shape=input_shape, name="mixvit_input")
        x = inputs
        cur = 0

        for i in range(num_layers):
            # Patch Embedding Stage
            patch_embed = OverlappingPatchingAndEmbedding(
                project_dim=hidden_dims[i],
                patch_size=layerwise_patch_sizes[i],
                stride=layerwise_strides[i],
                name=f"overlap_patch_and_embed_{i}",
            )
            x = patch_embed(x)

            # Transformer Blocks
            for k in range(layerwise_depths[i]):
                x = HierarchicalTransformerEncoder(
                    project_dim=hidden_dims[i],
                    num_heads=layerwise_num_heads[i],
                    sr_ratio=layerwise_sr_ratios[i],
                    drop_prob=dpr[cur + k],
                    spatial_dims=spatial_dims,
                    name=f"hierarchical_encoder_{i}_{k}",
                )(x)
            cur += layerwise_depths[i]

            # Layer Normalization
            x = get_norm_layer(norm_name="layer", epsilon=1e-5)(x)

            # The number of patches N is calculated from the current tensor shape
            n_patches = ops.shape(x)[1]
            current_spatial_dims = int(ops.round(n_patches ** (1 / spatial_dims)))
            current_spatial_dims = [current_spatial_dims] * spatial_dims
            x = keras.layers.Reshape(
                current_spatial_dims + [hidden_dims[i]], name=f"mixvit_features{i+1}"
            )(x)

        super().__init__(inputs=inputs, outputs=x, name=name or f"mixvit{spatial_dims}D", **kwargs)

        self.hidden_dims = hidden_dims
        self.layerwise_patch_sizes = layerwise_patch_sizes
        self.layerwise_strides = layerwise_strides
        self.layerwise_num_heads = layerwise_num_heads
        self.layerwise_depths = layerwise_depths
        self.layerwise_sr_ratios = layerwise_sr_ratios
        self.max_drop_path_rate = max_drop_path_rate

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         "hidden_dims": self.hidden_dims,
    #         "layerwise_patch_sizes": self.layerwise_patch_sizes,
    #         "layerwise_strides": self.layerwise_strides,
    #         "layerwise_num_heads": self.layerwise_num_heads,
    #         "layerwise_depths": self.layerwise_depths,
    #         "layerwise_sr_ratios": self.layerwise_sr_ratios,
    #         "max_drop_path_rate": self.max_drop_path_rate,
    #         "input_shape": self.input_shape[1:],
    #     })
    #     return config
