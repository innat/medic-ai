from medicai.utils import hide_warnings
from medicai.utils.model_utils import get_conv_layer, get_reshaping_layer

hide_warnings()

import itertools

import keras
import numpy as np
from keras import initializers, layers, ops

from .drop_path import DropPath
from .mlp import SwinMLP


def window_partition(x, window_size):
    input_shape = ops.shape(x)
    batch_size = input_shape[0]
    spatial_dims = len(window_size)
    channel = input_shape[-1]

    # Flatten blocks for each spatial dim
    flat_shape = [batch_size]
    for i in range(spatial_dims):
        flat_shape.extend([input_shape[i + 1] // window_size[i], window_size[i]])
    flat_shape.append(channel)
    x = ops.reshape(x, flat_shape)

    # compute transpose order: interleave blocks and windows
    if spatial_dims == 3:  # 3D
        x = ops.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
    else:  # 2D
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])

    # Merge batch and window blocks into first dim, flatten window elements
    window_volume = 1
    for w in window_size:
        window_volume *= w

    windows = ops.reshape(x, [-1, window_volume, channel])
    return windows


def window_reverse(windows, window_size, batch_size, spatial_shape):
    spatial_dims = len(spatial_shape)
    channel = ops.shape(windows)[-1]

    # Flatten blocks for each spatial dim
    blocks = [spatial_shape[i] // window_size[i] for i in range(spatial_dims)]
    flat_shape = [batch_size] + blocks + list(window_size) + [channel]
    x = ops.reshape(windows, flat_shape)

    # compute transpose order: interleave blocks and windows
    if spatial_dims == 3:  # 3D
        x = ops.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
    else:  # 2D
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])

    # flatten blocks/windows to original spatial shape
    final_shape = [batch_size] + list(spatial_shape) + [channel]
    x = ops.reshape(x, final_shape)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)

    if shift_size is not None:
        use_shift_size = list(shift_size)

    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def compute_mask(spatial_shape, window_size, shift_size):
    spatial_dims = len(spatial_shape)
    img_mask = np.zeros((1, *spatial_shape, 1))

    # Create slices for each spatial dimension
    slices_list = []
    for i in range(spatial_dims):
        slices_list.append(
            [
                slice(-window_size[i]),
                slice(-window_size[i], -shift_size[i]),
                slice(-shift_size[i], None),
            ]
        )

    # Iterate over all combinations of slices
    cnt = 0
    for idx_combination in itertools.product(*slices_list):
        img_mask[(0,) + idx_combination + (0,)] = cnt
        cnt += 1

    # Partition into windows
    mask_windows = window_partition(ops.convert_to_tensor(img_mask, dtype="float32"), window_size)
    mask_windows = ops.squeeze(mask_windows, axis=-1)

    # Compute attention mask
    attn_mask = ops.expand_dims(mask_windows, axis=1) - ops.expand_dims(mask_windows, axis=2)
    attn_mask = ops.where(attn_mask != 0, -100.0, 0.0)
    return attn_mask


class SwinPatchingAndEmbedding(keras.Model):
    def __init__(self, patch_size, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer

    def _compute_padding(self, dim, patch_size):
        pad_amount = patch_size - (dim % patch_size)
        return [0, pad_amount if pad_amount != patch_size else 0]

    def build(self, input_shape):
        self.spatial_dims = len(input_shape) - 2
        self.pads = [[0, 0]]
        for i in range(self.spatial_dims):
            self.pads.append(self._compute_padding(input_shape[1 + i], self.patch_size[i]))
        self.pads.append([0, 0])

        if self.norm_layer is not None:
            self.norm = self.norm_layer(axis=-1, epsilon=1e-5, name="embed_norm")
            self.norm.build(
                (None,) + tuple(None for _ in range(self.spatial_dims)) + (self.embed_dim,)
            )

        self.proj = get_conv_layer(
            spatial_dims=self.spatial_dims,
            layer_type="conv",
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            name="embed_proj",
        )
        self.proj.build(
            (None,) + tuple(None for _ in range(self.spatial_dims)) + (input_shape[-1],)
        )
        self.built = True

    def call(self, x):
        x = ops.pad(x, self.pads)
        x = self.proj(x)

        if self.norm_layer is not None:
            x = self.norm(x)

        return x

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        spatial_shape = input_shape[1:-1]

        padded = [
            dim + self._compute_padding(dim, self.patch_size[i])[1]
            for i, dim in enumerate(spatial_shape)
        ]
        out_spatial = [padded[i] // self.patch_size[i] for i in range(self.spatial_dims)]
        return (batch_size, *out_spatial, self.embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class SwinPatchMerging(layers.Layer):
    def __init__(self, input_dim, norm_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.norm_layer = norm_layer

    def build(self, input_shape):
        self.spatial_dims = len(input_shape) - 2
        self.reduction = layers.Dense(2 * self.input_dim, use_bias=False)
        reduced_shape = (
            (input_shape[0],)
            + tuple(dim // 2 for dim in input_shape[1:-1])
            + ((2**self.spatial_dims) * input_shape[-1],)
        )
        self.reduction.build(reduced_shape)

        if self.norm_layer is not None:
            norm_shape = (
                (input_shape[0],)
                + tuple(dim // 2 for dim in input_shape[1:-1])
                + ((2**self.spatial_dims) * input_shape[-1],)
            )
            self.norm = self.norm_layer(axis=-1, epsilon=1e-5)
            self.norm.build(norm_shape)

        # compute padding if needed
        self.pads = [[0, 0]]
        for dim in input_shape[1:-1]:
            self.pads.append([0, ops.mod(dim, 2)])
        self.pads.append([0, 0])

        self.built = True

    def call(self, x):
        # padding if needed
        x = ops.pad(x, self.pads)
        if self.spatial_dims == 3:
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 0::2, 0::2, 1::2, :]
            x4 = x[:, 1::2, 1::2, 0::2, :]
            x5 = x[:, 1::2, 0::2, 1::2, :]
            x6 = x[:, 0::2, 1::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = ops.concatenate([x0, x1, x2, x3, x4, x5, x6, x7], axis=-1)
        else:
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = ops.concatenate([x0, x1, x2, x3], axis=-1)
        if self.norm_layer is not None:
            x = self.norm(x)
        x = self.reduction(x)
        return x

    def compute_output_shape(self, input_shape):
        return (
            (input_shape[0],) + tuple(dim // 2 for dim in input_shape[1:-1]) + (2 * self.input_dim,)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
            }
        )
        return config


class SwinWindowAttention(keras.Model):

    def __init__(
        self,
        input_dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # variables
        self.input_dim = input_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.qk_scale = qk_scale
        self.scale = head_dim**-0.5  # qk_scale or head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate
        self.spatial_dims = len(self.window_size)

    def get_relative_position_index(self):
        if self.spatial_dims == 3:
            window_depth, window_height, window_width = self.window_size
            w, d, h = ops.meshgrid(
                ops.arange(window_width),
                ops.arange(window_depth),
                ops.arange(window_height),
                indexing="ij",
            )
            coords = ops.stack([d, h, w], axis=0)  # (3, D, H, W)

        elif self.spatial_dims == 2:
            window_height, window_width = self.window_size
            w, h = ops.meshgrid(
                ops.arange(window_width),
                ops.arange(window_height),
                indexing="ij",
            )
            coords = ops.stack([h, w], axis=0)  # (2, H, W)

        coords_flatten = ops.reshape(coords, [self.spatial_dims, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = ops.transpose(relative_coords, axes=[1, 2, 0])

        # map relative coords to a single index
        offsets = []
        multiplier = 1
        for size in reversed(self.window_size):
            offsets.insert(0, multiplier)
            multiplier *= 2 * size - 1
        offsets = ops.convert_to_tensor(offsets, dtype="int32")

        relative_coords = relative_coords + ops.convert_to_tensor([s - 1 for s in self.window_size])
        relative_coords = ops.cast(relative_coords, "int32")
        rel_index = ops.sum(relative_coords * offsets, axis=-1)
        return rel_index

    def build(self, input_shape):
        num_relative_positions = 1
        for s in self.window_size:
            num_relative_positions *= 2 * s - 1

        self.relative_position_bias_table = self.add_weight(
            shape=(num_relative_positions, self.num_heads),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name="relative_position_bias_table",
        )
        self.relative_position_index = self.get_relative_position_index()

        # layers
        self.qkv = layers.Dense(self.input_dim * 3, use_bias=self.qkv_bias)
        self.attn_drop = layers.Dropout(self.attn_drop_rate)
        self.proj = layers.Dense(self.input_dim)
        self.proj_drop = layers.Dropout(self.proj_drop_rate)
        self.qkv.build(input_shape)
        self.proj.build(input_shape)
        self.built = True

    def call(self, x, mask=None, training=None):
        input_shape = ops.shape(x)
        batch_size, depth, channel = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        qkv = self.qkv(x)
        qkv = ops.reshape(
            qkv,
            [batch_size, depth, 3, self.num_heads, channel // self.num_heads],
        )
        qkv = ops.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = ops.split(qkv, 3, axis=0)
        q = ops.squeeze(q, axis=0) * self.scale
        k = ops.squeeze(k, axis=0)
        v = ops.squeeze(v, axis=0)
        attn = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2]))

        rel_pos_bias = ops.take(
            self.relative_position_bias_table,
            self.relative_position_index[:depth, :depth],
            axis=0,
        )
        rel_pos_bias = ops.reshape(rel_pos_bias, [depth, depth, -1])
        rel_pos_bias = ops.transpose(rel_pos_bias, [2, 0, 1])
        attn = attn + rel_pos_bias[None, ...]

        if mask is not None:
            mask_size = ops.shape(mask)[0]
            mask = ops.cast(mask, dtype=attn.dtype)
            attn = (
                ops.reshape(
                    attn,
                    [
                        batch_size // mask_size,
                        mask_size,
                        self.num_heads,
                        depth,
                        depth,
                    ],
                )
                + mask[:, None, :, :]
            )
            attn = ops.reshape(attn, [-1, self.num_heads, depth, depth])

        attn = keras.activations.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        x = ops.matmul(attn, v)
        x = ops.transpose(x, [0, 2, 1, 3])
        x = ops.reshape(x, [batch_size, depth, channel])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "qk_scale": self.qk_scale,
                "qkv_bias": self.qkv_bias,
                "attn_drop_rate": self.attn_drop_rate,
                "proj_drop_rate": self.proj_drop_rate,
            }
        )
        return config


class SwinTransformerBlock(keras.Model):
    def __init__(
        self,
        input_dim,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        activation="gelu",
        norm_layer=layers.LayerNormalization,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # variables
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.norm_layer = norm_layer
        self._activation_identifier = activation
        self.spatial_dims = len(window_size)

        for i, (shift, window) in enumerate(zip(self.shift_size, self.window_size)):
            if not (0 <= shift < window):
                raise ValueError(
                    f"shift_size[{i}] must be in the range 0 to less than "
                    f"window_size[{i}], but got shift_size[{i}]={shift} "
                    f"and window_size[{i}]={window}."
                )

    def build(self, input_shape):
        # determine if cyclic shift is needed
        self.apply_cyclic_shift = any(i > 0 for i in self.shift_size)

        # DropPath or Identity
        self.drop_path = (
            DropPath(self.drop_path_rate) if self.drop_path_rate > 0.0 else layers.Identity()
        )

        # Normalizations
        self.norm1 = self.norm_layer(axis=-1, epsilon=1e-5)
        self.norm1.build(input_shape)
        self.norm2 = self.norm_layer(axis=-1, epsilon=1e-5)
        self.norm2.build((*input_shape[:-1], self.input_dim))

        # Attention
        self.attn = SwinWindowAttention(
            self.input_dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop_rate=self.attn_drop_rate,
            proj_drop_rate=self.drop_rate,
        )
        self.attn.build((None, None, self.input_dim))

        # MLP
        self.mlp = SwinMLP(
            output_dim=self.input_dim,
            hidden_dim=self.mlp_hidden_dim,
            activation=self._activation_identifier,
            drop_rate=self.drop_rate,
        )
        self.mlp.build((*input_shape[:-1], self.input_dim))

        # compute padding for all spatial dims
        self.pads = [[0, 0]]  # batch dim
        for i, size in enumerate(self.window_size):
            dim = input_shape[i + 1]
            pad = ops.mod(-dim + size, size)
            self.pads.append([0, pad])
        self.pads.append([0, 0])  # channel dim

        cropping = [(0, int(p[1])) for p in self.pads[1:-1]]
        self.crop_layer = get_reshaping_layer(
            spatial_dims=len(self.window_size), layer_type="cropping", cropping=cropping
        )

        self.built = True

    def first_forward(self, x, mask_matrix, training):
        x = self.norm1(x)
        x = ops.pad(x, self.pads)
        spatial_shape_pad = [ops.shape(x)[i + 1] for i in range(self.spatial_dims)]

        # cyclic shift
        if self.apply_cyclic_shift:
            shifted_x = ops.roll(
                x, shift=[-s for s in self.shift_size], axis=list(range(1, self.spatial_dims + 1))
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask, training=training)

        # reverse windows
        batch_size = ops.shape(x)[0]
        x = window_reverse(attn_windows, self.window_size, batch_size, spatial_shape_pad)

        # reverse cyclic shift
        if self.apply_cyclic_shift:
            x = ops.roll(
                x, shift=[s for s in self.shift_size], axis=list(range(1, self.spatial_dims + 1))
            )

        x = self.crop_layer(x)
        return x

    def second_forward(self, x, training):
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x, training=training)
        return x

    def call(self, x, mask_matrix=None, training=None):
        shortcut = x
        x = self.first_forward(x, mask_matrix, training)
        x = shortcut + self.drop_path(x)
        x = x + self.second_forward(x, training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "window_size": self.num_heads,
                "num_heads": self.window_size,
                "shift_size": self.shift_size,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "mlp_hidden_dim": self.mlp_hidden_dim,
                "activation": self._activation_identifier,
            }
        )
        return config


class SwinBasicLayer(keras.Model):
    def __init__(
        self,
        input_dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        downsampling_layer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = tuple([i // 2 for i in window_size])
        self.depth = depth
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.downsampling_layer = downsampling_layer

    def _compute_dim_padded(self, input_dim, window_dim_size):
        return ops.cast(
            ops.ceil(ops.cast(input_dim, "float32") / ops.cast(window_dim_size, "float32"))
            * window_dim_size,
            "int32",
        )

    def build(self, input_shape):
        spatial_dims = len(input_shape) - 2
        self.blocks = [
            SwinTransformerBlock(
                self.input_dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=(0,) * spatial_dims if (i % 2 == 0) else self.shift_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=(
                    self.drop_path_rate[i]
                    if isinstance(self.drop_path_rate, list)
                    else self.drop_path_rate
                ),
                norm_layer=self.norm_layer,
            )
            for i in range(self.depth)
        ]

        if self.downsampling_layer:
            self.downsample = self.downsampling_layer(
                input_dim=self.input_dim, norm_layer=self.norm_layer
            )
            self.downsample.build(input_shape)

        for blk in self.blocks:
            blk.build(input_shape)

        # compute padded spatial dims
        spatial_shape = input_shape[1:-1]
        padded_shape = tuple(
            self._compute_dim_padded(d, w) for d, w in zip(spatial_shape, self.window_size)
        )

        # attention mask
        self.attn_mask = compute_mask(padded_shape, self.window_size, self.shift_size)

        self.built = True

    def call(self, x, training=None):
        batch_size = ops.shape(x)[0]
        spatial_shape = ops.shape(x)[1:-1]
        channel = ops.shape(x)[-1]

        for blk in self.blocks:
            x = blk(x, self.attn_mask, training=training)

        x = ops.reshape(x, [batch_size, *spatial_shape, channel])

        if self.downsampling_layer:
            x = self.downsample(x)

        return x

    def compute_output_shape(self, input_shape):
        if self.downsampling_layer is not None:
            output_shape = self.downsample.compute_output_shape(input_shape)
            return output_shape

        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "depth": self.depth,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "drop_path_rate": self.drop_path_rate,
            }
        )
        return config
