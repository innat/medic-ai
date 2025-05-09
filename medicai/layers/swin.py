from medicai.utils import hide_warnings

hide_warnings()

import keras
import numpy as np
from keras import initializers, layers, ops

from .drop_path import DropPath
from .mlp import SwinMLP


def window_partition(x, window_size):
    input_shape = ops.shape(x)
    batch_size, depth, height, width, channel = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        input_shape[4],
    )

    x = ops.reshape(
        x,
        [
            batch_size,
            depth // window_size[0],
            window_size[0],
            height // window_size[1],
            window_size[1],
            width // window_size[2],
            window_size[2],
            channel,
        ],
    )

    x = ops.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
    windows = ops.reshape(x, [-1, window_size[0] * window_size[1] * window_size[2], channel])

    return windows


def window_reverse(windows, window_size, batch_size, depth, height, width):
    x = ops.reshape(
        windows,
        [
            batch_size,
            depth // window_size[0],
            height // window_size[1],
            width // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        ],
    )
    x = ops.transpose(x, [0, 1, 4, 2, 5, 3, 6, 7])
    x = ops.reshape(x, [batch_size, depth, height, width, -1])
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


def compute_mask(depth, height, width, window_size, shift_size):
    img_mask = np.zeros((1, depth, height, width, 1))
    cnt = 0
    for d in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d, h, w, :] = cnt
                cnt = cnt + 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = ops.squeeze(mask_windows, axis=-1)
    attn_mask = ops.expand_dims(mask_windows, axis=1) - ops.expand_dims(mask_windows, axis=2)
    attn_mask = ops.where(attn_mask != 0, -100.0, attn_mask)
    attn_mask = ops.where(attn_mask == 0, 0.0, attn_mask)
    return attn_mask


class SwinPatchingAndEmbedding(keras.Model):
    def __init__(self, patch_size=(2, 4, 4), embed_dim=96, norm_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer

    def _compute_padding(self, dim, patch_size):
        pad_amount = patch_size - (dim % patch_size)
        return [0, pad_amount if pad_amount != patch_size else 0]

    def build(self, input_shape):
        self.pads = [
            [0, 0],
            self._compute_padding(input_shape[1], self.patch_size[0]),
            self._compute_padding(input_shape[2], self.patch_size[1]),
            self._compute_padding(input_shape[3], self.patch_size[2]),
            [0, 0],
        ]

        if self.norm_layer is not None:
            self.norm = self.norm_layer(axis=-1, epsilon=1e-5, name="embed_norm")
            self.norm.build((None, None, None, None, self.embed_dim))

        self.proj = layers.Conv3D(
            self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            name="embed_proj",
        )
        self.proj.build((None, None, None, None, input_shape[-1]))
        self.built = True

    def call(self, x):
        x = ops.pad(x, self.pads)
        x = self.proj(x)

        if self.norm_layer is not None:
            x = self.norm(x)

        return x
    
    def compute_output_shape(self, input_shape):
        batch_size, depth, height, width, _ = input_shape
 
        # Compute padding
        pad_d = self._compute_padding(depth, self.patch_size[0])[1]
        pad_h = self._compute_padding(height, self.patch_size[1])[1]
        pad_w = self._compute_padding(width, self.patch_size[2])[1]

        # Add padding
        d_padded = depth + pad_d
        h_padded = height + pad_h
        w_padded = width + pad_w

        # Output shape after Conv3D with stride = patch_size
        d_out = d_padded // self.patch_size[0]
        h_out = h_padded // self.patch_size[1]
        w_out = w_padded // self.patch_size[2]

        return (batch_size, d_out, h_out, w_out, self.embed_dim)


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
        batch_size, depth, height, width, channel = input_shape
        self.reduction = layers.Dense(2 * self.input_dim, use_bias=False)
        self.reduction.build((batch_size, depth // 2, height // 2, width // 2, 8 * channel))

        if self.norm_layer is not None:
            self.norm = self.norm_layer(axis=-1, epsilon=1e-5)
            self.norm.build((batch_size, depth // 2, height // 2, width // 2, 8 * channel))

        # compute padding if needed
        self.pads = [
            [0, 0],
            [0, ops.mod(depth, 2)],
            [0, ops.mod(height, 2)],
            [0, ops.mod(width, 2)],
            [0, 0],
        ]

        self.built = True

    def call(self, x):
        # padding if needed
        x = ops.pad(x, self.pads)
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = ops.concatenate([x0, x1, x2, x3, x4, x5, x6, x7], axis=-1)
        if self.norm_layer is not None:
            x = self.norm(x)
        x = self.reduction(x)
        return x

    def compute_output_shape(self, input_shape):
        batch_size, depth, height, width, _ = input_shape
        return (batch_size, depth // 2, height // 2, width // 2, 2 * self.input_dim)

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

    def get_relative_position_index(self, window_depth, window_height, window_width):
        y_y, z_z, x_x = ops.meshgrid(
            ops.arange(window_width),
            ops.arange(window_depth),
            ops.arange(window_height),
            indexing="ij",
        )
        coords = ops.stack([z_z, y_y, x_x], axis=0)
        coords_flatten = ops.reshape(coords, [3, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = ops.transpose(relative_coords, axes=[1, 2, 0])
        z_z = (
            (relative_coords[:, :, 0] + window_depth - 1)
            * (2 * window_height - 1)
            * (2 * window_width - 1)
        )
        x_x = (relative_coords[:, :, 1] + window_height - 1) * (2 * window_width - 1)
        y_y = relative_coords[:, :, 2] + window_width - 1
        relative_coords = ops.stack([z_z, x_x, y_y], axis=-1)
        return ops.sum(relative_coords, axis=-1)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=(
                (2 * self.window_size[0] - 1)
                * (2 * self.window_size[1] - 1)
                * (2 * self.window_size[2] - 1),
                self.num_heads,
            ),
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name="relative_position_bias_table",
        )
        self.relative_position_index = self.get_relative_position_index(
            self.window_size[0], self.window_size[1], self.window_size[2]
        )

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
        window_size=(2, 7, 7),
        shift_size=(0, 0, 0),
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

        for i, (shift, window) in enumerate(zip(self.shift_size, self.window_size)):
            if not (0 <= shift < window):
                raise ValueError(
                    f"shift_size[{i}] must be in the range 0 to less than "
                    f"window_size[{i}], but got shift_size[{i}]={shift} "
                    f"and window_size[{i}]={window}."
                )

    def build(self, input_shape):

        self.apply_cyclic_shift = False
        if any(i > 0 for i in self.shift_size):
            self.apply_cyclic_shift = True

        # layers
        self.drop_path = (
            DropPath(self.drop_path_rate) if self.drop_path_rate > 0.0 else layers.Identity()
        )

        self.norm1 = self.norm_layer(axis=-1, epsilon=1e-05)
        self.norm1.build(input_shape)

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

        self.norm2 = self.norm_layer(axis=-1, epsilon=1e-05)
        self.norm2.build((*input_shape[:-1], self.input_dim))

        self.mlp = SwinMLP(
            output_dim=self.input_dim,
            hidden_dim=self.mlp_hidden_dim,
            activation=self._activation_identifier,
            drop_rate=self.drop_rate,
        )
        self.mlp.build((*input_shape[:-1], self.input_dim))

        # compute padding if needed.
        # pad input feature maps to multiples of window size.
        self.window_size, self.shift_size = get_window_size(
            input_shape[1:-1], self.window_size, self.shift_size
        )
        _, depth, height, width, _ = input_shape
        pad_l = pad_t = pad_d0 = 0
        self.pad_d1 = ops.mod(-depth + self.window_size[0], self.window_size[0])
        self.pad_b = ops.mod(-height + self.window_size[1], self.window_size[1])
        self.pad_r = ops.mod(-width + self.window_size[2], self.window_size[2])
        self.pads = [
            [0, 0],
            [pad_d0, self.pad_d1],
            [pad_t, self.pad_b],
            [pad_l, self.pad_r],
            [0, 0],
        ]
        self.built = True

    def first_forward(self, x, mask_matrix, training):
        input_shape = ops.shape(x)
        batch_size, depth, height, width, _ = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        )
        x = self.norm1(x)

        # apply padding if needed.
        x = ops.pad(x, self.pads)

        input_shape = ops.shape(x)
        depth_pad, height_pad, width_pad = (
            input_shape[1],
            input_shape[2],
            input_shape[3],
        )

        # cyclic shift
        if self.apply_cyclic_shift:
            shifted_x = ops.roll(
                x,
                shift=(
                    -self.shift_size[0],
                    -self.shift_size[1],
                    -self.shift_size[2],
                ),
                axis=(1, 2, 3),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)

        # get attentions params
        attn_windows = self.attn(x_windows, mask=attn_mask, training=training)

        # reverse the swin windows
        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            batch_size,
            depth_pad,
            height_pad,
            width_pad,
        )

        # reverse cyclic shift
        if self.apply_cyclic_shift:
            x = ops.roll(
                shifted_x,
                shift=(
                    self.shift_size[0],
                    self.shift_size[1],
                    self.shift_size[2],
                ),
                axis=(1, 2, 3),
            )
        else:
            x = shifted_x

        return x[:, :depth, :height, :width, :]

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
        window_size=(1, 7, 7),
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
        input_dim = ops.cast(input_dim, dtype="float32")
        window_dim_size = ops.cast(window_dim_size, dtype="float32")
        return ops.cast(ops.ceil(input_dim / window_dim_size) * window_dim_size, dtype="int32")

    def build(self, input_shape):
        # build blocks
        self.blocks = [
            SwinTransformerBlock(
                self.input_dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
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

        if self.downsampling_layer is not None:
            self.downsample = self.downsampling_layer(
                input_dim=self.input_dim, norm_layer=self.norm_layer
            )
            self.downsample.build(input_shape)

        for i in range(self.depth):
            self.blocks[i].build(input_shape)

        self.window_size, self.shift_size = get_window_size(
            input_shape[1:-1], self.window_size, self.shift_size
        )
        depth_pad = self._compute_dim_padded(input_shape[1], self.window_size[0])
        height_pad = self._compute_dim_padded(input_shape[2], self.window_size[1])
        width_pad = self._compute_dim_padded(input_shape[3], self.window_size[2])
        self.attn_mask = compute_mask(
            depth_pad,
            height_pad,
            width_pad,
            self.window_size,
            self.shift_size,
        )

        self.built = True

    def call(self, x, training=None):
        input_shape = ops.shape(x)
        batch_size, depth, height, width, channel = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        )

        for block in self.blocks:
            x = block(x, self.attn_mask, training=training)

        x = ops.reshape(x, [batch_size, depth, height, width, channel])

        if self.downsampling_layer is not None:
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
