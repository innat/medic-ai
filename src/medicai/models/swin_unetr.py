import keras
import numpy as np
from keras import ops
from functools import partial

import keras
import numpy as np
from keras import layers

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
    windows = ops.reshape(
        x, [-1, window_size[0] * window_size[1] * window_size[2], channel]
    )

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
    attn_mask = ops.expand_dims(mask_windows, axis=1) - ops.expand_dims(
        mask_windows, axis=2
    )
    attn_mask = ops.where(attn_mask != 0, -100.0, attn_mask)
    attn_mask = ops.where(attn_mask == 0, 0.0, attn_mask)
    return attn_mask


from keras import layers


class MLP(layers.Layer):
    def __init__(
        self, hidden_dim, output_dim, drop_rate=0.0, activation="gelu", **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self._activation_identifier = activation
        self.drop_rate = drop_rate
        self.activation = layers.Activation(self._activation_identifier)
        self.fc1 = layers.Dense(self.hidden_dim)
        self.fc2 = layers.Dense(self.output_dim)
        self.dropout = layers.Dropout(self.drop_rate)

    def build(self, input_shape):
        self.fc1.build(input_shape)
        self.fc2.build((*input_shape[:-1], self.hidden_dim))
        self.built = True

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "hidden_dim": self.hidden_dim,
                "drop_rate": self.drop_rate,
                "activation": self._activation_identifier,
            }
        )
        return config


import keras
from keras import layers, ops


class VideoSwinPatchingAndEmbedding(keras.Model):
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


from keras import layers, ops

class VideoSwinPatchMerging(layers.Layer):
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
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
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


import keras
from keras import layers, ops


class DropPath(layers.Layer):
    def __init__(self, rate=0.5, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self._seed_val = seed
        self.seed = keras.random.SeedGenerator(seed)

    def call(self, x, training=None):
        if self.rate == 0.0 or not training:
            return x
        else:
            batch_size = x.shape[0] or ops.shape(x)[0]
            drop_map_shape = (batch_size,) + (1,) * (len(x.shape) - 1)
            drop_map = ops.cast(
                keras.random.uniform(drop_map_shape, seed=self.seed) > self.rate,
                x.dtype,
            )
            x = x / (1.0 - self.rate)
            x = x * drop_map
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate, "seed": self._seed_val})
        return config


import keras
from keras import initializers
from keras import layers, ops

class VideoSwinWindowAttention(keras.Model):

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
        self.scale = head_dim**-0.5 #qk_scale or head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate

    def get_relative_position_index(self, window_depth, window_height, window_width):
        y_y, z_z, x_x = ops.meshgrid(
            ops.arange(window_width),
            ops.arange(window_depth),
            ops.arange(window_height),
            indexing="ij"
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


class VideoSwinTransformerBlock(keras.Model):
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
            DropPath(self.drop_path_rate)
            if self.drop_path_rate > 0.0
            else layers.Identity()
        )

        self.norm1 = self.norm_layer(axis=-1, epsilon=1e-05)
        self.norm1.build(input_shape)

        self.attn = VideoSwinWindowAttention(
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

        self.mlp = MLP(
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

        # pad if required
        do_pad = ops.logical_or(
            ops.greater(self.pad_d1, 0),
            ops.logical_or(ops.greater(self.pad_r, 0), ops.greater(self.pad_b, 0)),
        )
        x = ops.cond(do_pad, lambda: x[:, :depth, :height, :width, :], lambda: x)

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


class VideoSwinBasicLayer(keras.Model):
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
        return ops.cast(
            ops.ceil(input_dim / window_dim_size) * window_dim_size, dtype="int32"
        )

    def build(self, input_shape):
        # build blocks
        self.blocks = [
            VideoSwinTransformerBlock(
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

def parse_model_inputs(input_shape, input_tensor, **kwargs):
    if input_tensor is None:
        return keras.layers.Input(shape=input_shape, **kwargs)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return keras.layers.Input(tensor=input_tensor, shape=input_shape, **kwargs)
        else:
            return input_tensor

class VideoSwinBackboneV2(keras.Model):
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
        # Parse input specification.
        input_spec = parse_model_inputs(input_shape, input_tensor, name="videos")

        # Check that the input video is well specified.
        if (
            input_spec.shape[-4] is None
            or input_spec.shape[-3] is None
            or input_spec.shape[-2] is None
        ):
            raise ValueError(
                "Depth, height and width of the video must be specified"
                " in `input_shape`."
            )

        x = input_spec

        norm_layer = partial(layers.LayerNormalization, epsilon=1e-05)

        x = VideoSwinPatchingAndEmbedding(
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            name="patching_and_embedding",
        )(x)
        x = layers.Dropout(drop_rate, name="pos_drop")(x)

        dpr = np.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        num_layers = len(depths)
        for i in range(num_layers):
            layer = VideoSwinBasicLayer(
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
                downsampling_layer=VideoSwinPatchMerging,
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


import numpy as np
import keras
from keras import ops
from keras import layers, Model
from keras.layers import Conv3D, Conv3DTranspose, BatchNormalization, Activation, Add, Concatenate

def get_act_layer(act_name):
    if act_name[0] == "leakyrelu":
        return layers.LeakyReLU(negative_slope=act_name[1]["negative_slope"])
    else:
        return layers.Activation(act_name[0])

def get_norm_layer(norm_name):
    if norm_name == "instance":
        return layers.GroupNormalization(
            groups=-1, epsilon=1e-05, scale=False, center=False
        )
        
    elif norm_name == "batch":
        return layers.BatchNormalization()
    else:
        raise ValueError(f"Unsupported normalization: {norm_name}")


def UnetBasicBlock(out_channels, kernel_size=3, stride=1, norm_name="instance", dropout_rate=None):
    def wrapper(inputs):
        x = layers.Conv3D(out_channels, kernel_size, strides=stride, use_bias=False)(inputs)
        x = get_norm_layer(norm_name)(x)
        x = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))(x)
        
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Conv3D(out_channels, kernel_size, strides=1, use_bias=False)(x)
        x = get_norm_layer(norm_name)(x)
        x = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))(x)
        
        return x
    return wrapper


def UnetResBlock(in_channels, out_channels, kernel_size=3, stride=1, norm_name="instance", dropout_rate=None):
    def wrapper(inputs):
        # first convolution
        x = layers.Conv3D(out_channels, kernel_size, strides=stride, padding='same', use_bias=False)(inputs)
        x = get_norm_layer(norm_name)(x)
        x = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))(x)
        
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
        
        # second convolution
        x = layers.Conv3D(out_channels, kernel_size, strides=1, padding='same', use_bias=False)(x)
        x = get_norm_layer(norm_name)(x)
        
        # residual
        residual = inputs
        downsample = (in_channels != out_channels) or (np.atleast_1d(stride) != 1).any()
        if downsample:
            residual = layers.Conv3D(
                out_channels, kernel_size=1, strides=stride, padding='same', use_bias=False
            )(residual)
            residual = get_norm_layer(norm_name)(residual)
        
        # add residual connection
        x = layers.Add()([x, residual])
        x = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))(x)
        
        return x
    return wrapper

def UnetrBasicBlock(out_channels, kernel_size=3, stride=1, norm_name="instance", res_block=True):
    def wrapper(inputs):
        if res_block:
            x = UnetResBlock(
                in_channels=inputs.shape[-1], 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                norm_name=norm_name, 
            )(inputs)
        else:
            x = UnetBasicBlock(
                out_channels, 
                kernel_size, 
                stride,
                norm_name, 
            )(inputs)
        return x
    return wrapper


def UnetrUpBlock(
    out_channels, kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance", res_block=True
):
    def wrapper(inputs, skip):
        x = layers.Conv3DTranspose(
            out_channels,
            kernel_size=upsample_kernel_size,
            strides=upsample_kernel_size,
            use_bias=False
        )(inputs)

        # Concatenate with skip connection
        x = layers.Concatenate(axis=-1)([x, skip])

        # Apply the convolutional block
        if res_block:
            x = UnetResBlock(
                in_channels=x.shape[-1],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )(x)
        else:
            x = UnetBasicBlock(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )(x)
        return x
    return wrapper

def UnetOutBlock(out_channels, dropout_rate=None):
    def wrapper(inputs):
        x = layers.Conv3D(out_channels, kernel_size=1, strides=1, use_bias=True)(inputs)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
        return x
    return wrapper


def UnetrHead(
    out_channels=4, feature_size=16, res_block=True, norm_name="instance"
):
    def wrapper(inputs):
        enc_input = inputs[0]
        hidden_states_out = inputs[1:]

        # Encoder 1 (process raw input)
        enc0 = UnetrBasicBlock(
            feature_size, 
            kernel_size=3, 
            stride=1, 
            norm_name=norm_name, 
            res_block=res_block
        )(enc_input)

        # Encoder 2 (process hidden_states_out[0])
        enc1 = UnetrBasicBlock(
            feature_size, 
            kernel_size=3, 
            stride=1, 
            norm_name=norm_name, 
            res_block=res_block
        )(hidden_states_out[0])

        # Encoder 3 (process hidden_states_out[1])
        enc2 = UnetrBasicBlock(
            feature_size * 2, 
            kernel_size=3, 
            stride=1, 
            norm_name=norm_name, 
            res_block=res_block
        )(hidden_states_out[1])

        # Encoder 4 (process hidden_states_out[2])
        enc3 = UnetrBasicBlock(
            feature_size * 4, 
            kernel_size=3, 
            stride=1, 
            norm_name=norm_name, 
            res_block=res_block
        )(hidden_states_out[2])

        # Encoder 5 (process hidden_states_out[4] as bottleneck)
        dec4 = UnetrBasicBlock(
            feature_size * 16, 
            kernel_size=3, 
            stride=1, 
            norm_name=norm_name, 
            res_block=res_block
        )(hidden_states_out[4])

        # Decoder 5 (upsample dec4 and concatenate with hidden_states_out[3])
        dec3 = UnetrUpBlock(
            feature_size * 8, 
            kernel_size=3, 
            upsample_kernel_size=2,
            norm_name=norm_name, 
            res_block=res_block
        )(dec4, hidden_states_out[3])

        # Decoder 4 (upsample dec3 and concatenate with enc3)
        dec2 = UnetrUpBlock(
            feature_size * 4, 
            kernel_size=3, 
            upsample_kernel_size=2,
            norm_name=norm_name, 
            res_block=res_block
        )(dec3, enc3)

        # Decoder 3 (upsample dec2 and concatenate with enc2)
        dec1 = UnetrUpBlock(
            feature_size * 2, 
            kernel_size=3, 
            upsample_kernel_size=2,
            norm_name=norm_name, 
            res_block=res_block
        )(dec2, enc2)

        # Decoder 2 (upsample dec1 and concatenate with enc1)
        dec0 = UnetrUpBlock(
            feature_size, 
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name, 
            res_block=res_block
        )(dec1, enc1)

        out = UnetrUpBlock(
            feature_size, 
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name, 
            res_block=res_block
        )(dec0, enc0)

        # Final output (process dec0 and produce logits)
        logits = UnetOutBlock(out_channels)(out)
        return logits
    return wrapper



class SwinUNETR(keras.Model):
    def __init__(
        self,
        *,
        input_shape=(96,96,96,1),
        out_channels=4, 
        feature_size=48, 
        res_block=True, 
        norm_name="instance",
        **kwargs
    ):
        encoder = VideoSwinBackbone(
            input_shape=input_shape,
            patch_size=[2, 2, 2],
            depths=[2, 2, 2, 2],
            window_size=[7, 7, 7],
            num_heads=[3, 6, 12, 24],
            embed_dim=48,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            patch_norm=False
        )
        inputs = encoder.input
        skips = [
            encoder.get_layer("patching_and_embedding").output,
            encoder.get_layer("swin_feature1").output,
            encoder.get_layer("swin_feature2").output,
            encoder.get_layer("swin_feature3").output,
            encoder.get_layer("swin_feature4").output,
        ]
        unetr_head = UnetrHead(
            out_channels=out_channels,
            feature_size=feature_size, 
            res_block=True, 
            norm_name=norm_name, 
        )
        
        # Combine encoder and decoder
        outputs = unetr_head([inputs] + skips)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.out_channels = out_channels
        self.feature_size = feature_size
        self.res_block = res_block
        self.norm_name = norm_name

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "out_channels": self.out_channels,
            "feature_size": self.feature_size,
            "res_block": self.res_block,
            "norm_name": self.norm_name
        }
        return config
