import keras
import numpy as np
from keras import ops

from medicai.layers import ViTMLP
from medicai.utils import get_conv_layer


class ViTPatchingAndEmbedding(keras.layers.Layer):
    """Patches the image (2D or 3D) and embeds the patches.

    Args:
        image_size: tuple. (h, w) for 2D or (d, h, w) for 3D.
        patch_size: tuple. Same length as image_size.
        hidden_dim: int. Dimensionality of the patch embeddings.
        num_channels: int. Number of channels in the input image. Defaults to 3.
        use_class_token: bool. Whether to prepend a CLS token. Defaults to True.
        use_patch_bias: bool. Whether Conv patch projection has bias. Defaults to True.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        hidden_dim,
        num_channels=3,
        use_class_token=True,
        use_patch_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.spatial_dims = len(image_size)  # 2 or 3
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * self.spatial_dims

        grid_size = tuple([s // p for s, p in zip(image_size, patch_size)])
        num_patches = np.prod(grid_size)
        num_positions = num_patches + 1 if use_class_token else num_patches

        # === Config ===
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.num_positions = num_positions
        self.use_class_token = use_class_token
        self.use_patch_bias = use_patch_bias

    def build(self, input_shape):
        if self.use_class_token:
            self.class_token = self.add_weight(
                shape=(1, 1, self.hidden_dim),
                initializer="random_normal",
                name="class_token",
            )

        self.patch_embedding = get_conv_layer(
            self.spatial_dims,
            layer_type="conv",
            filters=self.hidden_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            activation=None,
            use_bias=self.use_patch_bias,
            name="patch_embedding",
        )
        self.patch_embedding.build(input_shape)

        self.position_embedding = keras.layers.Embedding(
            self.num_positions,
            self.hidden_dim,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),
            name="position_embedding",
        )
        self.position_embedding.build((1, self.num_positions))

        self.position_ids = keras.ops.expand_dims(keras.ops.arange(self.num_positions), axis=0)
        self.built = True

    def call(self, inputs):
        # Project patches
        patch_embeddings = self.patch_embedding(inputs)
        embeddings_shape = ops.shape(patch_embeddings)

        # Flatten spatial dimensions → sequence of patches
        patch_embeddings = ops.reshape(
            patch_embeddings, [embeddings_shape[0], -1, embeddings_shape[-1]]
        )

        # Add CLS token if enabled
        if self.use_class_token:
            class_token = ops.tile(self.class_token, (embeddings_shape[0], 1, 1))
            patch_embeddings = ops.concatenate([class_token, patch_embeddings], axis=1)

        # Add positional embeddings
        position_embeddings = self.position_embedding(self.position_ids)
        return ops.add(patch_embeddings, position_embeddings)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.num_positions,
            self.hidden_dim,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "num_channels": self.num_channels,
                "num_patches": self.num_patches,
                "num_positions": self.num_positions,
                "use_class_token": self.use_class_token,
                "use_patch_bias": self.use_patch_bias,
            }
        )
        return config


class ViTEncoderBlock(keras.layers.Layer):
    """Transformer encoder block.

    Args:
        num_heads: int. Number of attention heads.
        hidden_dim: int. Dimensionality of the hidden representations.
        mlp_dim: int. Dimensionality of the intermediate MLP layer.
        use_mha_bias: bool. Whether to use bias in the multi-head attention
            layer. Defaults to `True`.
        use_mlp_bias: bool. Whether to use bias in the MLP layer. Defaults to
            `True`.
        dropout_rate: float. Dropout rate. Between 0 and 1. Defaults to `0.0`.
        attention_dropout: float. Dropout rate for the attention mechanism.
            Between 0 and 1. Defaults to `0.0`.
        layer_norm_epsilon: float. Small float value for layer normalization
            stability. Defaults to `1e-6`.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_dim,
        use_mha_bias=True,
        use_mlp_bias=True,
        dropout_rate=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        key_dim = hidden_dim // num_heads

        # === Config ===
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.mlp_dim = mlp_dim
        self.use_mha_bias = use_mha_bias
        self.use_mlp_bias = use_mlp_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        # Attention block
        self.layer_norm_1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="ln_1",
        )
        self.layer_norm_1.build(input_shape)
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            use_bias=self.use_mha_bias,
            dropout=self.attention_dropout,
            name="mha",
        )
        self.mha.build(input_shape, input_shape)
        self.dropout = keras.layers.Dropout(self.dropout_rate, name="dropout")

        # MLP block
        self.layer_norm_2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="ln_2",
        )
        self.layer_norm_2.build((None, None, self.hidden_dim))
        self.mlp = ViTMLP(
            self.mlp_dim,
            output_dim=self.hidden_dim,
            drop_rate=self.dropout_rate,
            name="mlp",
        )
        self.mlp.build((None, None, self.hidden_dim))
        self.built = True

    def call(self, inputs):
        x = self.layer_norm_1(inputs)
        x = self.mha(x, x)
        x = self.dropout(x)
        x = x + inputs

        y = self.layer_norm_2(x)
        y = self.mlp(y)
        return x + y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "key_dim": self.key_dim,
                "mlp_dim": self.mlp_dim,
                "use_mha_bias": self.use_mha_bias,
                "use_mlp_bias": self.use_mlp_bias,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
