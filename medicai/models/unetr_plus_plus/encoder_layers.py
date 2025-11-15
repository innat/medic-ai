import keras
import numpy as np
from keras import layers, ops

from medicai.blocks import UNetResBlock
from medicai.layers import EfficientPairedAttention
from medicai.utils import get_conv_layer, get_dropout_layer


class UNETRPlusPlusTransformer(keras.layers.Layer):
    """
    A UNETR++ transformer block with Efficient Paired Attention (EPA).

    This layer implements a single stage of the UNETR++ transformer encoder.
    It combines a normalization layer, Efficient Paired Attention (EPA), optional
    positional embeddings, and residual convolutional blocks. EPA splits the
    attention into channel and spatial components and reduces the spatial token
    dimension for efficient computation on high-resolution 2D or 3D inputs.

    The transformer block maintains skip connections and residual convolutional
    refinement, producing a feature map of the same spatial dimensions as the
    input but with refined channel-wise representations.
    """

    def __init__(
        self,
        sequence_length,
        hidden_size,
        spatial_reduced_tokens,
        num_heads,
        dropout_rate=0.1,
        pos_embed=False,
        name="unetrpp_transformer",
        **kwargs,
    ):
        """
        Initializes a UNETRPlusPlusTransformer block.

        Args:
            sequence_length: Integer, flattened spatial token length for the
                transformer block.
            hidden_size: Integer, number of channels for the hidden features.
            spatial_reduced_tokens: Integer, number of spatial tokens after
                reduction inside EPA.
            num_heads: Integer, number of attention heads for multi-head
                attention.
            dropout_rate: Float, dropout probability for attention and spatial
                dropout layers.
            pos_embed: Boolean, whether to apply learnable positional embeddings.
            name: Optional string, name of the layer.
            **kwargs: Additional keyword arguments passed to the parent Layer.
        """
        super().__init__(name=name, **kwargs)

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.spatial_reduced_tokens = spatial_reduced_tokens
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.pos_embed = pos_embed
        self.prefix = name

    def build(self, input_shape):
        self.spatial_dims = len(input_shape) - 2
        self.total_spatial = np.prod(input_shape[1:-1])

        self.norm = layers.LayerNormalization(epsilon=1e-5, name=f"{self.prefix}_norm")
        self.norm.build((None, self.sequence_length, self.hidden_size))

        self.gamma = self.add_weight(
            shape=(self.hidden_size,),
            initializer=keras.initializers.Constant(1e-6),
            trainable=True,
            name="gamma",
        )
        self.epa_block = EfficientPairedAttention(
            sequence_length=self.sequence_length,
            hidden_size=self.hidden_size,
            spatial_reduced_tokens=self.spatial_reduced_tokens,
            num_heads=self.num_heads,
            channel_attn_drop=self.dropout_rate,
            spatial_attn_drop=self.dropout_rate,
        )
        self.epa_block.build((None, self.sequence_length, self.hidden_size))

        if self.pos_embed:
            self.pos_embed_layer = self.add_weight(
                shape=(1, self.sequence_length, self.hidden_size),
                initializer="zeros",
                trainable=True,
                name="pos_embed",
            )

        self.conv1 = UNetResBlock(
            filters=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name="batch",
            name=f"{self.prefix}_unet_residual_block",
        )
        self.conv1.build(input_shape)

        self.conv2 = keras.Sequential(
            [
                get_dropout_layer(
                    self.spatial_dims,
                    layer_type="spatial_dropout",
                    rate=self.dropout_rate,
                    name=f"{self.prefix}_spatial_dropout",
                ),
                get_conv_layer(
                    self.spatial_dims,
                    layer_type="conv",
                    filters=self.hidden_size,
                    kernel_size=1,
                    name=f"{self.prefix}_conv2",
                ),
            ]
        )
        spatial_input_shape = input_shape[:-1] + (self.hidden_size,)
        self.conv2.build(spatial_input_shape)
        self.built = True

    def call(self, x, training=False):
        input_shape = ops.shape(x)
        batch_size = input_shape[0]
        channel_size = input_shape[-1]
        x = ops.reshape(x, [batch_size, self.total_spatial, channel_size])

        # Apply positional embedding
        if self.pos_embed:
            x = x + self.pos_embed_layer

        # Apply transformer block
        norm_x = self.norm(x)
        attn_output = self.epa_block(norm_x, training=training)
        attn = x + self.gamma * attn_output

        # Reshape back to spatial dimensions
        attn_skip = ops.reshape(attn, input_shape)

        # Apply convolution blocks
        attn_conv = self.conv1(attn_skip)
        x_output = attn_skip + self.conv2(attn_conv, training=training)

        return x_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "spatial_reduced_tokens": self.spatial_reduced_tokens,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "pos_embed": self.pos_embed,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.hidden_size,)
