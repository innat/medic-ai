from keras import layers
from keras.initializers import HeNormal

from medicai.layers import TransUNetMLP


class MaskedCrossAttention(layers.Layer):
    """Masked cross-attention layer for transformer decoders.

    Performs cross-attention between queries and key-value pairs with optional masking.
    Includes residual connection, dropout, and layer normalization.

    Args:
        key_dim: Dimensionality of the key projections
        num_heads: Number of attention heads
        dropout_rate: Dropout rate for attention outputs (default: 0.1)

    Inputs:
        queries: Query tensor of shape (batch_size, target_len, embed_dim)
        keys: Key tensor of shape (batch_size, source_len, embed_dim)
        values: Value tensor of shape (batch_size, source_len, embed_dim)
        mask: Optional attention mask to prevent attention to certain positions
        training: Boolean for training mode

    Outputs:
        Tensor of shape (batch_size, target_len, embed_dim) with attended features
    """

    def __init__(self, num_heads, key_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            kernel_initializer=HeNormal(),
            dropout=self.dropout_rate,
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, query, key, value, mask=None, training=False):
        attn_output = self.attention(
            query=query, key=key, value=value, attention_mask=mask, training=training
        )
        attn_output = self.dropout(attn_output, training=training)
        output = self.layernorm(query + attn_output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class TransUNetDecoderBlock(layers.Layer):
    """Transformer decoder block with masked self-attention and cross-attention.

    Implements a single decoder block with:
    1. Masked self-attention (autoregressive)
    2. Cross-attention to encoder features
    3. MLP for feature transformation
    4. Residual connections and layer normalization

    Args:
        embed_dim: Dimensionality of input and output representations
        num_heads: Number of attention heads
        mlp_dim: Hidden dimension of the MLP (typically 4× embed_dim)
        dropout_rate: Dropout rate for attention and MLP (default: 0.1)

    Inputs:
        queries: Decoder input of shape (batch_size, num_queries, embed_dim)
        encoder_output: Encoder features of shape (batch_size, seq_len, embed_dim)
        attention_mask: Optional mask for autoregressive generation
        training: Boolean for training mode

    Outputs:
        Tensor of shape (batch_size, num_queries, embed_dim) with refined features
    """

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # input_shape will be a list: [query_shape, encoder_output_shape]
        query_shape, _ = input_shape

        # Masked self-attention
        self.masked_self_att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate,
            kernel_initializer=HeNormal(),
        )
        self.masked_self_att.build(query_shape, query_shape)

        # Use MaskedCrossAttention
        self.masked_cross_att = MaskedCrossAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        self.masked_cross_att.build(query_shape)

        # MLP
        self.mlp_layer = TransUNetMLP(
            self.mlp_dim,
            activation="gelu",
            output_dim=self.embed_dim,
            drop_rate=self.dropout_rate,
            kernel_initializer=HeNormal(),
            name="mlp_decode",
        )
        self.mlp_layer.build(query_shape)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.built = True

    def call(self, inputs, attention_mask=None, training=None):
        # Unpack inputs: [queries, encoder_output]
        queries, encoder_output = inputs

        # Masked self-attention
        p1 = self.masked_self_att(queries, queries, use_causal_mask=True, training=training)
        p1 = self.layernorm1(queries + p1)

        # Masked cross-attention
        p2 = self.masked_cross_att(
            p1, key=encoder_output, value=encoder_output, mask=attention_mask, training=training
        )

        # MLP
        p3 = self.mlp_layer(p2)
        output = self.layernorm2(p2 + p3)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Return shape of queries

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class CoarseToFineAttention(layers.Layer):
    """Coarse-to-fine attention refinement block for TransUNet.

    Refines transformer features by attending to CNN encoder features at multiple scales.
    Implements the Z-block mechanism for progressive feature refinement from deep to shallow features.

    Args:
        embed_dim: Dimensionality of input and output representations
        num_heads: Number of attention heads
        mlp_dim: Hidden dimension of the MLP
        dropout_rate: Dropout rate for attention and MLP (default: 0.1)

    Inputs:
        query: Transformer features of shape (batch_size, num_queries, embed_dim)
        features: CNN encoder features of shape (batch_size, spatial_positions, channels)
        mask: Optional attention mask
        training: Boolean for training mode

    Outputs:
        Tensor of shape (batch_size, num_queries, embed_dim) with refined features
    """

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        query_shape, _ = input_shape
        self.cross_attention = MaskedCrossAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        self.cross_attention.build(query_shape)
        self.mlp_layer = TransUNetMLP(
            self.mlp_dim,
            activation="gelu",
            output_dim=self.embed_dim,
            drop_rate=self.dropout_rate,
            name="mlp",
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None):
        # # Unpack the inputs: [query, features]
        query, features = inputs

        # Cross-attention, with query coming from transformer decoder and features from CNN encoder
        attn_output = self.cross_attention(query, key=features, value=features, training=training)

        # MLP for further refinement
        mlp_output = self.mlp_layer(attn_output)
        output = self.layernorm(attn_output + mlp_output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
