import keras
from keras import layers, ops
from keras.initializers import HeNormal

from medicai.layers import TransUNetMLP


class MaskedCrossAttention(layers.Layer):
    """Masked cross-attention layer for transformer decoders.

    This layer is a wrapper around `keras.layers.MultiHeadAttention` to perform
    cross-attention between a query and a set of key-value pairs.

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

        if len(input_shape) == 2:
            query_shape, key_value_shape = input_shape
            self.attention.build(query_shape, key_value_shape, key_value_shape)
        elif len(input_shape) == 3:
            query_shape, key_shape, value_shape = input_shape
            self.attention.build(query_shape, key_shape, value_shape)
        else:
            raise ValueError(
                f"MaskedCrossAttention layer expects list of 2 or 3 input shapes, but received {len(input_shape)}."
            )

        super().build(input_shape)

    def call(self, inputs, training=False):
        query, key, value = inputs
        output = self.attention(
            query=query, key=key, value=value, attention_mask=None, training=training
        )
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Return shape of queries

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


class QueryRefinementBlock(layers.Layer):
    """Transformer block for refining learnable queries using encoder output.

    This is the P-block from the paper's decoder. It refines a set of learnable
    query tokens by applying cross-attention with features from the Vision Transformer
    encoder. This is followed by a feed-forward network (MLP) and residual connections.

    Args:
        embed_dim: Dimensionality of input and output token representations.
        num_heads: Number of attention heads.
        mlp_dim: Hidden dimension of the MLP layer.
        dropout_rate: Dropout rate for attention and MLP (default: 0.1).

    Inputs:
        A list of two tensors:
        - queries: The learnable query tokens of shape `(batch_size, num_queries, embed_dim)`.
        - encoder_output: The output from the Vision Transformer encoder,
          of shape `(batch_size, num_patches, embed_dim)`.

    Outputs:
        A tensor of the same shape as the input queries, with refined features.
    """

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        query_shape, encoder_output_shape = input_shape
        self.cross_attention = MaskedCrossAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        self.cross_attention.build([query_shape, encoder_output_shape])
        self.mlp_layer = TransUNetMLP(
            self.mlp_dim,
            activation="gelu",
            output_dim=self.embed_dim,
            drop_rate=self.dropout_rate,
            name="mlp_decode",
        )
        self.mlp_layer.build(query_shape)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None):
        queries, encoder_output = inputs

        # 1. Cross-attention
        queries_norm = self.layernorm1(queries)
        attn_output = self.cross_attention(
            query=queries_norm, key=encoder_output, value=encoder_output, training=training
        )
        # Residual connection and layer norm
        x = queries + attn_output

        # 2. MLP
        x_norm = self.layernorm2(x)
        mlp_output = self.mlp_layer(x_norm, training=training)
        # Residual connection and layer norm
        output = x + mlp_output

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


class LearnableQueries(layers.Layer):
    """Initializes and provides a set of learnable query tokens.

    These tokens are learnable parameters of the model, initialized with random
    values. During the `call` method, they are tiled to match the batch size
    of the input tensor, providing a unique set of queries for each sample in the batch.

    Args:
        num_queries: The number of learnable queries to create.
        embed_dim: The dimensionality of each query token.

    Inputs:
        Any tensor, used only to infer the batch size for tiling the queries.

    Outputs:
        A tensor of shape `(batch_size, num_queries, embed_dim)` containing
        the tiled learnable queries.
    """

    def __init__(self, num_queries, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.queries = self.add_weight(
            name="learnable_queries",
            shape=[1, self.num_queries, self.embed_dim],
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        return ops.tile(self.queries, [batch_size, 1, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_queries, self.embed_dim)
