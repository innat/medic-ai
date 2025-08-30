import keras
from keras import layers, ops

from medicai.layers import TransUNetMLP
from medicai.utils import get_conv_layer, get_reshaping_layer


class LearnableQueries(layers.Layer):
    """Learnable query tokens for transformer decoder.

    This layer generates a set of learnable parameter vectors that serve as initial
    query tokens for transformer decoder architectures. These queries are learned
    during training and act as task-specific prompts that guide the attention mechanism.

    Attributes:
        num_queries (int): Number of learnable query tokens to generate.
        embed_dim (int): Dimensionality of each query token.
        queries (keras.Variable): Learnable query parameter matrix of shape
                              (1, num_queries, embed_dim).

    Args:
        num_queries (int): Number of learnable query tokens to generate. Typically
                          ranges from 10-500 depending on task complexity.
        embed_dim (int): Dimensionality of each query token. Must match the transformer's
                        embedding dimension.
        **kwargs: Additional layer arguments.

    Inputs:
        inputs: A tensor of any shape (used only for batch size inference).
                Typically the encoder output of shape (batch_size, sequence_length, embed_dim).

    Outputs:
        Tensor of shape (batch_size, num_queries, embed_dim) containing the learned queries
        replicated across the batch dimension.

    Example:
        ```python
        # Create learnable queries for 100 tokens with 256-dimensional embeddings
        learnable_queries = LearnableQueries(num_queries=100, embed_dim=256)
        queries = learnable_queries(encoder_output)  # Shape: (batch_size, 100, 256)
        ```

    Note:
        - The queries are initialized with a normal distribution (stddev=0.02)
        - The same queries are used across all batches (tiled for batch dimension)
        - These correspond to the 'p₀' tokens in the TransUNet architecture diagram

    Reference:
        Inspired by DETR (Carion et al., 2020) and similar object query mechanisms
        in detection and segmentation transformers.
    """

    def __init__(self, num_queries, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.queries = None

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
            num_heads=self.num_heads, key_dim=self.key_dim, dropout=self.dropout_rate
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
        p1 = self.masked_self_att(
            queries, queries, attention_mask=attention_mask, training=training
        )
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


class SpatialCrossAttention(layers.Layer):
    """Spatial cross-attention for CNN decoder feature refinement.

    Performs attention within spatial dimensions to fuse decoder features with skip connections.
    Uses convolutional projections and spatial attention weighting.

    Args:
        filters: Number of output filters/channels

    Inputs:
        decoder_features: Decoder features to be refined
        skip_features: Skip connection features from encoder
        training: Boolean for training mode

    Outputs:
        Tensor with same spatial dimensions as decoder_features, refined with skip features
    """

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        # input_shape will be a list of shapes when called with [d1, c3]
        decoder_shape, skip_shape = input_shape
        self.spatial_dims = len(decoder_shape) - 2  # 2 for 2D, 3 for 3D

        self.query_conv = get_conv_layer(
            spatial_dims=self.spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=1,
            name="query_conv",
        )
        self.key_conv = get_conv_layer(
            spatial_dims=self.spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=1,
            name="key_conv",
        )
        self.value_conv = get_conv_layer(
            spatial_dims=self.spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=1,
            name="value_conv",
        )
        self.out_conv = get_conv_layer(
            spatial_dims=self.spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=1,
            name="out_conv",
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

        # Automatic resizing based on input shapes
        resize_factors = []
        for i in range(self.spatial_dims):
            if decoder_shape[i + 1] % skip_shape[i + 1] != 0:
                raise ValueError(
                    f"Spatial dimension {i} of decoder features ({decoder_shape[i + 1]}) "
                    f"is not divisible by the corresponding skip feature dimension ({skip_shape[i + 1]})."
                )
            resize_factors.append(decoder_shape[i + 1] // skip_shape[i + 1])
        self.skip_resize = get_reshaping_layer(
            spatial_dims=self.spatial_dims, layer_type="upsampling", size=resize_factors
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs is a list [decoder_features, skip_features]
        decoder_features, skip_features = inputs

        # Auto-resize skip features to match decoder
        skip_resized = self.skip_resize(skip_features)

        # Project to query, key, value
        query = self.query_conv(decoder_features)
        key = self.key_conv(skip_resized)
        value = self.value_conv(skip_resized)

        # Compute spatial attention
        attention_scores = self.compute_attention_scores(query, key)
        attention_weights = self.apply_global_softmax(attention_scores)

        # Apply attention
        attended = self.apply_attention(attention_weights, value)
        output = self.layernorm(self.out_conv(attended) + decoder_features)
        return output

    def compute_attention_scores(self, query, key):
        pattern = {2: "bijd,bijd->bij", 3: "bijkd,bijkd->bijk"}
        return ops.einsum(pattern[self.spatial_dims], query, key)

    def apply_attention(self, weights, value):
        pattern = {2: "bij,bijd->bijd", 3: "bijk,bijkd->bijkd"}
        return ops.einsum(pattern[self.spatial_dims], weights, value)

    def apply_global_softmax(self, attention_scores):
        original_shape = ops.shape(attention_scores)
        flattened_scores = ops.reshape(attention_scores, (original_shape[0], -1))
        attention_weights_flat = ops.softmax(flattened_scores, axis=-1)
        attention_weights = ops.reshape(attention_weights_flat, original_shape)
        return attention_weights

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
            }
        )
        return config
