import math

import keras
from keras import initializers, layers, ops

from medicai.utils.model_utils import (
    get_act_layer,
    get_conv_layer,
    get_pooling_layer,
    get_reshaping_layer,
)


class AttentionGate(keras.layers.Layer):
    """https://arxiv.org/abs/1804.03999"""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        # variable
        self.filters = filters

        # ops
        self.relu = get_act_layer("relu")
        self.sigmoid = get_act_layer("sigmoid")
        self.mul = layers.Multiply()
        self.add = layers.Add()

    def build(self, input_shape):
        self.spatial_dims = len(input_shape[0]) - 2
        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"{self.__class__.__name__} only supports 2D or 3D inputs. Got spatial_dims={self.spatial_dims}"
            )

        # input_shape = [x_shape, g_shape]
        x_shape, g_shape = input_shape
        self.theta_x = get_conv_layer(
            self.spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=1,
            strides=2,
            padding="same",
        )
        self.theta_x.build(x_shape)

        self.upsampling = get_reshaping_layer(
            self.spatial_dims,
            layer_type="upsampling",
            size=2,
        )

        self.phi_g = get_conv_layer(
            self.spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
        )
        self.phi_g.build(g_shape)

        self.psi = get_conv_layer(
            self.spatial_dims,
            layer_type="conv",
            filters=1,
            kernel_size=1,
            strides=1,
            padding="same",
        )
        psi_shape = self.add.compute_output_shape(
            [self.theta_x.compute_output_shape(x_shape), self.phi_g.compute_output_shape(g_shape)]
        )
        self.psi.build(psi_shape)

        self.built = True

    def call(self, inputs):
        # expect list [skip, input]
        x, g = inputs

        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        add = self.add([theta_x, phi_g])

        relu = self.relu(add)
        psi = self.psi(relu)
        alpha = self.sigmoid(psi)
        alpha = self.upsampling(alpha)
        return self.mul([x, alpha])  # gated skip

    def compute_output_shape(self, input_shape):
        # output has the same shape as skip connection (x)
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
            }
        )
        return config


class SqueezeExcitation(keras.layers.Layer):
    def __init__(self, ratio=16, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.activation = activation

    def build(self, input_shape):
        spatial_dims = len(input_shape) - 2
        self.channels = input_shape[-1]
        self.reduced_channels = max(1, self.channels // self.ratio)

        # Global average pooling
        self.global_pool = get_pooling_layer(
            spatial_dims=spatial_dims, layer_type="avg", global_pool=True, keepdims=True
        )
        self.global_pool.build(input_shape)

        # Build excitation layers using 1x1 (or 1x1x1) convolutions
        self.conv1 = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.reduced_channels,
            kernel_size=1,
            activation=self.activation,
        )
        pooled_shape_conv1 = (input_shape[0],) + (1,) * spatial_dims + (self.channels,)
        self.conv1.build(pooled_shape_conv1)

        self.conv2 = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.channels,
            kernel_size=1,
            activation="sigmoid",
        )
        pooled_shape_conv2 = self.conv1.compute_output_shape(pooled_shape_conv1)
        self.conv2.build(pooled_shape_conv2)

        super().build(input_shape)

    def call(self, inputs):
        # Squeeze: Global average pooling
        se = self.global_pool(inputs)

        # Excitation: Two 1x1 convolutional layers
        se = self.conv1(se)
        se = self.conv2(se)

        # Scale the input features (broadcasting automatically handles dimensions)
        return inputs * se

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "ratio": self.ratio,
                "activation": self.activation,
            }
        )
        return config


class UniformSqrtDim(initializers.Initializer):
    def __call__(self, shape, dtype=None):
        limit = 1.0 / math.sqrt(shape[-1])
        return keras.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)


class EfficientPairedAttention(keras.layers.Layer):
    """
    Efficient Paired Attention (EPA) layer used in UNETR++.

    This layer implements a dual attention mechanism combining:
    1. Channel Attention (CA): Performs attention along the feature/head
       dimension to capture inter-channel dependencies.
    2. Spatial Attention (SA): Projects the input sequence to a reduced number
       of spatial tokens (N -> K) for efficient spatial interaction modeling.

    The layer is designed to handle high-resolution 2D or 3D inputs efficiently
    and supports multi-head attention. EPA maintains separate learnable
    temperature parameters for scaling attention logits and applies dropout
    for regularization.
    """

    def __init__(
        self,
        sequence_length,
        hidden_size,
        spatial_reduced_tokens,
        num_heads=4,
        qkv_bias=False,
        channel_attn_drop=0.1,
        spatial_attn_drop=0.1,
        name="efficient_paired_attention",
        **kwargs,
    ):
        """
        Initializes the EfficientPairedAttention layer.

        Args:
            sequence_length: Integer, length of the input sequence (N) to the
                attention mechanism.
            hidden_size: Integer, number of channels in the input features.
            spatial_reduced_tokens: Integer, reduced number of spatial tokens (K)
                used in spatial attention projection.
            num_heads: Integer, number of attention heads.
            qkv_bias: Boolean, whether to include bias in the QKV dense layers.
            channel_attn_drop: Float, dropout probability for channel attention.
            spatial_attn_drop: Float, dropout probability for spatial attention.
            name: Optional string, name of the layer.
            **kwargs: Additional keyword arguments passed to the parent Layer.

        Example:
        >>> from medicai.layers import EfficientPairedAttention
        >>> epa = EfficientPairedAttention(
        ...     sequence_length=64,
        ...     hidden_size=128,
        ...     spatial_reduced_tokens=16,
        ...     num_heads=4,
        ... )
        >>> x = keras.random.normal((2, 64, 128))
        >>> y = epa(x)
        >>> print(y.shape)  # (2, 64, 128)
        """
        super().__init__(name=name, **kwargs)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.spatial_reduced_tokens = spatial_reduced_tokens
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.channel_attn_drop = channel_attn_drop
        self.spatial_attn_drop = spatial_attn_drop
        self.prefix = name

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.head_dim = hidden_size // num_heads

    def build(self, input_shape):
        # qkvv = 4 linear layers (query_shared, key_shared, value_channel, value_spatial)
        # Output is (B, N, 4*C)
        self.qkvv = layers.Dense(
            self.hidden_size * 4, use_bias=self.qkv_bias, name=f"{self.prefix}_qkvv"
        )
        self.qkvv.build((None, None, self.hidden_size))

        # Shared projection matrix for spatial attention (N -> K projection)
        # shape: (sequence_length, spatial_reduced_tokens) i.e., (N, K)
        self.spatial_sequence_projection = self.add_weight(
            name=f"{self.prefix}_spatial_sequence_projection",
            shape=(self.sequence_length, self.spatial_reduced_tokens),
            initializer=UniformSqrtDim(),
            trainable=True,
        )

        # temperature parameters (per head) for scaling attention logits
        self.temperature_ca = self.add_weight(
            name=f"{self.prefix}_temperature_ca",
            shape=(self.num_heads, 1, 1),
            initializer="ones",
            trainable=True,
        )
        self.temperature_sa = self.add_weight(
            name=f"{self.prefix}_temperature_sa",
            shape=(self.num_heads, 1, 1),
            initializer="ones",
            trainable=True,
        )

        self.attn_drop_ca = layers.Dropout(self.channel_attn_drop)
        self.attn_drop_sa = layers.Dropout(self.spatial_attn_drop)
        self.built = True

    def channel_attention(self, q_norm, k_norm, v_CA, training):
        # Calculate attention logits
        attn_CA = ops.matmul(q_norm, ops.transpose(k_norm, (0, 1, 3, 2))) * self.temperature_ca

        # Apply softmax and dropout
        attn_CA = ops.nn.softmax(attn_CA, axis=-1)
        attn_CA = self.attn_drop_ca(attn_CA, training=training)

        # Compute channel output
        x_CA_raw = ops.matmul(attn_CA, v_CA)
        return x_CA_raw  # Shape: (B, H, D_h, N)

    def spatial_attention(self, q_norm, k_norm, v_SA, training):
        # Project k and v using the shared sequence projection matrix (N -> K)
        k_shared_projected = ops.einsum("bhdn,nk->bhdk", k_norm, self.spatial_sequence_projection)
        v_SA_projected = ops.einsum("bhdn,nk->bhdk", v_SA, self.spatial_sequence_projection)

        # Calculate attention logits
        attn_SA = (
            ops.matmul(
                ops.transpose(q_norm, (0, 1, 3, 2)),  # (B, H, N, D_h)
                k_shared_projected,  # (B, H, D_h, K)
            )
            * self.temperature_sa
        )

        # Apply softmax and dropout
        attn_SA = ops.nn.softmax(attn_SA, axis=-1)
        attn_SA = self.attn_drop_sa(attn_SA, training=training)

        # Compute spatial output
        x_SA_raw = ops.matmul(attn_SA, ops.transpose(v_SA_projected, (0, 1, 3, 2)))
        return x_SA_raw  # Shape: (B, H, N, D_h)

    def call(self, x, training=False):
        input_shape = ops.shape(x)
        B, N, C = input_shape[0], input_shape[1], input_shape[2]

        # 1. Project q, k, v_CA, v_SA
        qkvv = self.qkvv(x)  # (B, N, 4*C)
        qkvv = ops.reshape(qkvv, (B, N, 4, self.num_heads, self.head_dim))
        qkvv = ops.transpose(qkvv, (2, 0, 3, 1, 4))
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        # Transpose for consistency with Channel Attention operations
        q_shared_t = ops.transpose(q_shared, (0, 1, 3, 2))
        k_shared_t = ops.transpose(k_shared, (0, 1, 3, 2))
        v_CA_t = ops.transpose(v_CA, (0, 1, 3, 2))
        v_SA_t = ops.transpose(v_SA, (0, 1, 3, 2))

        # Normalize q and k - l2 norm
        q_norm = self.safe_normalize(q_shared_t, axis=-1, epsilon=1e-6)
        k_norm = self.safe_normalize(k_shared_t, axis=-1, epsilon=1e-6)

        # Compute Channel Attention
        x_CA_raw = self.channel_attention(q_norm, k_norm, v_CA_t, training)
        x_CA = ops.transpose(x_CA_raw, (0, 3, 1, 2))
        x_CA = ops.reshape(x_CA, (B, N, C))

        # Compute Spatial Attention
        x_SA_raw = self.spatial_attention(q_norm, k_norm, v_SA_t, training)
        x_SA = ops.transpose(x_SA_raw, (0, 2, 1, 3))
        x_SA = ops.reshape(x_SA, (B, N, C))

        # 4. Combine results
        return x_CA + x_SA

    @staticmethod
    def safe_normalize(x, axis=-1, epsilon=1e-6):
        inputs = ops.cast(x, "float32")
        square_sum = ops.sum(ops.square(inputs), axis=axis, keepdims=True)
        safe_square_sum = ops.where(
            square_sum < epsilon, epsilon * ops.ones_like(square_sum), square_sum
        )
        norm = ops.sqrt(safe_square_sum)
        result = inputs / norm
        return ops.cast(result, x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "spatial_reduced_tokens": self.spatial_reduced_tokens,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
            }
        )
        return config
