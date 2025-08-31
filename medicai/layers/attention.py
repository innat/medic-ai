from medicai.utils.general import hide_warnings

hide_warnings()

from keras import layers, ops

from medicai.utils import get_conv_layer, get_reshaping_layer


class ChannelWiseAttention(layers.Layer):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        dims = input_shape.shape[-1]
        # squeeze
        self.gap = layers.GlobalAveragePooling2D()
        # excitation
        self.fc0 = layers.Dense(int(self.alpha * dims), use_bias=False, activation="relu")
        self.fc1 = layers.Dense(dims, use_bias=False, activation="sigmoid")
        self.rs = layers.Reshape((1, 1, dims))

    def call(self, inputs):
        # calculate channel-wise attention vector
        z = self.gap(inputs)
        u = self.fc0(z)
        u = self.fc1(u)
        u = self.rs(u)
        return u * inputs


class ElementWiseAttention(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes

        self.conv0 = layers.Conv2D(
            512,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            activation=ops.relu,
        )
        self.conv1 = layers.Conv2D(
            512,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            activation=ops.relu,
        )
        self.conv2 = layers.Conv2D(
            self.num_classes,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            activation=ops.softmax,
        )

        # linear classifier
        self.linear = layers.Conv2D(
            self.num_classes,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            activation=None,
        )

    def call(self, inputs):
        # f(att)
        a = self.conv0(inputs)
        a = self.conv1(a)
        a = self.conv2(a)
        # confidence score
        s = self.linear(inputs)
        # element-wise multiply to prevent unnecessary attention
        m = s * a
        return m


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
        """This computes scores via an element-wise product of query and key,
        summed over the channel dimension.
        """
        pattern = {2: "bijd,bijd->bij", 3: "bijkd,bijkd->bijk"}
        return ops.einsum(pattern[self.spatial_dims], query, key)

    def apply_attention(self, weights, value):
        """This applies the computed spatial attention weights to the value tensor."""
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
