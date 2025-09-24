import keras
from keras import ops

from medicai.layers import DropPath
from medicai.utils import (
    get_conv_layer,
    get_norm_layer,
    get_reshaping_layer,
)


class OverlappingPatchingAndEmbedding(keras.layers.Layer):
    def __init__(self, project_dim, patch_size, stride, **kwargs):
        """
        Overlapping Patching and Embedding Layer.

        This layer processes an input tensor (e.g., an image or volume) by
        dividing it into overlapping patches, projecting these patches into
        a higher-dimensional space, and then flattening them into a sequence
        of tokens. This is a common technique used in vision transformers
        to capture local context.

        Args:
            project_dim (int): The dimensionality of the embedding space for each patch.
            patch_size (int): The size of the patches (e.g., 16 for a 16x16 patch).
            stride (int): The stride of the convolution operation, which determines the
                degree of overlap between patches. A stride smaller than `patch_size`
                creates overlapping patches.
            **kwargs: Standard Keras layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.patch_size = patch_size
        self.stride = stride

    def build(self, input_shape):
        spatial_dims = len(input_shape) - 2
        padding_size = self.patch_size // 2

        # Layer for applying padding to the input tensor.
        self.padding = get_reshaping_layer(
            spatial_dims=spatial_dims, layer_type="padding", padding=padding_size
        )
        self.padding.build(input_shape)

        # Convolutional layer for projecting and extracting overlapping patches.
        # This layer is the source of the trainable parameters.
        self.proj = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.project_dim,
            kernel_size=self.patch_size,
            strides=self.stride,
            padding="valid",
        )
        padded_shape = self.padding.compute_output_shape(input_shape)
        self.proj.build(padded_shape)

        # Layer normalization to normalize the projected patch embeddings.
        self.norm = get_norm_layer(norm_name="layer", epsilon=1e-5)
        norm_input_shape = (input_shape[0], -1, self.project_dim)
        self.norm.build(norm_input_shape)
        self.built = True

    def call(self, x):
        """
        Performs the forward pass of the layer.

        Args:
            x (Tensor): The input tensor with shape `(batch, ..., channels)`.

        Returns:
            Tensor: The output tensor with shape `(batch, sequence_length, project_dim)`.
        """
        # Apply padding to the input tensor.
        x = self.padding(x)

        # Project patches using the convolutional layer.
        x = self.proj(x)

        # Reshape the tensor from (B, D, H, W, C) or (B, H, W, C) to (B, N, C),
        # where N is the sequence length, to prepare it for a transformer block.
        batch_size = ops.shape(x)[0]
        channels = ops.shape(x)[-1]
        x = ops.reshape(x, (batch_size, -1, channels))

        # Normalize the patch embeddings.
        x = self.norm(x)
        return x

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer for fixed input sizes.

        Args:
            input_shape (tuple): The input shape, including the batch dimension.

        Returns:
            tuple: The output shape `(batch_size, sequence_length, project_dim)`.
        """
        batch_size = input_shape[0]
        spatial_dims = input_shape[1:-1]
        padding_size = self.patch_size // 2

        # Calculate output spatial dimensions after padding and convolution
        # Formula: output_size = (input_size + 2*padding - kernel_size) // stride + 1
        output_spatial_dims = []
        for dim in spatial_dims:
            # Input spatial dimensions are guaranteed to be concrete values
            padded_dim = dim + 2 * padding_size
            output_dim = (padded_dim - self.patch_size) // self.stride + 1
            output_spatial_dims.append(output_dim)

        # Calculate sequence length (product of output spatial dimensions)
        seq_len = 1
        for dim in output_spatial_dims:
            seq_len *= dim

        return (batch_size, seq_len, self.project_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "patch_size": self.patch_size,
                "stride": self.stride,
            }
        )
        return config


class SegFormerMultiheadAttention(keras.layers.Layer):
    def __init__(
        self,
        project_dim,
        num_heads,
        sr_ratio,
        spatial_dims,
        qkv_bias=True,
        attention_dropout=0.0,
        projection_dropout=0.0,
        **kwargs,
    ):
        """
        SegFormer Multi-Head Self-Attention Layer with Spatial Reduction.

        This layer implements the efficient self-attention mechanism proposed in the
        SegFormer architecture. It applies a spatial reduction to the key (K) and
        value (V) tensors, which significantly reduces the computational complexity
        and memory footprint, making it suitable for high-resolution feature maps
        in vision models. The query (Q) tensor remains at full resolution.

        Args:
            project_dim (int): The dimension of the projected queries, keys, and values.
            num_heads (int): The number of attention heads.
            sr_ratio (int): The spatial reduction ratio for keys and values.
                A value of 1 means no reduction (standard multi-head attention).
                A value greater than 1 applies a convolutional downsampling layer.
            spatial_dims (int): The number of spatial dimensions of the input (e.g., 2 for images, 3 for volumes).
            qkv_bias (bool): A boolean flag to indicate applying bias to projected queries, keys, and values.
            attention_dropout (float, optional): Dropout rate for the attention scores. Defaults to 0.0.
            projection_dropout (float, optional): Dropout rate for the final projection output. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.scale = (project_dim // num_heads) ** -0.5
        self.project_dim = project_dim
        self.spatial_dims = spatial_dims
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout

    def build(self, input_shape):
        B, N, C = input_shape[0], input_shape[1], input_shape[2]

        # Precompute spatial dimensions during build (when shapes are concrete)
        self.spatial_size = int(round(N ** (1 / self.spatial_dims)))

        # Build query projection layer
        self.q = keras.layers.Dense(self.project_dim, use_bias=self.qkv_bias)
        self.q.build(input_shape)

        # Build key and value projection layers
        if self.sr_ratio > 1:
            # Build spatial reduction layers
            self.sr = get_conv_layer(
                spatial_dims=self.spatial_dims,
                layer_type="conv",
                filters=self.project_dim,
                kernel_size=self.sr_ratio,
                strides=self.sr_ratio,
            )
            # Calculate spatial dimensions for SR (patial reduction)
            spatial_shape = [self.spatial_size] * self.spatial_dims
            sr_input_shape = (B, *spatial_shape, C)
            self.sr.build(sr_input_shape)

            # Calculate output shape after spatial reduction
            sr_output_shape = self.sr.compute_output_shape(sr_input_shape)
            sr_seq_len = 1
            for dim in sr_output_shape[1:-1]:
                sr_seq_len *= dim
            sr_output_shape_flat = (B, sr_seq_len, self.project_dim)

            self.norm = get_norm_layer(norm_name="layer", epsilon=1e-5)
            self.norm.build(sr_output_shape_flat)

            # Build K and V layers with reduced sequence length
            k_v_input_shape = sr_output_shape_flat
        else:
            # No spatial reduction or spatial_size is None
            k_v_input_shape = input_shape

        # Build key and value layers
        self.k = keras.layers.Dense(self.project_dim, use_bias=self.qkv_bias)
        self.v = keras.layers.Dense(self.project_dim, use_bias=self.qkv_bias)
        self.k.build(k_v_input_shape)
        self.v.build(k_v_input_shape)

        # Build final projection and dropout layers
        self.proj = keras.layers.Dense(self.project_dim)
        self.dropout = keras.layers.Dropout(self.attention_dropout)
        self.proj_drop = keras.layers.Dropout(self.projection_dropout)
        self.proj.build(input_shape)
        self.dropout.build((B, self.num_heads, N, N))
        self.proj_drop.build(input_shape)
        self.built = True

    def call(self, x):
        B, N, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]

        # Project and reshape query
        q = self.q(x)
        q = ops.reshape(q, (B, N, self.num_heads, C // self.num_heads))
        q = ops.transpose(q, [0, 2, 1, 3])

        # Apply spatial reduction if needed and spatial_size is available
        if self.sr_ratio > 1:
            # Use precomputed spatial dimensions
            spatial_shape = [self.spatial_size] * self.spatial_dims

            # Reshape to spatial format and apply reduction
            x_reshaped = ops.reshape(x, (B, *spatial_shape, C))
            x_reduced = self.sr(x_reshaped)

            # Flatten back to sequence
            x_reduced = ops.reshape(x_reduced, (B, -1, C))
            x_reduced = self.norm(x_reduced)

            # Use reduced sequence for K and V
            k = self.k(x_reduced)
            v = self.v(x_reduced)
        else:
            # No spatial reduction or fallback
            k = self.k(x)
            v = self.v(x)

        # Reshape K and V
        k = ops.transpose(
            ops.reshape(k, (B, -1, self.num_heads, C // self.num_heads)),
            [0, 2, 1, 3],
        )
        v = ops.transpose(
            ops.reshape(v, (B, -1, self.num_heads, C // self.num_heads)),
            [0, 2, 1, 3],
        )

        # Calculate attention
        attn = (q @ ops.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = ops.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        # Calculate output
        out = attn @ v
        out = ops.transpose(out, [0, 2, 1, 3])
        out = ops.reshape(out, (B, N, C))

        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "num_heads": self.num_heads,
                "sr_ratio": self.sr_ratio,
                "qkv_bias": self.qkv_bias,
                "spatial_dims": self.spatial_dims,
                "attention_dropout": self.attention_dropout,
                "projection_dropout": self.projection_dropout,
            }
        )
        return config


class MixFFN(keras.layers.Layer):
    def __init__(self, mlp_ratio, spatial_dims, dropout=0.0, **kwargs):
        """
        Mixed Feed-Forward Network (MixFFN) layer.

        This layer is a modified feed-forward network used in architectures like SegFormer.
        It combines a standard fully connected (Dense) layer with a depthwise separable
        convolutional layer. This allows the model to capture local spatial information
        that might be lost during the tokenization and self-attention processes,
        improving performance on vision tasks.

        Args:
            mlp_ratio (int): The ratio of the hidden dimension to the input dimension.
                The hidden layer size will be input_channels * mlp_ratio.
            spatial_dims (int): The number of spatial dimensions of the input (e.g., 2 for images, 3 for volumes).
            dropout (float, optional): The dropout rate applied after the GELU activation
                and the final dense layer. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.spatial_dims = spatial_dims
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        input_channels = input_shape[-1]
        project_dim = self.mlp_ratio * input_channels
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        # Calculate spatial dimensions during build (when shapes are concrete)
        self.spatial_size = int(round(seq_len ** (1 / self.spatial_dims)))

        # Build first fully connected layer
        self.fc1 = keras.layers.Dense(project_dim)
        self.fc1.build(input_shape)

        # Calculate output shape after fc1
        fc1_output_shape = self.fc1.compute_output_shape(input_shape)

        # Build depthwise convolution branch
        spatial_shape = [self.spatial_size] * self.spatial_dims
        dwconv_input_shape = (batch_size, *spatial_shape, project_dim)

        self.dwconv = keras.Sequential(
            [
                get_conv_layer(
                    spatial_dims=self.spatial_dims,
                    layer_type="conv",
                    filters=project_dim,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    groups=project_dim,
                ),
                get_norm_layer(norm_name="batch", name="dwconv_bn"),
            ]
        )
        self.dwconv.build(dwconv_input_shape)

        # Build second fully connected layer
        self.fc2 = keras.layers.Dense(input_channels)
        self.fc2.build(fc1_output_shape)

        # Build dropout layers
        self.dropout1.build(fc1_output_shape)
        self.dropout2.build((batch_size, seq_len, input_channels))

        self.built = True

    def call(self, x):
        # Pass through first fully connected layer
        x = self.fc1(x)

        # Reshape to spatial format for depthwise convolution
        spatial_shape = [self.spatial_size] * self.spatial_dims
        B, C = ops.shape(x)[0], ops.shape(x)[2]
        x_reshaped = ops.reshape(x, (B, *spatial_shape, C))

        # Apply depthwise convolution
        x_conv = self.dwconv(x_reshaped)

        # Reshape back to sequence format
        x = ops.reshape(x_conv, (B, -1, C))

        x = ops.nn.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mlp_ratio": self.mlp_ratio,
                "spatial_dims": self.spatial_dims,
                "dropout": self.dropout_rate,
            }
        )
        return config


class HierarchicalTransformerEncoder(keras.layers.Layer):
    def __init__(
        self,
        project_dim,
        num_heads,
        spatial_dims,
        sr_ratio=1,
        mlp_ratio=2,
        drop_prob=0.0,
        qkv_bias=True,
        layer_norm_epsilon=1e-5,
        attention_dropout=0.0,
        projection_dropout=0.0,
        **kwargs,
    ):
        """
        Hierarchical Transformer Encoder Block.

        This layer represents a single block of the encoder in a hierarchical
        vision transformer architecture, such as SegFormer. It consists of two
        main components: a spatially-reduced multi-head self-attention block
        and a mixed feed-forward network (MixFFN). This design efficiently
        processes high-resolution inputs while capturing both global and local
        context.

        Args:
            project_dim (int): The dimensionality of the token embeddings.
            num_heads (int): The number of attention heads in the multi-head attention layer.
            spatial_dims (int): The number of spatial dimensions of the input (e.g., 2 for images, 3 for volumes).
            sr_ratio (int, optional): The spatial reduction ratio for keys and values in the attention mechanism.
                A value of 1 means no reduction. Defaults to 1.
            drop_prob (float, optional): The dropout rate for Stochastic Depth (DropPath).
                This is applied to the output of both the attention and MLP blocks. Defaults to 0.0.
            qkv_bias (bool): A boolean flag to indicate applying bias to projected queries, keys, and values.
            mlp_ratio (int): A MLP expansion ratios for each encoder stage.
            layer_norm_epsilon (float, optional): A small value added to the denominator of the layer normalization
                for numerical stability. Defaults to 1e-5.
            attention_dropout (float, optional): The dropout rate for the attention scores. Defaults to 0.0.
            projection_dropout (float, optional): The dropout rate for the final output projection of the
                attention layer. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.sr_ratio = sr_ratio
        self.layer_norm_epsilon = layer_norm_epsilon
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
        self.spatial_dims = spatial_dims
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio

    def build(self, input_shape):
        # Build first layer normalization
        self.norm1 = get_norm_layer(norm_name="layer", epsilon=self.layer_norm_epsilon)
        self.norm1.build(input_shape)

        # Build attention block
        self.attn = SegFormerMultiheadAttention(
            project_dim=self.project_dim,
            num_heads=self.num_heads,
            sr_ratio=self.sr_ratio,
            spatial_dims=self.spatial_dims,
            attention_dropout=self.attention_dropout,
            projection_dropout=self.projection_dropout,
            qkv_bias=self.qkv_bias,
        )
        self.attn.build(input_shape)  # Attention expects same input shape

        # Build stochastic depth (DropPath)
        self.drop_path = DropPath(self.drop_prob)

        # Build second layer normalization
        self.norm2 = get_norm_layer(norm_name="layer", epsilon=self.layer_norm_epsilon)
        self.norm2.build(input_shape)  # Same input shape

        # Build MixFFN block
        self.mlp = MixFFN(
            mlp_ratio=self.mlp_ratio,
            spatial_dims=self.spatial_dims,
            dropout=0.0,
        )
        self.mlp.build(input_shape)  # MixFFN expects same input shape

        self.built = True

    def call(self, x):
        # Attention block with residual connection and DropPath.
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # MixFFN block with residual connection and DropPath.
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "num_heads": self.num_heads,
                "drop_prob": self.drop_prob,
                "sr_ratio": self.sr_ratio,
                "qkv_bias": self.qkv_bias,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "attention_dropout": self.attention_dropout,
                "projection_dropout": self.projection_dropout,
                "spatial_dims": self.spatial_dims,
            }
        )
        return config
