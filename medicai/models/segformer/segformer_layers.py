import keras
import numpy as np
from keras import ops

from medicai.layers import DropPath
from medicai.utils import (
    get_conv_layer,
    get_norm_layer,
    get_reshaping_layer,
    parse_model_inputs,
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
        self.padding = get_reshaping_layer(
            spatial_dims=spatial_dims, layer_type="padding", padding=padding_size
        )
        self.proj = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.project_dim,
            kernel_size=self.patch_size,
            strides=self.stride,
            padding="valid",
        )
        self.norm = get_norm_layer(norm_name="layer", epsilon=1e-5)

    def call(self, x):
        # Apply padding to the input tensor.
        x = self.padding(x)

        # Project patches using the convolutional layer.
        x = self.proj(x)

        batch_size = ops.shape(x)[0]
        spatial_dims = ops.shape(x)[1:-1]
        channels = ops.shape(x)[-1]

        # Calculate the sequence length, which is the total number of patches.
        # This is a product of the spatial dimensions of the convoluted tensor
        seq_len = ops.prod(spatial_dims)

        # Reshape the tensor from (B, D, H, W, C) or (B, H, W, C) to (B, N, C),
        # where N is the sequence length, to prepare it for a transformer block.
        x = ops.reshape(x, (batch_size, seq_len, channels))

        x = self.norm(x)
        return x

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
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        # Calculate the scaling factor for attention scores.
        self.scale = (project_dim // num_heads) ** -0.5
        self.project_dim = project_dim
        self.spatial_dims = spatial_dims
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout

        # Dense layers for projecting queries, keys, values, and the final output.
        self.q = keras.layers.Dense(project_dim, use_bias=qkv_bias)
        self.k = keras.layers.Dense(project_dim, use_bias=qkv_bias)
        self.v = keras.layers.Dense(project_dim, use_bias=qkv_bias)
        self.proj = keras.layers.Dense(project_dim)
        self.dropout = keras.layers.Dropout(attention_dropout)
        self.proj_drop = keras.layers.Dropout(projection_dropout)

        if self.sr_ratio > 1:
            # Convolutional layer for spatial reduction of K and V.
            # The kernel size and strides are equal to the reduction ratio,
            # effectively downsampling the spatial dimensions.
            self.sr = get_conv_layer(
                spatial_dims=self.spatial_dims,
                layer_type="conv",
                filters=self.project_dim,
                kernel_size=self.sr_ratio,
                strides=self.sr_ratio,
            )
            self.norm = get_norm_layer(norm_name="layer", epsilon=1e-5)

    def call(self, x):
        B, N, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]

        # Project and reshape the query (Q) for multi-head attention.
        q = self.q(x)
        q = ops.reshape(q, (B, N, self.num_heads, C // self.num_heads))
        q = ops.transpose(q, [0, 2, 1, 3])  # (B, heads, N, C//heads)

        # Apply spatial reduction for keys and values if sr_ratio > 1.
        if self.sr_ratio > 1:
            spatial_shape = ops.cast(
                ops.round(ops.power(ops.cast(N, "float32"), 1.0 / self.spatial_dims)), "int32"
            )
            spatial_shape = [spatial_shape] * self.spatial_dims
            x = ops.reshape(x, (B, *spatial_shape, C))

            # Apply the convolutional spatial reduction.
            x = self.sr(x)
            x = ops.reshape(x, (B, -1, C))
            x = self.norm(x)

        # Project and reshape the key (K) and value (V) for multi-head attention.
        k = self.k(x)
        v = self.v(x)
        k = ops.transpose(
            ops.reshape(k, (B, -1, self.num_heads, C // self.num_heads)),
            [0, 2, 1, 3],
        )
        v = ops.transpose(
            ops.reshape(v, (B, -1, self.num_heads, C // self.num_heads)),
            [0, 2, 1, 3],
        )

        # Calculate attention scores: Q @ K^T.
        attn = (q @ ops.transpose(k, [0, 1, 3, 2])) * self.scale
        # Apply softmax to get attention weights.
        attn = ops.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        # Calculate output: attn @ V.
        out = attn @ v
        out = ops.transpose(out, [0, 2, 1, 3])
        out = ops.reshape(out, (B, N, C))

        # Final projection and dropout.
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

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
    def __init__(self, mlp_ratio, spatial_dims, dropout=0.0):
        """
        Mixed Feed-Forward Network (MixFFN) layer.

        This layer is a modified feed-forward network used in architectures like SegFormer.
        It combines a standard fully connected (Dense) layer with a depthwise separable
        convolutional layer. This allows the model to capture local spatial information
        that might be lost during the tokenization and self-attention processes,
        improving performance on vision tasks.

        Args:
            mlp_ratio (int): The ratio of the hidden dimension to the input dimension.
                The hidden layer size will be `input_channels * mlp_ratio`.
            spatial_dims (int): The number of spatial dimensions of the input (e.g., 2 for images).
            dropout (float, optional): The dropout rate applied after the GELU activation
                and the final dense layer. Defaults to 0.0.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        input_channels = input_shape[-1]
        project_dim = self.mlp_ratio * input_channels

        # First fully connected layer to expand the feature dimension.
        self.fc1 = keras.layers.Dense(project_dim)
        # Second fully connected layer to project back to the original dimension.
        self.fc2 = keras.layers.Dense(input_channels)

        # Depthwise convolution branch to mix local spatial information.
        # It's implemented as a Sequential model with a conv and batch norm.
        self.dwconv = keras.Sequential(
            [
                get_conv_layer(
                    spatial_dims=self.spatial_dims,
                    layer_type="conv",
                    filters=project_dim,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    groups=project_dim,  # Crucial for depthwise behavior
                ),
                get_norm_layer(norm_name="batch", name="dwconv_bn"),
            ]
        )

    def call(self, x):
        # Pass the input through the first fully connected layer.
        x = self.fc1(x)

        # Infer the spatial dimensions from the sequence length N.
        B, N, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]
        L = ops.cast(
            ops.round(
                ops.power(ops.cast(N, "float32"), 1.0 / ops.cast(self.spatial_dims, "float32"))
            ),
            "int32",
        )
        spatial_shape = [L] * self.spatial_dims

        # Reshape the flattened token sequence back into its spatial form.
        x = ops.reshape(x, (B, *spatial_shape, C))

        # Apply the depthwise convolution to mix local spatial information.
        x = self.dwconv(x)
        x = ops.reshape(x, (B, -1, C))
        x = ops.nn.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

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

        # Layer normalization applied before the attention block.
        self.norm1 = get_norm_layer(norm_name="layer", epsilon=self.layer_norm_epsilon)

        # Spatially-reduced multi-head attention block.
        self.attn = SegFormerMultiheadAttention(
            project_dim=self.project_dim,
            num_heads=self.num_heads,
            sr_ratio=self.sr_ratio,
            spatial_dims=self.spatial_dims,
            attention_dropout=self.attention_dropout,
            proj_dropout=self.projection_dropout,
            qkv_bias=qkv_bias,
        )

        # Stochastic Depth (DropPath) regularization.
        self.drop_path = DropPath(self.drop_prob)

        # Layer normalization applied before the MixFFN block.
        self.norm2 = get_norm_layer(norm_name="layer", epsilon=self.layer_norm_epsilon)

        # Mixed Feed-Forward Network to capture local context.
        self.mlp = MixFFN(
            mlp_ratio=self.mlp_ratio,
            spatial_dims=self.spatial_dims,
            dropout=0.0,
        )

    def call(self, x):
        # Attention block with residual connection and DropPath.
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # MixFFN block with residual connection and DropPath.
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

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


class MixVisionTransformer(keras.Model):
    def __init__(
        self,
        input_shape,
        max_drop_path_rate=0.1,
        layer_norm_epsilon=1e-5,
        qkv_bias=True,
        project_dim=[32, 64, 160, 256],
        layerwise_sr_ratios=[4, 2, 1, 1],
        layerwise_patch_sizes=[7, 3, 3, 3],
        layerwise_strides=[4, 2, 2, 2],
        layerwise_num_heads=[1, 2, 5, 8],
        layerwise_depths=[2, 2, 2, 2],
        layerwise_mlp_ratios=[4, 4, 4, 4],
        name=None,
        **kwargs,
    ):
        """
        MixVisionTransformer (MixViT) Model.

        This class implements the encoder backbone of the SegFormer architecture. It is a
        hierarchical vision transformer that processes input data (2D images or 3D volumes)
        through multiple stages. Each stage consists of an overlapping patch embedding layer,
        followed by a series of efficient transformer encoder blocks. The use of overlapping
        patches and spatially reduced attention makes the model efficient for high-resolution
        inputs while capturing both local and global features.

        The model is built using the Keras Functional API, with a progressive
        downsampling of the spatial dimensions and an increase in the feature dimensions,
        similar to a convolutional neural network.

        Args:
            input_shape (tuple): The shape of the input data, excluding the batch dimension.
            max_drop_path_rate (float, optional): The maximum rate for stochastic depth.
                The dropout rate is linearly increased across all transformer blocks. Defaults to 0.1.
            layer_norm_epsilon (float, optional): A small value for numerical stability in layer normalization.
                Defaults to 1e-5.
            qkv_bias (bool): A boolean flag to indicate applying bias to projected queries, keys, and values.
            project_dim (list[int], optional): A list of feature dimensions for each stage.
                Defaults to [32, 64, 160, 256].
            layerwise_sr_ratios (list[int], optional): A list of spatial reduction ratios for each stage's
                attention layers. Defaults to [4, 2, 1, 1].
            layerwise_patch_sizes (list[int], optional): A list of patch sizes for the embedding layer
                in each stage. Defaults to [7, 3, 3, 3].
            layerwise_strides (list[int], optional): A list of strides for the embedding layer in each stage.
                Defaults to [4, 2, 2, 2].
            layerwise_num_heads (list[int], optional): A list of the number of attention heads for each stage.
                Defaults to [1, 2, 5, 8].
            layerwise_depths (list[int], optional): A list of the number of transformer blocks for each stage.
                Defaults to [2, 2, 2, 2].
            layerwise_mlp_ratios (list[int], optional): A list of MLP expansion ratios for each stage.
                Defaults to [4, 4, 4, 4].
            name (str, optional): The name of the model. Defaults to None.
            **kwargs: Standard Keras Model keyword arguments.
        """
        spatial_dims = len(input_shape) - 1
        num_layers = len(layerwise_depths)

        # Create a list of linearly increasing drop path rate
        dpr = [x for x in np.linspace(0.0, max_drop_path_rate, sum(layerwise_depths))]

        # initialize model input
        inputs = parse_model_inputs(input_shape=input_shape, name="mixvit_input")
        x = inputs
        cur = 0

        # Loop through each hierarchical stage of the model.
        for i in range(num_layers):
            # Overlapping Patch Embedding Stage
            patch_embed = OverlappingPatchingAndEmbedding(
                project_dim=project_dim[i],
                patch_size=layerwise_patch_sizes[i],
                stride=layerwise_strides[i],
                name=f"overlap_patch_and_embed_{i}",
            )
            x = patch_embed(x)

            # Transformer Blocks
            for k in range(layerwise_depths[i]):
                x = HierarchicalTransformerEncoder(
                    project_dim=project_dim[i],
                    num_heads=layerwise_num_heads[i],
                    sr_ratio=layerwise_sr_ratios[i],
                    mlp_ratio=layerwise_mlp_ratios[i],
                    drop_prob=dpr[cur + k],
                    qkv_bias=qkv_bias,
                    layer_norm_epsilon=layer_norm_epsilon,
                    spatial_dims=spatial_dims,
                    name=f"hierarchical_encoder_{i}_{k}",
                )(x)
            cur += layerwise_depths[i]

            # Layer Normalization
            x = get_norm_layer(norm_name="layer", epsilon=layer_norm_epsilon)(x)

            # Reshape output to a spatial feature map for the next stage.
            n_patches = ops.shape(x)[1]
            current_spatial_dims = int(ops.round(n_patches ** (1 / spatial_dims)))
            current_spatial_dims = [current_spatial_dims] * spatial_dims
            x = keras.layers.Reshape(
                current_spatial_dims + [project_dim[i]], name=f"mixvit_features{i+1}"
            )(x)

        super().__init__(inputs=inputs, outputs=x, name=name or f"mixvit{spatial_dims}D", **kwargs)

        self.project_dim = project_dim
        self.qkv_bias = qkv_bias
        self.layerwise_patch_sizes = layerwise_patch_sizes
        self.layerwise_strides = layerwise_strides
        self.layerwise_num_heads = layerwise_num_heads
        self.layerwise_depths = layerwise_depths
        self.layerwise_sr_ratios = layerwise_sr_ratios
        self.max_drop_path_rate = max_drop_path_rate
        self.layerwise_mlp_ratios = layerwise_mlp_ratios
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape[1:],
                "project_dim": self.project_dim,
                "qkv_bias": self.qkv_bias,
                "layerwise_patch_sizes": self.layerwise_patch_sizes,
                "layerwise_strides": self.layerwise_strides,
                "layerwise_num_heads": self.layerwise_num_heads,
                "layerwise_depths": self.layerwise_depths,
                "layerwise_sr_ratios": self.layerwise_sr_ratios,
                "layerwise_mlp_ratios": self.layerwise_mlp_ratios,
                "max_drop_path_rate": self.max_drop_path_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
