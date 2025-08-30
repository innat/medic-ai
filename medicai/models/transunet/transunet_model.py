import keras
import numpy as np
from keras import layers

from medicai.layers import ViTEncoderBlock, ViTPatchingAndEmbedding
from medicai.models import DenseNetBackbone
from medicai.utils import get_conv_layer, parse_model_inputs

from .transunet_layers import (
    CoarseToFineAttention,
    LearnableQueries,
    SpatialCrossAttention,
    TransUNetDecoderBlock,
)


class TransUNet(keras.Model):
    """
    2D and 3D TransUNet for image and volumetric medical sample segmentation.

    Architecture combines CNN encoder, transformer bottleneck, and CNN decoder
    with coarse-to-fine attention refinement and spatial cross-attention. A
    CNN encoder can be custom or any imagenet (2D or 3D) can be used. In this
    implementation, DenseNet121 is used.

    Args:
        input_shape: Input volume shape (depth, height, width, channels).
            Input image shape (height, width, channels)
        patch_size: Patch size for volume input (depth, height, width)
            Patch size for volume input (height, width)
        num_classes: Number of output segmentation classes
        classifier_activation: Activation for final layer (e.g., 'softmax')
        num_encoder_layers: Number of transformer encoder blocks
        num_decoder_layers: Number of transformer decoder blocks
        num_heads: Number of attention heads in transformer
        embed_dim: Transformer embedding dimension
        mlp_dim: MLP dimension in transformer blocks (typically 4× embed_dim)
        num_queries: Number of learnable queries for decoder
        dropout_rate: Dropout rate throughout network

    Returns:
        A Keras Model instance for 2D and 3D medical sample segmentation

    Architecture:
        - CNN Encoder: Extracts hierarchical features with skip connections
        - Transformer: Processes patches with self-attention for global context
        - Transformer Decoder: Refines features with masked cross-attention
        - Coarse-to-Fine Attention: Z-blocks fuse transformer and CNN features
        - CNN Decoder: Upsamples with spatial cross-attention to skip connections

    Example:
        ```python
        model3d = TransUNet(
            input_shape=(128, 128, 128, 1),
            patch_size=(16, 16, 16),
            num_classes=3,
            embed_dim=256
        )
        model2d = TransUNet(
            input_shape=(224, 224, 3),
            patch_size=(16, 16),
            num_classes=3,
            embed_dim=256
        )
        ```
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        patch_size,
        classifier_activation=None,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        embed_dim=256,
        mlp_dim=1024,
        num_queries=100,
        dropout_rate=0.1,
        name=None,
        **kwargs,
    ):

        # Auto-detect spatial dimensions and handle patch_size
        spatial_dims = len(input_shape) - 1  # 2 for 2D, 3 for 3D

        # Handle patch_size: if int, repeat for all spatial dimensions
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * spatial_dims
        elif isinstance(patch_size, (list, tuple)) and len(patch_size) != spatial_dims:
            raise ValueError(
                f"patch_size must have length {spatial_dims} for {spatial_dims}D input. "
                f"Got {patch_size} with length {len(patch_size)}"
            )

        inputs = parse_model_inputs(input_shape, name="transunet_input")

        # Get feature maps at different levels : CNN Encoder
        base_encoder = DenseNetBackbone(blocks=[6, 12, 24, 16], input_tensor=inputs)
        c1 = base_encoder.get_layer(index=309).output  # Deepest
        c2 = base_encoder.get_layer(index=137).output  # Mid
        c3 = base_encoder.get_layer(index=49).output  # Shallow
        p3 = base_encoder.get_layer(index=3).output  # Initial

        # Tokenize the feature maps : Transformer Encoder
        encoded_patches = ViTPatchingAndEmbedding(
            image_size=p3.shape[1:-1],
            patch_size=patch_size,
            hidden_dim=embed_dim,
            num_channels=p3.shape[-1],
            use_class_token=False,
            use_patch_bias=False,
            name="transunet_vit_patching_and_embedding",
        )(p3)

        encoder_output = encoded_patches
        for i in range(num_encoder_layers):
            encoder_output = ViTEncoderBlock(
                num_heads=num_heads,
                hidden_dim=embed_dim,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                name=f"transunet_vit_feature{i + 1}",
            )(encoder_output)

        # Transformer Decoder
        learnable_queries = LearnableQueries(num_queries, embed_dim, name="learnable_queries_1")(
            encoder_output
        )
        decoder_output = learnable_queries
        for i in range(num_decoder_layers):
            decoder_output = TransUNetDecoderBlock(
                embed_dim, num_heads, mlp_dim, dropout_rate, name=f"transunet_decoder_block_{i+1}"
            )([decoder_output, encoder_output])

        # Coarse-to-fine Z-Blocks
        # Prepare CNN features for Z-Blocks (flatten spatial dimensions)
        c1_flat = layers.Reshape((-1, c1.shape[-1]))(c1)  # (batch, spatial, channels)
        c2_flat = layers.Reshape((-1, c2.shape[-1]))(c2)
        c3_flat = layers.Reshape((-1, c3.shape[-1]))(c3)

        # Project CNN features to transformer dimension
        c1_proj = layers.Dense(embed_dim)(c1_flat)
        c2_proj = layers.Dense(embed_dim)(c2_flat)
        c3_proj = layers.Dense(embed_dim)(c3_flat)

        # Z³ Block: Refine with DEEPEST features first (coarse level)
        z3 = CoarseToFineAttention(
            embed_dim, num_heads, mlp_dim, dropout_rate, name="coarse_attention_3"
        )(
            [decoder_output, c1_proj]
        )  # c1_proj (deepest: index=309)

        # Z² Block: Refine with MID-LEVEL features
        z2 = CoarseToFineAttention(
            embed_dim, num_heads, mlp_dim, dropout_rate, name="coarse_attention_2"
        )(
            [z3, c2_proj]
        )  # c2_proj (mid: index=137)

        # Z¹ Block: Refine with SHALLOWEST features last (fine level)
        z1 = CoarseToFineAttention(
            embed_dim, num_heads, mlp_dim, dropout_rate, name="coarse_attention_1"
        )(
            [z2, c3_proj]
        )  # c3_proj (shallowest: index=49)

        # CNN Decoder with SpatialCrossAttention
        target_shape = self.get_target_spatial_shape(input_shape, downsampling_factor=8)
        num_spatial_positions = self.get_num_spatial_positions(target_shape)
        positional_queries = LearnableQueries(
            num_spatial_positions, embed_dim, name="learnable_queries_2"
        )(
            z1
        )  # (batch, spatial_positions, embed_dim)
        decoded_features = layers.Dense(64)(positional_queries)
        spatial_features = layers.Reshape(target_shape + [64])(decoded_features)

        # Level 1: Upsample and apply SpatialCrossAttention with c1 (DEEPEST - lowest resolution)
        d1 = get_conv_layer(
            spatial_dims=len(spatial_features.shape[1:-1]),
            layer_type="conv_transpose",
            filters=128,
            kernel_size=2,
            strides=2,
            padding="same",
        )(spatial_features)
        d1 = SpatialCrossAttention(128)([d1, c1])  # Fuse with c1 (deepest, lowest resolution)
        d1 = get_conv_layer(
            spatial_dims=len(d1.shape[1:-1]),
            layer_type="conv",
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same",
        )(d1)
        d1 = get_conv_layer(
            spatial_dims=len(d1.shape[1:-1]),
            layer_type="conv",
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same",
        )(d1)

        # Level 2: Upsample and apply SpatialCrossAttention with c2 (MID-LEVEL - medium resolution)
        d2 = get_conv_layer(
            spatial_dims=len(d1.shape[1:-1]),
            layer_type="conv_transpose",
            filters=64,
            kernel_size=2,
            strides=2,
            padding="same",
        )(d1)
        d2 = SpatialCrossAttention(64)([d2, c2])  # Fuse with c2 (mid-level, medium resolution)
        d2 = get_conv_layer(
            spatial_dims=len(d2.shape[1:-1]),
            layer_type="conv",
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
        )(d2)
        d2 = get_conv_layer(
            spatial_dims=len(d2.shape[1:-1]),
            layer_type="conv",
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
        )(d2)

        # Level 3: Upsample and apply SpatialCrossAttention with c3 (SHALLOWEST - highest resolution)
        d3 = get_conv_layer(
            spatial_dims=len(d2.shape[1:-1]),
            layer_type="conv_transpose",
            filters=32,
            kernel_size=2,
            strides=2,
            padding="same",
        )(d2)
        d3 = SpatialCrossAttention(32)([d3, c3])  # Fuse with c3 (shallowest, highest resolution)
        d3 = get_conv_layer(
            spatial_dims=len(d3.shape[1:-1]),
            layer_type="conv",
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same",
        )(d3)
        d3 = get_conv_layer(
            spatial_dims=len(d3.shape[1:-1]),
            layer_type="conv",
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same",
        )(d3)

        # Final output
        outputs = get_conv_layer(
            spatial_dims=len(d3.shape[1:-1]),
            layer_type="conv",
            filters=num_classes,
            kernel_size=1,
            activation=classifier_activation,
        )(d3)

        super().__init__(inputs=inputs, outputs=outputs, name=name or "TransUNet", **kwargs)

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.classifier_activation = classifier_activation
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_queries = num_queries
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "patch_size": self.patch_size,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "mlp_dim": self.mlp_dim,
            "num_queries": self.num_queries,
            "dropout_rate": self.dropout_rate,
        }
        return config

    @staticmethod
    def get_num_spatial_positions(target_shape):
        return np.prod(target_shape)

    @staticmethod
    def get_target_spatial_shape(input_shape, downsampling_factor=8):
        spatial_dims = len(input_shape) - 1  # For 2D or 3D
        target_shape = []
        for i in range(spatial_dims):
            dim = input_shape[i]
            if dim is not None:
                if dim % downsampling_factor != 0:
                    raise ValueError(
                        f"Input spatial dimension {i} ({dim}) is not divisible by "
                        f"downsampling_factor ({downsampling_factor})."
                    )
                target_shape.append(dim // downsampling_factor)
            else:
                target_shape.append(None)
        return target_shape
