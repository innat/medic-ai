import keras
import numpy as np
from keras import layers
from keras.initializers import HeNormal

from medicai.layers import ViTEncoderBlock, ViTPatchingAndEmbedding
from medicai.models import DenseNetBackbone
from medicai.utils import get_act_layer, get_conv_layer, get_reshaping_layer, parse_model_inputs

from .transunet_layers import (
    CoarseToFineAttention,
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
        num_encoder_layers=12,
        num_decoder_layers=12,
        num_heads=8,
        embed_dim=256,
        mlp_dim=1024,
        dropout_rate=0.1,
        decoder_projection_filters=64,
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
            use_class_token=True,
            use_patch_bias=True,
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

        # The first token is the class token (0), the rest are the spatial patches (1).
        patch_tokens = encoder_output[:, 1:, :]  # spatial tokens without class token

        # Transformer Decoder
        decoder_output = patch_tokens
        for i in range(num_decoder_layers):
            decoder_output = TransUNetDecoderBlock(
                embed_dim, num_heads, mlp_dim, dropout_rate, name=f"transunet_decoder_block_{i+1}"
            )([decoder_output, patch_tokens])

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

        # CNN Decoder : Create learnable positional queries
        target_shape = self.get_target_spatial_shape(input_shape, downsampling_factor=8)
        projected_features = layers.Dense(
            decoder_projection_filters,
            kernel_initializer=HeNormal(),
            name="decoder_projection_dense",
        )(z1)
        spatial_features = layers.Reshape(
            target_shape + [decoder_projection_filters], name="decoder_reshape"
        )(projected_features)

        # Decoder stages fuse with encoder features of corresponding resolutions:
        # - d1 (lowest resolution) ↔ c1 (deepest features, lowest resolution)
        # - d2 (medium resolution) ↔ c2 (mid-level features, medium resolution)
        # - d3 (highest resolution) ↔ c3 (shallowest features, highest resolution)
        spatial_dims_val = len(spatial_features.shape[1:-1])
        decoder_filters = [
            decoder_projection_filters * 2,  # d1 filters (e.g., 128 if projection=64)
            decoder_projection_filters,  # d2 filters (e.g., 64 if projection=64)
            decoder_projection_filters // 2,  # d3 filters (e.g., 32 if projection=64)
        ]
        d1 = self.build_decoder(decoder_filters[0], spatial_dims_val, "decoder_stage1")(
            spatial_features, c1
        )
        d2 = self.build_decoder(decoder_filters[1], spatial_dims_val, "decoder_stage2")(d1, c2)
        d3 = self.build_decoder(decoder_filters[2], spatial_dims_val, "decoder_stage3")(d2, c3)

        # Final output
        outputs = get_conv_layer(
            spatial_dims=len(d3.shape[1:-1]),
            layer_type="conv",
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=HeNormal(),
            activation=classifier_activation,
            dtype="float32",
        )(d3)

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or f"TransUNet{spatial_dims}D", **kwargs
        )

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.classifier_activation = classifier_activation
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.decoder_projection_filters = decoder_projection_filters

    def build_decoder(self, filters, spatial_dims, stage_name):
        def apply(input_tensor, skip_tensor):
            # Upsample previous decoder stage features using the CUP approach
            # The Cascaded Up-sampling (CUP) block consists of UpSampling followed by a ConvND and ReLU
            x = get_reshaping_layer(
                spatial_dims=spatial_dims,
                layer_type="upsampling",
                size=2,
                name=f"{stage_name}_upsample",
            )(input_tensor)
            x = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=filters,
                kernel_size=3,
                activation="relu",
                padding="same",
                name=f"{stage_name}_conv_upsample",
            )(x)
            x = layers.BatchNormalization(name=f"{stage_name}_conv_upsample_bn")(x)
            x = get_act_layer(name="relu")(x)

            # Upsample the skip_tensor to match the new shape of x
            # This is the same fix as before, ensuring shapes match
            upsample_factor = tuple(
                s_out // s_in for s_out, s_in in zip(x.shape[1:-1], skip_tensor.shape[1:-1])
            )
            upsampled_skip = get_reshaping_layer(
                spatial_dims=spatial_dims,
                layer_type="upsampling",
                size=upsample_factor,
                name=f"{stage_name}_upsample_skip",
            )(skip_tensor)

            # Concatenate with upsampled skip connection
            x = layers.Concatenate(axis=-1, name=f"{stage_name}_concat")([x, upsampled_skip])

            # Two standard convolutional blocks to process concatenated features
            for i in range(2):
                x = get_conv_layer(
                    spatial_dims=spatial_dims,
                    layer_type="conv",
                    filters=filters,
                    kernel_size=3,
                    padding="same",
                    name=f"{stage_name}_conv{i + 1}",
                )(x)
                x = layers.BatchNormalization(name=f"{stage_name}_conv{i + 1}_bn")(x)
                x = get_act_layer(name="relu")(x)
            return x

        return apply

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
            "dropout_rate": self.dropout_rate,
            "decoder_projection_filters": self.decoder_projection_filters,
        }
        return config

    @staticmethod
    def get_target_spatial_shape(input_shape, downsampling_factor=8):
        spatial_dims = len(input_shape) - 1  # For 2D or 3D
        target_shape = []
        for i in range(spatial_dims):
            dim = input_shape[i]
            if dim is None:
                raise ValueError(
                    f"Spatial dimension {i} of input_shape cannot be None for TransUNet, but got {dim}. "
                    f"It's required to determine the number of positional queries."
                )
            if dim % downsampling_factor != 0:
                raise ValueError(
                    f"Input spatial dimension {i} ({dim}) is not divisible by "
                    f"downsampling_factor ({downsampling_factor})."
                )
            target_shape.append(dim // downsampling_factor)
        return target_shape
