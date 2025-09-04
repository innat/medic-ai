import keras
import numpy as np
from keras import layers

from medicai.layers import ViTEncoderBlock, ViTPatchingAndEmbedding
from medicai.models import DenseNetBackbone
from medicai.utils import get_conv_layer, parse_model_inputs

from .transunet_layers import LearnableQueries, MaskedCrossAttention, QueryRefinementBlock


class TransUNet(keras.Model):
    """3D or 2D TransUNet model for medical image segmentation.

    This model combines a 3D or 2D CNN encoder (DenseNet) with a Vision Transformer
    (ViT) encoder and a hybrid decoder. The CNN extracts multi-scale local features,
    while the ViT captures global context. The decoder refines the features using
    a coarse-to-fine attention mechanism to produce the final segmentation map.

    Args:
        input_shape: The shape of the input data, e.g., `(depth, height, width, channels)`.
        num_classes: The number of segmentation classes.
        patch_size: The size of the patches for the Vision Transformer.
        classifier_activation: Activation function for the final segmentation head
            (e.g., 'sigmoid' for binary, 'softmax' for multi-class).
        num_encoder_layers: The number of transformer encoder blocks (default: 12).
        num_heads: The number of attention heads in the transformer blocks (default: 8).
        embed_dim: The dimensionality of the token embeddings (default: 256).
        mlp_dim: The hidden dimension of the MLP in the transformer blocks (default: 1024).
        dropout_rate: The dropout rate for regularization (default: 0.1).
        decoder_projection_filters: The number of filters for the final CNN
            upsampling layers in the decoder (default: 64).
        name: The name of the model (default: "TransUNetND").
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        patch_size,
        classifier_activation=None,
        num_encoder_layers=12,
        num_heads=8,
        embed_dim=256,
        mlp_dim=1024,
        dropout_rate=0.1,
        decoder_projection_filters=64,
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1

        if isinstance(patch_size, int):
            patch_size = (patch_size,) * spatial_dims
        elif isinstance(patch_size, (list, tuple)) and len(patch_size) != spatial_dims:
            raise ValueError(
                f"patch_size must have length {spatial_dims} for {spatial_dims}D input. "
                f"Got {patch_size} with length {len(patch_size)}"
            )

        inputs = parse_model_inputs(input_shape=input_shape, name="transunet_input")

        # -------------------- CNN Encoder --------------------
        base_encoder = DenseNetBackbone(blocks=[6, 12, 24, 16], input_tensor=inputs)

        # In a real model, get layers by name. Using a dummy index for this example.
        c1 = base_encoder.get_layer(index=49).output  # (24, 24, 24, 128)
        c2 = base_encoder.get_layer(index=137).output  # (12, 12, 12, 256)
        c3 = base_encoder.get_layer(index=309).output  # (6, 6, 6, 512)
        final_cnn_output = base_encoder.output  # (3, 3, 3, 1024)

        cnn_features = [c1, c2, c3]

        # -------------------- Transformer Encoder --------------------
        encoded_patches = ViTPatchingAndEmbedding(
            image_size=final_cnn_output.shape[1:-1],
            patch_size=patch_size,
            hidden_dim=embed_dim,
            num_channels=final_cnn_output.shape[-1],
            use_class_token=False,
            use_patch_bias=True,
            name="transunet_vit_patching_and_embedding",
        )(final_cnn_output)

        encoder_output = encoded_patches
        for i in range(num_encoder_layers):
            encoder_output = ViTEncoderBlock(
                num_heads=num_heads,
                hidden_dim=embed_dim,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                name=f"transunet_vit_encoder_block_{i + 1}",
            )(encoder_output)

        # -------------------- Decoder --------------------
        outputs = self.build_decoder(
            encoder_output=encoder_output,
            cnn_features=cnn_features,
            final_cnn_output=final_cnn_output,
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            decoder_projection_filters=decoder_projection_filters,
            num_classes=num_classes,
            classifier_activation=classifier_activation,
            spatial_dims=spatial_dims,
        )

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or f"TransUNet{spatial_dims}D", **kwargs
        )

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.classifier_activation = classifier_activation
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.decoder_projection_filters = decoder_projection_filters

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "patch_size": self.patch_size,
            "num_encoder_layers": self.num_encoder_layers,
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
            "decoder_projection_filters": self.decoder_projection_filters,
        }
        return config

    def build_decoder(
        self,
        encoder_output,
        cnn_features,
        final_cnn_output,
        num_heads,
        embed_dim,
        mlp_dim,
        dropout_rate,
        decoder_projection_filters,
        num_classes,
        classifier_activation,
        spatial_dims,
    ):
        # -------------------- 1. Learnable Queries & Refinement (P-Blocks) --------------------
        spatial_shape_c3 = cnn_features[-1].shape[1:-1]
        num_queries = int(np.prod(spatial_shape_c3))

        # Initialize P0
        p_queries = LearnableQueries(
            num_queries=num_queries, embed_dim=embed_dim, name="initial_learnable_queries"
        )(final_cnn_output)

        # Refine queries by attending to the encoder output.
        # This creates P1, P2, P3...
        refined_p_queries = [p_queries]
        for i in range(len(cnn_features)):
            # This is the "P-block" from the diagram
            p_queries = QueryRefinementBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                name=f"p_refinement_attention_{i+1}",
            )([p_queries, encoder_output])

            refined_p_queries.append(p_queries)

        # -------------------- 2. Coarse-to-fine Attention (Z-Blocks) --------------------

        # Project all CNN features to the `embed_dim` for attention
        proj_cnn_features = [
            get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=embed_dim,
                kernel_size=1,
                name=f"proj_cnn_{i}",
            )(f)
            for i, f in enumerate(cnn_features)
        ]

        # Start the Z-path with the deepest features (c3) and the most refined queries (P3).
        # The output of this attention block is the first Z-token set (Z3).
        z_tokens = MaskedCrossAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout_rate=dropout_rate,
            name="coarse_to_fine_stage_c3",
        )(
            query=refined_p_queries[-1],
            key=self.flatten_spatial_to_tokens(proj_cnn_features[-1]),
            value=self.flatten_spatial_to_tokens(proj_cnn_features[-1]),
        )

        # Now, progressively upsample the Z tokens and fuse them with the next level of CNN features.
        # The loop runs from i=2 down to 1.
        for i in range(len(cnn_features) - 1, 0, -1):
            # Reshape Z tokens to the current spatial grid
            z_spatial = self.reshape_to_spatial_tokens(z_tokens, proj_cnn_features[i])

            # Upsample the spatial grid using Conv3DTranspose
            z_upsampled_spatial = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv_transpose",
                filters=embed_dim,
                kernel_size=2,
                strides=2,
                padding="same",
                name=f"z_upsample_{i}",
            )(z_spatial)

            # Flatten the upsampled grid back to tokens. The number of tokens increases.
            z_upsampled_tokens = self.flatten_spatial_to_tokens(z_upsampled_spatial)

            # Get the tokens for the next CNN feature map
            cnn_tokens = self.flatten_spatial_to_tokens(proj_cnn_features[i - 1])

            # Fuse the upsampled Z tokens with the next projected CNN features using attention.
            # The upsampled Z tokens are the query, and the next CNN features are key/value.
            z_tokens = MaskedCrossAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                dropout_rate=dropout_rate,
                name=f"coarse_to_fine_stage_c{i}",
            )(query=z_upsampled_tokens, key=cnn_tokens, value=cnn_tokens)

        # -------------------- 3. Final CNN Decoder Path --------------------
        # `z_tokens` is now at the resolution of c1. We convert it to a spatial map.
        x = self.reshape_to_spatial_tokens(z_tokens, cnn_features[0])

        # Perform the final upsampling stages.
        final_decoder_filters = [decoder_projection_filters * 2, decoder_projection_filters]

        for i, f in enumerate(final_decoder_filters):
            x = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv_transpose",
                filters=f,
                kernel_size=2,
                strides=2,
                padding="same",
                name=f"final_upsample_conv_{i}",
            )(x)
            x = keras.layers.BatchNormalization(name=f"final_upsample_bn_{i}")(x)
            x = keras.layers.Activation("relu", name=f"final_upsample_relu_{i}")(x)

        # Final segmentation head
        outputs = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=num_classes,
            kernel_size=1,
            padding="same",
            activation=classifier_activation,
            name="final_output_conv",
        )(x)
        return outputs

    @staticmethod
    def flatten_spatial_to_tokens(x):
        num_tokens = np.prod(x.shape[1:-1])
        return layers.Reshape((num_tokens, x.shape[-1]))(x)

    @staticmethod
    def reshape_to_spatial_tokens(x, target_tensor):
        target_spatial_shape = target_tensor.shape[1:-1]
        return layers.Reshape(target_spatial_shape + (x.shape[-1],))(x)
