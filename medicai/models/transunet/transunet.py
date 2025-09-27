import keras
from keras import layers, ops

from medicai.layers import ViTEncoderBlock, ViTPatchingAndEmbedding
from medicai.utils import DescribeMixin, get_conv_layer, get_reshaping_layer, resolve_encoder

from .transunet_layers import LearnableQueries, MaskedCrossAttention


@keras.saving.register_keras_serializable(package="transunet")
class TransUNet(keras.Model, DescribeMixin):
    """3D or 2D TransUNet model for medical image segmentation.

    This model combines a 3D or 2D CNN encoder (DenseNet) with a Vision Transformer
    (ViT) encoder and a hybrid decoder. The CNN extracts multi-scale local features,
    while the ViT captures global context. The decoder upsamples the fused
    features to produce the final segmentation map using a coarse-to-fine
    attention mechanism and U-Net-style skip connections.

    Example:
    >>> from medicai.models import TransUNet
    >>> model = TransUNet(input_shape=(96, 96, 1), encoder_name="densenet121")
    >>> model = TransUNet(input_shape=(96, 96, 96, 1), encoder_name="densenet121")
    """

    ALLOWED_BACKBONE_FAMILIES = ["densenet", "resnet"]

    def __init__(
        self,
        *,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        num_classes=1,
        patch_size=3,
        classifier_activation=None,
        num_encoder_layers=6,
        num_heads=8,
        num_queries=100,
        embed_dim=256,
        mlp_dim=1024,
        dropout_rate=0.1,
        decoder_projection_filters=64,
        name=None,
        **kwargs,
    ):
        """
        Initializes the TransUNet model.

        Args:
            encoder: (Optional) A Keras model to use as the encoder (backbone).
                This argument is intended for passing a custom or pre-trained
                model not available in the `BACKBONE_ZOO`. If provided, the
                model must have a `pyramid_outputs` attribute, which should be
                a dictionary of intermediate feature vectors from shallow to
                deep layers (e.g., `'P1'`, `'P2'`, ...).
            encoder_name: (Optional) A string specifying the name of a
                pre-configured backbone from the `BACKBONE_ZOO` to use as the
                encoder. This is a convenient option for using a backbone from
                the library without having to instantiate it manually.
            input_shape (tuple): The shape of the input data. For 2D, it is
                `(height, width, channels)`. For 3D, it is `(depth, height, width, channels)`.
            num_classes (int): The number of segmentation classes.
            patch_size (int or tuple): The size of the patches for the Vision
                Transformer. Must be a tuple of length `spatial_dims`. Defaults to 3.
            num_queries (int, optional): The number of learnable queries used in the
                decoder's attention mechanism. Defaults to 100.
            classifier_activation (str, optional): Activation function for the final
                segmentation head (e.g., 'sigmoid' for binary, 'softmax' for multi-class).
            num_encoder_layers (int, optional): The number of transformer encoder blocks
                in the ViT encoder. Defaults to 6.
            num_heads (int, optional): The number of attention heads in the transformer blocks.
                Defaults to 8.
            embed_dim (int, optional): The dimensionality of the token embeddings.
                Defaults to 256.
            mlp_dim (int, optional): The hidden dimension of the MLP in the transformer
                blocks. Defaults to 1024.
            dropout_rate (float, optional): The dropout rate for regularization.
                Defaults to 0.1.
            decoder_projection_filters (int, optional): The number of filters for the
                convolutional layers in the decoder upsampling path. Defaults to 64.
            name (str, optional): The name of the model. Defaults to `TransUNetND`.
        """
        encoder, input_shape = resolve_encoder(
            encoder=encoder,
            encoder_name=encoder_name,
            input_shape=input_shape,
            allowed_families=TransUNet.ALLOWED_BACKBONE_FAMILIES,
        )

        spatial_dims = len(input_shape) - 1

        if isinstance(patch_size, int):
            patch_size = (patch_size,) * spatial_dims
        elif isinstance(patch_size, (list, tuple)) and len(patch_size) != spatial_dims:
            raise ValueError(
                f"patch_size must have length {spatial_dims} for {spatial_dims}D input. "
                f"Got {patch_size} with length {len(patch_size)}"
            )

        # Get CNN feature maps from the encoder.
        inputs = encoder.input
        pyramid_outputs = encoder.pyramid_outputs
        required_keys = {"P1", "P2", "P3", "P4", "P5"}
        if not required_keys.issubset(pyramid_outputs.keys()):
            raise ValueError(
                f"The backbone's `pyramid_outputs` is missing one or more required keys. "
                f"Required: {required_keys}, Available: {set(pyramid_outputs.keys())}"
            )
        c1 = pyramid_outputs.get("P1")
        c2 = pyramid_outputs.get("P2")
        c3 = pyramid_outputs.get("P3")
        final_cnn_output = pyramid_outputs.get("P4")
        cnn_features = [c1, c2, c3]  # shallow to deep

        # Transformer Encoder
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

        # Decoder
        outputs = self.build_decoder(
            encoder_output=encoder_output,
            cnn_features=cnn_features,
            final_cnn_output=final_cnn_output,
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            num_queries=num_queries,
            spatial_dims=spatial_dims,
            decoder_projection_filters=decoder_projection_filters,
        )

        outputs = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=num_classes,
            kernel_size=1,
            activation=classifier_activation,
            dtype="float32",
            name="final_conv",
        )(outputs)

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or f"TransUNet{spatial_dims}D", **kwargs
        )

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.encoder_name = encoder_name
        self.encoder = encoder
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
            "num_queries": self.num_queries,
            "encoder_name": self.encoder_name,
            "classifier_activation": self.classifier_activation,
            "patch_size": self.patch_size,
            "num_encoder_layers": self.num_encoder_layers,
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
            "decoder_projection_filters": self.decoder_projection_filters,
        }
        if self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})
        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)

    def build_decoder(
        self,
        encoder_output,
        cnn_features,
        final_cnn_output,
        num_heads,
        embed_dim,
        num_classes,
        num_queries,
        mlp_dim,
        dropout_rate,
        spatial_dims,
        decoder_projection_filters,
    ):
        """
        Builds the hybrid decoder, which consists of a transformer-based
        refinement loop and a U-Net style upsampling path.
        """

        # Step 1: Initialize Learnable Queries.
        # These queries act as "segmentation concepts" and will be refined over
        # the course of the decoder.
        current_queries = LearnableQueries(num_queries, embed_dim)(
            encoder_output
        )  # Initial queries

        # Step 2: Project CNN features for decoder skip connections
        # Project the 3 intermediate CNN features (C1, C2, C3) to the
        # transformer's embedding dimension.
        projected_features = []
        for i, feat in enumerate(cnn_features):
            proj_feat = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=embed_dim,
                kernel_size=1,
                name=f"proj_cnn_{i}",
            )(feat)
            projected_features.append(proj_feat)

        # Step 3: Initial Coarse Prediction
        # Generate an initial coarse mask prediction by performing a dot product
        # between the learnable queries (F) and the global transformer context (E).
        # Paper notation: M_0 = F^T * E
        initial_coarse_logits = layers.Dot(axes=(2, 2))([current_queries, encoder_output])
        current_coarse_mask = layers.Activation("sigmoid", name="initial_coarse_mask")(
            initial_coarse_logits
        )

        # Step 4: Coarse-to-Fine Refinement Loop
        # This loop iterates through the CNN feature maps in reverse order (C3 -> C2 -> C1),
        # refining the queries and mask at each step.
        for t, cnn_feature in enumerate(reversed(projected_features)):  # [C3, C2, C1]
            level_name = f"level_{t}"  # level_0 (C3), level_1 (C2), level_2 (C1)

            # Step 4a: Prepare CNN features for cross-attention
            # Flatten the spatial dimensions of the CNN feature map into a sequence of tokens.
            P_t = self.flatten_spatial_to_tokens(cnn_feature, embed_dim=embed_dim)
            current_num_tokens = ops.shape(P_t)[1]

            # Step 4b: Create an attention mask from the previous coarse prediction
            # The previous coarse mask (M_(t-1)) is resized to match the number of tokens
            # in the current CNN feature map (P_t). This mask guides the cross-attention
            resized_mask = layers.Resizing(
                height=ops.shape(current_coarse_mask)[1],
                width=current_num_tokens,
                interpolation="nearest",
                name=f"mask_resize_{level_name}",
            )(current_coarse_mask[..., None])
            resized_mask = ops.squeeze(resized_mask, axis=-1)

            # The mask is converted to a format suitable for Keras's MultiHeadAttention,
            # where large negative values effectively "mask out" tokens.
            attention_mask = layers.Lambda(
                lambda x: ops.where(x > 0.5, 0.0, -1e9), name=f"attention_mask_{level_name}"
            )(resized_mask)

            # Step 4c: Masked Cross-Attention
            # The current queries (F_t) perform a masked cross-attention on the CNN feature tokens (P_t).
            # This refines the queries by incorporating local, fine-grained information.
            refined_queries = MaskedCrossAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                name=f"masked_xattn_{level_name}",
            )([current_queries, P_t, attention_mask])

            # Step 4d: Update current queries and coarse mask
            # The queries are updated with the refined output from the cross-attention.
            current_queries = refined_queries

            # A new, more refined coarse mask (M_t) is generated from the updated queries
            # and the global transformer context (E).
            current_coarse_logits = layers.Dot(axes=(2, 2))([current_queries, encoder_output])
            current_coarse_mask = layers.Activation("sigmoid", name=f"coarse_mask_{level_name}")(
                current_coarse_logits
            )

        # Step 5: Final Predictions from Refined Queries
        # After the refinement loop, the final queries are used to produce two outputs:
        # a class prediction for each query, and a final mask prediction.

        # Class prediction: maps each query to a class probability.
        # Paper notation: Class logits
        class_logits = layers.Dense(num_classes, name="class_prediction")(current_queries)

        # Final mask prediction: maps each query to a spatial mask.
        final_mask_logits = layers.Dot(axes=(2, 2))(
            [current_queries, encoder_output]  # Ground in global context
        )
        final_mask = layers.Activation("sigmoid", name="final_mask")(final_mask_logits)

        # Step 6: Combine Masks with Class Predictions
        # The final segmentation map is produced by combining the per-query class
        # predictions with their corresponding spatial masks.
        final_output = self.combine_masks_and_classes(final_mask, class_logits, final_cnn_output)

        # Step 7: U-Net style upsampling path
        # This final path uses standard convolutions and upsampling to refine the
        # combined output and leverage the CNN skip connections (C1, C2, C3).
        c1, c2, c3 = cnn_features

        # Step 7a: UpSampling and Concatenation with C3
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=decoder_projection_filters,
            kernel_size=3,
            padding="same",
            name="decoder_proj_0",
        )(final_output)
        x = get_reshaping_layer(
            spatial_dims=spatial_dims, layer_type="upsampling", size=2, name="upsample_to_c3"
        )(x)
        x = layers.Concatenate(axis=-1, name="concat_with_c3")([x, c3])
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=decoder_projection_filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            name="decoder_conv_1",
        )(x)

        # Step 7b: UpSampling and Concatenation with C2
        x = get_reshaping_layer(
            spatial_dims=spatial_dims, layer_type="upsampling", size=2, name="upsample_to_c2"
        )(x)
        x = layers.Concatenate(axis=-1, name="concat_with_c2")([x, c2])
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=decoder_projection_filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            name="decoder_conv_2",
        )(x)

        # Step 7c: UpSampling and Concatenation with C1
        x = get_reshaping_layer(
            spatial_dims=spatial_dims, layer_type="upsampling", size=2, name="upsample_to_c1"
        )(x)
        x = layers.Concatenate(axis=-1, name="concat_with_c1")([x, c1])
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=decoder_projection_filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            name="decoder_conv_3",
        )(x)

        # Final upsampling to the original input resolution
        x = get_reshaping_layer(
            spatial_dims=spatial_dims, layer_type="upsampling", size=2, name="final_upsample"
        )(x)

        return x

    @staticmethod
    def flatten_spatial_to_tokens(x, embed_dim=None):
        """Flatten spatial dims to tokens (B, N, C)."""
        if embed_dim is None:
            embed_dim = x.shape[-1]
        return layers.Reshape((-1, embed_dim))(x)

    def combine_masks_and_classes(self, mask_predictions, class_predictions, spatial_reference):
        """
        Combine mask predictions and class predictions to produce final segmentation output.

        This method handles both 2D and 3D inputs by dynamically adjusting the spatial dimensions
        and einsum operation based on the input shape.

        Args:
            mask_predictions: Tensor of shape (batch_size, num_queries, num_spatial_tokens)
                containing mask logits for each query.
            class_predictions: Tensor of shape (batch_size, num_queries, num_classes)
                containing class logits for each query.
            spatial_reference: Tensor with target spatial shape used for reference.

        Returns:
            final_output: Tensor of shape (batch_size, *spatial_dims, num_classes)
                containing the final segmentation probabilities.
        """
        num_queries = ops.shape(mask_predictions)[1]
        num_classes = ops.shape(class_predictions)[2]

        # Get target spatial shape from reference tensor
        spatial_shape = ops.shape(spatial_reference)[1:-1]  # Exclude batch and channel dimensions
        spatial_dims = len(spatial_shape)  # 2 for 2D, 3 for 3D

        target_elements = int(ops.prod(spatial_shape))  # Total spatial elements (H*W or D*H*W)
        actual_elements = ops.shape(mask_predictions)[2]

        # Reshape mask predictions to match target spatial dimensions
        if actual_elements != target_elements:
            # Resize mask predictions to match target using a dense layer
            mask_predictions = layers.Dense(target_elements, name="mask_resize")(mask_predictions)

        # Reshape to spatial dimensions: (batch, queries, *spatial_dims)
        mask_spatial = layers.Reshape(
            (num_queries,) + tuple(spatial_shape), name="reshape_masks_spatial"
        )(mask_predictions)

        # Get class probabilities using softmax
        class_probs = layers.Softmax(axis=-1, name="class_probabilities")(class_predictions)

        # Combine masks with class probabilities using einsum
        # The einsum pattern depends on whether we're working with 2D or 3D data
        einsum_pattern = (
            f"bi{'d' if spatial_dims == 3 else ''}hw,bik->b{'d' if spatial_dims == 3 else ''}hwk"
        )

        final_output = layers.Lambda(
            lambda inputs: ops.einsum(einsum_pattern, inputs[0], inputs[1]),
            output_shape=lambda input_shape: (
                input_shape[0][0],  # batch size
                *tuple(spatial_shape),  # spatial dimensions (D,H,W or H,W)
                num_classes,  # number of classes
            ),
            name=f"combine_masks_classes_{spatial_dims}d",
        )([mask_spatial, class_probs])

        return final_output
