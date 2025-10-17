import keras
from keras import layers, ops

from medicai.layers import ResizingND
from medicai.models.vit.vit_layers import ViTEncoderBlock, ViTPatchingAndEmbedding
from medicai.utils import (
    VALID_ACTIVATION_LIST,
    DescribeMixin,
    get_act_layer,
    get_conv_layer,
    registration,
    resolve_encoder,
)

from .transunet_layers import LearnableQueries, MaskedCrossAttention


@keras.saving.register_keras_serializable(package="transunet")
@registration.register(name="trans_unet", type="segmentation")
class TransUNet(keras.Model, DescribeMixin):
    """3D or 2D TransUNet model for medical image segmentation.

    This model combines a 3D or 2D CNN encoder with a Vision Transformer
    (ViT) encoder and a hybrid decoder. The CNN extracts multi-scale local features,
    while the ViT captures global context. The decoder upsamples the fused
    features to produce the final segmentation map using a coarse-to-fine
    attention mechanism and U-Net-style skip connections.

    Example:
    >>> from medicai.models import TransUNet
    >>> model = TransUNet(input_shape=(96, 96, 1), encoder_name="densenet121")
    >>> model = TransUNet(input_shape=(96, 96, 96, 1), encoder_name="densenet121")
    """

    ALLOWED_BACKBONE_FAMILIES = ["densenet", "resnet", "efficientnet"]

    def __init__(
        self,
        *,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        encoder_depth=5,
        num_classes=1,
        classifier_activation=None,
        num_vit_layers=12,
        num_heads=8,
        num_queries=100,
        embed_dim=512,
        mlp_dim=1024,
        dropout_rate=0.1,
        decoder_activation="leaky_relu",
        decoder_filters=(256, 128, 64, 32, 16),
        name=None,
        **kwargs,
    ):
        """
        Initializes the TransUNet model.

        Args:
            encoder: (Optional) A Keras model to use as the encoder (backbone).
                This argument is intended for passing a custom or pre-trained
                model. If provided, the model must have a `pyramid_outputs` attribute,
                which should be a dictionary of intermediate feature vectors from shallow
                to deep layers (e.g., `'P1'`, `'P2'`, ...).
            encoder_name: (Optional) A string specifying the name of a
                pre-configured backbone from the `medicai.models.list_models()` to use as
                the encoder. This is a convenient option for using a backbone from
                the library without having to instantiate it manually.
            encoder_depth: An integer specifying how many stages of the encoder
                backbone to use. A number of stages used in encoder in range [3, 5].
                Expected available intermediate or pyramid level, P1, P2, ... P5.
                If `encoder_depth=5`, bottleneck key would be P5, and P4...P1 will
                be used for skip connection. If `encoder_depth=4`, bottleneck key
                would be P4, and P3..P1 will be used for skip connection.
                The `encoder_depth` should be in [3, 4, 5]. This can be used to
                reduce the size of the model. Default: 5.
            input_shape (tuple): The shape of the input data. For 2D, it is
                `(height, width, channels)`. For 3D, it is `(depth, height, width, channels)`.
            num_classes (int): The number of segmentation classes.
            num_queries (int, optional): The number of learnable queries used in the
                decoder's attention mechanism. Defaults to 100.
            classifier_activation (str, optional): Activation function for the final
                segmentation head (e.g., 'sigmoid' for binary, 'softmax' for multi-class).
            num_vit_layers (int, optional): The number of transformer encoder blocks
                in the ViT encoder. Defaults to 12.
            num_heads (int, optional): The number of attention heads in the transformer blocks.
                Defaults to 8.
            embed_dim (int, optional): The dimensionality of the token embeddings.
                Defaults to 512.
            mlp_dim (int, optional): The hidden dimension of the MLP in the transformer
                blocks. Defaults to 1024.
            dropout_rate (float, optional): The dropout rate for regularization.
                Defaults to 0.1.
            decoder_activation (str): Controls the use of activation layers in decoder blocks.
                It should the activation string identifier in available in keras.
                Default: 'leaky_relu'
            decoder_filters: The number of filters for the convolutional layers in the
                decoder upsampling path. The number of filters should correspond to
                the `encoder_depth`. Default: [256, 128, 64, 32, 16]
            name (str, optional): The name of the model. Defaults to `TransUNetND`.
        """
        encoder, input_shape = resolve_encoder(
            encoder=encoder,
            encoder_name=encoder_name,
            input_shape=input_shape,
            allowed_families=TransUNet.ALLOWED_BACKBONE_FAMILIES,
        )

        spatial_dims = len(input_shape) - 1

        # Get CNN feature maps from the encoder.
        inputs = encoder.input
        pyramid_outputs = encoder.pyramid_outputs

        # Determine required pyramid levels dynamically
        required_keys = {f"P{i}" for i in range(1, encoder_depth + 1)}
        available_keys = set(pyramid_outputs.keys())

        # Find missing ones
        missing_keys = required_keys - available_keys
        if missing_keys:
            raise ValueError(
                f"The encoder's `pyramid_outputs` is missing required pyramid levels. "
                f"Missing: {missing_keys}. "
                f"Expected keys (based on encoder_depth={encoder_depth}): {required_keys}, "
                f"but got: {available_keys}"
            )

        if not (3 <= encoder_depth <= 5):
            raise ValueError(f"encoder_depth must be in range [3, 5], but got {encoder_depth}")

        if len(decoder_filters) < encoder_depth:
            raise ValueError(
                f"Length of decoder_filters ({len(decoder_filters)}) must be >= encoder_depth ({encoder_depth})."
            )

        if isinstance(decoder_activation, str):
            decoder_activation = decoder_activation.lower()

        if decoder_activation not in VALID_ACTIVATION_LIST:
            raise ValueError(
                f"Invalid value for `decoder_activation`: {decoder_activation!r}. "
                f"Supported values are: {VALID_ACTIVATION_LIST}"
            )

        # prepare head and skip layers
        sorted_keys = sorted(required_keys, key=lambda x: int(x[1:]))
        final_cnn_feature = pyramid_outputs[sorted_keys[-1]]
        cnn_features = [pyramid_outputs[key] for key in sorted_keys[:-1]]
        decoder_filters = decoder_filters[:encoder_depth]

        # Compute adaptive patch size based on the last CNN feature map
        # Use patch size = 1 if feature map is small, else divide it
        feature_shape = final_cnn_feature.shape[1:-1]  # e.g., (3, 3, 3) or (6, 6, 6)
        if any(s is None for s in feature_shape):
            patch_size = (1,) * len(feature_shape)
        else:
            min_dim = min(feature_shape)
            patch_size = (
                (1,) * len(feature_shape)
                if min_dim <= 4
                else tuple(max(1, s // 2) for s in feature_shape)
            )

        # Transformer Encoder
        encoded_patches = ViTPatchingAndEmbedding(
            image_size=final_cnn_feature.shape[1:-1],
            patch_size=patch_size,
            hidden_dim=embed_dim,
            num_channels=final_cnn_feature.shape[-1],
            use_class_token=False,
            use_patch_bias=True,
            name="transunet_vit_patching_and_embedding",
        )(final_cnn_feature)

        encoder_output = encoded_patches
        for i in range(num_vit_layers):
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
            final_cnn_feature=final_cnn_feature,
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            num_queries=num_queries,
            spatial_dims=spatial_dims,
            decoder_filters=decoder_filters,
            decoder_activation=decoder_activation,
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
        self.encoder_depth = encoder_depth
        self.classifier_activation = classifier_activation
        self.num_vit_layers = num_vit_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.decoder_activation = decoder_activation
        self.decoder_filters = decoder_filters

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "num_classes": self.num_classes,
            "num_queries": self.num_queries,
            "encoder_name": self.encoder_name,
            "encoder_depth": self.encoder_depth,
            "classifier_activation": self.classifier_activation,
            "num_vit_layers": self.num_vit_layers,
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
            "decoder_activation": self.decoder_activation,
            "decoder_filters": self.decoder_filters,
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
        final_cnn_feature,
        num_heads,
        embed_dim,
        num_classes,
        num_queries,
        mlp_dim,
        dropout_rate,
        spatial_dims,
        decoder_filters,
        decoder_activation,
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
        # Project the intermediate CNN features to the
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
        # This loop iterates through the CNN feature maps in reverse order (P4 -> P3 -> P2 -> P1),
        # refining the queries and mask at each step.
        for t, cnn_feature in enumerate(reversed(projected_features)):  # [P4, P3, P2, P1]
            level_name = f"level_{t}"  # level_0 (P4), level_1 (P3), level_2 (P2), level_3 (P1)

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
        final_output = self.combine_masks_and_classes(final_mask, class_logits, final_cnn_feature)

        # Step 7: U-Net style upsampling path
        # This final path uses standard convolutions and upsampling to refine the
        # combined output and leverage the CNN skip connections [P1, P2, P3, P4].
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=decoder_filters[0],
            kernel_size=3,
            padding="same",
            name="decoder_proj_0",
        )(final_output)

        # Iterate from deepest skip (last element) to shallowest (first)
        for i, (skip, filters) in enumerate(
            zip(reversed(cnn_features), decoder_filters[1:]), start=1
        ):
            pyramid_level = len(cnn_features) - i + 1
            x = ResizingND(
                scale_factor=2,
                interpolation="bilinear" if spatial_dims == 2 else "trilinear",
                name=f"upsample_to_p{pyramid_level}",
            )(x)
            x = layers.Concatenate(axis=-1, name=f"concat_with_p{pyramid_level}")([x, skip])
            x = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=filters,
                kernel_size=3,
                strides=1,
                padding="same",
                activation=None,
                name=f"decoder_conv_{pyramid_level}",
            )(x)
            x = get_act_layer(layer_type=decoder_activation, name=f"decoder_act_{pyramid_level}")(x)

        # Final upsample to restore full resolution
        x = ResizingND(
            scale_factor=2,
            interpolation="bilinear" if spatial_dims == 2 else "trilinear",
            name="final_upsample",
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
