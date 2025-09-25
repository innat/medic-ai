import keras
from keras import layers, ops

from medicai.utils import (
    get_act_layer,
    get_conv_layer,
    get_norm_layer,
    resize_volumes,
)

from ..encoder_utils import resolve_encoder


@keras.saving.register_keras_serializable(package="segformer")
class SegFormer(keras.Model):
    """SegFormer model for 2D or 3D semantic segmentation.

    This class implements the full SegFormer architecture, which combines a
    hierarchical MixVisionTransformer (MiT) encoder with a lightweight MLP decoder
    head. This design is highly efficient for semantic segmentation tasks on
    high-resolution images or volumes.

    The encoder (MiT) progressively downsamples the spatial dimensions and increases the
    feature dimensions across four stages, producing multi-scale feature maps (P1, P2, P3, P4).
    The decoder then takes these features, processes them through a linear layer and upsampling
    to a common resolution (P1's resolution), fuses them via concatenation and a convolution,
    and finally generates a high-resolution segmentation mask matching the input size.

    Example:
    >>> from medicai.models import SegFormer, MixViTB0
    >>> # 1. Initialize with an encoder class (e.g., MiT-B0)
    >>> model = SegFormer(encoder=MixViTB0(input_shape=(256, 256, 3), include_top=False), num_classes=2)
    >>> # 2. Initialize using a registered name
    >>> model = SegFormer(encoder_name='mit_b2', input_shape=(128, 128, 1), num_classes=5)
    """

    ALLOWED_BACKBONE_FAMILIES = ["mit"]

    def __init__(
        self,
        *,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        num_classes=1,
        classifier_activation=None,
        decoder_head_embedding_dim=256,
        dropout=0.0,
        name=None,
        **kwargs,
    ):
        """
        Initializes the SegFormer model.

        The encoder can be provided either as an instantiated Keras model (`encoder`)
        or by its registered name (`encoder_name`), in which case `input_shape` must be provided.

        Args:
            input_shape (tuple, optional): The shape of the input data, excluding the batch dimension.
                Required if `encoder_name` is provided. Format is (H, W, C) for 2D or (D, H, W, C) for 3D.
            encoder_name (str, optional): The name of a registered hierarchical backbone (e.g., 'mit_b0').
            encoder (keras.Model, optional): An already instantiated hierarchical feature extractor.
                Must have a `pyramid_outputs` attribute.
            num_classes (int, optional): The number of output classes for segmentation. Default: 1.
            classifier_activation (str, optional): The activation function for the final output layer.
                Typically 'softmax' for multi-class or 'sigmoid' for multi-label/binary segmentation.
                Default: None.
            decoder_head_embedding_dim (int, optional): The hidden dimension used for linear embedding
                of the feature maps in the decoder head before fusion. Controls the capacity of the
                lightweight MLP decoder. Default: 256.
            dropout (float, optional): Dropout rate applied after the fusion convolution in the decoder head.
                Regularizes the decoder to prevent overfitting. Default: 0.0.
            name (str, optional): The name of the Keras model.
                Sets the model's identifier. Default: Auto-generated as "SegFormer{D}D".
            **kwargs: Standard Keras Model keyword arguments.
        """
        encoder, input_shape = resolve_encoder(
            encoder=encoder,
            encoder_name=encoder_name,
            input_shape=input_shape,
            allowed_families=SegFormer.ALLOWED_BACKBONE_FAMILIES,
        )

        inputs = encoder.input
        spatial_dims = len(input_shape) - 1

        # Check that the spatial dimensions are all equal.
        spatial_shapes = list(input_shape[:spatial_dims])
        if not all(x == spatial_shapes[0] for x in spatial_shapes):
            raise ValueError(
                f"Input shape {input_shape} is not square or cubic. "
                "SegFormer currently only supports inputs with equal spatial dimensions "
                "for proper hierarchical downsampling and reshaping."
            )

        # Get intermediate vectores
        pyramid_outputs = encoder.pyramid_outputs

        # SegFormer needs 4 skip connection layers
        required_keys = {"P1", "P2", "P3", "P4"}
        if not required_keys.issubset(pyramid_outputs.keys()):
            raise ValueError(
                f"The encoder's `pyramid_outputs` is missing one or more required keys. "
                f"Required: {required_keys}, Available: {set(pyramid_outputs.keys())}"
            )

        skips = [pyramid_outputs.get(f"P{i+1}") for i in range(4)]
        skips = skips + [inputs]

        # build_decoder method
        decoder_head = self.build_decoder(
            num_classes, decoder_head_embedding_dim, spatial_dims, dropout
        )
        outputs = decoder_head(skips)

        if classifier_activation:
            outputs = layers.Activation(classifier_activation, dtype="float32")(outputs)

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or f"SegFormer{spatial_dims}D", **kwargs
        )

        self._input_shape = input_shape
        self.num_classes = num_classes
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.dropout = dropout
        self.decoder_head_embedding_dim = decoder_head_embedding_dim
        self.classifier_activation = classifier_activation

    def build_decoder(self, num_classes, decoder_head_embedding_dim, spatial_dims, dropout):
        """
        Constructs the lightweight MLP decoder head as a callable function.

        This decoder head performs four main steps:
        1. Linear Embedding: Each of the four multi-scale feature maps (P1-P4) is
           processed by a 1x1 convolution (implemented as Dense layer after flattening)
           to unify the channel dimension to `decoder_head_embedding_dim`.
        2. Upsampling: Feature maps from P2, P3, and P4 are upsampled to the resolution
           of the highest-resolution feature map (P1).
        3. Feature Fusion: All four feature maps are concatenated and passed through
           a single 3x3 (or 3D equivalent) fusion convolution block.
        4. Final Prediction: A final 1x1 convolution is used to predict the class scores,
           followed by upsampling to the original input resolution.

        Args:
            num_classes (int): The number of output channels for the final prediction.
            decoder_head_embedding_dim (int): The hidden dimension for the MLP/linear
                embedding layers.
            spatial_dims (int): 2 for 2D or 3 for 3D inputs.
            dropout (float): Dropout rate to apply in the decoder fusion block.

        Returns:
            function: A Keras-style function that takes the list of skip connections
                      and returns the final segmentation output.
        """

        def apply(inputs):
            c1, c2, c3, c4, original_input = inputs

            # Get target spatial shape from c1
            target_spatial_shape = ops.shape(c1)[1:-1]

            # Process each feature level with linear embedding and resize to c1 size
            # stage 1
            c4_shape = ops.shape(c4)
            c4 = self.linear_embedding(c4, decoder_head_embedding_dim)
            c4 = self.reshape_to_spatial(c4, c4_shape)
            c4 = self.resize_to_target(c4, target_spatial_shape, spatial_dims)

            # stage 2
            c3_shape = ops.shape(c3)
            c3 = self.linear_embedding(c3, decoder_head_embedding_dim)
            c3 = self.reshape_to_spatial(c3, c3_shape)
            c3 = self.resize_to_target(c3, target_spatial_shape, spatial_dims)

            # stage 3
            c2_shape = ops.shape(c2)
            c2 = self.linear_embedding(c2, decoder_head_embedding_dim)
            c2 = self.reshape_to_spatial(c2, c2_shape)
            c2 = self.resize_to_target(c2, target_spatial_shape, spatial_dims)

            # stage 4
            c1_shape = ops.shape(c1)
            c1 = self.linear_embedding(c1, decoder_head_embedding_dim)
            c1 = self.reshape_to_spatial(c1, c1_shape)

            # Fuse all features (channel-last: concatenate along last axis)
            x = layers.Concatenate(axis=-1)([c1, c2, c3, c4])

            # Fusion convolution
            x = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=decoder_head_embedding_dim,
                kernel_size=1,
            )(x)
            x = get_norm_layer(norm_name="batch")(x)
            x = get_act_layer(name="relu")(x)
            x = layers.Dropout(dropout)(x)

            # Final prediction
            x = get_conv_layer(
                spatial_dims=spatial_dims, layer_type="conv", filters=num_classes, kernel_size=1
            )(x)

            # Get output spatial shape from original input
            output_spatial_shape = ops.shape(original_input)[1:-1]
            x = self.resize_to_target(x, output_spatial_shape, spatial_dims)
            return x

        return apply

    def linear_embedding(self, x, hidden_dims):
        spatial_shape_tensor = ops.shape(x)[1:-1]
        num_patches = int(ops.prod(spatial_shape_tensor))
        x = layers.Reshape((num_patches, ops.shape(x)[-1]))(x)
        x = layers.Dense(hidden_dims)(x)
        x = get_norm_layer(norm_name="layer", epsilon=1e-5)(x)
        return x

    def reshape_to_spatial(self, x, target_shape):
        spatial_shape = target_shape[1:-1]
        x = ops.reshape(x, [-1, *spatial_shape, ops.shape(x)[-1]])
        return x

    def resize_to_target(self, x, target_spatial_shape, spatial_dims):
        if spatial_dims == 3:
            uid = keras.backend.get_uid(prefix="resize_op")
            target_depth, target_height, target_width = target_spatial_shape
            lambda_layer = ResizeVolume(
                lambda volume: resize_volumes(
                    volume, target_depth, target_height, target_width, method="trilinear"
                ),
                name=f"resize_op{uid}",
            )
            x = lambda_layer(x)
        elif spatial_dims == 2:
            x = ops.image.resize(x, target_spatial_shape, interpolation="bilinear")
        return x

    def get_config(self):
        config = {
            "input_shape": self._input_shape,
            "encoder_name": self.encoder_name,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "decoder_head_embedding_dim": self.decoder_head_embedding_dim,
            "dropout": self.dropout,
        }

        if self.encoder_name is None and self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})
        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)


class ResizeVolume(keras.layers.Lambda):
    pass
