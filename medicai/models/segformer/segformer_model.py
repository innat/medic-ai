import keras
from keras import layers, ops

from medicai.utils import (
    get_act_layer,
    get_conv_layer,
    get_norm_layer,
    get_reshaping_layer,
    registration,
    resize_volumes,
)

from .segformer_layers import MixVisionTransformer


class SegFormer(keras.Model):
    """SegFormer model for 2D or 3D semantic segmentation.

    This class implements the full SegFormer architecture, which combines a
    hierarchical MixVisionTransformer (MiT) encoder with a lightweight MLP decoder
    head. This design is highly efficient for semantic segmentation tasks on
    high-resolution images or volumes.

    The encoder progressively downsamples the spatial dimensions and increases the
    feature dimensions across four stages, producing multi-scale feature maps.
    The decoder then takes these features, processes them through linear layers,
    upsamples them to a common resolution, and fuses them to generate a
    high-resolution segmentation mask.
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

        Args:
            input_shape (tuple): The shape of the input data, excluding the batch dimension.
            num_classes (int): The number of output classes for segmentation.
            decoder_head_embedding_dim (int, optional): The embedding dimension of the decoder head.
                Defaults to 256.
            classifier_activation (str, optional): The activation function for the final output layer.
                Common choices are 'softmax' for multi-class segmentation and 'sigmoid' for multi-label
                or binary segmentation. Defaults to None.
            name (str, optional): The name of the model. Defaults to None.
            **kwargs: Standard Keras Model keyword arguments.
        """
        if bool(encoder) == bool(encoder_name):
            raise ValueError("Exactly one of `encoder` or `encoder_name` must be provided.")

        if encoder is not None:
            input_shape = encoder.input_shape[1:]
        elif encoder_name is not None:
            if not input_shape:
                raise ValueError(
                    "Argument `input_shape` must be provided. "
                    "It should be a tuple of integers specifying the dimensions of the input "
                    "data, not including the batch size. "
                    "For 2D data, the format is `(height, width, channels)`. "
                    "For 3D data, the format is `(depth, height, width, channels)`."
                )

            if encoder_name.lower() not in registration._registry:
                raise ValueError(
                    f"Encoder '{encoder_name}' not found in the registry. Available: {list(registration._registry.keys())}"
                )

            entry = registration.get_entry(encoder_name)
            invalid_families = [
                f for f in entry["family"] if f not in SegFormer.ALLOWED_BACKBONE_FAMILIES
            ]
            if invalid_families:
                raise ValueError(
                    f"The provided encoder_name='{encoder_name}' uses unsupported families: "
                    f"{invalid_families}. Allowed families: {SegFormer.ALLOWED_BACKBONE_FAMILIES}"
                )

            encoder = entry["class"](input_shape=input_shape, include_top=False)

        if not hasattr(encoder, "pyramid_outputs"):
            raise AttributeError(
                f"The provided `encoder` must have a `pyramid_outputs` attribute, "
                f"but the provided encoder of type {type(encoder).__name__} does not."
            )

        inputs = encoder.input
        spatial_dims = len(input_shape) - 1
        pyramid_outputs = encoder.pyramid_outputs

        required_keys = {"P1", "P2", "P3", "P4"}
        if not required_keys.issubset(pyramid_outputs.keys()):
            raise ValueError(
                f"The encoder's `pyramid_outputs` is missing one or more required keys. "
                f"Required: {required_keys}, Available: {set(pyramid_outputs.keys())}"
            )

        skips = [pyramid_outputs.get(f"P{i+1}") for i in range(4)]

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
        def apply(inputs):
            c1, c2, c3, c4 = inputs

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
            x = get_reshaping_layer(spatial_dims=spatial_dims, layer_type="upsampling", size=4)(x)
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
        }

        if self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})
        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)


class ResizeVolume(keras.layers.Lambda):
    pass
