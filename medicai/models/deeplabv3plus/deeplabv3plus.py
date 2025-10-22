import keras
from keras import layers

from medicai.layers import ResizingND
from medicai.utils import (
    DescribeMixin,
    get_conv_layer,
    keras_constants,
    registration,
    resolve_encoder,
    validate_activation,
)

from .decoder import DeepLabV3PlusDecoder


@keras.saving.register_keras_serializable(package="deeplab")
@registration.register(name="deeplabv3plus", type="segmentation")
class DeepLabV3Plus(keras.Model, DescribeMixin):
    """
    Implements the DeepLabV3+ architecture for semantic segmentation.

    This model combines the DeepLabV3 Atrous Spatial Pyramid Pooling (ASPP)
    module with an encoder-decoder structure, leveraging both high-level,
    context-rich features (via ASPP) and low-level, boundary-detailed features
    (via a skip connection) to produce fine segmentation masks.

    The implementation is generalized to handle both 2D (image) and 3D (volumetric/video)
    inputs, dynamically adjusting convolutional and upsampling layers based on the
    `input_shape`.

    Example (2D):
    ```python
    from medicai.models import DeepLabV3Plus

    # Create an image segmentation model using ResNet50 as the encoder
    model = DeepLabV3Plus(
        encoder_name='resnet50',
        input_shape=(224, 224, 3),
        num_classes=1,
        classifier_activation="sigmoid",
    )
    ```

    Example (3D):
    ```python
    from medicai.models import DeepLabV3Plus

    # Create a volumetric segmentation model using a 3D backbone
    model_3d = DeepLabV3Plus(
        encoder_name='convnext_tiny',
        input_shape=(64, 128, 128, 1),
        num_classes=4,
        classifier_activation='softmax',
        encoder_depth=4,
        head_upsample=8,
    )
    ```
    """

    ALLOWED_BACKBONE_FAMILIES = ["resnet", "densenet", "efficientnet", "convnext", "senet"]

    def __init__(
        self,
        encoder=None,
        encoder_name=None,
        input_shape=None,
        encoder_depth=5,
        num_classes=1,
        decoder_channels=256,
        decoder_dilation_rates=(12, 24, 36),
        decoder_aspp_separable=True,
        decoder_aspp_dropout=0.5,
        decoder_normalization="batch",
        decoder_activation="relu",
        projection_filters=48,
        classifier_activation="sigmoid",
        head_upsample=4,
        name=None,
        **kwargs,
    ):
        """
        Initializes the DeepLabV3Plus model.

        Args:
        encoder (keras.Model, optional): The backbone model used for feature extraction.
            If provided, `encoder_name` and `input_shape` are ignored. Defaults to `None`.
        encoder_name (str, optional): Name of the backbone model to instantiate from
            the supported families. Defaults to `None`.
        input_shape (tuple, optional): The shape of the input tensor, excluding the batch
            dimension. E.g., `(224, 224, 3)` for 2D or `(96, 96, 96, 1)` for 3D.
            Defaults to `None`.
        encoder_depth (int, optional): The number of feature levels (pyramid stages P1-P5)
            to use from the encoder. Must be between 3 and 5. The deepest feature (P{depth})
            is used for ASPP, and P2 is typically used for the low-level skip connection.
            Defaults to 5.
        num_classes (int, optional): The number of segmentation classes. Must be positive.
            Defaults to 1.
        decoder_channels (int, optional): The number of channels used in the ASPP output
            and the final decoder convolution. Defaults to 256.
        decoder_dilation_rates (tuple of ints, optional): The dilation rates for the
            parallel dilated 3x3x3 convolutions in the ASPP module. Defaults to (12, 24, 36).
        decoder_aspp_separable (bool, optional): If True, uses SeparableConv layers
            within the ASPP module for efficiency. Defaults to `True`.
        decoder_aspp_dropout (float, optional): Dropout rate applied after the final
            ASPP projection. Defaults to 0.5.
        decoder_normalization (str, optional): Normalization type used in the decoder
            (e.g., 'batch', 'layer', 'group'). Defaults to `"batch"`.
        decoder_activation (str, optional): Activation function used within the decoder blocks.
            Defaults to `"relu"`.
        projection_filters (int, optional): The number of channels to project the
            low-level feature map down to (e.g., 48 in the original paper). Defaults to 48.
        classifier_activation (str, optional): The activation function applied to the final
            output layer (e.g., 'sigmoid' for binary or multi-label, 'softmax' for multi-class).
            Defaults to `"sigmoid"`.
        head_upsample (int, float, list, or tuple, optional): The final upsampling scale
            factor(s) applied to the head output to match the original input resolution.
            Defaults to 4 (e.g., to go from 1/4 resolution to full resolution).
        name (str, optional): The name of the model. Defaults to `None`.
        """

        encoder, input_shape = resolve_encoder(
            encoder=encoder,
            encoder_name=encoder_name,
            input_shape=input_shape,
            allowed_families=DeepLabV3Plus.ALLOWED_BACKBONE_FAMILIES,
        )
        inputs = encoder.input
        spatial_dims = len(input_shape) - 1
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

        # number of classes must be positive.
        if num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        if isinstance(decoder_normalization, str):
            decoder_normalization = decoder_normalization.lower()

        if decoder_normalization not in keras_constants.VALID_DECODER_NORMS:
            raise ValueError(
                f"Invalid value for `decoder_normalization`: {decoder_normalization!r}. "
                f"Supported values are: {keras_constants.VALID_DECODER_NORMS}"
            )

        # verify input activation.
        decoder_activation = validate_activation(decoder_activation)
        classifier_activation = validate_activation(classifier_activation)

        # Select feature keys and their output tensor
        low_level_idx = 1 if encoder_depth >= 3 else 0
        deep_idx = encoder_depth - 1
        low_level_feature = pyramid_outputs[f"P{low_level_idx + 1}"]
        deep_feature = pyramid_outputs[f"P{deep_idx + 1}"]

        # deeplabv3plus decoder
        decoder = DeepLabV3PlusDecoder(
            spatial_dims=spatial_dims,
            decoder_channels=decoder_channels,
            decoder_dilation_rates=decoder_dilation_rates,
            decoder_aspp_separable=decoder_aspp_separable,
            decoder_aspp_dropout=decoder_aspp_dropout,
            decoder_normalization=decoder_normalization,
            decoder_activation=decoder_activation,
            projection_filters=projection_filters,
        )
        x = decoder(deep_feature, low_level_feature)

        # Final segmentation head
        x = get_conv_layer(
            spatial_dims, layer_type="conv", filters=num_classes, kernel_size=1, padding="same"
        )(x)

        # Some encoder like, i.e. convnext need final upsampling.
        if isinstance(head_upsample, (int, float)):
            if head_upsample > 1:
                x = ResizingND(
                    scale_factor=head_upsample,
                    interpolation="bilinear" if spatial_dims == 2 else "trilinear",
                )(x)
        elif isinstance(head_upsample, (list, tuple)):
            if any(s > 1 for s in head_upsample):
                x = ResizingND(
                    scale_factor=head_upsample,
                    interpolation="bilinear" if spatial_dims == 2 else "trilinear",
                )(x)
        else:
            raise ValueError(
                f"`head_upsample` must be int, float, tuple, or list, got {type(head_upsample)}"
            )

        outputs = layers.Activation(classifier_activation, dtype="float32", name="predictions")(x)

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or f"DeepLabV3Plus{spatial_dims}D", **kwargs
        )

        self._input_shape = input_shape
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.head_upsample = head_upsample
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.encoder_depth = encoder_depth
        self.decoder_channels = decoder_channels
        self.decoder_dilation_rates = decoder_dilation_rates
        self.decoder_normalization = decoder_normalization
        self.decoder_aspp_dropout = decoder_aspp_dropout
        self.decoder_activation = decoder_activation
        self.projection_filters = projection_filters

    def get_config(self):
        config = {
            "input_shape": self._input_shape,
            "encoder_name": self.encoder_name,
            "encoder_depth": self.encoder_depth,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "decoder_channels": self.decoder_channels,
            "projection_filters": self.projection_filters,
            "decoder_dilation_rates": self.decoder_dilation_rates,
            "decoder_normalization": self.decoder_normalization,
            "decoder_aspp_dropout": self.decoder_aspp_dropout,
            "decoder_activation": self.decoder_activation,
            "head_upsample": self.head_upsample,
        }

        if self.encoder_name is None and self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})
        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)
