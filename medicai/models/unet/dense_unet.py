import keras
from keras import layers

from medicai.models.densenet import DenseNetBackbone
from medicai.utils.model_utils import (
    BACKBONE_ARGS,
    SKIP_CONNECTION_ARGS,
    get_conv_layer,
    parse_model_inputs,
)

from .unet_imagenet_decoder import UNetDecoder


class DenseUNetBase(keras.Model):
    def __init__(
        self,
        backbone_name,
        input_shape=None,
        input_tensor=None,
        num_classes=1,
        classifier_activation=None,
        decoder_block_type="upsampling",
        decoder_filters=(256, 128, 64, 32, 16),
        use_attention=False,
        name=None,
        **kwargs,
    ):
        # Get spatial dims for 2D and 3D cases.
        spatial_dims = len(input_shape) - 1

        inputs = parse_model_inputs(input_shape, input_tensor)
        base_model = DenseNetBackbone(blocks=BACKBONE_ARGS.get(backbone_name), input_tensor=inputs)
        inputs = base_model.input

        # Get skip connections
        selected_layers = SKIP_CONNECTION_ARGS.get(backbone_name)
        skip_layers = [
            (
                base_model.get_layer(name=i).output
                if isinstance(i, str)
                else base_model.get_layer(index=i).output
            )
            for i in selected_layers
        ]

        # Apply decoder
        x = base_model.output
        decoder = UNetDecoder(
            skip_layers,
            decoder_filters,
            spatial_dims,
            block_type=decoder_block_type,
            use_attention=use_attention,
        )
        x = decoder(x)

        # Final segmentation head
        x = get_conv_layer(
            spatial_dims, layer_type="conv", filters=num_classes, kernel_size=1, padding="same"
        )(x)
        outputs = layers.Activation(classifier_activation, dtype="float32")(x)
        super().__init__(
            inputs=inputs, outputs=outputs, name=name or f"DenseUNetBase{spatial_dims}D", **kwargs
        )

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.decoder_block_type = decoder_block_type
        self.decoder_filters = decoder_filters

    def get_config(self):
        config = {
            "backbone_name": self.backbone_name,
            "input_shape": self.input_shape[1:],
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "decoder_block_type": self.decoder_block_type,
            "decoder_filters": self.decoder_filters,
        }
        return config


class DenseUNet121(DenseUNetBase):
    def __init__(
        self,
        input_shape=None,
        input_tensor=None,
        num_classes=1,
        classifier_activation=None,
        decoder_block_type="upsampling",
        decoder_filters=(256, 128, 64, 32, 16),
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1
        name = name or f"DenseNet121{spatial_dims}D"
        super().__init__(
            backbone_name="densenet121",
            input_shape=input_shape,
            input_tensor=input_tensor,
            num_classes=num_classes,
            classifier_activation=classifier_activation,
            decoder_block_type=decoder_block_type,
            decoder_filters=decoder_filters,
            name=name,
            **kwargs,
        )


class DenseUNet169(DenseUNetBase):
    def __init__(
        self,
        input_shape=None,
        input_tensor=None,
        num_classes=1,
        classifier_activation=None,
        decoder_block_type="upsampling",
        decoder_filters=(256, 128, 64, 32, 16),
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1
        name = name or f"DenseUNet169{spatial_dims}D"
        super().__init__(
            backbone_name="densenet169",
            input_shape=input_shape,
            input_tensor=input_tensor,
            num_classes=num_classes,
            classifier_activation=classifier_activation,
            decoder_block_type=decoder_block_type,
            decoder_filters=decoder_filters,
            name=name,
            **kwargs,
        )


class DenseUNet201(DenseUNetBase):
    def __init__(
        self,
        input_shape=None,
        input_tensor=None,
        num_classes=1,
        classifier_activation=None,
        decoder_block_type="upsampling",
        decoder_filters=(256, 128, 64, 32, 16),
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1
        name = name or f"DenseUNet201{spatial_dims}D"
        super().__init__(
            backbone_name="densenet201",
            input_shape=input_shape,
            input_tensor=input_tensor,
            num_classes=num_classes,
            classifier_activation=classifier_activation,
            decoder_block_type=decoder_block_type,
            decoder_filters=decoder_filters,
            name=name,
            **kwargs,
        )
