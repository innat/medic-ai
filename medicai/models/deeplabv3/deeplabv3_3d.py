

import keras
from keras import layers
from keras import ops
from ...layers import SpatialPyramidPooling3D
from medicai.utils.model_utils import BACKBONE_ZOO, KERAS_APPLICATION, SKIP_CONNECTION_ARGS

class DeepLabV3Plus3D(keras.Model):
    """DeepLabV3Plus 3D architecture for semantic segmentation.
    """

    def __init__(
        self,
        variant,
        num_classes,
        upsampling_size=8,
        dilation_rates=[6, 12, 18],
        input_shape=(96, 96, 96, 1),
        classifier_activation=None,
        spatial_pyramid_pooling_key=None,
        low_level_feature_key=None,
        projection_filters=48,
        **kwargs,
    ):
        
        if variant not in BACKBONE_ZOO:
            raise ValueError(
                f"Invalid variant '{variant}'. Choose from {list(BACKBONE_ZOO.keys())}."
            )

        channel_axis = -1 # "channels_last"
        inputs = layers.Input(input_shape, name="inputs")

        # Build encoder
        base_model = BACKBONE_ZOO[variant](
            input_shape=input_shape,
            variant=variant,
            include_top=False,
            **kwargs,
        )

        # build pyramid level featur extractor.
        spatial_pyramid_pooling_key = spatial_pyramid_pooling_key or SKIP_CONNECTION_ARGS[variant][0]
        low_level_feature_key = low_level_feature_key or SKIP_CONNECTION_ARGS[variant][-1]

        print(spatial_pyramid_pooling_key, low_level_feature_key)
        skip_layers = [
            base_model.get_layer(index=i).output for i in [low_level_feature_key, spatial_pyramid_pooling_key]
        ]
        encoder_model = keras.Model(
            inputs=base_model.inputs, outputs={f"P{i+1}": out for i, out in enumerate(skip_layers)}
        )
        encoder_outputs = encoder_model(inputs)

        # ASPP
        spp_input_tensor = encoder_outputs[spatial_pyramid_pooling_key]
        spp_block = SpatialPyramidPooling3D(dilation_rates=dilation_rates)
        spp_outputs = spp_block(spp_input_tensor)

        combined_features = None

        if low_level_feature_key:
            decoder_feature = encoder_outputs[low_level_feature_key]
            # Dynamically calculate the upsampling factor for the ASPP branch
            # to match the spatial resolution of the low_level_feature.
            spp_spatial_shape = ops.shape(spp_outputs)[1:-1]
            decoder_spatial_shape = ops.shape(decoder_feature)[1:-1]

            upsampling_d = decoder_spatial_shape[0] // spp_spatial_shape[0]
            upsampling_h = decoder_spatial_shape[1] // spp_spatial_shape[1]
            upsampling_w = decoder_spatial_shape[2] // spp_spatial_shape[2]
            upsampling_size_decoder = (upsampling_d, upsampling_h, upsampling_w)

            encoder_outputs_upsampled = layers.UpSampling3D(
                size=upsampling_size_decoder,
                name="encoder_output_upsampling",
            )(spp_outputs)

            low_level_projected_features = apply_low_level_feature_network(
                decoder_feature, projection_filters, channel_axis
            )

            # Now, the spatial dimensions of both tensors will match.
            combined_features = layers.Concatenate(axis=channel_axis, name="encoder_decoder_concat")(
                [encoder_outputs_upsampled, low_level_projected_features]
            )
        else:
            # If no low-level features, upsample ASPP output directly to original image size
            spp_spatial_shape = ops.shape(spp_outputs)[1:-1]
            input_spatial_shape = ops.shape(inputs)[1:-1]

            upsampling_d = input_spatial_shape[0] // spp_spatial_shape[0]
            upsampling_h = input_spatial_shape[1] // spp_spatial_shape[1]
            upsampling_w = input_spatial_shape[2] // spp_spatial_shape[2]
            final_upsampling_size = (upsampling_d, upsampling_h, upsampling_w)

            combined_features = layers.UpSampling3D(
                size=final_upsampling_size,
                name="backbone_output_upsampling",
            )(spp_outputs)


        # The rest of the segmentation head logic remains the same
        x = layers.Conv3D(256, 1, padding="same", use_bias=False, name="segmentation_head_conv")(combined_features)
        x = layers.BatchNormalization(axis=channel_axis, name="segmentation_head_norm")(x)
        x = layers.ReLU(name="segmentation_head_relu")(x)

        # Final dynamic upsampling to match input size
        current_shape = ops.shape(x)
        input_shape_ops = ops.shape(inputs)
        upsampling_d = input_shape_ops[1] // current_shape[1]
        upsampling_h = input_shape_ops[2] // current_shape[2]
        upsampling_w = input_shape_ops[3] // current_shape[3]
        final_upsampling_size = (upsampling_d, upsampling_h, upsampling_w)

        x = layers.UpSampling3D(
            size=final_upsampling_size,
            name="final_upsampling",
        )(x)

        output_conv = keras.layers.Conv3D(
            name="segmentation_output",
            filters=num_classes,
            kernel_size=1,
            use_bias=False,
            padding="same",
            activation=classifier_activation,
        )
        outputs = output_conv(x)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.variant = variant
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.projection_filters = projection_filters
        self.upsampling_size = upsampling_size
        self.dilation_rates = dilation_rates
        self.low_level_feature_key = low_level_feature_key
        self.spatial_pyramid_pooling_key = spatial_pyramid_pooling_key

    def get_config(self):
        config = {
            "variant": self.variant,
            "input_shape": self.input_shape[1:],
            'num_classes': self.num_classes,
            "classifier_activation": self.classifier_activation,
            "projection_filters": self.projection_filters,
            "upsampling_size": self.upsampling_size,
            "dilation_rates": self.dilation_rates,
            "low_level_feature_key": self.low_level_feature_key,
            "spatial_pyramid_pooling_key": self.spatial_pyramid_pooling_key,
        }
        return config
    

def apply_low_level_feature_network(
    input_tensor, projection_filters, channel_axis
):
    x = layers.Conv3D(
        name="decoder_conv",
        filters=projection_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
    )(input_tensor)
    x = layers.BatchNormalization(name="decoder_norm", axis=channel_axis)(x)
    x = layers.ReLU(name="decoder_relu")(x)
    return x