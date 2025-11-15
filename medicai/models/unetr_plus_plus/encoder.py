import keras
import numpy as np

from medicai import utils
from medicai.utils import DescribeMixin, parse_model_inputs, registration
from medicai.utils.swi_utils import ensure_tuple_rep

from .encoder_layers import UNETRPlusPlusTransformer


@keras.saving.register_keras_serializable(package="unetr_plusplus")
@registration.register(name="unetr_plusplus_encoder", family="unetr_plusplus")
class UNETRPlusPlusEncoder(keras.Model, DescribeMixin):
    """
    A UNETR++ hierarchical encoder built with Efficient Paired Attention (EPA).

    This class constructs the encoder portion of the UNETR++ architecture for
    both 2D and 3D inputs. The encoder begins with a convolutional stem,
    followed by four hierarchical stages that progressively downsample the
    feature maps and apply transformer blocks. Each transformer block uses
    Efficient Paired Attention (EPA), which reduces the spatial token
    dimension before attention, enabling efficient computation on high-resolution
    volumes.

    The encoder produces multi-scale features, stored as pyramid outputs
    (P1-P4), which can be used by a decoder for segmentation or other dense
    prediction tasks.
    """

    def __init__(
        self,
        *,
        input_shape,
        input_tensor=None,
        sequence_lengths=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],
        stem_kernel_sizes=4,
        stem_strides=4,
        downsampling_kernel_sizes=2,
        downsampling_strides=2,
        encoder_filters=[32, 64, 128, 256],
        spatial_reduced_tokens=[64, 64, 64, 32],
        depths=[3, 3, 3, 3],
        num_heads=4,
        transformer_dropout_rate=0.1,
        **kwargs,
    ):
        """
        Initializes the UNETRPlusPlusEncoder.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch dimension. Supports both 2D and 3D.
            input_tensor: (Optional) A Keras tensor to use as the model input.
                If not provided, a new input tensor will be created.
            sequence_lengths: A list of integers representing the flattened
                spatial sequence lengths for each stage. Used by transformer
                blocks to determine the token count.
            stem_kernel_sizes: Integer or sequence specifying the kernel size
                for the convolutional stem. Automatically broadcast per
                spatial dimension.
            stem_strides: Integer or sequence specifying the stride of the
                convolutional stem.
            downsampling_kernel_sizes: Integer or sequence specifying the
                kernel size for downsampling layers in stages 1-3.
            downsampling_strides: Integer or sequence specifying the stride
                for downsampling layers in stages 1-3.
            encoder_filters: A list of integers specifying the number of
                feature channels produced at each stage of the encoder.
            spatial_reduced_tokens: A list of integers specifying the number
                of reduced spatial tokens used by EPA inside each transformer
                block.
            depths: A list of integers specifying the number of transformer
                blocks at each encoder stage.
            num_heads: Number of attention heads used in the transformer
                blocks.
            transformer_dropout_rate: Dropout rate applied inside transformer
                blocks.
            **kwargs: Additional keyword arguments.

        """
        spatial_dims = len(input_shape) - 1
        pyramid_outputs = {}

        stem_kernel_sizes = ensure_tuple_rep(stem_kernel_sizes, spatial_dims)
        stem_strides = ensure_tuple_rep(stem_strides, spatial_dims)
        downsampling_kernel_sizes = ensure_tuple_rep(downsampling_kernel_sizes, spatial_dims)
        downsampling_strides = ensure_tuple_rep(downsampling_strides, spatial_dims)
        inputs = parse_model_inputs(input_shape, input_tensor, name="input_spec")
        x = inputs

        # Stem layer
        x = utils.get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=encoder_filters[0],
            kernel_size=stem_kernel_sizes,
            strides=stem_strides,
            use_bias=False,
            name="stem_conv",
        )(x)
        x = utils.get_norm_layer(layer_type="group", groups=inputs.shape[-1], name="stem_norm")(x)

        # Stage 0 transformer blocks
        for j in range(depths[0]):
            x = UNETRPlusPlusTransformer(
                sequence_lengths=sequence_lengths[0],
                hidden_size=encoder_filters[0],
                spatial_reduced_tokens=spatial_reduced_tokens[0],
                num_heads=num_heads,
                dropout_rate=transformer_dropout_rate,
                pos_embed=True,
                name=f"stage0_transformer_block_{j}",
            )(x)

        pyramid_outputs["P1"] = x

        # Stages 1-3
        for i in range(1, 4):
            # Downsampling
            x = utils.get_conv_layer(
                spatial_dims,
                layer_type="conv",
                filters=encoder_filters[i],
                kernel_size=downsampling_kernel_sizes,
                strides=downsampling_strides,
                use_bias=False,
                name=f"downsample_conv_{i}",
            )(x)
            x = utils.get_norm_layer(
                layer_type="group", groups=encoder_filters[i], name=f"downsample_norm_{i}"
            )(x)

            # Transformer blocks
            for j in range(depths[i]):
                x = UNETRPlusPlusTransformer(
                    input_size=sequence_lengths[i],
                    hidden_size=encoder_filters[i],
                    spatial_reduced_tokens=spatial_reduced_tokens[i],
                    num_heads=num_heads,
                    dropout_rate=transformer_dropout_rate,
                    pos_embed=True,
                    name=f"stage{i}_transformer_block_{j}",
                )(x)

            pyramid_outputs[f"P{i+1}"] = x

        super().__init__(
            inputs=inputs,
            outputs=x,
            name=f"UNETRPlusPlusEncoder{spatial_dims}D",
            **kwargs,
        )

        self.pyramid_outputs = pyramid_outputs
        self.encoder_filters = encoder_filters
        self.stem_kernel_sizes = stem_kernel_sizes
        self.stem_strides = stem_strides
        self.downsampling_kernel_sizes = downsampling_kernel_sizes
        self.downsampling_strides = downsampling_strides
        self.spatial_reduced_tokens = spatial_reduced_tokens
        self.depths = depths
        self.num_heads = num_heads
        self.transformer_dropout_rate = transformer_dropout_rate

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "encoder_filters": self.encoder_filters,
            "stem_kernel_sizes": self.stem_kernel_sizes,
            "stem_strides": self.stem_strides,
            "downsampling_kernel_sizes": self.downsampling_kernel_sizes,
            "downsampling_strides": self.downsampling_strides,
            "spatial_reduced_tokens": self.spatial_reduced_tokens,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "transformer_dropout_rate": self.transformer_dropout_rate,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
