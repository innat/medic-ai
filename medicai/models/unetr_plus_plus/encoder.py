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
        patch_size=[4, 4, 4],
        filters=[32, 64, 128, 256],
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
                Examples:
                - 3D: (D, H, W, C)
                - 2D: (H, W, C)
            input_tensor: (Optional) A Keras tensor to use as the model input.
                If not provided, a new input tensor will be created.
            sequence_lengths: A list of integers representing the flattened
                spatial sequence lengths for each stage. Used by transformer
                blocks to determine the token count.
            patch_size: Integer or tuple specifying the patch size used by
                the convolutional stem. The stem downsamples the input
                independently along each spatial dimension according to
                this value. Anisotropic patching is supported.
                Examples:
                - (4, 4, 4): isotropic 3D patching
                - (1, 4, 4): anisotropic patching (no downsampling in depth)
            filters: List of integers specifying the number of feature
                channels produced at each encoder stage. The length of
                this list determines the number of hierarchical stages.
            spatial_reduced_tokens: List of integers specifying the number
                of spatial tokens retained after spatial reduction inside
                the Efficient Paired Attention (EPA) module for each stage.
                This controls the attention computation cost.
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

        # Ensure to correct shape for 2D and 3D ops.
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        sequence_lengths = self.calculate_downsampled_input_sizes(input_shape, patch_size)

        inputs = parse_model_inputs(input_shape, input_tensor, name="unetr_pp_input")
        x = inputs

        # stem layer
        x = utils.get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=filters[0],
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=False,
            name="stem_conv",
        )(x)
        x = utils.get_norm_layer(layer_type="group", groups=input_shape[-1], name="stem_norm")(x)

        # stage 0 transformer blocks
        for j in range(depths[0]):
            x = UNETRPlusPlusTransformer(
                sequence_length=sequence_lengths[0],
                hidden_size=filters[0],
                spatial_reduced_tokens=spatial_reduced_tokens[0],
                num_heads=num_heads,
                dropout_rate=transformer_dropout_rate,
                pos_embed=True,
                name=f"stage_0_transformer_block_{j}",
            )(x)

        pyramid_outputs["P1"] = x

        # stages 1-3
        for i in range(1, 4):
            # downsampling
            x = utils.get_conv_layer(
                spatial_dims,
                layer_type="conv",
                filters=filters[i],
                kernel_size=2,
                strides=2,
                use_bias=False,
                name=f"downsample_conv_{i}",
            )(x)
            x = utils.get_norm_layer(
                layer_type="group", groups=filters[i], name=f"downsample_norm_{i}"
            )(x)

            # Transformer blocks
            for j in range(depths[i]):
                x = UNETRPlusPlusTransformer(
                    sequence_length=sequence_lengths[i],
                    hidden_size=filters[i],
                    spatial_reduced_tokens=spatial_reduced_tokens[i],
                    num_heads=num_heads,
                    dropout_rate=transformer_dropout_rate,
                    pos_embed=True,
                    name=f"stage_{i}_transformer_block_{j}",
                )(x)

            pyramid_outputs[f"P{i+1}"] = x

        super().__init__(
            inputs=inputs,
            outputs=x,
            name=f"UNETRPlusPlusEncoder{spatial_dims}D",
            **kwargs,
        )

        self.sequence_lengths = sequence_lengths
        self.pyramid_outputs = pyramid_outputs
        self.filters = filters
        self.spatial_reduced_tokens = spatial_reduced_tokens
        self.patch_size = patch_size
        self.depths = depths
        self.num_heads = num_heads
        self.transformer_dropout_rate = transformer_dropout_rate

    @staticmethod
    def calculate_downsampled_input_sizes(input_shape, patch_size):
        spatial_shape = input_shape[:-1]
        spatial_dims = len(spatial_shape)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        sizes = []

        # Stage 0 (stem only)
        current = [spatial_shape[d] // patch_size[d] for d in range(spatial_dims)]
        sizes.append(current)

        # Stages 1–3 (each halves spatial dims)
        for _ in range(3):
            current = [dim // 2 for dim in current]
            sizes.append(current)

        # Flatten to token counts (for transformer)
        sequence_lengths = [int(np.prod(s)) for s in sizes]

        return sequence_lengths

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "filters": self.filters,
            "patch_size": self.patch_size,
            "spatial_reduced_tokens": self.spatial_reduced_tokens,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "transformer_dropout_rate": self.transformer_dropout_rate,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
