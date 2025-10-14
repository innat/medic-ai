import keras
import numpy as np
from keras import layers

from medicai.utils import DescribeMixin, get_conv_layer, get_norm_layer, parse_model_inputs

from .convnext_layers import ConvNeXtBlock, ConvNeXtV2Block, PreStem


@keras.utils.register_keras_serializable(package="convnext.backbone")
class ConvNeXtBackbone(keras.Model, DescribeMixin):
    """
    ConvNeXt V1 Backbone model as a Keras Model.

    The ConvNeXt V1 architecture is a modern, purely convolutional
    network designed to compete with Vision Transformers (ViTs), featuring
    macro-design inspired by ViTs (e.g., stage ratios, downsampling methods)
    and micro-design enhancements (e.g., inverted bottleneck, large kernel
    depth-wise convolution, LayerNorm).

    This model implements the feature extraction stages (Stem, Stages 1-4)
    and outputs the feature maps at the end of each stage, commonly used
    for downstream tasks like object detection or segmentation.

    Args:
        depths: A list or tuple of integers specifying the number of
            ConvNeXt blocks in each of the 4 stages. E.g., `[3, 3, 9, 3]`.
        projection_dims: A list or tuple of integers specifying the number
            of channels (filters) for the stem and each of the 4 stages.
            E.g., `[96, 192, 384, 768]`. Must have a length of 4.
        input_shape: The shape of the input tensor, excluding the batch
            dimension. E.g., `(224, 224, 3)` for 2D inputs.
        input_tensor: Optional Keras tensor (e.g., `keras.Input`) to use
            as the input to the model.
        drop_path_rate: Float, the maximum dropout rate for Stochastic
            Depth (DropPath) regularization. The rate is applied linearly
            across all ConvNeXt blocks. Defaults to 0.0 (no dropout).
        layer_scale_init_value: Float, initial value for the LayerScale
            parameter in each ConvNeXt block. A small value (e.g., 1e-6)
            is typically used to stabilize training. Defaults to 1e-6.
        include_rescaling: Boolean, whether to include the input rescaling
            layer (normalizing pixel values to [0, 1] or performing
            ImageNet mean/std normalization if input channels is 3).
            Defaults to False.
        name: Optional string, the name for the Keras model.
        **kwargs: Additional keyword arguments for the Keras Model constructor.

    Attributes:
        depths (list): The block depths used for each stage.
        projection_dims (list): The channel dimensions for each stage.
        pyramid_outputs (dict): Dictionary containing the output feature
            maps for each stage, keyed as 'P1', 'P2', 'P3', 'P4'.
    """

    def __init__(
        self,
        *,
        depths,
        projection_dims,
        input_shape,
        input_tensor=None,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        include_rescaling=False,
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1
        inputs = parse_model_inputs(input_shape, input_tensor, name="input_spec")

        x = inputs
        if include_rescaling:
            num_channels = input_shape[-1]
            if num_channels == 3:
                x = PreStem()(x)
            else:
                x = layers.Rescaling(1.0 / 255)(x)

        # feature pyramid outputs
        pyramid_outputs = {}

        # 1. Stem
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=projection_dims[0],
            kernel_size=4,
            strides=4,
            name="stem_conv",
        )(x)
        x = get_norm_layer(layer_type="layer", epsilon=1e-6, name="stem_layernorm")(x)

        # 2. Downsampling and Stages
        num_stages = len(depths)

        # A list to keep track of cumulative drop path rates for Stochastic Depth
        dpr = np.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        cur = 0

        for i in range(num_stages):
            # Downsampling block (except for the first stage which uses the stem)
            if i > 0:
                x = get_norm_layer(
                    layer_type="layer",
                    epsilon=1e-6,
                    name=f"downsampling_layernorm_{i-1}",
                )(x)
                x = get_conv_layer(
                    spatial_dims=spatial_dims,
                    layer_type="conv",
                    filters=projection_dims[i],
                    kernel_size=2,
                    strides=2,
                    name=f"downsampling_conv_{i-1}",
                )(x)

            # ConvNeXt Blocks (Main Stage)
            for j in range(depths[i]):
                # Determine current drop path rate
                current_drop_path_rate = dpr[cur]
                cur += 1
                x = ConvNeXtBlock(
                    projection_dim=projection_dims[i],
                    drop_path_rate=current_drop_path_rate,
                    layer_scale_init_value=layer_scale_init_value,
                    name=f"stage_{i}_block_{j}",
                )(x)

            pyramid_outputs[f"P{i+1}"] = get_norm_layer(
                layer_type="layer", epsilon=1e-6, name=f"pyramid_feature_norm_{i+1}"
            )(x)

        super().__init__(
            inputs=inputs, outputs=x, name=name or f"ConvNeXtBackbone{spatial_dims}D", **kwargs
        )

        self.depths = depths
        self.pyramid_outputs = pyramid_outputs
        self.projection_dims = projection_dims
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.include_rescaling = include_rescaling
        self.name = name

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape[1:],
                "depths": self.depths,
                "projection_dims": self.projection_dims,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init_value": self.layer_scale_init_value,
                "include_rescaling": self.include_rescaling,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package="convnext.backbone")
class ConvNeXtBackboneV2(keras.Model, DescribeMixin):
    """
    ConvNeXt V2 Backbone model as a Keras Model.

    ConvNeXt V2 (A ConvNet for the era of vision transformers) is an improved
    version of ConvNeXt V1, primarily distinguishing itself by introducing the
    **Global Response Normalization (GRN)** layer in each block and removing
    the Layer Scale (which was present in V1). This design makes the model
    more effective for masked autoencoding (MAE) pre-training, leading to
    improved performance on various downstream tasks.

    This model implements the feature extraction stages (Stem, Stages 1-4)
    and outputs the feature maps at the end of each stage.

    Args:
        depths: A list or tuple of integers specifying the number of
            ConvNeXt V2 blocks in each of the 4 stages. E.g., `[3, 3, 9, 3]`
            for the ConvNeXt-T V2 configuration.
        projection_dims: A list or tuple of integers specifying the number
            of channels (filters) for the stem and each of the 4 stages.
            E.g., `[96, 192, 384, 768]`. Must have a length of 4.
        input_shape: The shape of the input tensor, excluding the batch
            dimension. E.g., `(224, 224, 3)` for 2D inputs.
        input_tensor: Optional Keras tensor (e.g., `keras.Input`) to use
            as the input to the model.
        drop_path_rate: Float, the maximum dropout rate for Stochastic
            Depth (DropPath) regularization. The rate is applied linearly
            across all ConvNeXt V2 blocks. Defaults to 0.0 (no dropout).
        include_rescaling: Boolean, whether to include the input rescaling
            layer (normalizing pixel values to [0, 1] or performing
            ImageNet mean/std normalization if input channels is 3).
            Defaults to False.
        name: Optional string, the name for the Keras model.
        **kwargs: Additional keyword arguments for the Keras Model constructor.

    Attributes:
        depths (list): The block depths used for each stage.
        projection_dims (list): The channel dimensions for each stage.
        pyramid_outputs (dict): Dictionary containing the output feature
            maps for each stage, keyed as 'P1', 'P2', 'P3', 'P4'.
    """

    def __init__(
        self,
        *,
        depths,
        projection_dims,
        input_shape,
        input_tensor=None,
        drop_path_rate=0.0,
        include_rescaling=False,
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1
        inputs = parse_model_inputs(input_shape, input_tensor, name="input_spec")

        x = inputs
        if include_rescaling:
            num_channels = input_shape[-1]
            if num_channels == 3:
                x = PreStem()(x)
            else:
                x = layers.Rescaling(1.0 / 255)(x)

        # feature pyramid outputs
        pyramid_outputs = {}

        # 1. Stem
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=projection_dims[0],
            kernel_size=4,
            strides=4,
            name="stem_conv",
        )(x)
        x = get_norm_layer(layer_type="layer", epsilon=1e-6, name="stem_layernorm")(x)

        # 2. Downsampling and Stages
        num_stages = len(depths)

        # A list to keep track of cumulative drop path rates for Stochastic Depth
        dpr = np.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        cur = 0

        for i in range(num_stages):
            # Downsampling block (except for the first stage which uses the stem)
            if i > 0:
                x = get_norm_layer(
                    layer_type="layer",
                    epsilon=1e-6,
                    name=f"downsampling_layernorm_{i-1}",
                )(x)
                x = get_conv_layer(
                    spatial_dims=spatial_dims,
                    layer_type="conv",
                    filters=projection_dims[i],
                    kernel_size=2,
                    strides=2,
                    name=f"downsampling_conv_{i-1}",
                )(x)

            # ConvNeXt Blocks (Main Stage)
            for j in range(depths[i]):
                # Determine current drop path rate
                current_drop_path_rate = dpr[cur]
                cur += 1
                x = ConvNeXtV2Block(
                    projection_dim=projection_dims[i],
                    drop_path_rate=current_drop_path_rate,
                    name=f"stage_{i}_block_{j}",
                )(x)

            pyramid_outputs[f"P{i+1}"] = get_norm_layer(
                layer_type="layer", epsilon=1e-6, name=f"pyramid_feature_norm_{i+1}"
            )(x)

        super().__init__(
            inputs=inputs, outputs=x, name=name or f"ConvNeXtBackboneV2{spatial_dims}D", **kwargs
        )
        self.depths = depths
        self.pyramid_outputs = pyramid_outputs
        self.projection_dims = projection_dims
        self.drop_path_rate = drop_path_rate
        self.include_rescaling = include_rescaling
        self.name = name

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape[1:],
                "depths": self.depths,
                "projection_dims": self.projection_dims,
                "drop_path_rate": self.drop_path_rate,
                "include_rescaling": self.include_rescaling,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
