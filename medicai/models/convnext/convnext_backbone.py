import keras
import numpy as np
from keras import layers

from medicai.utils import DescribeMixin, get_conv_layer, get_norm_layer, parse_model_inputs

from .convnext_layers import ConvNeXtBlock, ConvNeXtV2Block, PreStem


@keras.utils.register_keras_serializable(package="convnext.backbone")
class ConvNeXtBackbone(keras.Model, DescribeMixin):
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
        config = {
            "input_shape": self.input_shape[1:],
            "depths": self.depths,
            "projection_dims": self.projection_dims,
            "drop_path_rate": self.drop_path_rate,
            "layer_scale_init_value": self.layer_scale_init_value,
            "include_rescaling": self.include_rescaling,
            "name": self.name,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package="convnext.backbone")
class ConvNeXtBackboneV2(keras.Model, DescribeMixin):
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
        config = {
            "input_shape": self.input_shape[1:],
            "depths": self.depths,
            "projection_dims": self.projection_dims,
            "drop_path_rate": self.drop_path_rate,
            "include_rescaling": self.include_rescaling,
            "name": self.name,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
