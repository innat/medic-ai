import keras
from keras import backend, initializers, layers, ops

from medicai.layers import DropPath
from medicai.utils import get_act_layer, get_conv_layer, get_norm_layer


class LayerScale(layers.Layer):
    """Layer scale module.

    References:

    - https://arxiv.org/abs/2103.17239

    Args:
        init_values (float): Initial value for layer scale. Should be within
            [0, 1].
        projection_dim (int): Projection dimensionality.

    Returns:
        Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, _):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


def PreStem(name=None):
    """Normalizes inputs with ImageNet-1k mean and std.

    Args:
      name (str): Name prefix.

    Returns:
      A prestem function.
    """
    if name is None:
        name = "prestem" + str(backend.get_uid("prestem"))

    def apply(x):
        x = layers.Normalization(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            variance=[
                (0.229 * 255) ** 2,
                (0.224 * 255) ** 2,
                (0.225 * 255) ** 2,
            ],
            name=name + "_prestem_normalization",
        )(x)
        return x

    return apply


def ConvNeXtBlock(projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-6, name=None):
    """ConvNeXt block.

    References:
    - https://arxiv.org/abs/2201.03545
    - https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Notes:
        In the original ConvNeXt implementation (linked above), the authors use
        `Dense` layers for pointwise convolutions for increased efficiency.
        Following that, this implementation also uses the same.

    Args:
        projection_dim (int): Number of filters for convolution layers. In the
            ConvNeXt paper, this is referred to as projection dimension.
        drop_path_rate (float): Probability of dropping paths. Should be within
            [0, 1].
        layer_scale_init_value (float): Layer scale value.
            Should be a small float number.
        name: name to path to the keras layer.

    Returns:
        A function representing a ConvNeXtBlock block.
    """
    if name is None:
        name = f"convnext_block{backend.get_uid('convnext_block')}"

    def apply(inputs):
        spatial_dims = len(inputs.shape) - 2

        x = inputs
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="depthwise_conv",
            kernel_size=7,
            padding="same",
            name=f"{name}_depthwise_conv",
        )(x)
        x = get_norm_layer(layer_type="layer", epsilon=1e-6, name=f"{name}_layernorm")(x)
        x = layers.Dense(4 * projection_dim, name=f"{name}_pointwise_conv_1")(x)
        x = get_act_layer(layer_type="gelu", name=f"{name}_gelu")(x)
        x = layers.Dense(projection_dim, name=f"{name}_pointwise_conv_2")(x)

        if layer_scale_init_value is not None:
            x = LayerScale(
                layer_scale_init_value,
                projection_dim,
                name=f"{name}_layer_scale",
            )(x)

        if drop_path_rate:
            layer = DropPath(drop_path_rate, name=f"{name}_stochastic_depth")
        else:
            layer = get_act_layer(layer_type="linear", name=f"{name}_identity")

        return inputs + layer(x)

    return apply


class GlobalResponseNormalization(layers.Layer):
    """
    GlobalResponseNormalization layer for ConvNeXt-V2.
    Reference: https://arxiv.org/abs/2301.00808
    """

    def build(self, input_shape):
        self.spatial_dims = len(input_shape) - 2

        # Use input_shape[-1] for channels_last format
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, 1, 1, input_shape[-1]),
            initializer="zeros",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(1, 1, 1, input_shape[-1]),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        # Ensure computation happens in float32 for numerical stability
        gamma = ops.cast(self.gamma, "float32")
        beta = ops.cast(self.beta, "float32")
        x = ops.cast(inputs, "float32")

        # Compute spatial norm: sqrt(sum(x^2) over spatial dimensions)
        spatial_axes = tuple(range(1, self.spatial_dims + 1))
        spatial_norm = ops.sqrt(
            ops.sum(ops.square(x), axis=spatial_axes, keepdims=True) + keras.config.epsilon()
        )

        # Normalize by mean across channels
        normalized = spatial_norm / (
            ops.mean(spatial_norm, axis=-1, keepdims=True) + keras.config.epsilon()
        )

        # Apply scaling and shifting with residual connection
        result = gamma * (x * normalized) + beta + x

        # Cast back to original dtype
        return ops.cast(result, inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        return config


def ConvNeXtV2Block(projection_dim, drop_path_rate=0.0, name=None):
    """ConvNeXtV2 block constructed using the Keras Functional API.

    References:
      - https://arxiv.org/abs/2301.00808

    Args:
      projection_dim (int): Number of filters for convolution layers.
      drop_path_rate (float): Probability of dropping paths. Should be within [0, 1].
      name (str, optional): Name to path to the Keras layer.

    Returns:
      A function representing a ConvNeXtV2 block that takes an input tensor
      and returns the output tensor.
    """

    def apply(inputs):
        """Applies the ConvNeXtV2 block logic to the input tensor."""

        spatial_dism = len(inputs.shape) - 2

        # Original implementation path: x = inputs
        x = inputs

        # 1. Depthwise Conv (7x7)
        x = get_conv_layer(
            spatial_dims=spatial_dism,
            layer_type="depthwise_conv",
            kernel_size=7,
            padding="same",
            name=f"{name}_depthwise_conv",
        )(x)

        # 2. Layer Norm
        x = get_norm_layer(layer_type="layer", epsilon=1e-6, name=f"{name}_layernorm")(x)

        # 3. Expansion (Pointwise Conv / Dense Layer)
        # Ratio is 4x as per ConvNeXt design
        x = layers.Dense(4 * projection_dim, name=f"{name}_dense_expand")(x)

        # 4. Activation (GELU)
        x = get_act_layer(layer_type="gelu", name=f"{name}_gelu")(x)

        # 5. Global Response Normalization (GRN)
        x = GlobalResponseNormalization()(x)

        # 6. Contraction (Pointwise Conv / Dense Layer)
        x = layers.Dense(projection_dim, name=f"{name}_dense_contract")(x)

        # 7. Residual Connection with Stochastic Depth
        if drop_path_rate > 0.0:
            # Stochastic Depth is used for training regularization
            x = DropPath(drop_path_rate, name=f"{name}_drop_path")(x)

        # Standard residual connection: inputs + x
        # Use Add layer for clear functional graph definition
        output = layers.Add(name=f"{name}_add")([inputs, x])
        return output

    # Return the function that applies the block logic
    return apply
