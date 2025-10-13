from keras import backend, initializers, layers

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
      A presemt function.
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
        name = f"prestem{str(backend.get_uid('prestem'))}"

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
