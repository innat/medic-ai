import keras

from medicai.utils import (
    DescribeMixin,
    get_act_layer,
    get_conv_layer,
    get_norm_layer,
    get_pooling_layer,
    parse_model_inputs,
)


@keras.utils.register_keras_serializable(package="xception.backbone")
class XceptionBackbone(keras.Model, DescribeMixin):
    """
    The backbone is constructed in the following stages:

    1. An input layer is created from ``input_shape``.
    2. An optional ``Rescaling`` layer normalizes raw image intensities.
    3. An entry flow applies initial convolutions and residual downsampling
       blocks to progressively reduce spatial resolution.
    4. A middle flow applies repeated separable-convolution residual blocks at
       constant resolution.
    5. An exit flow expands the channel dimension and produces the final
       backbone feature tensor.
    6. Intermediate feature maps from the main stage boundaries are stored in
       ``pyramid_outputs``.

    Args:
        input_shape: A tuple specifying the input shape of the model, not
            including the batch size. This can describe either 2D or 3D
            inputs.
        include_rescaling: A boolean indicating whether to include a
            ``Rescaling`` layer at the beginning of the model.
        name: (Optional) The name of the model.
        **kwargs: Additional keyword arguments.

    Returns:
        A ``keras.Model`` whose forward pass returns the final backbone
        feature tensor. Intermediate multi-scale features are available in
        the ``pyramid_outputs`` attribute.

    Example:
        Build the backbone and inspect the pyramid features::

            import torch
            from medicai.models import XceptionBackbone

            model = XceptionBackbone(
                input_shape=(224, 224, 3),
                include_rescaling=True,
                name="xception_backbone",
            )
            x = torch.randn((1, 224, 224, 3))
            y = model(x)
            print(y.shape)  # torch.Size([1, 7, 7, 2048])


    References:
        - Xception: Deep Learning with Depthwise Separable Convolutions.
          `arXiv:1610.02357 <https://arxiv.org/abs/1610.02357>`_

    """

    def __init__(
        self,
        input_shape=(None, None, 3),
        include_rescaling=False,
        name=None,
        **kwargs,
    ):

        spatial_dims = len(input_shape) - 1
        pyramid_outputs = {}

        inputs = parse_model_inputs(input_shape, name="xception_input")

        x = inputs
        if include_rescaling:
            x = keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

        # Block 1
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=32,
            kernel_size=3,
            padding="same",
            strides=2,
            use_bias=False,
            name="block1_conv1",
        )(x)
        x = get_norm_layer(layer_type="batch", name="block1_conv1_bn")(x)
        x = get_act_layer(layer_type="relu", name="block1_conv1_act")(x)
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=64,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="block1_conv2",
        )(x)
        x = get_norm_layer(layer_type="batch", name="block1_conv2_bn")(x)
        x = get_act_layer(layer_type="relu", name="block1_conv2_act")(x)
        pyramid_outputs["P1"] = x

        # Block 2
        residual = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=128,
            kernel_size=1,
            strides=2,
            padding="same",
            use_bias=False,
        )(x)
        residual = get_norm_layer(layer_type="batch")(residual)

        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="separable_conv",
            filters=128,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="block2_sepconv1",
        )(x)
        x = get_norm_layer(layer_type="batch", name="block2_sepconv1_bn")(x)
        x = get_act_layer(layer_type="relu", name="block2_sepconv2_act")(x)
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="separable_conv",
            filters=128,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="block2_sepconv2",
        )(x)
        x = get_norm_layer(layer_type="batch", name="block2_sepconv2_bn")(x)
        x = get_pooling_layer(
            spatial_dims=spatial_dims,
            layer_type="max",
            pool_size=3,
            strides=2,
            padding="same",
            name="block2_pool",
        )(x)
        x = keras.layers.add([x, residual])
        pyramid_outputs["P2"] = x

        # Block 3
        residual = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=256,
            kernel_size=1,
            strides=2,
            padding="same",
            use_bias=False,
        )(x)
        residual = get_norm_layer(layer_type="batch")(residual)

        x = get_act_layer(layer_type="relu", name="block3_sepconv1_act")(x)
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="separable_conv",
            filters=256,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="block3_sepconv1",
        )(x)
        x = get_norm_layer(layer_type="batch", name="block3_sepconv1_bn")(x)
        x = get_act_layer(layer_type="relu", name="block3_sepconv2_act")(x)
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="separable_conv",
            filters=256,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="block3_sepconv2",
        )(x)
        x = get_norm_layer(layer_type="batch", name="block3_sepconv2_bn")(x)

        x = get_pooling_layer(
            spatial_dims=spatial_dims,
            layer_type="max",
            pool_size=3,
            strides=2,
            padding="same",
            name="block3_pool",
        )(x)
        x = keras.layers.add([x, residual])
        pyramid_outputs["P3"] = x

        # Block 4
        residual = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=728,
            kernel_size=1,
            strides=2,
            padding="same",
            use_bias=False,
        )(x)
        residual = get_norm_layer(layer_type="batch")(residual)

        x = get_act_layer(layer_type="relu", name="block4_sepconv1_act")(x)
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="separable_conv",
            filters=728,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="block4_sepconv1",
        )(x)
        x = get_norm_layer(layer_type="batch", name="block4_sepconv1_bn")(x)

        x = get_act_layer(layer_type="relu", name="block4_sepconv2_act")(x)
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="separable_conv",
            filters=728,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="block4_sepconv2",
        )(x)
        x = get_norm_layer(layer_type="batch", name="block4_sepconv2_bn")(x)

        x = get_pooling_layer(
            spatial_dims=spatial_dims,
            layer_type="max",
            pool_size=3,
            strides=2,
            padding="same",
            name="block4_pool",
        )(x)
        x = keras.layers.add([x, residual])
        pyramid_outputs["P4"] = x

        # Blocks 5–12
        for i in range(8):
            residual = x
            prefix = "block" + str(i + 5)

            x = get_act_layer(layer_type="relu", name=prefix + "_sepconv1_act")(x)
            x = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="separable_conv",
                filters=728,
                kernel_size=3,
                padding="same",
                use_bias=False,
                name=prefix + "_sepconv1",
            )(x)

            x = get_norm_layer(layer_type="batch", name=prefix + "_sepconv1_bn")(x)
            x = get_act_layer(layer_type="relu", name=prefix + "_sepconv2_act")(x)

            x = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="separable_conv",
                filters=728,
                kernel_size=3,
                padding="same",
                use_bias=False,
                name=prefix + "_sepconv2",
            )(x)

            x = get_norm_layer(layer_type="batch", name=prefix + "_sepconv2_bn")(x)
            x = get_act_layer(layer_type="relu", name=prefix + "_sepconv3_act")(x)

            x = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="separable_conv",
                filters=728,
                kernel_size=3,
                padding="same",
                use_bias=False,
                name=prefix + "_sepconv3",
            )(x)
            x = get_norm_layer(layer_type="batch", name=prefix + "_sepconv3_bn")(x)
            x = keras.layers.add([x, residual])

        residual = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=1024,
            kernel_size=1,
            strides=2,
            padding="same",
            use_bias=False,
        )(x)
        residual = get_norm_layer(layer_type="batch")(residual)
        x = get_act_layer(layer_type="relu", name="block13_sepconv1_act")(x)

        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="separable_conv",
            filters=728,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="block13_sepconv1",
        )(x)
        x = get_norm_layer(layer_type="batch", name="block13_sepconv1_bn")(x)
        x = get_act_layer(layer_type="relu", name="block13_sepconv2_act")(x)

        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="separable_conv",
            filters=1024,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="block13_sepconv2",
        )(x)
        x = get_norm_layer(layer_type="batch", name="block13_sepconv2_bn")(x)

        x = get_pooling_layer(
            spatial_dims=spatial_dims,
            layer_type="max",
            pool_size=3,
            strides=2,
            padding="same",
            name="block13_pool",
        )(x)
        x = keras.layers.add([x, residual])
        pyramid_outputs["P5"] = x

        # Block 14
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="separable_conv",
            filters=1536,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="block14_sepconv1",
        )(x)

        x = get_norm_layer(layer_type="batch", name="block14_sepconv1_bn")(x)
        x = get_act_layer(layer_type="relu", name="block14_sepconv1_act")(x)

        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="separable_conv",
            filters=2048,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="block14_sepconv2",
        )(x)

        x = get_norm_layer(layer_type="batch", name="block14_sepconv2_bn")(x)
        x = get_act_layer(layer_type="relu", name="block14_sepconv2_act")(x)

        super().__init__(
            inputs=inputs,
            outputs=x,
            name=name or f"XceptionBackbone{spatial_dims}D",
            **kwargs,
        )
        self.pyramid_outputs = pyramid_outputs
        self.include_rescaling = include_rescaling
        self._input_shape = input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_rescaling": self.include_rescaling,
                "input_shape": self._input_shape,
            }
        )
        return config
