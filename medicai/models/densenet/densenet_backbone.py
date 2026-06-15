import keras
from keras import layers

from medicai.utils import DescribeMixin, get_conv_layer, get_pooling_layer, parse_model_inputs

from .densenet_layers import apply_dense_block, apply_transition_layer


@keras.utils.register_keras_serializable(package="densenet.backbone")
class DenseNetBackbone(keras.Model, DescribeMixin):
    """
    This class builds only the backbone portion of a DenseNet model. It is
    intended for workflows that need reusable feature maps rather than a final
    classification layer, such as custom classifiers, detection heads, or
    segmentation decoders.

    The backbone is constructed in the following stages:

    1. An input layer is created from ``input_shape`` or ``input_tensor``.
    2. An optional ``Rescaling`` layer normalizes raw image intensities.
    3. A convolutional stem applies a strided ``7x7`` convolution, batch
       normalization, ReLU activation, and max pooling to produce the first
       lower-resolution feature map.
    4. A sequence of dense blocks is applied. After each block, the feature map
       is stored in ``pyramid_outputs``. Between blocks, transition layers
       compress the channel dimension and downsample the spatial resolution.
    5. A final batch normalization and ReLU activation are applied to the last
       feature map, which is returned as the output of the model.

    Args:
        blocks: A list of integers specifying the number of dense layers in
            each dense block. The length of this list determines how many
            pyramid levels are produced after the stem.
        input_shape: A tuple specifying the input shape of the model,
            not including the batch size.
        input_tensor: (Optional) A Keras tensor to use as the input to the
            model. If not provided, a new input tensor will be created.
        growth_rate: An integer specifying how many new feature channels
            each dense layer contributes.
        bn_size: An integer bottleneck multiplier used to size the
            intermediate ``1x1`` convolution inside each dense layer.
        compression: A float between 0.0 and 1.0 representing the
            compression factor of transition layers. Values below ``1.0``
            reduce the number of output channels before downsampling.
        dropout_rate: A float specifying the dropout rate applied inside
            dense layers.
        include_rescaling: A boolean indicating whether to include a
            `Rescaling` layer at the beginning of the model. If `True`,
            the input pixels will be scaled from `[0, 255]` to `[0, 1]`.
        name: (Optional) The name of the model.

    Returns:
        A ``keras.Model`` whose forward pass returns the final backbone
        feature tensor. Intermediate multi-scale features are available in
        the ``pyramid_outputs`` attribute.

    Examples:
        .. code-block:: python

            import torch
            from medicai.models.densenet import DenseNetBackbone

            model = DenseNetBackbone(
                input_shape=(224, 224, 3),
                blocks=[6, 12, 24, 16],
                growth_rate=32,
                bn_size=4,
                compression=0.5,
                dropout_rate=0.0,
                name="densenet_backbone",
            )
            x = torch.randn((1, 224, 224, 3))
            y = model(x)
            print(y.shape)  # torch.Size([1, 7, 7, 1024])


    References:
     
     - Densely Connected Convolutional Networks. CVPR 2017. `arXiv:1608.06993 <https://arxiv.org/abs/1608.06993>`_
    """

    def __init__(
        self,
        *,
        blocks,
        input_shape,
        input_tensor=None,
        growth_rate=32,
        bn_size=4,
        compression=0.5,
        dropout_rate=0.0,
        include_rescaling=False,
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1
        input = parse_model_inputs(input_shape, input_tensor, name="input_spec")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        x = input
        if include_rescaling:
            x = layers.Rescaling(1.0 / 255)(x)

        # Initial convolution stem
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=64,
            kernel_size=7,
            strides=2,
            padding="same",
            use_bias=False,
            name="stem_dense_conv",
        )(x)
        x = layers.BatchNormalization(name="stem_dense_bn")(x)
        x = layers.Activation("relu", name="stem_dense_relu")(x)

        pyramid_outputs = {"P1": x}

        x = get_pooling_layer(
            spatial_dims=spatial_dims,
            layer_type="max",
            pool_size=3,
            strides=2,
            padding="same",
        )(x)

        # Dense blocks and transitions
        num_channels = 64

        for i, num_layers in enumerate(blocks):
            x = apply_dense_block(x, num_layers, growth_rate, bn_size, dropout_rate, block_idx=i)
            num_channels += num_layers * growth_rate

            pyramid_outputs[f"P{i+2}"] = x

            if i != len(blocks) - 1:
                out_channels = int(num_channels * compression)
                x = apply_transition_layer(x, out_channels, block_idx=i)
                num_channels = out_channels

        # Final batch norm
        x = layers.BatchNormalization(name="final_bn")(x)
        x = layers.Activation("relu", name="final_relu")(x)

        super().__init__(
            inputs=input, outputs=x, name=name or f"DenseNetBackbone{spatial_dims}D", **kwargs
        )

        self.blocks = blocks
        self.growth_rate = growth_rate
        self.bn_size = bn_size
        self.compression = compression
        self.dropout_rate = dropout_rate
        self.pyramid_outputs = pyramid_outputs
        self.include_rescaling = include_rescaling
        self.name = name

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "blocks": self.blocks,
            "growth_rate": self.growth_rate,
            "bn_size": self.bn_size,
            "compression": self.compression,
            "dropout_rate": self.dropout_rate,
            "include_rescaling": self.include_rescaling,
            "name": self.name,
        }
        return config
