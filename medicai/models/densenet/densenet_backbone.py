import keras
from keras import layers

from ...utils import get_conv_layer, get_pooling_layer, parse_model_inputs
from .densenet_layers import apply_dense_block, apply_transition_layer


@keras.utils.register_keras_serializable(package="densenet.backbone")
class DenseNetBackbone(keras.Model):
    """
    A Dense Convolutional Network (DenseNet) model.

    This class builds a DenseNet model that serves as a feature extractor or
    'backbone' for other tasks like 2D and 3D classification, or
    segmentation. The core idea behind DenseNets is to connect each layer to
    every other layer in a feed-forward fashion, which is different from
    ResNets, which use identity shortcuts.

    Each layer in a DenseNet receives feature maps from all preceding layers in
    its 'Dense Block' and passes its own feature maps to all subsequent layers.
    This creates a dense connectivity pattern, which helps to alleviate the
    vanishing-gradient problem, strengthen feature propagation, encourage
    feature reuse, and significantly reduce the number of parameters. The model
    is composed of multiple Dense Blocks separated by 'Transition Layers'
    that downsample the feature maps.

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
        """
        Initializes the DenseNetBackbone.

        Args:
            blocks: A list of integers specifying the number of dense layers in
                each Dense Block. For example, `[6, 12, 24, 16]` would create
                a DenseNet-121 architecture.
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            input_tensor: (Optional) A Keras tensor to use as the input to the
                model. If not provided, a new input tensor will be created.
            growth_rate: An integer representing the 'growth rate' of the
                network. This is the number of new feature maps added by each
                dense layer.
            bn_size: An integer for the bottleneck layer size. This factor
                multiplies the growth rate to determine the number of filters
                in the 1x1 convolution within the bottleneck.
            compression: A float between 0.0 and 1.0 representing the
                compression factor of the transition layers. A value less than
                1.0 reduces the number of feature maps.
            dropout_rate: A float for the dropout rate to be applied after
                each dense layer.
            include_rescaling: A boolean indicating whether to include a
                `Rescaling` layer at the beginning of the model. If `True`,
                the input pixels will be scaled from `[0, 255]` to `[0, 1]`.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
        spatial_dims = len(input_shape) - 1
        input = parse_model_inputs(input_shape, input_tensor, name="input_spec")

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
