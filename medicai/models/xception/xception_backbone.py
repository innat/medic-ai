import keras

from medicai.utils import (
    get_act_layer,
    get_conv_layer,
    get_norm_layer,
    get_pooling_layer,
    parse_model_inputs,
)

# TODO: Incomplete!
class XceptionBackbone(keras.Model):
    """Xception core network with hyperparameters.

    This class implements a Xception backbone as described in
    [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357).

    Most users will want the pretrained presets available with this model. If
    you are creating a custom backbone, this model provides customizability
    through the `stackwise_conv_filters` and `stackwise_pooling` arguments. This
    backbone assumes the same basic structure as the original Xception mode:
    * Residuals and pre-activation everywhere but the first and last block.
    * Conv layers for the first block only, separable conv layers elsewhere.

    Args:
        stackwise_conv_filters: list of list of ints. Each outermost list
            entry represents a block, and each innermost list entry a conv
            layer. The integer value specifies the number of filters for the
            conv layer.
        stackwise_pooling: list of bools. A list of booleans per block, where
            each entry is true if the block should includes a max pooling layer
            and false if it should not.
        image_shape: tuple. The input shape without the batch size.
            Defaults to `(None, None, 3)`.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. If unspecified, the Keras default will be used.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Examples:
    """

    def __init__(
        self,
        conv_filters,
        pooling,
        include_rescaling=False,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        if len(conv_filters) != len(pooling):
            raise ValueError("")

        num_blocks = len(conv_filters)
        spatial_dims = len(input_shape) - 1
        pyramid_outputs = {}

        # Layer shorcuts with common args.
        norm = get_norm_layer(layer_type="batch")
        act = get_act_layer(layer_type="relu")
        conv = get_conv_layer(
            spatial_dims=spatial_dims, layer_type="conv", kernel_size=3, use_bias=False
        )
        sep_conv = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="separable_conv",
            kernel_size=3,
            padding="same",
            use_bias=False,
        )

        point_conv = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            kernel_size=1,
            strides=2,
            padding="same",
            use_bias=False,
        )
        pool = get_pooling_layer(
            layer_type="max",
            pool_size=3,
            strides=2,
            padding="same",
        )

        image_input = parse_model_inputs(shape=input_shape, name="xception_input")
        x = image_input

        if include_rescaling:
            x = keras.layers.Rescaling(1.0 / 255)(x)

        # Iterate through the blocks.
        for block_i in range(num_blocks):
            first_block, last_block = block_i == 0, block_i == num_blocks - 1
            block_filters = conv_filters[block_i]
            use_pooling = pooling[block_i]

            # Save the block input as a residual.
            residual = x
            for conv_i, filters in enumerate(block_filters):
                # First block has post activation and strides on first conv.
                if first_block:
                    prefix = f"block{block_i + 1}_conv{conv_i + 1}"
                    strides = (2, 2) if conv_i == 0 else (1, 1)
                    x = conv(filters, strides=strides, name=prefix)(x)
                    x = norm(name=f"{prefix}_bn")(x)
                    x = act(name=f"{prefix}_act")(x)
                # Last block has post activation.
                elif last_block:
                    prefix = f"block{block_i + 1}_sepconv{conv_i + 1}"
                    x = sep_conv(filters, name=prefix)(x)
                    x = norm(name=f"{prefix}_bn")(x)
                    x = act(name=f"{prefix}_act")(x)
                else:
                    prefix = f"block{block_i + 1}_sepconv{conv_i + 1}"
                    # The first conv in second block has no activation.
                    if block_i != 1 or conv_i != 0:
                        x = act(name=f"{prefix}_act")(x)
                    x = sep_conv(filters, name=prefix)(x)
                    x = norm(name=f"{prefix}_bn")(x)

            # Optional block pooling.
            if use_pooling:
                x = pool(name=f"block{block_i + 1}_pool")(x)

            # Sum residual, first and last block do not have a residual.
            if not first_block and not last_block:
                prefix = f"block{block_i + 1}_residual"
                filters = x.shape[-1]
                # Match filters with a pointwise conv if needed.
                if filters != residual.shape[-1]:
                    residual = point_conv(filters, name=f"{prefix}_conv")(residual)
                    residual = norm(name=f"{prefix}_bn")(residual)
                x = keras.layers.Add(name=f"{prefix}_add")([x, residual])

            print(x.shape)

        super().__init__(
            inputs=image_input,
            outputs=x,
            **kwargs,
        )
        self.conv_filters = conv_filters
        self.pooling = pooling
        self.pyramid_outputs = pyramid_outputs
        self.include_rescaling = include_rescaling
        self._input_shape = input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "conv_filters": self.conv_filters,
                "pooling": self.pooling,
                "include_rescaling": self.include_rescaling,
                "input_shape": self._input_shape,
            }
        )
        return config
