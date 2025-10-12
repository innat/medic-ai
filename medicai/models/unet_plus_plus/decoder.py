from keras import layers

from medicai.models.unet.decoder import Conv3x3BnReLU
from medicai.utils import get_act_layer, get_conv_layer, get_norm_layer, get_reshaping_layer

INVALID_BLOCK_TYPE_MSG = "Invalid block_type '{}'. Must be 'upsampling' or 'transpose'."


def UNetPlusPlusDecoder(
    spatial_dims,
    skip_layers,
    decoder_filters,
    block_type,
    use_batchnorm=True,
):
    def upsample_block(x, name_prefix):
        return get_reshaping_layer(
            spatial_dims, layer_type="upsampling", size=2, name=f"{name_prefix}_up"
        )(x)

    def transpose_block(x, filters, name_prefix):
        x = get_conv_layer(
            spatial_dims,
            layer_type="conv_transpose",
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="same",
            name=f"{name_prefix}_transpose",
        )(x)
        if use_batchnorm:
            x = get_norm_layer(layer_type="batch", axis=-1, name=f"{name_prefix}_transpose_bn")(x)
        x = get_act_layer(layer_type="relu", name=f"{name_prefix}_transpose_relu")(x)
        return x

    def apply(x):
        """
        UNet++ Decoder following the original paper with an extra upsampling step
        to handle backbone-specific resolution differences.

        The decoder builds a dense grid of convolutional nodes where each node X_{i,j}
        receives inputs from:
        - Parent node: X_{i+1,j-1} (upsampled from lower resolution)
        - Previous nodes: X_{i,0} to X_{i,j-1} (all nodes in the same row)

        Grid structure for encoder_depth=5 (N=4 upsampling blocks):
        X0,0 -- X0,1 -- X0,2 -- X0,3 -- X0,4 -- [Bridge]
          |       |       |       |       |
        X1,0 -- X1,1 -- X1,2 -- X1,3 -- X1,4
          |       |       |       |
        X2,0 -- X2,1 -- X2,2 -- X2,3
          |       |       |
        X3,0 -- X3,1 -- X3,2
          |
        X4,0

        Where:
        - X_{i,0} are encoder features:
            (X4,0 = bottleneck, X3,0 = P4, X2,0 = P3, X1,0 = P2, X0,0 = P1)
        - Final output is from X_{0,4} with an additional bridge upsampling
        - The bridge upsampling compensates for backbone-specific resolution differences

        Args:
            x: Bottleneck tensor from encoder (deepest feature)

        Returns:
            Final upsampled feature map at input resolution
        """

        # Number of upsampling blocks = encoder_levels - 1 (following paper)
        N = len(decoder_filters) - 1

        # Placeholder for dense grid.
        dense_grid = {}

        # Initialize encoder features
        dense_grid[(N, 0)] = x

        for i in range(len(skip_layers)):
            level_index = N - 1 - i
            dense_grid[(level_index, 0)] = skip_layers[i]

        # Build dense connection.
        for j in range(1, N + 1):
            for i in range(0, N - j + 1):
                node_inputs = []

                # Parent node
                parent_node = (i + 1, j - 1)
                if parent_node in dense_grid:
                    parent_feature = dense_grid[parent_node]

                    # Upsampling.
                    if block_type == "transpose":
                        upsampled = transpose_block(
                            parent_feature,
                            decoder_filters[i],
                            name_prefix=f"x_{i}_{j}",
                        )
                    elif block_type == "upsampling":
                        upsampled = upsample_block(parent_feature, name_prefix=f"x_{i}_{j}")
                    else:
                        raise ValueError(INVALID_BLOCK_TYPE_MSG.format(block_type))

                    node_inputs.append(upsampled)

                # Previous nodes in same row
                for k in range(j):
                    prev_node = (i, k)
                    if prev_node in dense_grid:
                        node_inputs.append(dense_grid[prev_node])

                # Process node
                concat_inputs = layers.Concatenate(axis=-1, name=f"x_{i}_{j}_concat")(node_inputs)

                x_ij = Conv3x3BnReLU(
                    spatial_dims, decoder_filters[i], use_batchnorm, f"x_{i}_{j}_conv1"
                )(concat_inputs)

                if block_type != "transpose":
                    x_ij = Conv3x3BnReLU(
                        spatial_dims, decoder_filters[i], use_batchnorm, f"x_{i}_{j}_conv2"
                    )(x_ij)

                dense_grid[(i, j)] = x_ij

        final_node = dense_grid[(0, N)]

        # Add the extra upsampling that the official code does
        if block_type == "transpose":
            final_output = transpose_block(final_node, decoder_filters[0], "bridge")
        elif block_type == "upsampling":
            final_output = upsample_block(final_node, "bridge")

        return final_output

    return apply
