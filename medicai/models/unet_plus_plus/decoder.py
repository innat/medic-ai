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
        x = get_reshaping_layer(
            spatial_dims, layer_type="upsampling", size=2, name=f"{name_prefix}_up"
        )(x)
        return x

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
        UNet++ Decoder following the original paper.
        Grid structure (N=5 example for encoder_depth=5):
        X0,0 -- X0,1 -- X0,2 -- X0,3 -- X0,4 -- X0,5
          |       |       |       |       |
        X1,0 -- X1,1 -- X1,2 -- X1,3 -- X1,4
          |       |       |       |
        X2,0 -- X2,1 -- X2,2 -- X2,3
          |       |       |
        X3,0 -- X3,1 -- X3,2
          |       |
        X4,0 -- X4,1
          |
        X5,0
        """
        N = len(decoder_filters)  # Number of upsampling levels

        # Initialize the dense grid
        dense_grid = {}

        # Bottom layer (j=0): encoder features
        # X_{N,0} is the bottleneck (deepest feature)
        dense_grid[(N, 0)] = x

        # Initialize encoder skip connections (X_{i,0} nodes)
        # skip_layers should be ordered from deepest to shallowest
        for i in range(N):
            if i < len(skip_layers):
                dense_grid[(N - 1 - i, 0)] = skip_layers[i]

        # DENSE CONNECTIONS
        for j in range(1, N + 1):  # columns
            for i in range(0, N - j + 1):  # rows
                # Current node: X_{i,j}
                node_inputs = []

                # 1. Input from parent node (upsampled from lower row, previous column)
                parent_node = (i + 1, j - 1)
                if parent_node in dense_grid:
                    parent_feature = dense_grid[parent_node]

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

                # 2. Input from previous node in same row (lateral connection)
                # prev_node = (i, j - 1)
                # if prev_node in dense_grid:
                #     node_inputs.append(dense_grid[prev_node])
                for k in range(j):
                    prev_node = (i, k)
                    if prev_node in dense_grid:
                        node_inputs.append(dense_grid[prev_node])

                # Concatenate available inputs
                if not node_inputs:
                    continue

                if len(node_inputs) > 1:
                    concatenated = layers.Concatenate(axis=-1, name=f"x_{i}_{j}_concat")(
                        node_inputs
                    )
                else:
                    concatenated = node_inputs[0]

                # Apply convolutions
                x_ij = Conv3x3BnReLU(
                    spatial_dims, decoder_filters[i], use_batchnorm, f"x_{i}_{j}_conv1"
                )(concatenated)

                if block_type != "transpose":
                    x_ij = Conv3x3BnReLU(
                        spatial_dims, decoder_filters[i], use_batchnorm, f"x_{i}_{j}_conv2"
                    )(x_ij)

                dense_grid[(i, j)] = x_ij

        # Final output is from node X_{0,N}
        return dense_grid[(0, N)]

    return apply
