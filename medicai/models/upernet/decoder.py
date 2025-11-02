from keras import layers, ops

from medicai.layers import ConvBnAct, ResizingND
from medicai.utils import get_norm_layer, get_pooling_layer


def PyramidPoolingModule(
    out_channels,
    pool_scales=(1, 2, 3, 6),
    decoder_normalization="batch",
    decoder_activation="relu",
    prefix="upernet_ppm",
):
    """Pyramid Pooling Module (PPM) that captures multi-scale context by pooling
    the input feature map into multiple grid scales, applying 1x1 conv to each,
    resizing them back to the original size, and concatenating them.
    """
    # 1. Create pooling blocks (The 1x1 ConvBnAct layers)
    pooling_blocks = []
    for size in pool_scales:

        def create_pooling_block(size=size):
            def apply_block(x):
                spatial_dims = len(x.shape) - 2

                # 1. Adaptive average pooling to target grid size (e.g., 1x1, 2x2, 3x3, 6x6)
                x = get_pooling_layer(
                    spatial_dims=spatial_dims, layer_type="adaptive_avg", output_size=size
                )(x)

                # 2. 1x1 Conv + BN + ReLU for channel reduction
                x = ConvBnAct(
                    out_channels,
                    kernel_size=1,
                    padding="valid",
                    activation=decoder_activation,
                    normalization=decoder_normalization,
                    name=f"{prefix}_conv_1x1_s{size}",
                )(x)
                return x

            return apply_block

        pooling_blocks.append(create_pooling_block(size))

    def apply(feature):
        spatial_dims = len(feature.shape) - 2
        input_spatial_shape = ops.shape(feature)[1:-1]

        # List to collect multi-scale pooled features
        pyramid_features = [feature]

        # Apply each pooling block and resize back to the original feature size
        for i, block in enumerate(pooling_blocks):
            pooled_feature = block(feature)

            # Resize back to original spatial dimensions
            resized_feature = ResizingND(
                target_shape=input_spatial_shape,
                interpolation="bilinear" if spatial_dims == 2 else "trilinear",
                name=f"{prefix}_resize_s{i+1}",
            )(pooled_feature)
            pyramid_features.append(resized_feature)

        # Concatenate all pooled and resized features
        fused = layers.Concatenate(axis=-1, name=f"{prefix}_concat")(pyramid_features)

        # Final 3x3 Conv + BN + ReLU to fuse multi-scale context
        fused_output = ConvBnAct(
            out_channels,
            kernel_size=3,
            padding="same",
            activation=decoder_activation,
            normalization=decoder_normalization,
            name=f"{prefix}_final_3x3",
        )(fused)
        return fused_output

    return apply


def UPerNetDecoder(
    spatial_dims,
    skip_layers,
    decoder_filters,
    decoder_normalization="batch",
    decoder_activation="relu",
    prefix="upernet_decoder",
):
    """Combines PSP (Pyramid Pooling Module) with a Feature Pyramid Network (FPN)
    to merge high- and low-level features for rich semantic segmentation output.
    """

    def apply(bottleneck):
        # 1. Normalize all features (bottleneck + skip layers)
        all_features = [bottleneck] + skip_layers  # [P5, P4, P3, P2, P1]
        normalized_features = []
        for i, feat in enumerate(all_features):
            norm_layer = get_norm_layer(
                layer_type="batch",
                axis=-1,
                epsilon=1e-6,
                name=f"{prefix}_norm_p{len(all_features) - i}",
            )
            normalized_features.append(norm_layer(feat))

        psp_input = normalized_features[0]  # [P5]
        fpn_lateral_features = normalized_features[1:]  # [P4, P3, P2, P1]

        # Create level names dynamically based on number of skip layers + bottleneck
        num_levels = len(skip_layers) + 1
        level_names = [f"p{num_levels - i}" for i in range(num_levels)]

        # 2. Pass lowest resolution feature to PSP module
        psp_out = PyramidPoolingModule(
            out_channels=decoder_filters,
            pool_scales=(1, 2, 3, 6),
            decoder_normalization=decoder_normalization,
            decoder_activation=decoder_activation,
            prefix=f"{prefix}_ppm",
        )(psp_input)

        # 3. Feature Pyramid Network (FPN)
        fpn_features = [psp_out]
        fpn_conv_blocks = [
            ConvBnAct(
                decoder_filters,
                kernel_size=3,
                padding="same",
                activation=decoder_activation,
                normalization=decoder_normalization,
                name=f"{prefix}_fpn_conv_3x3_{level_names[i+1]}",
            )
            for i in range(len(fpn_lateral_features))
        ]

        # Iterate through lateral features (P4 → P3 → P2 → P1) - bottom-up.
        for i, lateral_feature in enumerate(fpn_lateral_features):
            level = level_names[i + 1]
            state_feature = fpn_features[-1]

            # 1. Lateral 1x1 Conv (align channels)
            lateral_feature = ConvBnAct(
                decoder_filters,
                kernel_size=1,
                padding="valid",
                activation=decoder_activation,
                normalization=decoder_normalization,
                name=f"{prefix}_fpn_lateral_1x1_{level}",
            )(lateral_feature)

            # 2. Resize state to match lateral resolution
            target_spatial_shape = ops.shape(lateral_feature)[1:-1]
            state_feature = ResizingND(
                target_shape=target_spatial_shape,
                interpolation="bilinear" if spatial_dims == 2 else "trilinear",
                name=f"{prefix}_fpn_resize_{level}",
            )(state_feature)

            # 3. Fuse (Add)
            fused = layers.Add(name=f"{prefix}_fpn_add_{level}")([state_feature, lateral_feature])

            # 4. Post-FPN Conv
            fused = fpn_conv_blocks[i](fused)

            fpn_features.append(fused)

        # 4. Upsample all FPN outputs to the highest resolution (P1)
        target_spatial_shape = ops.shape(fpn_features[-1])[1:-1]
        resized_fpn_features = []

        for i, feature in enumerate(fpn_features):
            resized_feature = ResizingND(
                target_shape=target_spatial_shape,
                interpolation="bilinear" if spatial_dims == 2 else "trilinear",
                name=f"{prefix}_fpn_resize_to_p1_{i}",
            )(feature)
            resized_fpn_features.append(resized_feature)

        # 5. Concatenate all features (reverse order: P1→P2→P3→P4→P5)
        stacked_features = layers.Concatenate(axis=-1, name=f"{prefix}_fpn_concat")(
            resized_fpn_features[::-1]
        )

        # 6. Final fusion 3x3 Conv to unify decoder output
        fused_output = ConvBnAct(
            decoder_filters,
            kernel_size=3,
            padding="same",
            activation=decoder_activation,
            normalization=decoder_normalization,
            name=f"{prefix}_fpn_fusion_3x3",
        )(stacked_features)

        return fused_output

    return apply
