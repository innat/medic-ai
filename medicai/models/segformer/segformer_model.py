import keras
from keras import layers, ops

from medicai.utils import get_conv_layer, get_reshaping_layer, resize_volumes

from .segformer_layers import MixVisionTransformer


class SegFormer(keras.Model):
    def __init__(
        self,
        input_shape,
        num_classes,
        decoder_head_embedding_dim=256,
        classifier_activation=None,
        qkv_bias=True,
        dropout=0.0,
        project_dim=[32, 64, 160, 256],
        layerwise_sr_ratios=[4, 2, 1, 1],
        layerwise_patch_sizes=[7, 3, 3, 3],
        layerwise_strides=[4, 2, 2, 2],
        layerwise_num_heads=[1, 2, 5, 8],
        layerwise_depths=[2, 2, 2, 2],
        layerwise_mlp_ratios=[4, 4, 4, 4],
        name=None,
        **kwargs,
    ):
        """
        SegFormer model for 2D or 3D semantic segmentation.

        This class implements the full SegFormer architecture, which combines a
        hierarchical MixVisionTransformer (MiT) encoder with a lightweight MLP decoder
        head. This design is highly efficient for semantic segmentation tasks on
        high-resolution images or volumes.

        The encoder progressively downsamples the spatial dimensions and increases the
        feature dimensions across four stages, producing multi-scale feature maps.
        The decoder then takes these features, processes them through linear layers,
        upsamples them to a common resolution, and fuses them to generate a
        high-resolution segmentation mask.

        Args:
            input_shape (tuple): The shape of the input data, excluding the batch dimension.
            num_classes (int): The number of output classes for segmentation.
            decoder_head_embedding_dim (int, optional): The embedding dimension of the decoder head.
                Defaults to 256.
            classifier_activation (str, optional): The activation function for the final output layer.
                Common choices are 'softmax' for multi-class segmentation and 'sigmoid' for multi-label
                or binary segmentation. Defaults to None.
            qkv_bias (bool, optional): Whether to include a bias in the query, key, and value projections.
                Defaults to True.
            dropout (float, optional): The dropout rate for the decoder head. Defaults to 0.0.
            project_dim (list[int], optional): A list of feature dimensions for each encoder stage.
                Defaults to [32, 64, 160, 256].
            layerwise_sr_ratios (list[int], optional): A list of spatial reduction ratios for each
                encoder stage's attention layers. Defaults to [4, 2, 1, 1].
            layerwise_patch_sizes (list[int], optional): A list of patch sizes for the embedding layer
                in each encoder stage. Defaults to [7, 3, 3, 3].
            layerwise_strides (list[int], optional): A list of strides for the embedding layer in
                each encoder stage. Defaults to [4, 2, 2, 2].
            layerwise_num_heads (list[int], optional): A list of the number of attention heads for
                each encoder stage. Defaults to [1, 2, 5, 8].
            layerwise_depths (list[int], optional): A list of the number of transformer blocks for
                each encoder stage. Defaults to [2, 2, 2, 2].
            layerwise_mlp_ratios (list[int], optional): A list of MLP expansion ratios for each
                encoder stage. Defaults to [4, 4, 4, 4].
            name (str, optional): The name of the model. Defaults to None.
            **kwargs: Standard Keras Model keyword arguments.
        """
        spatial_dims = len(input_shape) - 1
        mit_backbone = MixVisionTransformer(
            input_shape=input_shape,
            qkv_bias=qkv_bias,
            project_dim=project_dim,
            layerwise_sr_ratios=layerwise_sr_ratios,
            layerwise_patch_sizes=layerwise_patch_sizes,
            layerwise_strides=layerwise_strides,
            layerwise_num_heads=layerwise_num_heads,
            layerwise_depths=layerwise_depths,
            layerwise_mlp_ratios=layerwise_mlp_ratios,
        )
        inputs = mit_backbone.inputs
        skips = [mit_backbone.get_layer(name=f"mixvit_features{i+1}").output for i in range(4)]

        # Pass self to the build_decoder method
        decoder_head = self.build_decoder(
            num_classes, decoder_head_embedding_dim, spatial_dims, dropout
        )
        outputs = decoder_head(skips)

        if classifier_activation:
            outputs = layers.Activation(classifier_activation)(outputs)

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or f"SegFormer{spatial_dims}D", **kwargs
        )

        self.num_classes = num_classes
        self.decoder_head_embedding_dim = decoder_head_embedding_dim
        self.dropout = dropout
        self.classifier_activation = classifier_activation
        self.spatial_dims = spatial_dims
        self.qkv_bias = qkv_bias
        self.project_dim = project_dim
        self.layerwise_sr_ratios = layerwise_sr_ratios
        self.layerwise_patch_sizes = layerwise_patch_sizes
        self.layerwise_strides = layerwise_strides
        self.layerwise_num_heads = layerwise_num_heads
        self.layerwise_depths = layerwise_depths
        self.layerwise_mlp_ratios = layerwise_mlp_ratios

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape[1:],
                "num_classes": self.num_classes,
                "decoder_head_embedding_dim": self.decoder_head_embedding_dim,
                "classifier_activation": self.classifier_activation,
                "qkv_bias": self.qkv_bias,
                "dropout": self.dropout,
                "project_dim": self.project_dim,
                "layerwise_sr_ratios": self.layerwise_sr_ratios,
                "layerwise_patch_sizes": self.layerwise_patch_sizes,
                "layerwise_strides": self.layerwise_strides,
                "layerwise_num_heads": self.layerwise_num_heads,
                "layerwise_depths": self.layerwise_depths,
                "layerwise_mlp_ratios": self.layerwise_mlp_ratios,
            }
        )
        return config

    def build_decoder(self, num_classes, decoder_head_embedding_dim, spatial_dims, dropout):
        def apply(inputs):
            c1, c2, c3, c4 = inputs

            # Get target spatial shape from c1
            target_spatial_shape = ops.shape(c1)[1:-1]

            # Process each feature level with linear embedding and resize to c1 size
            # stage 1
            c4_shape = ops.shape(c4)
            c4 = self.linear_embedding(c4, decoder_head_embedding_dim)
            c4 = self.reshape_to_spatial(c4, c4_shape)
            c4 = self.resize_to_target(c4, target_spatial_shape, spatial_dims)

            # stage 2
            c3_shape = ops.shape(c3)
            c3 = self.linear_embedding(c3, decoder_head_embedding_dim)
            c3 = self.reshape_to_spatial(c3, c3_shape)
            c3 = self.resize_to_target(c3, target_spatial_shape, spatial_dims)

            # stage 3
            c2_shape = ops.shape(c2)
            c2 = self.linear_embedding(c2, decoder_head_embedding_dim)
            c2 = self.reshape_to_spatial(c2, c2_shape)
            c2 = self.resize_to_target(c2, target_spatial_shape, spatial_dims)

            # stage 4
            c1_shape = ops.shape(c1)
            c1 = self.linear_embedding(c1, decoder_head_embedding_dim)
            c1 = self.reshape_to_spatial(c1, c1_shape)

            # Fuse all features (channel-last: concatenate along last axis)
            x = layers.Concatenate(axis=-1)([c1, c2, c3, c4])

            # Fusion convolution
            x = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=decoder_head_embedding_dim,
                kernel_size=1,
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(dropout)(x)

            # Final prediction
            x = get_conv_layer(
                spatial_dims=spatial_dims, layer_type="conv", filters=num_classes, kernel_size=1
            )(x)
            x = get_reshaping_layer(spatial_dims=spatial_dims, layer_type="upsampling", size=4)(x)
            return x

        return apply

    def linear_embedding(self, x, hidden_dims):
        spatial_shape_tensor = ops.shape(x)[1:-1]
        num_patches = int(ops.prod(spatial_shape_tensor))
        x = layers.Reshape((num_patches, ops.shape(x)[-1]))(x)
        x = layers.Dense(hidden_dims)(x)
        x = layers.LayerNormalization()(x)
        return x

    def reshape_to_spatial(self, x, target_shape):
        spatial_shape = target_shape[1:-1]
        x = ops.reshape(x, [-1, *spatial_shape, ops.shape(x)[-1]])
        return x

    def resize_to_target(self, x, target_spatial_shape, spatial_dims):
        if spatial_dims == 3:
            target_depth, target_height, target_width = target_spatial_shape
            x = resize_volumes(x, target_depth, target_height, target_width, method="trilinear")
        elif spatial_dims == 2:
            x = ops.image.resize(x, target_spatial_shape)
        return x
