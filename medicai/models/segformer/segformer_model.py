import keras
import numpy as np
from keras import layers, ops

from medicai.utils import (
    get_conv_layer,
    get_reshaping_layer,
)

from .segformer_layers import MixVisionTransformer


class SegFormer(keras.Model):
    def __init__(
        self,
        input_shape,
        num_classes,
        decoder_head_embedding_dim=256,
        dropout=0.0,
        classifier_activation=None,
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1
        mit_backbone = MixVisionTransformer(input_shape=input_shape)
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

    def build_decoder(self, num_classes, decoder_head_embedding_dim, spatial_dims, dropout):
        def apply(inputs):
            c1, c2, c3, c4 = inputs

            # Get target spatial shape from c1
            target_spatial_shape = ops.shape(c1)[1:-1]

            # Process each feature level with linear embedding and resize to c1 size
            c4_shape = ops.shape(c4)
            c4 = self.linear_embedding(c4, decoder_head_embedding_dim)
            c4 = self.reshape_to_spatial(c4, c4_shape)
            c4 = self.resize_to_target(c4, target_spatial_shape, spatial_dims)

            c3_shape = ops.shape(c3)
            c3 = self.linear_embedding(c3, decoder_head_embedding_dim)
            c3 = self.reshape_to_spatial(c3, c3_shape)
            c3 = self.resize_to_target(c3, target_spatial_shape, spatial_dims)

            c2_shape = ops.shape(c2)
            c2 = self.linear_embedding(c2, decoder_head_embedding_dim)
            c2 = self.reshape_to_spatial(c2, c2_shape)
            c2 = self.resize_to_target(c2, target_spatial_shape, spatial_dims)

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
        num_patches = int(np.prod(spatial_shape_tensor))
        x = layers.Reshape((num_patches, ops.shape(x)[-1]))(x)
        x = layers.Dense(hidden_dims)(x)
        x = layers.LayerNormalization()(x)
        return x

    def reshape_to_spatial(self, x, target_shape):
        batch_size = ops.shape(x)[0]
        spatial_shape = target_shape[1:-1]
        x = ops.reshape(x, [-1, *spatial_shape, ops.shape(x)[-1]])
        return x

    def resize_to_target(self, x, target_spatial_shape, spatial_dims):
        current_shape = ops.shape(x)[1:-1]
        if spatial_dims == 3:
            size_factors = [
                int(ops.round(target_spatial_shape[i] / current_shape[i]))
                for i in range(spatial_dims)
            ]
            x = layers.UpSampling3D(size=size_factors)(x)
        elif spatial_dims == 2:
            x = ops.image.resize(x, target_spatial_shape)
        return x
