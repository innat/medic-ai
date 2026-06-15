import keras
from keras import layers

from medicai.utils import DescribeMixin, get_pooling_layer, registration, validate_activation

from .mit_backbone import MiTBackbone


@keras.saving.register_keras_serializable(package="mit")
class MiTBase(keras.Model):
    """
    Base class for the Mix Transformer (MiT) model family.

    MiT is a highly efficient and versatile **vision transformer** architecture.
    It is designed to be suitable for various computer vision tasks, particularly
    dense prediction tasks like semantic segmentation.
    """

    def __init__(
        self,
        *,
        input_shape,
        max_drop_path_rate,
        layer_norm_epsilon,
        qkv_bias,
        project_dim,
        sr_ratios,
        patch_sizes,
        strides,
        num_heads,
        depths,
        mlp_ratios,
        include_rescaling=False,
        pooling=None,
        include_top=True,
        num_classes=1000,
        classifier_activation=None,
        name=None,
        **kwargs,
    ):
        """
        Initializes the MiTBase model.

        Args:
            input_shape: A tuple specifying the input shape (H, W, C) or (D, H, W, C).
                The batch size is excluded.
            max_drop_path_rate: The maximum drop path rate to be used in the backbone.
                The actual rate is scaled linearly across layers.
            layer_norm_epsilon: Epsilon for the Layer Normalization layers.
            qkv_bias: Boolean indicating whether to include a bias term for the
                Query, Key, and Value projections in the attention mechanism.
            project_dim: A list of integers specifying the feature dimension
                (number of channels) for the output of each of the four MiT stages.
            sr_ratios: A list of integers specifying the **Spatial-Reduction Ratio (r)**
                for the Spatially-Reduced Attention in each stage. A larger 'r'
                reduces computation but can affect performance (e.g., `[8, 4, 2, 1]`).
            patch_sizes: A list of integers specifying the kernel size for the
                Overlapping Patch Embedding module in each stage.
            strides: A list of integers specifying the stride for the Overlapping
                Patch Embedding module in each stage.
            num_heads: A list of integers specifying the number of attention heads
                in the attention block for each stage.
            depths: A list of integers specifying the number of Transformer
                Encoder blocks (depth) in each of the four stages.
            mlp_ratios: A list of integers specifying the expansion ratio for the
                hidden layer in the MLP block of each Transformer Encoder block.
                The hidden dimension is `project_dim * mlp_ratio`.
            include_rescaling: Boolean, whether to include a rescaling layer
                at the input for input normalization. Default: False.
            pooling: Optional string, only relevant when `include_top=False`.
                Specifies the type of global pooling to apply to the output features:
                'avg' for Global Average Pooling, 'max' for Global Max Pooling.
                Default: None (no pooling).
            include_top: Boolean, whether to include the final classification layer
                (a Dense layer with `num_classes` outputs). Default: True.
            num_classes: Optional integer, number of classes to classify. Only
                relevant if `include_top` is True. Default: 1000.
            classifier_activation: Optional activation function for the final
                classification layer. Only relevant if `include_top` is True.
                Default: None (linear).
            name: Optional name for the model. Default: 'MiTBaseND'.
            **kwargs: Additional keyword arguments passed to the parent `keras.Model`
                constructor.
        """

        spatial_dims = len(input_shape) - 1
        if name is None and self.__class__ is not MiTBase:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        # number of classes must be positive.
        if include_top and num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        # verify input activation.
        classifier_activation = validate_activation(classifier_activation)

        backbone = MiTBackbone(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            max_drop_path_rate=max_drop_path_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            qkv_bias=qkv_bias,
            project_dim=project_dim,
            sr_ratios=sr_ratios,
            patch_sizes=patch_sizes,
            strides=strides,
            num_heads=num_heads,
            depths=depths,
            mlp_ratios=mlp_ratios,
        )
        inputs = backbone.input
        x = backbone.output

        GlobalAvgPool = get_pooling_layer(
            spatial_dims=spatial_dims, layer_type="avg", global_pool=True
        )
        GlobalMaxPool = get_pooling_layer(
            spatial_dims=spatial_dims, layer_type="max", global_pool=True
        )
        if include_top:
            x = GlobalAvgPool(x)
            x = layers.Dense(num_classes, activation=classifier_activation, name="predictions")(x)
        elif pooling == "avg":
            x = GlobalAvgPool(x)
        elif pooling == "max":
            x = GlobalMaxPool(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.pyramid_outputs = backbone.pyramid_outputs
        self.max_drop_path_rate = max_drop_path_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.qkv_bias = qkv_bias
        self.project_dim = project_dim
        self.sr_ratios = sr_ratios
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.num_heads = num_heads
        self.depths = depths
        self.mlp_ratios = mlp_ratios
        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.num_classes = num_classes
        self.pooling = pooling
        self.classifier_activation = classifier_activation
        self.name = name

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "include_top": self.include_top,
            "include_rescaling": self.include_rescaling,
            "num_classes": self.num_classes,
            "pooling": self.pooling,
            "classifier_activation": self.classifier_activation,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="mit")
@registration.register(name="mit_b0", family="mit")
class MixViTB0(MiTBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        """
        Default Parameters (MiT-B0 specific):
            max_drop_path_rate: Float, the maximum drop path rate for regularization.
                *Effect:* A higher value increases regularization to prevent overfitting.
                Default: 0.1.
            layer_norm_epsilon: Float, a small value added to the variance in the layer
                normalization calculation to prevent division by zero.
                *Effect:* Ensures numerical stability during training. Default: 1e-6.
            qkv_bias: Boolean, whether to use bias terms for the query, key, and value
                linear layers in the attention mechanism.
                *Effect:* If True, adds an extra learnable parameter to each attention head.
                Default: True.
            project_dim: List of integers, the dimensions for the feature projection
                in each of the four stages of the encoder.
                *Effect:* Defines the channel depth of the feature maps in each stage.
                Default: [32, 64, 160, 256].
            sr_ratios: List of integers, the spatial reduction ratios used in the
                efficient self-attention mechanism for each stage.
                *Effect:* Reduces the computational cost of self-attention by shrinking
                the key and value maps. Default: [4, 2, 1, 1].
            patch_sizes: List of integers, the sizes of the patches created in each stage.
                *Effect:* Controls the initial tokenization and resolution of the feature maps.
                Default: [7, 3, 3, 3].
            strides: List of integers, the strides used for the patch embedding layers.
                *Effect:* Determines the downsampling rate between stages. Default: [4, 2, 2, 2].
            num_heads: List of integers, the number of attention heads in each stage.
                *Effect:* Allows the model to jointly attend to information from different
                representational subspaces. Default: [1, 2, 5, 8].
            depths: List of integers, the number of transformer blocks in each stage.
                *Effect:* Defines the number of layers in each of the four stages. Default: [2, 2, 2, 2].
            mlp_ratios: List of integers, the expansion ratio for the hidden dimension
                of the MLP block in each stage.
                *Effect:* Controls the capacity of the feed-forward network. Default: [4, 4, 4, 4].
        """
        spatial_dims = len(input_shape) - 1
        sr_ratio = [8, 4, 2, 1] if spatial_dims == 2 else [4, 2, 1, 1]

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[32, 64, 160, 256],
            sr_ratios=sr_ratio,
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            depths=[2, 2, 2, 2],
            mlp_ratios=[4, 4, 4, 4],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="mit")
@registration.register(name="mit_b1", family="mit")
class MixViTB1(MiTBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        """
        Default Parameters (MiT-B1 specific):
            max_drop_path_rate: Float, the maximum drop path rate for regularization.
                *Effect:* A higher value increases regularization to prevent overfitting.
                Default: 0.1.
            layer_norm_epsilon: Float, a small value added to the variance in the layer
                normalization calculation to prevent division by zero.
                *Effect:* Ensures numerical stability during training. Default: 1e-6.
            qkv_bias: Boolean, whether to use bias terms for the query, key, and value
                linear layers in the attention mechanism.
                *Effect:* If True, adds an extra learnable parameter to each attention head.
                Default: True.
            project_dim: List of integers, the dimensions for the feature projection
                in each of the four stages of the encoder.
                *Effect:* Defines the channel depth of the feature maps in each stage.
                Default: [64, 128, 320, 512].
            sr_ratios: List of integers, the spatial reduction ratios used in the
                efficient self-attention mechanism for each stage.
                *Effect:* Reduces the computational cost of self-attention by shrinking
                the key and value maps. **Default: [8, 4, 2, 1]**.
            patch_sizes: List of integers, the sizes of the patches created in each stage.
                *Effect:* Controls the initial tokenization and resolution of the feature maps.
                Default: [7, 3, 3, 3].
            strides: List of integers, the strides used for the patch embedding layers.
                *Effect:* Determines the downsampling rate between stages. Default: [4, 2, 2, 2].
            num_heads: List of integers, the number of attention heads in each stage.
                *Effect:* Allows the model to jointly attend to information from different
                representational subspaces. Default: [1, 2, 5, 8].
            depths: List of integers, the number of transformer blocks in each stage.
                *Effect:* Defines the number of layers in each of the four stages.
                Default: [2, 2, 2, 2].
            mlp_ratios: List of integers, the expansion ratio for the hidden dimension
                of the MLP block in each stage.
                *Effect:* Controls the capacity of the feed-forward network. Default: [4, 4, 4, 4].
        """

        spatial_dims = len(input_shape) - 1
        sr_ratio = [8, 4, 2, 1] if spatial_dims == 2 else [4, 2, 1, 1]

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[64, 128, 320, 512],
            sr_ratios=sr_ratio,
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            depths=[2, 2, 2, 2],
            mlp_ratios=[4, 4, 4, 4],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="mit")
@registration.register(name="mit_b2", family="mit")
class MixViTB2(MiTBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        """
        Default Parameters (MiT-B2 specific):
            max_drop_path_rate: Float, the maximum drop path rate for regularization.
                *Effect:* A higher value increases regularization to prevent overfitting.
                Default: 0.1.
            layer_norm_epsilon: Float, a small value added to the variance in the layer
                normalization calculation to prevent division by zero.
                *Effect:* Ensures numerical stability during training. Default: 1e-6.
            qkv_bias: Boolean, whether to use bias terms for the query, key, and value
                linear layers in the attention mechanism.
                *Effect:* If True, adds an extra learnable parameter to each attention head.
                Default: True.
            project_dim: List of integers, the dimensions for the feature projection
                in each of the four stages of the encoder.
                *Effect:* Defines the channel depth of the feature maps in each stage.
                Default: [64, 128, 320, 512].
            sr_ratios: List of integers, the spatial reduction ratios used in the
                efficient self-attention mechanism for each stage.
                *Effect:* Reduces the computational cost of self-attention by shrinking
                the key and value maps. Default: [8, 4, 2, 1].
            patch_sizes: List of integers, the sizes of the patches created in each stage.
                *Effect:* Controls the initial tokenization and resolution of the feature maps.
                Default: [7, 3, 3, 3].
            strides: List of integers, the strides used for the patch embedding layers.
                *Effect:* Determines the downsampling rate between stages. Default: [4, 2, 2, 2].
            num_heads: List of integers, the number of attention heads in each stage.
                *Effect:* Allows the model to jointly attend to information from different
                representational subspaces. Default: [1, 2, 5, 8].
            depths: List of integers, the number of transformer blocks in each stage.
                *Effect:* Defines the number of layers in each of the four stages.
                Default: [3, 4, 6, 3].
            mlp_ratios: List of integers, the expansion ratio for the hidden dimension
                of the MLP block in each stage.
                *Effect:* Controls the capacity of the feed-forward network. Default: [4, 4, 4, 4].
        """

        spatial_dims = len(input_shape) - 1
        sr_ratio = [8, 4, 2, 1] if spatial_dims == 2 else [4, 2, 1, 1]

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[64, 128, 320, 512],
            sr_ratios=sr_ratio,
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            depths=[3, 4, 6, 3],
            mlp_ratios=[4, 4, 4, 4],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="mit")
@registration.register(name="mit_b3", family="mit")
class MixViTB3(MiTBase, DescribeMixin):

    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        """
        Default Parameters (MiT-B3 specific):
            max_drop_path_rate: Float, the maximum drop path rate for regularization.
                *Effect:* A higher value increases regularization to prevent overfitting.
                Default: 0.1.
            layer_norm_epsilon: Float, a small value added to the variance in the layer
                normalization calculation to prevent division by zero.
                *Effect:* Ensures numerical stability during training. Default: 1e-6.
            qkv_bias: Boolean, whether to use bias terms for the query, key, and value
                linear layers in the attention mechanism.
                *Effect:* If True, adds an extra learnable parameter to each attention head.
                Default: True.
            project_dim: List of integers, the dimensions for the feature projection
                in each of the four stages of the encoder.
                *Effect:* Defines the channel depth of the feature maps in each stage.
                Default: [64, 128, 320, 512].
            sr_ratios: List of integers, the spatial reduction ratios used in the
                efficient self-attention mechanism for each stage.
                *Effect:* Reduces the computational cost of self-attention by shrinking
                the key and value maps. Default: [8, 4, 2, 1].
            patch_sizes: List of integers, the sizes of the patches created in each stage.
                *Effect:* Controls the initial tokenization and resolution of the feature maps.
                Default: [7, 3, 3, 3].
            strides: List of integers, the strides used for the patch embedding layers.
                *Effect:* Determines the downsampling rate between stages. Default: [4, 2, 2, 2].
            num_heads: List of integers, the number of attention heads in each stage.
                *Effect:* Allows the model to jointly attend to information from different
                representational subspaces. Default: [1, 2, 5, 8].
            depths: List of integers, the number of transformer blocks in each stage.
                *Effect:* Defines the number of layers in each of the four stages. Default: [3, 4, 18, 3].
            mlp_ratios: List of integers, the expansion ratio for the hidden dimension
                of the MLP block in each stage.
                *Effect:* Controls the capacity of the feed-forward network. Default: [4, 4, 4, 4].
        """

        spatial_dims = len(input_shape) - 1
        sr_ratio = [8, 4, 2, 1] if spatial_dims == 2 else [4, 2, 1, 1]

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[64, 128, 320, 512],
            sr_ratios=sr_ratio,
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            depths=[3, 4, 18, 3],
            mlp_ratios=[4, 4, 4, 4],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="mit")
@registration.register(name="mit_b4", family="mit")
class MixViTB4(MiTBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        """
        Default Parameters (MiT-B4 specific):
            max_drop_path_rate: Float, the maximum drop path rate for regularization.
                *Effect:* A higher value increases regularization to prevent overfitting.
                Default: 0.1.
            layer_norm_epsilon: Float, a small value added to the variance in the layer
                normalization calculation to prevent division by zero.
                *Effect:* Ensures numerical stability during training. Default: 1e-6.
            qkv_bias: Boolean, whether to use bias terms for the query, key, and value
                linear layers in the attention mechanism.
                *Effect:* If True, adds an extra learnable parameter to each attention head.
                Default: True.
            project_dim: List of integers, the dimensions for the feature projection
                in each of the four stages of the encoder.
                *Effect:* Defines the channel depth of the feature maps in each stage.
                Default: [64, 128, 320, 512].
            sr_ratios: List of integers, the spatial reduction ratios used in the
                efficient self-attention mechanism for each stage.
                *Effect:* Reduces the computational cost of self-attention by shrinking
                the key and value maps. Default: [8, 4, 2, 1].
            patch_sizes: List of integers, the sizes of the patches created in each stage.
                *Effect:* Controls the initial tokenization and resolution of the feature maps.
                Default: [7, 3, 3, 3].
            strides: List of integers, the strides used for the patch embedding layers.
                *Effect:* Determines the downsampling rate between stages. Default: [4, 2, 2, 2].
            num_heads: List of integers, the number of attention heads in each stage.
                *Effect:* Allows the model to jointly attend to information from different
                representational subspaces. Default: [1, 2, 5, 8].
            depths: List of integers, the number of transformer blocks in each stage.
                *Effect:* Defines the number of layers in each of the four stages. Default: [3, 8, 27, 3].
            mlp_ratios: List of integers, the expansion ratio for the hidden dimension
                of the MLP block in each stage.
                *Effect:* Controls the capacity of the feed-forward network. Default: [4, 4, 4, 4].
        """

        spatial_dims = len(input_shape) - 1
        sr_ratio = [8, 4, 2, 1] if spatial_dims == 2 else [4, 2, 1, 1]

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[64, 128, 320, 512],
            sr_ratios=sr_ratio,
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            depths=[3, 8, 27, 3],
            mlp_ratios=[4, 4, 4, 4],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="mit")
@registration.register(name="mit_b5", family="mit")
class MixViTB5(MiTBase, DescribeMixin):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        """
        Default Parameters (MiT-B5 specific):
            max_drop_path_rate: Float, the maximum drop path rate for regularization.
                *Effect:* A higher value increases regularization to prevent overfitting.
                Default: 0.1.
            layer_norm_epsilon: Float, a small value added to the variance in the layer
                normalization calculation to prevent division by zero.
                *Effect:* Ensures numerical stability during training. Default: 1e-6.
            qkv_bias: Boolean, whether to use bias terms for the query, key, and value
                linear layers in the attention mechanism.
                *Effect:* If True, adds an extra learnable parameter to each attention head.
                Default: True.
            project_dim: List of integers, the dimensions for the feature projection
                in each of the four stages of the encoder.
                *Effect:* Defines the channel depth of the feature maps in each stage.
                Default: [64, 128, 320, 512].
            sr_ratios: List of integers, the spatial reduction ratios used in the
                efficient self-attention mechanism for each stage.
                *Effect:* Reduces the computational cost of self-attention by shrinking
                the key and value maps. Default: [8, 4, 2, 1].
            patch_sizes: List of integers, the sizes of the patches created in each stage.
                *Effect:* Controls the initial tokenization and resolution of the feature maps.
                Default: [7, 3, 3, 3].
            strides: List of integers, the strides used for the patch embedding layers.
                *Effect:* Determines the downsampling rate between stages. Default: [4, 2, 2, 2].
            num_heads: List of integers, the number of attention heads in each stage.
                *Effect:* Allows the model to jointly attend to information from different
                representational subspaces. Default: [1, 2, 5, 8].
            depths: List of integers, the number of transformer blocks in each stage.
                *Effect:* Defines the number of layers in each of the four stages.
                Default: [3, 6, 40, 3].
            mlp_ratios: List of integers, the expansion ratio for the hidden dimension
                of the MLP block in each stage.
                *Effect:* Controls the capacity of the feed-forward network.
                Default: [4, 4, 4, 4].
        """

        spatial_dims = len(input_shape) - 1
        sr_ratio = [8, 4, 2, 1] if spatial_dims == 2 else [4, 2, 1, 1]

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[64, 128, 320, 512],
            sr_ratios=sr_ratio,
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            depths=[3, 6, 40, 3],
            mlp_ratios=[4, 4, 4, 4],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )

MIT_DOCSTRING = """
{name} model for classification, supporting both 2D and 3D inputs.

This class provides a complete {name} model, including the hierarchical
transformer backbone and the classification head (the "top"). The backbone
follows the Mix Transformer (MiT) design with overlapping patch embedding and
multi-stage transformer encoding, and the full model can be used either for
end-to-end classification or as a feature extractor.

It can operate on 2D inputs (e.g., images of shape `(H, W, C)`) or 3D inputs
(e.g., volumetric data of shape `(D, H, W, C)`).

Args:
    input_shape: A tuple specifying the input shape of the model,
        not including the batch size. Can be `(height, width, channels)` for
        2D or `(depth, height, width, channels)` for 3D.
    include_rescaling: A boolean indicating whether to include a
        ``Rescaling`` layer at the beginning of the model.
    include_top: A boolean indicating whether to include the fully
        connected classification layer at the top of the network. If
        `False`, the model's output will be the features from the
        backbone, without the final classifier.
    num_classes: An integer specifying the number of classes for the
        classification layer. This is only relevant if `include_top`
        is `True`.
    pooling: (Optional) A string specifying the type of pooling to
        apply to the output of the backbone. Can be `"avg"` for global
        average pooling or `"max"` for global max pooling. This is only
        relevant if `include_top` is `False`.
    classifier_activation: A string specifying the activation function
        to use for the classification layer.
    name: (Optional) The name of the model.

Returns:
    A ``keras.Model`` whose output depends on the configuration:

        - If ``include_top=True``, the output is a classification tensor of shape
        ``(batch_size, num_classes)``.
        - If ``include_top=False`` and ``pooling`` is ``None``, the output is the
        final backbone feature tensor.
        - If ``include_top=False`` and ``pooling`` is ``"avg"`` or ``"max"``,
        the output is a pooled feature tensor with last dimension
        ``{projection_dim_last}``.

Examples:
    .. code-block:: python

        import torch
        from medicai.models.mit import {name}

        model = {name}(
            input_shape=(224, 224, 3), include_top=True, num_classes=2
        )
        x = torch.randn((1, 224, 224, 3))
        y = model(x)
        print(y.shape) # torch.Size([1, 2])

References:
    - SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS 2021.
        `arXiv:2105.15203 <https://arxiv.org/abs/2105.15203>`_
"""

MixViTB0.__doc__ = MIT_DOCSTRING.format(name="MixViTB0", projection_dim_last=256)
MixViTB1.__doc__ = MIT_DOCSTRING.format(name="MixViTB1", projection_dim_last=512)
MixViTB2.__doc__ = MIT_DOCSTRING.format(name="MixViTB2", projection_dim_last=512)
MixViTB3.__doc__ = MIT_DOCSTRING.format(name="MixViTB3", projection_dim_last=512)
MixViTB4.__doc__ = MIT_DOCSTRING.format(name="MixViTB4", projection_dim_last=512)
MixViTB5.__doc__ = MIT_DOCSTRING.format(name="MixViTB5", projection_dim_last=512)