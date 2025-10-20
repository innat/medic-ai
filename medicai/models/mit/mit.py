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
    """
    Mix Transformer B0 (MiT-B0) model.

    MiT-B0 is the smallest and most efficient variant of the **Mix Transformer**
    (MiT) family, providing an excellent trade-off between speed and performance.
    It is parameterized with the configuration specified for the B0 variant in
    the original SegFormer paper.

    This class inherits from `MiTBase` and sets the hyper-parameters for
    depths, feature dimensions, heads, and attention reduction ratios specific
    to the B0 architecture.

    Example:
    >>> from medicai.models import MixViTB0
    >>> # 2D Model (e.g., for ImageNet)
    >>> model_2d = MixViTB0(input_shape=(224, 224, 3), num_classes=5, ...)
    >>> # 3D Model (e.g., for medical volumes)
    >>> model_3d = MixViTB0(input_shape=(64, 64, 64, 1), num_classes=5, ...)

    Note:
    The list of spatial reduction ratios for MixViTB0 is set `[4, 2, 1, 1]`.
    This follows official `SegFormer3D` model. In official 2D `SegFormer`, it
    is set `[8, 4, 2, 1]`. To build official 2D `SegFormer` with `MixViTB0`, use
    `MiTBackbone`.

    Reference:
        'SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation'
        - Encoder: MiT-B0
        - Paper: https://arxiv.org/abs/2404.10156

    Initializes the Mix Transformer B0 (MiT-B0) model.

    This constructor automatically sets the MiT-B0-specific hyper-parameters
    and accepts configuration for the input and output head.

    Args:
        input_shape: A tuple specifying the shape of the input tensor,
            excluding the batch dimension. Format is typically (H, W, C) for 2D
            images or (D, H, W, C) for 3D volumes.
            *Effect:* Defines the input shape for the entire model.
        include_rescaling: Boolean, determines whether to include a **rescaling**
            layer at the start of the model to normalize pixel values.
            *Effect:* If True, applies an input pre-processing step. Default: False.
        include_top: Boolean, whether to include the final **Dense classification
            layer** (the "top") on top of the feature extractor.
            *Effect:* If True, the model output is a probability distribution/logits.
            If False, the model outputs the feature map for downstream tasks. Default: True.
        num_classes: Optional integer, the number of output classes. Only
            relevant if `include_top` is True.
            *Effect:* Sets the number of units in the final Dense layer. Default: 1000.
        pooling: Optional string, only relevant when `include_top=False`.
            Specifies the global pooling type to apply to the feature map:
            'avg' for Global Average Pooling or 'max' for Global Max Pooling.
            *Effect:* Collapses the spatial dimensions of the output feature map. Default: None.
        classifier_activation: Optional activation function for the final
            Dense classification layer. Only relevant if `include_top` is True.
            *Effect:* Typically 'softmax' for multi-class classification or
            'sigmoid' for multi-label classification. Default: "softmax".
        name: Optional string, the name to give the Keras model.
            *Effect:* Sets the model's identifier. Default: Auto-generated.
        **kwargs: Additional keyword arguments passed to the parent `keras.Model`
            constructor.
    """

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

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[32, 64, 160, 256],
            sr_ratios=[4, 2, 1, 1],
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
    """
    Mix Transformer B1 (MiT-B1) model.

    MiT-B1 is a small and efficient variant of the **Mix Transformer**
    (MiT) family, providing an excellent trade-off between speed and performance.
    It is parameterized with the configuration specified for the B1 variant in
    the original SegFormer paper.

    This class inherits from `MiTBase` and sets the hyper-parameters for
    depths, feature dimensions, heads, and attention reduction ratios specific
    to the B1 architecture.

    Example:
    >>> from medicai.models import MixViTB1
    >>> # 2D Model (e.g., for ImageNet)
    >>> model_2d = MixViTB1(input_shape=(224, 224, 3), num_classes=5, ...)
    >>> # 3D Model (e.g., for medical volumes)
    >>> model_3d = MixViTB1(input_shape=(64, 64, 64, 1), num_classes=5, ...)

    Reference:
        'SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers'
        - Encoder: MiT-B1
        - Paper: https://arxiv.org/abs/2105.15203

    Initializes the Mix Transformer B1 (MiT-B1) model.

    Args:
        input_shape: A tuple specifying the shape of the input tensor,
            excluding the batch dimension. Format is typically (H, W, C) for 2D
            images or (D, H, W, C) for 3D volumes.
            *Effect:* Defines the input shape for the entire model.
        include_rescaling: Boolean, determines whether to include a **rescaling**
            layer at the start of the model to normalize pixel values.
            *Effect:* If True, applies an input pre-processing step. Default: False.
        include_top: Boolean, whether to include the final **Dense classification
            layer** (the "top") on top of the feature extractor.
            *Effect:* If True, the model output is a probability distribution/logits.
            If False, the model outputs the feature map for downstream tasks. Default: True.
        num_classes: Optional integer, the number of output classes. Only
            relevant if `include_top` is True.
            *Effect:* Sets the number of units in the final Dense layer. Default: 1000.
        pooling: Optional string, only relevant when `include_top=False`.
            Specifies the global pooling type to apply to the feature map:
            'avg' for Global Average Pooling or 'max' for Global Max Pooling.
            *Effect:* Collapses the spatial dimensions of the output feature map. Default: None.
        classifier_activation: Optional activation function for the final
            Dense classification layer. Only relevant if `include_top` is True.
            *Effect:* Typically 'softmax' for multi-class classification or
            'sigmoid' for multi-label classification. Default: "softmax".
        name: Optional string, the name to give the Keras model.
            *Effect:* Sets the model's identifier. Default: Auto-generated.
        **kwargs: Additional keyword arguments passed to the parent `keras.Model`
            constructor.
    """

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

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[64, 128, 320, 512],
            sr_ratios=[8, 4, 2, 1],
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
    """
    Mix Transformer B2 (MiT-B2) model.

    MiT-B2 is a small variant of the **Mix Transformer** (MiT) family. It offers
    improved performance over MiT-B0 and MiT-B1 by increasing the number of
    transformer blocks in each stage of the encoder, providing a larger capacity.
    The architecture is parameterized with the configuration specific to the B2
    variant from the original SegFormer paper.

    This class inherits from `MiTBase` and sets the hyper-parameters for
    depths, feature dimensions, heads, and attention reduction ratios specific
    to the B2 architecture.

    Example:
    >>> from medicai.models import MixViTB2
    >>> # 2D Model (e.g., for ImageNet)
    >>> model_2d = MixViTB2(input_shape=(224, 224, 3), num_classes=5, ...)
    >>> # 3D Model (e.g., for medical volumes)
    >>> model_3d = MixViTB2(input_shape=(64, 64, 64, 1), num_classes=5, ...)

    Reference:
        'SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers'
        - Encoder: MiT-B2
        - Paper: https://arxiv.org/abs/2105.15203

    Initializes the Mix Transformer B2 (MiT-B2) model.

    Args:
        input_shape: A tuple specifying the shape of the input tensor,
            excluding the batch dimension. Format is typically (H, W, C) for 2D
            images or (D, H, W, C) for 3D volumes.
            *Effect:* Defines the input shape for the entire model.
        include_rescaling: Boolean, determines whether to include a **rescaling**
            layer at the start of the model to normalize pixel values.
            *Effect:* If True, applies an input pre-processing step. Default: False.
        include_top: Boolean, whether to include the final **Dense classification
            layer** (the "top") on top of the feature extractor.
            *Effect:* If True, the model output is a probability distribution/logits.
            If False, the model outputs the feature map for downstream tasks. Default: True.
        num_classes: Optional integer, the number of output classes. Only
            relevant if `include_top` is True.
            *Effect:* Sets the number of units in the final Dense layer. Default: 1000.
        pooling: Optional string, only relevant when `include_top=False`.
            Specifies the global pooling type to apply to the feature map:
            'avg' for Global Average Pooling or 'max' for Global Max Pooling.
            *Effect:* Collapses the spatial dimensions of the output feature map. Default: None.
        classifier_activation: Optional activation function for the final
            Dense classification layer. Only relevant if `include_top` is True.
            *Effect:* Typically 'softmax' for multi-class classification or
            'sigmoid' for multi-label classification. Default: "softmax".
        name: Optional string, the name to give the Keras model.
            *Effect:* Sets the model's identifier. Default: Auto-generated.
        **kwargs: Additional keyword arguments passed to the parent `keras.Model`
            constructor.
    """

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

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[64, 128, 320, 512],
            sr_ratios=[8, 4, 2, 1],
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
    """
    Mix Transformer B3 (MiT-B3) model.

    MiT-B3 is a medium-sized variant of the **Mix Transformer** (MiT) family. It offers
    strong performance by significantly increasing the number of transformer blocks
    in the third stage of the encoder, building upon the B2 architecture. This provides
    greater representational capacity for more complex vision tasks. The architecture
    is parameterized with the configuration specific to the B3 variant from the
    original SegFormer paper.

    This class inherits from `MiTBase` and sets the hyper-parameters for
    depths, feature dimensions, heads, and attention reduction ratios specific
    to the B3 architecture.

    Example:
    >>> from medicai.models import MixViTB3
    >>> # 2D Model (e.g., for ImageNet)
    >>> model_2d = MixViTB3(input_shape=(224, 224, 3), num_classes=5, ...)
    >>> # 3D Model (e.g., for medical volumes)
    >>> model_3d = MixViTB3(input_shape=(64, 64, 64, 1), num_classes=5, ...)

    Reference:
        'SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers'
        - Encoder: MiT-B3
        - Paper: https://arxiv.org/abs/2105.15203

    Initializes the Mix Transformer B3 (MiT-B3) model.

    This constructor automatically sets the MiT-B3-specific hyper-parameters
    (e.g., `depths=[3, 4, 18, 3]`, `project_dim=[64, 128, 320, 512]`, etc.)
    and accepts configuration for the input and output head.

    Args:
        input_shape: A tuple specifying the shape of the input tensor,
            excluding the batch dimension. Format is typically (H, W, C) for 2D
            images or (D, H, W, C) for 3D volumes.
            *Effect:* Defines the input shape for the entire model.
        include_rescaling: Boolean, determines whether to include a **rescaling**
            layer at the start of the model to normalize pixel values.
            *Effect:* If True, applies an input pre-processing step. Default: False.
        include_top: Boolean, whether to include the final **Dense classification
            layer** (the "top") on top of the feature extractor.
            *Effect:* If True, the model output is a probability distribution/logits.
            If False, the model outputs the feature map for downstream tasks. Default: True.
        num_classes: Optional integer, the number of output classes. Only
            relevant if `include_top` is True.
            *Effect:* Sets the number of units in the final Dense layer. Default: 1000.
        pooling: Optional string, only relevant when `include_top=False`.
            Specifies the global pooling type to apply to the feature map:
            'avg' for Global Average Pooling or 'max' for Global Max Pooling.
            *Effect:* Collapses the spatial dimensions of the output feature map. Default: None.
        classifier_activation: Optional activation function for the final
            Dense classification layer. Only relevant if `include_top` is True.
            *Effect:* Typically 'softmax' for multi-class classification or
            'sigmoid' for multi-label classification. Default: "softmax".
        name: Optional string, the name to give the Keras model.
            *Effect:* Sets the model's identifier. Default: Auto-generated.
        **kwargs: Additional keyword arguments passed to the parent `keras.Model`
            constructor.
    """

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

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[64, 128, 320, 512],
            sr_ratios=[8, 4, 2, 1],
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
    """
    Mix Transformer B4 (MiT-B4) model.

    MiT-B4 is a large variant of the **Mix Transformer** (MiT) family. It offers
    very high performance for demanding vision tasks by significantly increasing
    the number of transformer blocks in the second and third stages of the encoder.
    This provides a larger representational capacity compared to the B3 variant.
    The architecture is parameterized with the configuration specific to the B4
    variant from the original SegFormer paper.

    This class inherits from `MiTBase` and sets the hyper-parameters for
    depths, feature dimensions, heads, and attention reduction ratios specific
    to the B4 architecture.

    Example:
    >>> from medicai.models import MixViTB4
    >>> # 2D Model (e.g., for ImageNet)
    >>> model_2d = MixViTB4(input_shape=(224, 224, 3), num_classes=5, ...)
    >>> # 3D Model (e.g., for medical volumes)
    >>> model_3d = MixViTB4(input_shape=(64, 64, 64, 1), num_classes=5, ...)

    Reference:
        'SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers'
        - Encoder: MiT-B4
        - Paper: https://arxiv.org/abs/2105.15203

    Initializes the Mix Transformer B4 (MiT-B4) model.

    Args:
        input_shape: A tuple specifying the shape of the input tensor,
            excluding the batch dimension. Format is typically (H, W, C) for 2D
            images or (D, H, W, C) for 3D volumes.
            *Effect:* Defines the input shape for the entire model.
        include_rescaling: Boolean, determines whether to include a **rescaling**
            layer at the start of the model to normalize pixel values.
            *Effect:* If True, applies an input pre-processing step. Default: False.
        include_top: Boolean, whether to include the final **Dense classification
            layer** (the "top") on top of the feature extractor.
            *Effect:* If True, the model output is a probability distribution/logits.
            If False, the model outputs the feature map for downstream tasks. Default: True.
        num_classes: Optional integer, the number of output classes. Only
            relevant if `include_top` is True.
            *Effect:* Sets the number of units in the final Dense layer. Default: 1000.
        pooling: Optional string, only relevant when `include_top=False`.
            Specifies the global pooling type to apply to the feature map:
            'avg' for Global Average Pooling or 'max' for Global Max Pooling.
            *Effect:* Collapses the spatial dimensions of the output feature map. Default: None.
        classifier_activation: Optional activation function for the final
            Dense classification layer. Only relevant if `include_top` is True.
            *Effect:* Typically 'softmax' for multi-class classification or
            'sigmoid' for multi-label classification. Default: "softmax".
        name: Optional string, the name to give the Keras model.
            *Effect:* Sets the model's identifier. Default: Auto-generated.
        **kwargs: Additional keyword arguments passed to the parent `keras.Model`
            constructor.
    """

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

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[64, 128, 320, 512],
            sr_ratios=[8, 4, 2, 1],
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
    """
    Mix Transformer B5 (MiT-B5) model.

    MiT-B5 is the **largest and most powerful variant** of the **Mix Transformer**
    (MiT) family. It is designed for maximum performance on complex vision tasks by
    significantly increasing the number of transformer blocks in the second and
    third stages of the encoder compared to other variants. The architecture is
    parameterized with the configuration specific to the B5 variant from the
    original SegFormer paper.

    This class inherits from `MiTBase` and sets the hyper-parameters for
    depths, feature dimensions, heads, and attention reduction ratios specific
    to the B5 architecture.

    Example:
    >>> from medicai.models import MixViTB5
    >>> # 2D Model (e.g., for ImageNet)
    >>> model_2d = MixViTB5(input_shape=(224, 224, 3), num_classes=5, ...)
    >>> # 3D Model (e.g., for medical volumes)
    >>> model_3d = MixViTB5(input_shape=(64, 64, 64, 1), num_classes=5, ...)

    Reference:
        'SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers'
        - Encoder: MiT-B5
        - Paper: https://arxiv.org/abs/2105.15203

    Initializes the Mix Transformer B5 (MiT-B5) model.

    This constructor automatically sets the MiT-B5-specific hyper-parameters
    (e.g., `depths=[3, 6, 40, 3]`, `project_dim=[64, 128, 320, 512]`, etc.)
    and accepts configuration for the input and output head.

    Args:
        input_shape: A tuple specifying the shape of the input tensor,
            excluding the batch dimension. Format is typically (H, W, C) for 2D
            images or (D, H, W, C) for 3D volumes.
            *Effect:* Defines the input shape for the entire model.
        include_rescaling: Boolean, determines whether to include a **rescaling**
            layer at the start of the model to normalize pixel values.
            *Effect:* If True, applies an input pre-processing step. Default: False.
        include_top: Boolean, whether to include the final **Dense classification
            layer** (the "top") on top of the feature extractor.
            *Effect:* If True, the model output is a probability distribution/logits.
            If False, the model outputs the feature map for downstream tasks. Default: True.
        num_classes: Optional integer, the number of output classes. Only
            relevant if `include_top` is True.
            *Effect:* Sets the number of units in the final Dense layer. Default: 1000.
        pooling: Optional string, only relevant when `include_top=False`.
            Specifies the global pooling type to apply to the feature map:
            'avg' for Global Average Pooling or 'max' for Global Max Pooling.
            *Effect:* Collapses the spatial dimensions of the output feature map. Default: None.
        classifier_activation: Optional activation function for the final
            Dense classification layer. Only relevant if `include_top` is True.
            *Effect:* Typically 'softmax' for multi-class classification or
            'sigmoid' for multi-label classification. Default: "softmax".
        name: Optional string, the name to give the Keras model.
            *Effect:* Sets the model's identifier. Default: Auto-generated.
        **kwargs: Additional keyword arguments passed to the parent `keras.Model`
            constructor.
    """

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

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[64, 128, 320, 512],
            sr_ratios=[8, 4, 2, 1],
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
