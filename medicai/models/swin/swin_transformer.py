import keras

from medicai.utils import DescribeMixin, get_pooling_layer, keras_constants, registration

from .swin_backbone import SwinBackbone, SwinBackboneV2, resolve_input_params

SWIN_CFG = {
    "2D": {
        "tiny": {
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "embed_dim": 96,
        },
        "small": {
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24],
            "embed_dim": 96,
        },
        "base": {
            "depths": [2, 2, 18, 2],
            "num_heads": [4, 8, 16, 32],
            "embed_dim": 128,
        },
    },
    "3D": {
        "tiny": {
            "depths": [2, 2, 2, 2],
            "num_heads": [3, 6, 12, 24],
            "embed_dim": 48,
        },
        "small": {
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "embed_dim": 48,
        },
        "base": {
            "depths": [2, 2, 6, 2],
            "num_heads": [4, 8, 16, 32],
            "embed_dim": 96,
        },
    },
}


@keras.saving.register_keras_serializable(package="swin")
class SwinVariantsBase(keras.Model):
    """
    Base class for Swin Transformer models (2D or 3D) for classification or feature extraction.

    This class handles the core logic for model configuration, input validation,
    backbone creation, and adding the classification head or pooling layers.
    Specific variants (Tiny, Small, Base) should inherit from this class.
    """

    backbone_cls = None

    def __init__(
        self,
        *,
        input_shape,
        include_rescaling,
        include_top,
        patch_size,
        window_size,
        pooling,
        dropout,
        attn_drop_rate,
        drop_path_rate,
        downsampling_strategy,
        num_classes=1000,
        classifier_activation=None,
        stage_wise_conv=False,
        variant=None,
        name=None,
        **kwargs,
    ):
        # Check that the input is well specified.
        input_shape, patch_size, window_size, downsampling_strategy = resolve_input_params(
            input_shape, patch_size, window_size, downsampling_strategy
        )
        spatial_dims = len(input_shape) - 1

        # Get variant specific config
        cfg = SWIN_CFG.get(f"{spatial_dims}D")[variant]

        # Get and set class name
        if name is None and self.__class__ is not SwinVariantsBase:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        backbone = self.backbone_cls(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            patch_size=patch_size,
            window_size=window_size,
            num_heads=cfg["num_heads"],
            depths=cfg["depths"],
            embed_dim=cfg["embed_dim"],
            downsampling_strategy=downsampling_strategy,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            stage_wise_conv=stage_wise_conv,
            patch_norm=False,
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
            if dropout > 0.0:
                x = keras.layers.Dropout(dropout, name="output_dropout")(x)

            VALID_ACTIVATION_LIST = keras_constants.get_valid_activations()
            if classifier_activation not in VALID_ACTIVATION_LIST:
                raise ValueError(
                    f"Invalid value for `classifier_activation`: {classifier_activation!r}. "
                    f"Supported values are: {VALID_ACTIVATION_LIST}"
                )

            x = keras.layers.Dense(
                num_classes, activation=classifier_activation, dtype="float32", name="predictions"
            )(x)
        elif pooling == "avg":
            x = GlobalAvgPool(x)
        elif pooling == "max":
            x = GlobalMaxPool(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.pyramid_outputs = backbone.pyramid_outputs
        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.window_size = window_size
        self.pooling = pooling
        self.dropout = dropout
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.stage_wise_conv = stage_wise_conv
        self.classifier_activation = classifier_activation
        self.downsampling_strategy = downsampling_strategy
        self.name = name

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "include_rescaling": self.include_rescaling,
            "include_top": self.include_top,
            "patch_size": self.patch_size,
            "window_size": self.window_size,
            "pooling": self.pooling,
            "dropout": self.dropout,
            "downsampling_strategy": self.downsampling_strategy,
            "name": self.name,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="swin")
@registration.register(name="swin_tiny", family="swin")
class SwinTiny(SwinVariantsBase, DescribeMixin):
    """
    Swin Tiny model, a small-scale Swin Transformer for vision tasks.

    The Swin Transformer, based on shifted windows, is a hierarchical Vision Transformer
    that excels in general-purpose vision tasks, offering both efficiency and performance
    for 2D (images) and 3D (volumetric/video) data. The 'Tiny' variant uses a compact
    configuration suitable for fast experimentation or resource-constrained environments.

    Example:
    >>> from medicai.models import SwinTiny
    >>> import numpy as np
    >>> input_data = np.ones((1, 96, 96, 1), dtype=np.float32)
    >>> model = SwinTiny(input_shape=(96, 96, 1), num_classes=5)
    >>> output = model(input_data)

    Stage-wise Residual Convolution (SwinUNETR-V2):
        If `stage_wise_conv=True`, an additional convolutional residual block
        (`UnetrBasicBlock`) is inserted at the **beginning of each Swin stage**.
        This improves local feature extraction before self-attention.

        Data flow with `stage_wise_conv=True`:
            PatchEmbed → Dropout
            └─ Stage 0: [UnetrBasicBlock] → SwinBasicLayer(0)
            └─ Stage 1: [UnetrBasicBlock] → SwinBasicLayer(1)
            └─ Stage 2: [UnetrBasicBlock] → SwinBasicLayer(2)
            └─ Stage 3: [UnetrBasicBlock] → SwinBasicLayer(3)

    Example:
    >>> from medicai.models import SwinTiny
    >>> import numpy as np
    >>> input_data = np.ones((1, 96, 96, 1), dtype=np.float32)
    >>> model = SwinTiny(input_shape=(96, 96, 1), num_classes=5, stage_wise_conv=True)
    >>> output = model(input_data)

    Reference:
      paper: https://arxiv.org/abs/2103.14030
    """

    backbone_cls = SwinBackbone
    variant = "tiny"

    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        patch_size=4,
        window_size=7,
        num_classes=1000,
        pooling="avg",
        dropout=0.0,
        classifier_activation=None,
        stage_wise_conv=False,
        downsampling_strategy="swin_transformer_like",
        name=None,
        **kwargs,
    ):
        """Initializes the SwinTiny model.

        Args:
            input_shape (tuple): The shape of the input tensor (H, W, C) for 2D or (D, H, W, C) for 3D,
                excluding the batch size. Must have fixed spatial dimensions.
            include_rescaling (bool): Whether to include a rescaling layer at the input.
            include_top (bool): Whether to include the final classification layer.
            patch_size (int or tuple): The size of the non-overlapping patches.
                Should be (patch_size,) * spatial_dims.
            window_size (int or tuple): The size of the attention window.
                Should be (window_size,) * spatial_dims.
            pooling (str): Optional pooling type applied to the backbone output when `include_top` is False.
                Must be 'avg' or 'max'. Ignored if `include_top` is True.
            dropout (float): Dropout rate for the classification head. Ignored if `include_top` is False.
            attn_drop_rate (float): Dropout rate for the attention layers.
            drop_path_rate (float): Stochastic depth rate for the residual paths.
            num_classes (int): The number of output classes for classification. Default is 1000.
            classifier_activation (str, optional): The activation function for the final
                classification layer (e.g., 'softmax', 'sigmoid'). If None, no activation is applied.
                Default is None.
            downsampling_strategy: swin-transformer and swin-unetr uses bit different downsampling strategy.
                It should be either "swin_unetr_like" or "swin_transformer_like".
                'swin_transformer_like' (for 2D/3D classification tasks).
                'swin_unetr_like' (for 2D/3D segmentation tasks).
                Default: "swin_transformer_like".
            stage_wise_conv (bool): If True, use the SwinUNETR-V2 variant with convolutional residual blocks
                before each stage. Default: False.
            variant (str): The specific Swin configuration variant to use ('tiny', 'small', or 'base').
            name (str, optional): The name of the model. Default is automatically generated.
            **kwargs: Additional keyword arguments passed to the base Model class.
        """
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            patch_size=patch_size,
            window_size=window_size,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            dropout=dropout,
            variant=self.variant,
            stage_wise_conv=stage_wise_conv,
            downsampling_strategy=downsampling_strategy,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="swin")
@registration.register(name="swin_small", family="swin")
class SwinSmall(SwinVariantsBase, DescribeMixin):
    """Swin Small model, a medium-scale Swin Transformer for vision tasks.

    The Swin Transformer, based on shifted windows, is a hierarchical Vision Transformer
    that excels in general-purpose vision tasks, offering both efficiency and performance
    for 2D (images) and 3D (volumetric/video) data. The 'Small' variant is a good
    balance of complexity and accuracy.

    Example:
    >>> from medicai.models import SwinSmall
    >>> import numpy as np
    >>> input_data = np.ones((1, 96, 96, 1), dtype=np.float32)
    >>> model = SwinSmall(input_shape=(96, 96, 1), num_classes=5)
    >>> output = model(input_data)

    Stage-wise Residual Convolution (SwinUNETR-V2):
        If `stage_wise_conv=True`, an additional convolutional residual block
        (`UnetrBasicBlock`) is inserted at the **beginning of each Swin stage**.
        This improves local feature extraction before self-attention.

        Data flow with `stage_wise_conv=True`:
            PatchEmbed → Dropout
            └─ Stage 0: [UnetrBasicBlock] → SwinBasicLayer(0)
            └─ Stage 1: [UnetrBasicBlock] → SwinBasicLayer(1)
            └─ Stage 2: [UnetrBasicBlock] → SwinBasicLayer(2)
            └─ Stage 3: [UnetrBasicBlock] → SwinBasicLayer(3)

    Example:
    >>> from medicai.models import SwinSmall
    >>> import numpy as np
    >>> input_data = np.ones((1, 96, 96, 1), dtype=np.float32)
    >>> model = SwinSmall(input_shape=(96, 96, 1), num_classes=5, stage_wise_conv=True)
    >>> output = model(input_data)

    Reference:
      paper: https://arxiv.org/abs/2103.14030
    """

    backbone_cls = SwinBackbone
    variant = "small"

    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        patch_size=4,
        window_size=7,
        num_classes=1000,
        pooling="avg",
        dropout=0.0,
        classifier_activation=None,
        stage_wise_conv=False,
        downsampling_strategy="swin_transformer_like",
        name=None,
        **kwargs,
    ):
        """Initializes the Swin Small model.

        Args:
            input_shape (tuple): The shape of the input tensor (H, W, C) for 2D or (D, H, W, C) for 3D,
                excluding the batch size. Must have fixed spatial dimensions.
            include_rescaling (bool): Whether to include a rescaling layer at the input.
            include_top (bool): Whether to include the final classification layer.
            patch_size (int or tuple): The size of the non-overlapping patches.
                Should be (patch_size,) * spatial_dims.
            window_size (int or tuple): The size of the attention window.
                Should be (window_size,) * spatial_dims.
            pooling (str): Optional pooling type applied to the backbone output when `include_top` is False.
                Must be 'avg' or 'max'. Ignored if `include_top` is True.
            dropout (float): Dropout rate for the classification head. Ignored if `include_top` is False.
            attn_drop_rate (float): Dropout rate for the attention layers.
            drop_path_rate (float): Stochastic depth rate for the residual paths.
            num_classes (int): The number of output classes for classification. Default is 1000.
            classifier_activation (str, optional): The activation function for the final
                classification layer (e.g., 'softmax', 'sigmoid'). If None, no activation is applied.
                Default is None.
            downsampling_strategy: swin-transformer and swin-unetr uses bit different downsampling strategy.
                It should be either "swin_unetr_like" or "swin_transformer_like".
                'swin_transformer_like' (for 2D/3D classification tasks).
                'swin_unetr_like' (for 2D/3D segmentation tasks).
                Default: "swin_transformer_like".
            stage_wise_conv (bool): If True, use the SwinUNETR-V2 variant with convolutional residual blocks
                before each stage. Default: False.
            variant (str): The specific Swin configuration variant to use ('tiny', 'small', or 'base').
            name (str, optional): The name of the model. Default is automatically generated.
            **kwargs: Additional keyword arguments passed to the base Model class.

        Default Configuration Details (Swin-Small):
            - **Embedding Dimension:** 96 for 2D, 48 for 3D.
            - **Depths:** [2, 2, 18, 2] for 2D, [2, 2, 6, 2] for 3D.
            - **Number of Heads:** [3, 6, 12, 24]
        """
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            patch_size=patch_size,
            window_size=window_size,
            classifier_activation=classifier_activation,
            dropout=dropout,
            variant=self.variant,
            stage_wise_conv=stage_wise_conv,
            downsampling_strategy=downsampling_strategy,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="swin")
@registration.register(name="swin_base", family="swin")
class SwinBase(SwinVariantsBase, DescribeMixin):
    """Swin Base model, a large-scale Swin Transformer for vision tasks.

    The Swin Transformer, based on shifted windows, is a hierarchical Vision Transformer
    that excels in general-purpose vision tasks, offering both efficiency and performance
    for 2D (images) and 3D (volumetric/video) data. The 'Base' variant offers high
    capacity suitable for complex tasks and large datasets.

    Example:
    >>> from medicai.models import SwinBase
    >>> import numpy as np
    >>> input_data = np.ones((1, 96, 96, 1), dtype=np.float32)
    >>> model = SwinBase(input_shape=(96, 96, 1), num_classes=5)
    >>> output = model(input_data)

    Stage-wise Residual Convolution (SwinUNETR-V2):
        If `stage_wise_conv=True`, an additional convolutional residual block
        (`UnetrBasicBlock`) is inserted at the **beginning of each Swin stage**.
        This improves local feature extraction before self-attention.

        Data flow with `stage_wise_conv=True`:
            PatchEmbed → Dropout
            └─ Stage 0: [UnetrBasicBlock] → SwinBasicLayer(0)
            └─ Stage 1: [UnetrBasicBlock] → SwinBasicLayer(1)
            └─ Stage 2: [UnetrBasicBlock] → SwinBasicLayer(2)
            └─ Stage 3: [UnetrBasicBlock] → SwinBasicLayer(3)

    Example:
    >>> from medicai.models import SwinBase
    >>> import numpy as np
    >>> input_data = np.ones((1, 96, 96, 1), dtype=np.float32)
    >>> model = SwinBase(input_shape=(96, 96, 1), num_classes=5, stage_wise_conv=True)
    >>> output = model(input_data)

    Reference:
      paper: https://arxiv.org/abs/2103.14030
    """

    backbone_cls = SwinBackbone
    variant = "base"

    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        patch_size=4,
        window_size=7,
        pooling="avg",
        dropout=0.0,
        stage_wise_conv=False,
        classifier_activation=None,
        downsampling_strategy="swin_transformer_like",
        name=None,
        **kwargs,
    ):
        """Initializes the Swin Base model.

        Args:
            input_shape (tuple): The shape of the input tensor (H, W, C) for 2D or (D, H, W, C) for 3D,
                excluding the batch size. Must have fixed spatial dimensions.
            include_rescaling (bool): Whether to include a rescaling layer at the input.
            include_top (bool): Whether to include the final classification layer.
            patch_size (int or tuple): The size of the non-overlapping patches.
                Should be (patch_size,) * spatial_dims.
            window_size (int or tuple): The size of the attention window.
                Should be (window_size,) * spatial_dims.
            pooling (str): Optional pooling type applied to the backbone output when `include_top` is False.
                Must be 'avg' or 'max'. Ignored if `include_top` is True.
            dropout (float): Dropout rate for the classification head. Ignored if `include_top` is False.
            attn_drop_rate (float): Dropout rate for the attention layers.
            drop_path_rate (float): Stochastic depth rate for the residual paths.
            num_classes (int): The number of output classes for classification. Default is 1000.
            classifier_activation (str, optional): The activation function for the final
                classification layer (e.g., 'softmax', 'sigmoid'). If None, no activation is applied.
                Default is None.
            downsampling_strategy: swin-transformer and swin-unetr uses bit different downsampling strategy.
                It should be either "swin_unetr_like" or "swin_transformer_like".
                'swin_transformer_like' (for 2D/3D classification tasks).
                'swin_unetr_like' (for 2D/3D segmentation tasks).
                Default: "swin_transformer_like".
            stage_wise_conv (bool): If True, use the SwinUNETR-V2 variant with convolutional residual blocks
                before each stage. Default: False.
            variant (str): The specific Swin configuration variant to use ('tiny', 'small', or 'base').
            name (str, optional): The name of the model. Default is automatically generated.
            **kwargs: Additional keyword arguments passed to the base Model class.

        Default Configuration Details (Swin-Base):
            - **Embedding Dimension:** 128 for 2D, 96 for 3D.
            - **Depths:** [2, 2, 18, 2] for 2D, [2, 2, 6, 2] for 3D.
            - **Number of Heads:** [4, 8, 16, 32] same for 2D and 3D.
        """
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            patch_size=patch_size,
            window_size=window_size,
            classifier_activation=classifier_activation,
            dropout=dropout,
            variant=self.variant,
            stage_wise_conv=stage_wise_conv,
            downsampling_strategy=downsampling_strategy,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="swin")
@registration.register(name="swin_tiny_v2", family="swin")
class SwinTinyV2(SwinVariantsBase, DescribeMixin):
    """Swin Tiny V2 model, a small-scale Swin Transformer V2 for vision tasks.

    The Swin Transformer V2, based on shifted windows, is a hierarchical Vision Transformer
    that excels in general-purpose vision tasks, offering both efficiency and performance
    for 2D (images) and 3D (volumetric/video) data.  The 'Tiny' variant offers a
    compact configuration suitable for fast experimentation.

    Example:
    >>> from medicai.models import SwinTinyV2
    >>> import numpy as np
    >>> input_data = np.ones((1, 96, 96, 1), dtype=np.float32)
    >>> model = SwinTinyV2(input_shape=(96, 96, 1), num_classes=5)
    >>> output = model(input_data)

    Stage-wise Residual Convolution (SwinUNETR-V2):
        If `stage_wise_conv=True`, an additional convolutional residual block
        (`UnetrBasicBlock`) is inserted at the **beginning of each Swin stage**.
        This improves local feature extraction before self-attention.

        Data flow with `stage_wise_conv=True`:
            PatchEmbed → Dropout
            └─ Stage 0: [UnetrBasicBlock] → SwinBasicLayer(0)
            └─ Stage 1: [UnetrBasicBlock] → SwinBasicLayer(1)
            └─ Stage 2: [UnetrBasicBlock] → SwinBasicLayer(2)
            └─ Stage 3: [UnetrBasicBlock] → SwinBasicLayer(3)

    Example:
    >>> from medicai.models import SwinTinyV2
    >>> import numpy as np
    >>> input_data = np.ones((1, 96, 96, 1), dtype=np.float32)
    >>> model = SwinTinyV2(input_shape=(96, 96, 1), num_classes=5, stage_wise_conv=True)
    >>> output = model(input_data)

    Reference:
      - paper: https://arxiv.org/abs/2111.09883
    """

    backbone_cls = SwinBackboneV2
    variant = "tiny"

    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        patch_size=4,
        window_size=7,
        num_classes=1000,
        pooling="avg",
        dropout=0.0,
        stage_wise_conv=False,
        classifier_activation=None,
        downsampling_strategy="swin_transformer_like",
        name=None,
        **kwargs,
    ):
        """Initializes the Swin Tiny model.

        Args:
            input_shape (tuple): The shape of the input tensor (H, W, C) for 2D or (D, H, W, C) for 3D,
                excluding the batch size. Must have fixed spatial dimensions.
            include_rescaling (bool): Whether to include a rescaling layer at the input.
            include_top (bool): Whether to include the final classification layer.
            patch_size (int or tuple): The size of the non-overlapping patches.
                Should be (patch_size,) * spatial_dims.
            window_size (int or tuple): The size of the attention window.
                Should be (window_size,) * spatial_dims.
            pooling (str): Optional pooling type applied to the backbone output when `include_top` is False.
                Must be 'avg' or 'max'. Ignored if `include_top` is True.
            dropout (float): Dropout rate for the classification head. Ignored if `include_top` is False.
            attn_drop_rate (float): Dropout rate for the attention layers.
            drop_path_rate (float): Stochastic depth rate for the residual paths.
            num_classes (int): The number of output classes for classification. Default is 1000.
            classifier_activation (str, optional): The activation function for the final
                classification layer (e.g., 'softmax', 'sigmoid'). If None, no activation is applied.
                Default is None.
            downsampling_strategy: swin-transformer and swin-unetr uses bit different downsampling strategy.
                It should be either "swin_unetr_like" or "swin_transformer_like".
                'swin_transformer_like' (for 2D/3D classification tasks).
                'swin_unetr_like' (for 2D/3D segmentation tasks).
                Default: "swin_transformer_like".
            stage_wise_conv (bool): If True, use the SwinUNETR-V2 variant with convolutional residual blocks
                before each stage. Default: False.
            variant (str): The specific Swin configuration variant to use ('tiny', 'small', or 'base').
            name (str, optional): The name of the model. Default is automatically generated.
            **kwargs: Additional keyword arguments passed to the base Model class.

        Default Configuration Details (Swin-Tiny):
            - **Embedding Dimension:** 96 for 2D, 48 for 3D.
            - **Depths:** [2, 2, 6, 2] for 2D, [2, 2, 2, 2] for 3D.
            - **Number of Heads:** [3, 6, 12, 24] same for 2D and 3D.
        """
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            patch_size=patch_size,
            window_size=window_size,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            dropout=dropout,
            variant=self.variant,
            stage_wise_conv=stage_wise_conv,
            downsampling_strategy=downsampling_strategy,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="swin")
@registration.register(name="swin_small_v2", family="swin")
class SwinSmallV2(SwinVariantsBase, DescribeMixin):
    """Swin Small V2 model, a medium-scale Swin Transformer V2 for vision tasks.

    The Swin Transformer V2, based on shifted windows, is a hierarchical Vision Transformer
    that excels in general-purpose vision tasks, offering both efficiency and performance
    for 2D (images) and 3D (volumetric/video) data. The 'Small' variant offers a
    good balance between model capacity and computational cost.

    Example:
    >>> from medicai.models import SwinSmallV2
    >>> import numpy as np
    >>> input_data = np.ones((1, 96, 96, 1), dtype=np.float32)
    >>> model = SwinSmallV2(input_shape=(96, 96, 1), num_classes=5)
    >>> output = model(input_data)

    Stage-wise Residual Convolution (SwinUNETR-V2):
        If `stage_wise_conv=True`, an additional convolutional residual block
        (`UnetrBasicBlock`) is inserted at the **beginning of each Swin stage**.
        This improves local feature extraction before self-attention.

        Data flow with `stage_wise_conv=True`:
            PatchEmbed → Dropout
            └─ Stage 0: [UnetrBasicBlock] → SwinBasicLayer(0)
            └─ Stage 1: [UnetrBasicBlock] → SwinBasicLayer(1)
            └─ Stage 2: [UnetrBasicBlock] → SwinBasicLayer(2)
            └─ Stage 3: [UnetrBasicBlock] → SwinBasicLayer(3)

    Example:
    >>> from medicai.models import SwinSmallV2
    >>> import numpy as np
    >>> input_data = np.ones((1, 96, 96, 1), dtype=np.float32)
    >>> model = SwinSmallV2(input_shape=(96, 96, 1), num_classes=5, stage_wise_conv=True)
    >>> output = model(input_data)

    Reference:
      - paper: https://arxiv.org/abs/2111.09883
    """

    backbone_cls = SwinBackboneV2
    variant = "small"

    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        patch_size=4,
        window_size=7,
        num_classes=1000,
        pooling="avg",
        dropout=0.0,
        stage_wise_conv=False,
        classifier_activation=None,
        downsampling_strategy="swin_transformer_like",
        name=None,
        **kwargs,
    ):
        """Initializes the Swin Small model.

        Args:
            input_shape (tuple): The shape of the input tensor (H, W, C) for 2D or (D, H, W, C) for 3D,
                excluding the batch size. Must have fixed spatial dimensions.
            include_rescaling (bool): Whether to include a rescaling layer at the input.
            include_top (bool): Whether to include the final classification layer.
            patch_size (int or tuple): The size of the non-overlapping patches.
                Should be (patch_size,) * spatial_dims.
            window_size (int or tuple): The size of the attention window.
                Should be (window_size,) * spatial_dims.
            pooling (str): Optional pooling type applied to the backbone output when `include_top` is False.
                Must be 'avg' or 'max'. Ignored if `include_top` is True.
            dropout (float): Dropout rate for the classification head. Ignored if `include_top` is False.
            attn_drop_rate (float): Dropout rate for the attention layers.
            drop_path_rate (float): Stochastic depth rate for the residual paths.
            num_classes (int): The number of output classes for classification. Default is 1000.
            classifier_activation (str, optional): The activation function for the final
                classification layer (e.g., 'softmax', 'sigmoid'). If None, no activation is applied.
                Default is None.
            downsampling_strategy: swin-transformer and swin-unetr uses bit different downsampling strategy.
                It should be either "swin_unetr_like" or "swin_transformer_like".
                'swin_transformer_like' (for 2D/3D classification tasks).
                'swin_unetr_like' (for 2D/3D segmentation tasks).
                Default: "swin_transformer_like".
            stage_wise_conv (bool): If True, use the SwinUNETR-V2 variant with convolutional residual blocks
                before each stage. Default: False.
            variant (str): The specific Swin configuration variant to use ('tiny', 'small', or 'base').
            name (str, optional): The name of the model. Default is automatically generated.
            **kwargs: Additional keyword arguments passed to the base Model class.

        Default Configuration Details (Swin-Small):
            - **Embedding Dimension:** 96 for 2D, 48 for 3D.
            - **Depths:** [2, 2, 18, 2] for 2D, [2, 2, 6, 2] for 3D.
            - **Number of Heads:** [3, 6, 12, 24]
        """
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            patch_size=patch_size,
            window_size=window_size,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            dropout=dropout,
            variant=self.variant,
            stage_wise_conv=stage_wise_conv,
            downsampling_strategy=downsampling_strategy,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="swin")
@registration.register(name="swin_base_v2", family="swin")
class SwinBaseV2(SwinVariantsBase, DescribeMixin):
    """Swin Base V2 model, a large-scale Swin Transformer V2 for vision tasks.

    The Swin Transformer V2, based on shifted windows, is a hierarchical Vision Transformer
    that excels in general-purpose vision tasks, offering both efficiency and performance
    for 2D (images) and 3D (volumetric/video) data. The 'Base' variant offers high
    capacity suitable for complex tasks and large datasets.

    Example:
    >>> from medicai.models import SwinBaseV2
    >>> import numpy as np
    >>> input_data = np.ones((1, 96, 96, 1), dtype=np.float32)
    >>> model = SwinBaseV2(input_shape=(96, 96, 1), num_classes=5)
    >>> output = model(input_data)

    Stage-wise Residual Convolution (SwinUNETR-V2):
        If `stage_wise_conv=True`, an additional convolutional residual block
        (`UnetrBasicBlock`) is inserted at the **beginning of each Swin stage**.
        This improves local feature extraction before self-attention.

        Data flow with `stage_wise_conv=True`:
            PatchEmbed → Dropout
            └─ Stage 0: [UnetrBasicBlock] → SwinBasicLayer(0)
            └─ Stage 1: [UnetrBasicBlock] → SwinBasicLayer(1)
            └─ Stage 2: [UnetrBasicBlock] → SwinBasicLayer(2)
            └─ Stage 3: [UnetrBasicBlock] → SwinBasicLayer(3)

    Example:
    >>> from medicai.models import SwinBaseV2
    >>> import numpy as np
    >>> input_data = np.ones((1, 96, 96, 1), dtype=np.float32)
    >>> model = SwinBaseV2(input_shape=(96, 96, 1), num_classes=5, stage_wise_conv=True)
    >>> output = model(input_data)

    Reference:
      - paper: https://arxiv.org/abs/2111.09883
    """

    backbone_cls = SwinBackboneV2
    variant = "base"

    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        patch_size=4,
        window_size=7,
        num_classes=1000,
        pooling="avg",
        dropout=0.0,
        stage_wise_conv=False,
        classifier_activation=None,
        downsampling_strategy="swin_transformer_like",
        name=None,
        **kwargs,
    ):
        """Initializes the Swin Base V2 model.

        Args:
            input_shape (tuple): The shape of the input tensor (H, W, C) for 2D or (D, H, W, C) for 3D,
                excluding the batch size. Must have fixed spatial dimensions.
            include_rescaling (bool): Whether to include a rescaling layer at the input.
            include_top (bool): Whether to include the final classification layer.
            patch_size (int or tuple): The size of the non-overlapping patches.
                Should be (patch_size,) * spatial_dims.
            window_size (int or tuple): The size of the attention window.
                Should be (window_size,) * spatial_dims.
            pooling (str): Optional pooling type applied to the backbone output when `include_top` is False.
                Must be 'avg' or 'max'. Ignored if `include_top` is True.
            dropout (float): Dropout rate for the classification head. Ignored if `include_top` is False.
            attn_drop_rate (float): Dropout rate for the attention layers.
            drop_path_rate (float): Stochastic depth rate for the residual paths.
            num_classes (int): The number of output classes for classification. Default is 1000.
            classifier_activation (str, optional): The activation function for the final
                classification layer (e.g., 'softmax', 'sigmoid'). If None, no activation is applied.
                Default is None.
            downsampling_strategy: swin-transformer and swin-unetr uses bit different downsampling strategy.
                It should be either "swin_unetr_like" or "swin_transformer_like".
                'swin_transformer_like' (for 2D/3D classification tasks).
                'swin_unetr_like' (for 2D/3D segmentation tasks).
                Default: "swin_transformer_like".
            stage_wise_conv (bool): If True, use the SwinUNETR-V2 variant with convolutional residual blocks
                before each stage. Default: False.
            variant (str): The specific Swin configuration variant to use ('tiny', 'small', or 'base').
            name (str, optional): The name of the model. Default is automatically generated.
            **kwargs: Additional keyword arguments passed to the base Model class.

        Default Configuration Details (Swin-Base):
            - **Embedding Dimension:** 128 for 2D, 96 for 3D.
            - **Depths:** [2, 2, 18, 2] for 2D, [2, 2, 6, 2] for 3D.
            - **Number of Heads:** [4, 8, 16, 32] same for 2D and 3D.
        """
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            patch_size=patch_size,
            window_size=window_size,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            dropout=dropout,
            variant=self.variant,
            stage_wise_conv=stage_wise_conv,
            downsampling_strategy=downsampling_strategy,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            name=name,
            **kwargs,
        )
