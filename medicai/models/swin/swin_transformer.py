import keras

from medicai.utils import get_pooling_layer, registration

from .swin_backbone import SwinBackbone


@keras.saving.register_keras_serializable(package="swin")
class SwinVariantsBase(keras.Model):
    """A 3D Swin Transformer model for classification.

    ...
    """

    def __init__(
        self,
        *,
        input_shape,
        include_rescaling,
        include_top,
        pooling,
        dropout,
        patch_size,
        depths,
        window_size,
        num_heads,
        embed_dim,
        attn_drop_rate,
        drop_path_rate,
        num_classes=1000,
        classifier_activation=None,
        name=None,
        **kwargs,
    ):
        """Initializes the SwinTransformer model.

        Args:
            input_shape (tuple): The shape of the input tensor (depth, height, width, channels).
                Default is (96, 96, 96, 1).
            num_classes (int): The number of output classes for classification. Default is 4.
            classifier_activation (str, optional): The activation function for the final
                classification layer (e.g., 'softmax'). If None, no activation is applied.
                Default is None.
            **kwargs: Additional keyword arguments passed to the base Model class.
        """
        spatial_dims = len(input_shape) - 1
        if name is None and self.__class__ is not SwinVariantsBase:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        backbone = SwinBackbone(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            patch_size=patch_size,
            depths=depths,
            window_size=window_size,
            num_heads=num_heads,
            embed_dim=embed_dim,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=False,
        )
        input = backbone.input
        x = backbone.output

        GlobalAvgPool = get_pooling_layer(
            spatial_dims=spatial_dims, layer_type="avg", global_pool=True
        )
        GlobalMaxPool = get_pooling_layer(
            spatial_dims=spatial_dims, layer_type="max", global_pool=True
        )
        if include_top:
            x = GlobalAvgPool(x)
            x = keras.layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(x)
        elif pooling == "avg":
            x = GlobalAvgPool(x)
        elif pooling == "max":
            x = GlobalMaxPool(x)

        super().__init__(inputs=input, outputs=x, **kwargs)

        self.pyramid_outputs = backbone.pyramid_outputs
        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.num_classes = num_classes
        self.pooling = pooling
        self.dropout = dropout
        self.patch_size = patch_size
        self.depths = depths
        self.window_size = window_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.classifier_activation = classifier_activation
        self.name = name

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="swin")
@registration.register(name="swin_tiny", family="swin")
class SwinTiny(SwinVariantsBase):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling="avg",
        use_class_token=True,
        dropout=0.0,
        classifier_activation=None,
        name=None,
        **kwargs,
    ):
        if not input_shape:
            raise ValueError(
                "Argument `input_shape` must be provided. "
                "It should be a tuple of integers specifying the dimensions of the input "
                "data, not including the batch size. "
                "For 2D data, the format is `(height, width, channels)`. "
                "For 3D data, the format is `(depth, height, width, channels)`."
            )
        spatial_dims = len(input_shape) - 1
        if spatial_dims not in (2, 3):
            raise ValueError(
                f"Invalid `input_shape`: {input_shape}. "
                f"Expected 3D (H, W, C) for 2D data or 4D (D, H, W, C) for 3D data, "
                f"but got {len(input_shape)}D."
            )

        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            use_class_token=use_class_token,
            dropout=dropout,
            patch_size=[2, 2, 2] if spatial_dims == 3 else [2, 4, 4],
            depths=[2, 2, 2, 2] if spatial_dims == 3 else [2, 2, 6, 2],
            window_size=[7, 7, 7] if spatial_dims == 3 else [8, 7, 7],
            num_heads=[3, 6, 12, 24],
            embed_dim=48 if spatial_dims == 3 else 96,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="swin")
@registration.register(name="swin_small", family="swin")
class SwinSmall(SwinVariantsBase):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling="avg",
        use_class_token=True,
        dropout=0.0,
        classifier_activation=None,
        name=None,
        **kwargs,
    ):
        if not input_shape:
            raise ValueError(
                "Argument `input_shape` must be provided. "
                "It should be a tuple of integers specifying the dimensions of the input "
                "data, not including the batch size. "
                "For 2D data, the format is `(height, width, channels)`. "
                "For 3D data, the format is `(depth, height, width, channels)`."
            )
        spatial_dims = len(input_shape) - 1
        if spatial_dims not in (2, 3):
            raise ValueError(
                f"Invalid `input_shape`: {input_shape}. "
                f"Expected 3D (H, W, C) for 2D data or 4D (D, H, W, C) for 3D data, "
                f"but got {len(input_shape)}D."
            )

        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            use_class_token=use_class_token,
            dropout=dropout,
            patch_size=[2, 2, 2] if spatial_dims == 3 else [2, 4, 4],
            depths=[2, 2, 6, 2] if spatial_dims == 3 else [2, 2, 18, 2],
            window_size=[7, 7, 7] if spatial_dims == 3 else [8, 7, 7],
            num_heads=[3, 6, 12, 24],
            embed_dim=48 if spatial_dims == 3 else 96,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="swin")
@registration.register(name="swin_base", family="swin")
class SwinBase(SwinVariantsBase):
    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling="avg",
        use_class_token=True,
        dropout=0.0,
        classifier_activation=None,
        name=None,
        **kwargs,
    ):
        if not input_shape:
            raise ValueError(
                "Argument `input_shape` must be provided. "
                "It should be a tuple of integers specifying the dimensions of the input "
                "data, not including the batch size. "
                "For 2D data, the format is `(height, width, channels)`. "
                "For 3D data, the format is `(depth, height, width, channels)`."
            )
        spatial_dims = len(input_shape) - 1
        if spatial_dims not in (2, 3):
            raise ValueError(
                f"Invalid `input_shape`: {input_shape}. "
                f"Expected 3D (H, W, C) for 2D data or 4D (D, H, W, C) for 3D data, "
                f"but got {len(input_shape)}D."
            )

        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            use_class_token=use_class_token,
            dropout=dropout,
            patch_size=[2, 2, 2] if spatial_dims == 3 else [2, 4, 4],
            depths=[2, 2, 6, 2] if spatial_dims == 3 else [2, 2, 18, 2],
            window_size=[7, 7, 7] if spatial_dims == 3 else [8, 7, 7],
            num_heads=[4, 8, 16, 32],
            embed_dim=96 if spatial_dims == 3 else 128,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="swin.transformer")
@registration.register(family="swin")
class SwinTransformer(keras.Model):
    """A 3D Swin Transformer model for classification.

    This model utilizes the Swin Transformer backbone for feature extraction
    from 3D input data and includes a global average pooling layer followed
    by a dense layer for classification.
    """

    def __init__(
        self, *, input_shape=(96, 96, 96, 1), num_classes=4, classifier_activation=None, **kwargs
    ):
        """Initializes the SwinTransformer model.

        Args:
            input_shape (tuple): The shape of the input tensor (depth, height, width, channels).
                Default is (96, 96, 96, 1).
            num_classes (int): The number of output classes for classification. Default is 4.
            classifier_activation (str, optional): The activation function for the final
                classification layer (e.g., 'softmax'). If None, no activation is applied.
                Default is None.
            **kwargs: Additional keyword arguments passed to the base Model class.
        """
        spatial_dims = len(input_shape) - 1
        inputs = keras.Input(shape=input_shape)
        encoder = SwinBackbone(
            input_shape=input_shape,
            patch_size=[2, 2, 2],
            depths=[2, 2, 2, 2],
            window_size=[7, 7, 7],
            num_heads=[3, 6, 12, 24],
            embed_dim=48,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            patch_norm=False,
        )(inputs)

        x = get_pooling_layer(spatial_dims=spatial_dims, layer_type="avg", global_pool=True)(
            encoder
        )
        outputs = keras.layers.Dense(num_classes, activation=classifier_activation)(x)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
        }
        return config
