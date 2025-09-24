import keras
from keras import layers

from medicai.utils import get_pooling_layer, registration

from .mit_backbone import MiTBackbone


@keras.saving.register_keras_serializable(package="mitbase")
class MiTBase(keras.Model):
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

        spatial_dims = len(input_shape) - 1
        if name is None and self.__class__ is not MiTBase:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        backbone = MiTBackbone(
            input_shape=input_shape,
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
        inputs = backbone.inputs
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


@keras.saving.register_keras_serializable(package="mitb0")
@registration.register(name="mit_b0", family="mit")
class MixVisionTransformerB0(MiTBase):
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

        super().__init__(
            input_shape=input_shape,
            max_drop_path_rate=0.1,
            layer_norm_epsilon=1e-6,
            qkv_bias=True,
            project_dim=[32, 64, 160, 256],
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


@keras.saving.register_keras_serializable(package="mitb0")
@registration.register(name="mit_b1", family="mit")
class MixVisionTransformerB1(MiTBase):
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


@keras.saving.register_keras_serializable(package="mitb0")
@registration.register(name="mit_b2", family="mit")
class MixVisionTransformerB2(MiTBase):
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


@keras.saving.register_keras_serializable(package="mitb0")
@registration.register(name="mit_b3", family="mit")
class MixVisionTransformerB3(MiTBase):
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


@keras.saving.register_keras_serializable(package="mitb0")
@registration.register(name="mit_b4", family="mit")
class MixVisionTransformerB4(MiTBase):
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


@keras.saving.register_keras_serializable(package="mitb0")
@registration.register(name="mit_b5", family="mit")
class MixVisionTransformerB5(MiTBase):
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
