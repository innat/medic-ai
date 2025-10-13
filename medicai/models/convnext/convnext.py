import keras
from keras import layers

from medicai.utils import (
    DescribeMixin,
    get_act_layer,
    get_norm_layer,
    get_pooling_layer,
    registration,
)

from .convnext_backbone import ConvNeXtBackbone

MODEL_CONFIGS = {
    "tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "small": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
    },
    "xlarge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
    },
}


class ConvNeXtVariantsBase(keras.Model):
    def __init__(
        self,
        *,
        depths,
        projection_dims,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):
        spatial_dims = len(input_shape) - 1
        if name is None and self.__class__ is not ConvNeXtVariantsBase:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        backbone = ConvNeXtBackbone(
            input_shape=input_shape,
            depths=depths,
            projection_dims=projection_dims,
            include_rescaling=include_rescaling,
        )
        inputs = backbone.input
        x = backbone.output

        GlobalAvgPool = get_pooling_layer(
            spatial_dims=spatial_dims, layer_type="avg", global_pool=True
        )
        GlobalMaxPool = get_pooling_layer(
            spatial_dims=spatial_dims, layer_type="max", global_pool=True
        )
        GlobalNorm = get_norm_layer(layer_type="layer", epsilon=1e-6, name="head_norm")

        if include_top:
            x = GlobalAvgPool(x)
            x = GlobalNorm(x)
            x = layers.Dense(
                num_classes, activation=classifier_activation, dtype="float32", name="predictions"
            )(x)
        elif pooling == "avg":
            x = GlobalAvgPool(x)
            x = GlobalNorm(x)
        elif pooling == "max":
            x = GlobalMaxPool(x)
            x = GlobalNorm(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.pyramid_outputs = backbone.pyramid_outputs
        self.depths = depths
        self.projection_dims = projection_dims
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


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_tiny", family="convnext")
class ConvNeXtTiny(ConvNeXtVariantsBase, DescribeMixin):
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
            depths=MODEL_CONFIGS["tiny"]["depths"],
            projection_dims=MODEL_CONFIGS["tiny"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_small", family="convnext")
class ConvNeXtSmall(ConvNeXtVariantsBase, DescribeMixin):
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
            depths=MODEL_CONFIGS["small"]["depths"],
            projection_dims=MODEL_CONFIGS["small"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_base", family="convnext")
class ConvNeXtBase(ConvNeXtVariantsBase, DescribeMixin):
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
            depths=MODEL_CONFIGS["base"]["depths"],
            projection_dims=MODEL_CONFIGS["base"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_large", family="convnext")
class ConvNeXtLarge(ConvNeXtVariantsBase, DescribeMixin):
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
            depths=MODEL_CONFIGS["large"]["depths"],
            projection_dims=MODEL_CONFIGS["large"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="convnext")
@registration.register(name="convnext_xlarge", family="convnext")
class ConvNeXtXLarge(ConvNeXtVariantsBase, DescribeMixin):
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
            depths=MODEL_CONFIGS["xlarge"]["depths"],
            projection_dims=MODEL_CONFIGS["xlarge"]["projection_dims"],
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )
