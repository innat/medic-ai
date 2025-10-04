import keras
from keras import layers

from medicai.utils import get_pooling_layer

from .efficientnet_backbone import EfficientNetBackbone


@keras.saving.register_keras_serializable(package="efficientnetbase")
class EfficientNetBase(keras.Model):
    def __init__(
        self,
        *,
        width_coefficient,
        depth_coefficient,
        drop_connect_rate,
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
        if name is None and self.__class__ is not EfficientNetBase:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        backbone = EfficientNetBackbone(
            input_shape=input_shape,
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            drop_connect_rate=drop_connect_rate,
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
        if include_top:
            x = GlobalAvgPool(x)
            x = layers.Dense(
                num_classes, activation=classifier_activation, dtype="float32", name="predictions"
            )(x)
        elif pooling == "avg":
            x = GlobalAvgPool(x)
        elif pooling == "max":
            x = GlobalMaxPool(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.pyramid_outputs = backbone.pyramid_outputs
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.drop_connect_rate = drop_connect_rate
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

    # if include_top:
    #         x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    #         if dropout_rate > 0:
    #             x = layers.Dropout(dropout_rate, name="top_dropout")(x)
    #         x = layers.Dense(
    #             classes,
    #             activation=classifier_activation,
    #             kernel_initializer="glorot_uniform",
    #             name="predictions",
    #         )(x)
    #     else:
    #         if pooling == "avg":
    #             x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    #         elif pooling == "max":
    #             x = layers.GlobalMaxPooling2D(name="max_pool")(x)
