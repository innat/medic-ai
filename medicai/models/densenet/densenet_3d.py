import keras
from keras import layers

from .densenet_backbone import DenseNet3DBackbone, parse_model_inputs


@keras.saving.register_keras_serializable(package="densenet3d.model")
class DenseNet3D(keras.Model):
    def __init__(
        self,
        *,
        variant="densenet121",
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling=None,
        input_shape=(None, None, None, 1),
        input_tensor=None,
        classifier_activation="softmax",
        name=None,
        **kwargs,
    ):

        from .densenet_model import BACKBONE_ARGS

        if variant not in BACKBONE_ARGS:
            raise ValueError(
                f"Invalid variant '{variant}'. Choose from {list(BACKBONE_ARGS.keys())}."
            )

        blocks = BACKBONE_ARGS[variant]
        name = name or variant
        GlobalAvgPool = layers.GlobalAveragePooling3D
        GlobalMaxPool = layers.GlobalMaxPooling3D

        inputs = parse_model_inputs(input_shape, input_tensor)

        # The actual backbone feature extractor (Functional)
        encoder_model = DenseNet3DBackbone(
            blocks=blocks,
            include_rescaling=include_rescaling,
            input_tensor=inputs,
            name="densenet3d_backbone",
        )
        encoder_outputs = encoder_model.output  # encoder_model(inputs)

        if include_top:
            x = GlobalAvgPool(name="avg_pool")(encoder_outputs)
            x = layers.Dense(num_classes, activation=classifier_activation, name="predictions")(x)
        elif pooling == "avg":
            x = GlobalAvgPool(name="avg_pool")(encoder_outputs)
        elif pooling == "max":
            x = GlobalMaxPool(name="max_pool")(encoder_outputs)
        else:
            x = encoder_outputs

        super().__init__(inputs=inputs, outputs=x, name=name)

        # Save reference to encoder (used for skip connections externally)
        self.encoder = encoder_model
        self.include_top = include_top
        self.pooling = pooling
        self.classifier_activation = classifier_activation
        self.variant = variant
        self.include_rescaling = include_rescaling
        self.num_classes = num_classes

    def get_config(self):
        return {
            "variant": self.variant,
            "include_rescaling": self.include_rescaling,
            "include_top": self.include_top,
            "num_classes": self.num_classes,
            "pooling": self.pooling,
            "classifier_activation": self.classifier_activation,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
