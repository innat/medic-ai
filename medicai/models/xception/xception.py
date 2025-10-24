import keras
from keras import layers

from medicai.utils import DescribeMixin, registration, validate_activation
from medicai.utils.model_utils import get_pooling_layer

from .xception_backbone import XceptionBackbone


@keras.saving.register_keras_serializable(package="xception")
@registration.register(family="xception")
class Xception(keras.Model, DescribeMixin):
    """
    A full Xception model for classification.

    This class provides a complete Xception model, including both the
    convolutional backbone and the classification head (the "top"). It is
    capable of handling both 2D and 3D inputs. The model's architecture is
    defined by the `XceptionBackbone` and a final classification layer.

    The model can be used for a variety of tasks, including image
    classification on 2D images (with input shape `(height, width, channels)`)
    and volumetric data classification on 3D images (with input shape
    `(depth, height, width, channels)`).
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
        Initializes the Xception model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size. Can be `(height, width, channels)`
                for 2D or `(depth, height, width, channels)` for 3D.
            include_rescaling: A boolean indicating whether to include a
                `Rescaling` layer at the beginning of the model. If `True`,
                the input pixels will be scaled from `[0, 255]` to `[0, 1]`.
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
            **kwargs: Additional keyword arguments.
        """
        spatial_dims = len(input_shape) - 1
        if name is None:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        # number of classes must be positive.
        if include_top and num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        # verify input activation.
        classifier_activation = validate_activation(classifier_activation)

        backbone = XceptionBackbone(input_shape=input_shape, include_rescaling=include_rescaling)
        inputs = backbone.input
        x = backbone.output

        if include_top:
            x = get_pooling_layer(
                spatial_dims=spatial_dims, layer_type="avg", global_pool=True, name="avg_pool"
            )(x)
            x = layers.Dense(
                num_classes, activation=classifier_activation, dtype="float32", name="predictions"
            )(x)
        elif pooling in ("avg", "max"):
            x = get_pooling_layer(
                spatial_dims=spatial_dims,
                layer_type=pooling,
                global_pool=True,
                name=f"{pooling}_pool",
            )(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.pyramid_outputs = backbone.pyramid_outputs
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
