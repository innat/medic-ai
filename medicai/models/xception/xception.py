import keras
from keras import layers

from medicai.utils import DescribeMixin, registration, validate_activation
from medicai.utils.model_utils import get_pooling_layer

from .xception_backbone import XceptionBackbone


@keras.saving.register_keras_serializable(package="xception")
@registration.register(name="xception", family="xception")
class Xception(keras.Model, DescribeMixin):
    """
    This class combines the ``XceptionBackbone`` with an optional
    classification head. The backbone follows the Xception design with
    depthwise separable convolutions and residual stage transitions, and it
    exposes intermediate pyramid features ``P1`` through ``P5`` for
    feature-based workflows.

    Args:
        input_shape: A tuple specifying the input shape of the model, not
            including the batch size. This can describe either 2D or 3D
            inputs.
        include_rescaling: Whether to include a ``Rescaling`` layer at the
            beginning of the model.
        include_top: Whether to include the final classification head.
        num_classes: Number of classes for the classification layer. This is
            only relevant if ``include_top`` is ``True``.
        pooling: Optional pooling mode applied when ``include_top`` is
            ``False``. Supported values are ``"avg"`` and ``"max"``.
        classifier_activation: Activation function used by the classification
            head.
        name: Optional model name.
        **kwargs: Additional keyword arguments passed to ``keras.Model``.

    Example:
        Build a 2D classification model::

            import torch
            from medicai.models import Xception

            model = Xception(
                input_shape=(224, 224, 3),
                include_top=True,
                num_classes=2,
            )
            x = torch.randn((1, 224, 224, 3))
            y = model(x)
            print(y.shape)  # torch.Size([1, 2])

    Returns:
        ``keras.KerasTensor``: The output depends on the ``include_top`` and
        ``pooling`` arguments:

        - If ``include_top=True``, returns a classification tensor of shape
          ``(batch_size, num_classes)``.
        - If ``include_top=False`` and ``pooling`` is ``None``, returns the
          final backbone feature tensor.
        - If ``include_top=False`` and ``pooling`` is ``"avg"`` or ``"max"``,
          returns a pooled feature tensor.

    References:
        - Xception: Deep Learning with Depthwise Separable Convolutions. `arXiv:1610.02357 <https://arxiv.org/abs/1610.02357>`_
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
