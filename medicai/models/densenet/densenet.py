import keras

from ...utils.model_utils import BACKBONE_ARGS, BACKBONE_ZOO
from .densenet_backbone import DenseNetBackbone


@keras.saving.register_keras_serializable(package="densenet121")
class DenseNet121(DenseNetBackbone):
    def __init__(
        self,
        *,
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
        blocks = BACKBONE_ARGS["densenet121"]
        spatial_dims = len(input_shape) - 1
        name = name or f"DenseNet121{spatial_dims}D"
        super().__init__(
            blocks=blocks,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            input_shape=input_shape,
            input_tensor=input_tensor,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="densenet169")
class DenseNet169(DenseNetBackbone):
    def __init__(
        self,
        *,
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
        blocks = BACKBONE_ARGS["densenet169"]
        spatial_dims = len(input_shape) - 1
        name = name or f"DenseNet169{spatial_dims}D"
        super().__init__(
            blocks=blocks,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            input_shape=input_shape,
            input_tensor=input_tensor,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="densenet201")
class DenseNet201(DenseNetBackbone):
    def __init__(
        self,
        *,
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
        blocks = BACKBONE_ARGS["densenet201"]
        spatial_dims = len(input_shape) - 1
        name = name or f"DenseNet201{spatial_dims}D"
        super().__init__(
            blocks=blocks,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            input_shape=input_shape,
            input_tensor=input_tensor,
            classifier_activation=classifier_activation,
            name=name,
            **kwargs,
        )


BACKBONE_ZOO["densenet121"] = DenseNet121
BACKBONE_ZOO["densenet169"] = DenseNet169
BACKBONE_ZOO["densenet201"] = DenseNet201
