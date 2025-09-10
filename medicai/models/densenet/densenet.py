import keras
from keras import layers

from ...utils.model_utils import BACKBONE_ARGS, BACKBONE_ZOO, get_pooling_layer, parse_model_inputs
from .densenet_backbone import DenseNetBackbone


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


class DenseNet169(keras.Model):
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


class DenseNet201(keras.Model):
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
