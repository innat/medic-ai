import numpy as np
import pytest
from keras import ops

from medicai.models import ConvNeXtV2Atto, DeepLabV3Plus, DenseNet121, EfficientNetB0, UNet, UNetPlusPlus, ViTBase


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
@pytest.mark.parametrize("model_cls", [ConvNeXtV2Atto, DenseNet121, EfficientNetB0])
def test_classification_models_2d_forward_shape(model_cls):
    model = model_cls(input_shape=(32, 32, 1), num_classes=2)
    x = as_tensor(np.random.randn(1, 32, 32, 1).astype(np.float32))
    y = model(x)
    assert tuple(ops.shape(y)) == (1, 2)


@pytest.mark.unit
def test_vit_2d_forward_shape():
    model = ViTBase(
        input_shape=(32, 32, 3),
        num_classes=3,
        pooling="token",
        intermediate_dim=64,
        classifier_activation=None,
    )
    x = as_tensor(np.random.randn(2, 32, 32, 3).astype(np.float32))
    y = model(x)
    assert tuple(ops.shape(y)) == (2, 3)


@pytest.mark.unit
@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (UNet, {"encoder_name": "densenet121"}),
        (UNetPlusPlus, {"encoder_name": "efficientnet_b0"}),
        (DeepLabV3Plus, {"encoder_name": "efficientnet_v2_s", "encoder_depth": 3}),
    ],
)
def test_segmentation_models_2d_forward_shape(builder, kwargs):
    num_classes = 2
    model = builder(input_shape=(32, 32, 1), num_classes=num_classes, **kwargs)
    x = as_tensor(np.random.randn(1, 32, 32, 1).astype(np.float32))
    y = model(x)
    assert tuple(ops.shape(y)) == (1, 32, 32, num_classes)

