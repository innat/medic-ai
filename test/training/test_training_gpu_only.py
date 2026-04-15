import keras
import numpy as np
import pytest
from keras import ops

from medicai.losses import SparseDiceCELoss
from medicai.metrics import SparseDiceMetric
from medicai.models import DenseNet121, UNet


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.gpu
@pytest.mark.integration
def test_gpu_only_classification_training_smoke():
    model = DenseNet121(input_shape=(32, 32, 1), num_classes=1, classifier_activation=None)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)],
        jit_compile=False,
    )

    x = as_tensor(np.random.randn(4, 32, 32, 1).astype(np.float32))
    y = as_tensor(np.random.randint(0, 2, (4, 1)).astype(np.float32))

    history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1


@pytest.mark.gpu
@pytest.mark.integration
def test_gpu_only_segmentation_training_smoke():
    num_classes = 3
    model = UNet(
        input_shape=(32, 32, 1),
        num_classes=num_classes,
        encoder_name="densenet121",
        classifier_activation=None,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=SparseDiceCELoss(from_logits=True, num_classes=num_classes),
        metrics=[
            SparseDiceMetric(
                from_logits=True,
                num_classes=num_classes,
                ignore_empty=True,
                name="dice_score",
            )
        ],
        jit_compile=False,
    )

    x = as_tensor(np.random.randn(2, 32, 32, 1).astype(np.float32))
    y = as_tensor(np.random.randint(0, num_classes, (2, 32, 32, 1)).astype(np.float32))

    history = model.fit(x, y, epochs=1, batch_size=1, verbose=0)
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1
