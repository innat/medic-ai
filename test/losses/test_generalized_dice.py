import numpy as np
import pytest
from keras import ops

from medicai.losses.generalized_dice import (
    BinaryGeneralizedDiceLoss,
    CategoricalGeneralizedDiceLoss,
    SparseGeneralizedDiceLoss,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
@pytest.mark.parametrize("weight_type", ["square", "simple", "uniform"])
def test_sparse_generalized_dice_loss_is_finite(weight_type):
    y_true = as_tensor(np.array([[[[0], [1]], [[1], [2]]]], dtype=np.int32), dtype="int32")
    y_pred = as_tensor(
        np.array(
            [
                [
                    [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
                    [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
                ]
            ],
            dtype=np.float32,
        )
    )
    loss_fn = SparseGeneralizedDiceLoss(from_logits=False, num_classes=3, weight_type=weight_type)
    loss = loss_fn(y_true, y_pred)
    assert np.isfinite(float(ops.convert_to_numpy(loss)))


@pytest.mark.unit
def test_categorical_generalized_dice_perfect_prediction_is_near_zero():
    y_true_idx = np.array([[[[0], [1]], [[1], [2]]]], dtype=np.int32)
    y_true = ops.one_hot(as_tensor(np.squeeze(y_true_idx, axis=-1), dtype="int32"), num_classes=3)
    y_pred = as_tensor(ops.convert_to_numpy(y_true), dtype="float32")
    loss_fn = CategoricalGeneralizedDiceLoss(from_logits=False, num_classes=3)
    loss = loss_fn(y_true, y_pred)
    assert float(ops.convert_to_numpy(loss)) < 1e-5


@pytest.mark.unit
def test_binary_generalized_dice_multilabel_reduction_none_shape():
    y_true = as_tensor(np.array([[[[1, 0], [0, 1]]]], dtype=np.float32))
    y_pred = as_tensor(np.array([[[[0.9, 0.1], [0.1, 0.9]]]], dtype=np.float32))
    loss_fn = BinaryGeneralizedDiceLoss(from_logits=False, num_classes=2, reduction="none")
    loss = loss_fn(y_true, y_pred)
    assert tuple(ops.shape(loss)) == (1,)

