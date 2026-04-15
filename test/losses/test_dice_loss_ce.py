import numpy as np
import pytest
from keras import ops

from medicai.losses.dice_ce import BinaryDiceCELoss, CategoricalDiceCELoss, SparseDiceCELoss


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_sparse_dice_ce_loss_is_low_for_good_predictions():
    y_true = as_tensor(np.array([[[[0], [1]], [[1], [2]]]], dtype=np.int32), dtype="int32")
    y_pred = as_tensor(
        np.array(
            [
                [
                    [[0.97, 0.02, 0.01], [0.02, 0.97, 0.01]],
                    [[0.01, 0.98, 0.01], [0.01, 0.01, 0.98]],
                ]
            ],
            dtype=np.float32,
        )
    )
    loss_fn = SparseDiceCELoss(from_logits=False, num_classes=3)
    loss = loss_fn(y_true, y_pred)
    assert float(ops.convert_to_numpy(loss)) < 0.2


@pytest.mark.unit
def test_categorical_dice_ce_reduction_none_has_batch_class_shape():
    y_true_idx = np.array([[[[0], [1]], [[1], [2]]]], dtype=np.int32)
    y_true = ops.one_hot(as_tensor(np.squeeze(y_true_idx, axis=-1), dtype="int32"), num_classes=3)
    y_pred = as_tensor(ops.convert_to_numpy(y_true), dtype="float32")
    loss_fn = CategoricalDiceCELoss(from_logits=False, num_classes=3, reduction="none")
    loss = loss_fn(y_true, y_pred)
    assert tuple(ops.shape(loss)) == (1, 3)


@pytest.mark.unit
def test_binary_dice_ce_multilabel_logits_is_finite():
    y_true = as_tensor(
        np.array(
            [[[[[1, 0], [0, 1]], [[1, 1], [0, 0]]]]],
            dtype=np.float32,
        )
    )
    y_logit = as_tensor(
        np.array(
            [[[[[3.0, -3.0], [-3.0, 3.0]], [[2.0, 2.0], [-2.0, -2.0]]]]],
            dtype=np.float32,
        )
    )
    loss_fn = BinaryDiceCELoss(from_logits=True, num_classes=2)
    loss = loss_fn(y_true, y_logit)
    assert np.isfinite(float(ops.convert_to_numpy(loss)))
