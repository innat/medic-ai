import numpy as np
import pytest
from keras import ops

from medicai.losses.dice import BinaryDiceLoss, CategoricalDiceLoss, SparseDiceLoss


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_categorical_dice_loss_perfect_prediction_is_near_zero():
    y_true_idx = np.array([[[[0], [1]], [[1], [2]]]], dtype=np.int32)
    y_true = ops.one_hot(as_tensor(np.squeeze(y_true_idx, axis=-1), dtype="int32"), num_classes=3)
    y_pred = as_tensor(ops.convert_to_numpy(y_true), dtype="float32")

    loss_fn = CategoricalDiceLoss(from_logits=False, num_classes=3)
    loss = loss_fn(y_true, y_pred)
    assert float(ops.convert_to_numpy(loss)) < 1e-5


@pytest.mark.unit
def test_sparse_dice_loss_perfect_prediction_is_near_zero():
    y_true = as_tensor(np.array([[[[0], [1]], [[1], [2]]]], dtype=np.int32), dtype="int32")
    y_pred = as_tensor(
        np.array(
            [
                [
                    [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]],
                    [[0.01, 0.98, 0.01], [0.01, 0.01, 0.98]],
                ]
            ],
            dtype=np.float32,
        )
    )

    loss_fn = SparseDiceLoss(from_logits=False, num_classes=3)
    loss = loss_fn(y_true, y_pred)
    assert float(ops.convert_to_numpy(loss)) < 0.05


@pytest.mark.unit
def test_binary_dice_loss_returns_expected_shape_with_none_reduction():
    y_true = as_tensor(
        np.array(
            [[[[[1], [0]], [[1], [0]]]], [[[[0], [1]], [[0], [1]]]]],
            dtype=np.float32,
        )
    )
    y_pred = as_tensor(
        np.array(
            [[[[[0.9], [0.1]], [[0.8], [0.2]]]], [[[[0.2], [0.8]], [[0.1], [0.9]]]]],
            dtype=np.float32,
        )
    )

    loss_fn = BinaryDiceLoss(from_logits=False, num_classes=1, reduction="none")
    loss = loss_fn(y_true, y_pred)
    assert tuple(ops.shape(loss)) == (2, 1)


@pytest.mark.unit
def test_binary_dice_loss_multilabel_logits_is_finite():
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

    loss_fn = BinaryDiceLoss(from_logits=True, num_classes=2)
    loss = loss_fn(y_true, y_logit)
    assert np.isfinite(float(ops.convert_to_numpy(loss)))

