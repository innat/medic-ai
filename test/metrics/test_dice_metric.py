import numpy as np
import pytest
from keras import ops

from medicai.metrics.dice import BinaryDiceMetric, CategoricalDiceMetric, SparseDiceMetric


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_categorical_metric_perfect_prediction_is_one():
    y_true_idx = np.array([[[[0], [1]], [[1], [2]]]], dtype=np.int32)
    y_true = ops.one_hot(as_tensor(np.squeeze(y_true_idx, axis=-1), dtype="int32"), num_classes=3)
    y_pred = as_tensor(ops.convert_to_numpy(y_true), dtype="float32")

    metric = CategoricalDiceMetric(from_logits=False, num_classes=3, ignore_empty=False)
    metric.update_state(y_true, y_pred)
    score = float(ops.convert_to_numpy(metric.result()))
    metric.reset_states()

    assert score == pytest.approx(1.0, rel=1e-6)


@pytest.mark.unit
def test_sparse_metric_target_class_filtering():
    y_true = as_tensor(np.array([[[[0], [1]], [[1], [2]]]], dtype=np.float32))
    y_pred = as_tensor(
        np.array(
            [
                [
                    [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]],
                    [[0.05, 0.9, 0.05], [0.1, 0.2, 0.7]],
                ]
            ],
            dtype=np.float32,
        )
    )

    metric = SparseDiceMetric(
        from_logits=False,
        num_classes=3,
        target_class_ids=[1, 2],
        ignore_empty=False,
    )
    metric.update_state(y_true, y_pred)
    score = float(ops.convert_to_numpy(metric.result()))
    metric.reset_states()

    assert score > 0.8


@pytest.mark.unit
def test_binary_metric_output_is_scalar_and_finite():
    y_true = as_tensor(np.array([[[[1], [0]], [[0], [1]]]], dtype=np.float32))
    y_logit = as_tensor(np.array([[[[3.0], [-3.0]], [[-2.0], [2.0]]]], dtype=np.float32))

    metric = BinaryDiceMetric(from_logits=True, num_classes=1, ignore_empty=False)
    metric.update_state(y_true, y_logit)
    score = metric.result()
    metric.reset_states()

    assert tuple(ops.shape(score)) == ()
    assert np.isfinite(float(ops.convert_to_numpy(score)))
