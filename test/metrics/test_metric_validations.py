import pytest

from medicai.metrics import BinaryDiceMetric, SparseDiceMetric


@pytest.mark.unit
def test_binary_metric_rejects_ignore_class_ids_for_multilabel():
    with pytest.raises(ValueError, match="ignore_class_ids"):
        BinaryDiceMetric(
            from_logits=True,
            num_classes=3,
            ignore_class_ids=[0],
        )


@pytest.mark.unit
def test_sparse_metric_rejects_out_of_range_target_class_id():
    with pytest.raises(ValueError, match="out of the valid range"):
        SparseDiceMetric(
            from_logits=True,
            num_classes=2,
            target_class_ids=[2],
        )


@pytest.mark.unit
def test_sparse_metric_rejects_invalid_target_class_type():
    with pytest.raises(ValueError, match="class_ids must be"):
        SparseDiceMetric(
            from_logits=True,
            num_classes=2,
            target_class_ids={"id": 1},
        )
