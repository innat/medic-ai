import pytest

from medicai.losses import BinaryDiceCELoss, BinaryDiceLoss, BinaryGeneralizedDiceLoss, SparseDiceLoss


@pytest.mark.unit
@pytest.mark.parametrize(
    "loss_cls,kwargs",
    [
        (BinaryDiceLoss, {}),
        (BinaryGeneralizedDiceLoss, {"weight_type": "square"}),
        (BinaryDiceCELoss, {}),
    ],
)
def test_binary_losses_reject_ignore_class_ids_for_multilabel(loss_cls, kwargs):
    with pytest.raises(ValueError, match="ignore_class_ids"):
        loss_cls(
            from_logits=True,
            num_classes=2,
            ignore_class_ids=[0],
            **kwargs,
        )


@pytest.mark.unit
def test_sparse_dice_loss_rejects_out_of_range_target_class_id():
    with pytest.raises(ValueError, match="out of the valid range"):
        SparseDiceLoss(
            from_logits=True,
            num_classes=3,
            target_class_ids=[0, 3],
        )


@pytest.mark.unit
def test_sparse_dice_loss_rejects_invalid_target_class_type():
    with pytest.raises(ValueError, match="class_ids must be"):
        SparseDiceLoss(
            from_logits=True,
            num_classes=3,
            target_class_ids="1",
        )

