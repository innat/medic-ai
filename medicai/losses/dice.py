from keras import ops

from medicai.utils import DescribeMixin

from .base import BASE_COMMON_ARGS, BaseDiceLoss


class SparseDiceLoss(BaseDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name or "sparse_dice_loss",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        else:
            return y_pred

    def _process_targets(self, y_true):
        if y_true.shape[-1] == 1:
            y_true = ops.squeeze(y_true, axis=-1)

        y_true = ops.one_hot(y_true, num_classes=self.num_classes)
        return y_true


class CategoricalDiceLoss(BaseDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=None,
            smooth=smooth,
            reduction=reduction,
            name=name or "categorical_dice_loss",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        else:
            return y_pred


class BinaryDiceLoss(BaseDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        if ignore_class_ids is not None and num_classes > 1:
            raise ValueError(
                "`ignore_class_ids` is only supported when `num_classes=1` "
                "(binary or sparse segmentation). One-hot or multi-label cases "
                "with `num_classes > 1` are not supported."
            )
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name or "binary_dice_loss",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.sigmoid(y_pred)
        else:
            return y_pred


CATEGORICAL_LOSS_DOCSTRING = """Dice loss for categorical (one-hot encoded) segmentation labels.

This loss function calculates the Dice loss directly using the provided
one-hot encoded labels and prediction probabilities. It uses a **Softmax**
activation on logits.

""" + BASE_COMMON_ARGS.format(
    specific_args="", default_name="categorical_dice_loss"
)

SPARSE_LOSS_DOCSTRING = """Dice loss for sparse categorical segmentation labels.

This loss function adapts the Dice loss to work with sparse labels
(integer class indices) by one-hot encoding them before calculating
the Dice coefficient. It uses a **Softmax** activation on logits.

""" + BASE_COMMON_ARGS.format(
    specific_args="", default_name="sparse_dice_loss"
)


BINARY_LOSS_DOCSTRING = """Dice loss for binary or multi-label segmentation tasks.

This loss function is specifically designed for binary or multi-label
segmentation where the labels typically have a single or multi-label channel
(representing the foreground). It uses a **Sigmoid** activation on logits.

""" + BASE_COMMON_ARGS.format(
    specific_args="", default_name="binary_dice_loss"
)

CategoricalDiceLoss.__doc__ = CATEGORICAL_LOSS_DOCSTRING
SparseDiceLoss.__doc__ = SPARSE_LOSS_DOCSTRING
BinaryDiceLoss.__doc__ = BINARY_LOSS_DOCSTRING
