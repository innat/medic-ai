from keras import ops

from medicai.utils import DescribeMixin

from .base import BASE_COMMON_ARGS, BaseGeneralizedDiceLoss


class SparseGeneralizedDiceLoss(BaseGeneralizedDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        weight_type="square",
        class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "sparse_generalized_dice_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            weight_type=weight_type,
            class_ids=class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name,
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


class CategoricalGeneralizedDiceLoss(BaseGeneralizedDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        weight_type="square",
        class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "categorical_generalized_dice_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            weight_type=weight_type,
            class_ids=class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name,
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        else:
            return y_pred


class BinaryGeneralizedDiceLoss(BaseGeneralizedDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        class_ids=None,
        smooth=1e-7,
        weight_type="square",
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "binary_generalized_dice_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            class_ids=class_ids,
            smooth=smooth,
            weight_type=weight_type,
            reduction=reduction,
            name=name,
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.sigmoid(y_pred)
        else:
            return y_pred


EXTRA_INFO = """
\nNote: Unlike other losses, Generalized Dice loss aggregates all classes into a single 
score per batch element, so with reduction='none', it returns shape [batch] instead 
of [batch, num_classes].
"""

SPARSE_LOSS_DOCSTRING = (
    """Generalized Dice loss for sparse categorical segmentation labels.

This loss function adapts the Generalized Dice loss to work with sparse labels
(integer class indices) by one-hot encoding them before calculating
the Generalized Dice. It uses a **Softmax** activation on logits.

"""
    + BASE_COMMON_ARGS.format(specific_args="", default_name="sparse_generalized_dice_loss")
    + EXTRA_INFO
)


CATEGORICAL_LOSS_DOCSTRING = (
    """Generalized Dice loss for categorical (one-hot encoded) segmentation labels.

This loss function calculates the Generalized Dice Loss (GDL) directly using the provided
one-hot encoded labels and prediction probabilities, applying a softmax activation
if predictions are logits.

"""
    + BASE_COMMON_ARGS.format(specific_args="", default_name="categorical_generalized_dice_loss")
    + EXTRA_INFO
)


BINARY_LOSS_DOCSTRING = (
    """Generalized Dice loss for binary or multi-label segmentation tasks.

This loss function is specifically designed for binary or multi-label
segmentation where the labels typically have a single or multi-label channel
(representing the foreground). It uses a **Sigmoid** activation on logits.

"""
    + BASE_COMMON_ARGS.format(specific_args="", default_name="binary_generalized_dice_loss")
    + EXTRA_INFO
)

SparseGeneralizedDiceLoss.__doc__ = SPARSE_LOSS_DOCSTRING
CategoricalGeneralizedDiceLoss.__doc__ = CATEGORICAL_LOSS_DOCSTRING
BinaryGeneralizedDiceLoss.__doc__ = BINARY_LOSS_DOCSTRING
