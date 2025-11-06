from keras import ops

from medicai.utils import DescribeMixin

from .base import BASE_COMMON_ARGS, BaseIoULoss


class SparseIoULoss(BaseIoULoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "sparse_iou_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
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


class CategoricalIoULoss(BaseIoULoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "categorical_iou_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
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


class BinaryIoULoss(BaseIoULoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "binary_iou_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            class_ids=class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name,
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.sigmoid(y_pred)
        else:
            return y_pred


SPARSE_LOSS_DOCSTRING = """IoU loss for sparse categorical segmentation labels.

This loss function adapts the IoU loss to work with sparse labels
(integer class indices) by one-hot encoding them before calculating
the IoU. It uses a **Softmax** activation on logits.

""" + BASE_COMMON_ARGS.format(
    specific_args="", default_name="sparse_iou_loss"
)


CATEGORICAL_LOSS_DOCSTRING = """IoU loss for categorical (one-hot encoded) 
segmentation labels.

This loss function calculates the IoU loss directly using the provided
one-hot encoded labels and prediction probabilities. 
It uses a **Softmax** activation on logits.

""" + BASE_COMMON_ARGS.format(
    specific_args="", default_name="categorical_iou_loss"
)


BINARY_LOSS_DOCSTRING = """IoU loss for binary or multi-label segmentation tasks.

This loss function is specifically designed for binary or multi-label segmentation 
where the labels typically have a single channel (representing the foreground).
It uses a **Sigmoid** activation on logits.
                         
""" + BASE_COMMON_ARGS.format(
    specific_args="", default_name="binary_iou_loss"
)

SparseIoULoss.__doc__ = SPARSE_LOSS_DOCSTRING
CategoricalIoULoss.__doc__ = CATEGORICAL_LOSS_DOCSTRING
BinaryIoULoss.__doc__ = BINARY_LOSS_DOCSTRING
