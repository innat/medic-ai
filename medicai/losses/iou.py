from keras import ops

from medicai.utils import DescribeMixin

from .base import BASE_COMMON_ARGS, BaseIoULoss


class SparseIoULoss(BaseIoULoss, DescribeMixin):
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
        name = name or "sparse_iou_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
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
        target_class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "categorical_iou_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=None,
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
        name = name or "binary_iou_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
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

This loss adapts IoU loss to sparse class-index targets by one-hot encoding
them internally. When ``from_logits=True``, the predictions are passed through
softmax before the IoU score is computed.

""" + BASE_COMMON_ARGS.format(
    specific_args="",
    example="""    Compute sparse IoU loss with three common reductions::

        import keras
        from medicai.losses import SparseIoULoss

        y_true = keras.ops.array(
            [
                [[1], [0]],
                [[1], [0]],
            ],
            dtype="int32",
        )
        y_pred = keras.ops.array(
            [
                [[0.2, 0.8], [0.8, 0.2]],
                [[0.4, 0.6], [0.6, 0.4]],
            ],
            dtype="float32",
        )

        loss_none = SparseIoULoss(
            from_logits=False,
            num_classes=2,
            target_class_ids=1,
            reduction="none",
        )
        loss_sum = SparseIoULoss(
            from_logits=False,
            num_classes=2,
            target_class_ids=1,
            reduction="sum",
        )
        loss_mean = SparseIoULoss(
            from_logits=False,
            num_classes=2,
            target_class_ids=1,
            reduction="mean",
        )

        # reduction="none" -> [[0.3333], [0.5714]]
        # reduction="sum"  -> 0.9048
        # reduction="mean" -> 0.4524""",
    raises="""    ValueError: If ``target_class_ids`` is provided with an unsupported
        type or contains invalid class IDs.""",
    default_name="sparse_iou_loss",
)


CATEGORICAL_LOSS_DOCSTRING = """IoU loss for categorical one-hot encoded segmentation labels.

This loss computes ``1 - IoU`` directly from one-hot encoded targets and
prediction probabilities. When ``from_logits=True``, the predictions are
passed through softmax before the IoU score is computed.

""" + BASE_COMMON_ARGS.format(
    specific_args="",
    example="""    Compute categorical IoU loss with three common reductions::

        import keras
        from medicai.losses import CategoricalIoULoss

        y_true = keras.ops.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
            dtype="float32",
        )
        y_pred = keras.ops.array(
            [
                [[0.2, 0.8], [0.8, 0.2]],
                [[0.4, 0.6], [0.6, 0.4]],
            ],
            dtype="float32",
        )

        loss_none = CategoricalIoULoss(
            from_logits=False,
            num_classes=2,
            target_class_ids=1,
            reduction="none",
        )
        loss_sum = CategoricalIoULoss(
            from_logits=False,
            num_classes=2,
            target_class_ids=1,
            reduction="sum",
        )
        loss_mean = CategoricalIoULoss(
            from_logits=False,
            num_classes=2,
            target_class_ids=1,
            reduction="mean",
        )

        # reduction="none" -> [[0.3333], [0.5714]]
        # reduction="sum"  -> 0.9048
        # reduction="mean" -> 0.4524""",
    raises="""    ValueError: If ``target_class_ids`` is provided with an unsupported
        type or contains invalid class IDs.""",
    default_name="categorical_iou_loss",
)


BINARY_LOSS_DOCSTRING = """IoU loss for binary or multi-label segmentation tasks.

This loss computes ``1 - IoU`` for binary or multi-label targets. When
``from_logits=True``, the predictions are passed through a sigmoid activation
before the IoU score is computed.
                         
""" + BASE_COMMON_ARGS.format(
    specific_args="",
    example="""    Compute binary IoU loss with three common reductions::

        import keras
        from medicai.losses import BinaryIoULoss

        y_true = keras.ops.array(
            [
                [[1.0], [0.0]],
                [[1.0], [0.0]],
            ],
            dtype="float32",
        )
        y_pred = keras.ops.array(
            [
                [[0.8], [0.2]],
                [[0.6], [0.4]],
            ],
            dtype="float32",
        )

        loss_none = BinaryIoULoss(
            from_logits=False,
            num_classes=1,
            reduction="none",
        )
        loss_sum = BinaryIoULoss(
            from_logits=False,
            num_classes=1,
            reduction="sum",
        )
        loss_mean = BinaryIoULoss(
            from_logits=False,
            num_classes=1,
            reduction="mean",
        )

        # reduction="none" -> [[0.3333], [0.5714]]
        # reduction="sum"  -> 0.9048
        # reduction="mean" -> 0.4524""",
    raises="""    ValueError: If ``target_class_ids`` is provided with an unsupported
        type or contains invalid class IDs.
    ValueError: If ``ignore_class_ids`` is used while ``num_classes > 1``.""",
    default_name="binary_iou_loss",
)

SparseIoULoss.__doc__ = SPARSE_LOSS_DOCSTRING
CategoricalIoULoss.__doc__ = CATEGORICAL_LOSS_DOCSTRING
BinaryIoULoss.__doc__ = BINARY_LOSS_DOCSTRING
