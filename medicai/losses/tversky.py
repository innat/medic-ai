from keras import ops

from medicai.utils import DescribeMixin

from .base import BASE_COMMON_ARGS, BaseTverskyLoss


class SparseTverskyLoss(BaseTverskyLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        alpha=0.5,
        beta=0.5,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "sparse_tversky_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
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


class CategoricalTverskyLoss(BaseTverskyLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        alpha=0.5,
        beta=0.5,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "categorical_tversky_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
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


class BinaryTverskyLoss(BaseTverskyLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        alpha=0.5,
        beta=0.5,
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
        name = name or "binary_tversky_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
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


TWERSKY_DOCSTRING = """
    alpha (float, optional): Weight for false positives (FP). Defaults to 0.5.
    beta (float, optional): Weight for false negatives (FN). Defaults to 0.5.
"""

SPARSE_LOSS_DOCSTRING = """Tversky loss for sparse categorical segmentation labels.

This loss adapts Tversky loss to sparse class-index targets by one-hot
encoding them internally. When ``from_logits=True``, the predictions are
passed through softmax before the Tversky score is computed.

""" + BASE_COMMON_ARGS.format(
    specific_args=TWERSKY_DOCSTRING,
    example="""    Compute sparse Tversky loss with ``alpha=beta=0.5`` and three
    common reductions::

        import keras
        from medicai.losses import SparseTverskyLoss

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

        loss_none = SparseTverskyLoss(
            from_logits=False,
            num_classes=2,
            target_class_ids=1,
            alpha=0.5,
            beta=0.5,
            reduction="none",
        )
        loss_sum = SparseTverskyLoss(
            from_logits=False,
            num_classes=2,
            target_class_ids=1,
            alpha=0.5,
            beta=0.5,
            reduction="sum",
        )
        loss_mean = SparseTverskyLoss(
            from_logits=False,
            num_classes=2,
            target_class_ids=1,
            alpha=0.5,
            beta=0.5,
            reduction="mean",
        )

        # reduction="none" -> [[0.2], [0.4]]
        # reduction="sum"  -> 0.6
        # reduction="mean" -> 0.3""",
    raises="""    ValueError: If ``target_class_ids`` is provided with an unsupported
        type or contains invalid class IDs.""",
    default_name="sparse_tversky_loss",
)

CATEGORICAL_LOSS_DOCSTRING = """Tversky loss for categorical one-hot encoded segmentation labels.

This loss computes ``1 - Tversky`` directly from one-hot encoded targets and
prediction probabilities. When ``from_logits=True``, the predictions are
passed through softmax before the Tversky score is computed.

""" + BASE_COMMON_ARGS.format(
    specific_args=TWERSKY_DOCSTRING,
    example="""    Compute categorical Tversky loss with ``alpha=beta=0.5`` and
    three common reductions::

        import keras
        from medicai.losses import CategoricalTverskyLoss

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

        loss_none = CategoricalTverskyLoss(
            from_logits=False,
            num_classes=2,
            target_class_ids=1,
            alpha=0.5,
            beta=0.5,
            reduction="none",
        )
        loss_sum = CategoricalTverskyLoss(
            from_logits=False,
            num_classes=2,
            target_class_ids=1,
            alpha=0.5,
            beta=0.5,
            reduction="sum",
        )
        loss_mean = CategoricalTverskyLoss(
            from_logits=False,
            num_classes=2,
            target_class_ids=1,
            alpha=0.5,
            beta=0.5,
            reduction="mean",
        )

        # reduction="none" -> [[0.2], [0.4]]
        # reduction="sum"  -> 0.6
        # reduction="mean" -> 0.3""",
    raises="""    ValueError: If ``target_class_ids`` is provided with an unsupported
        type or contains invalid class IDs.""",
    default_name="categorical_tversky_loss",
)

BINARY_LOSS_DOCSTRING = """Tversky loss for binary segmentation tasks.

This loss computes ``1 - Tversky`` for binary targets. When
``from_logits=True``, the predictions are passed through a sigmoid activation
before the Tversky score is computed.

""" + BASE_COMMON_ARGS.format(
    specific_args=TWERSKY_DOCSTRING,
    example="""    Compute binary Tversky loss with ``alpha=beta=0.5`` and three
    common reductions::

        import keras
        from medicai.losses import BinaryTverskyLoss

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

        loss_none = BinaryTverskyLoss(
            from_logits=False,
            num_classes=1,
            alpha=0.5,
            beta=0.5,
            reduction="none",
        )
        loss_sum = BinaryTverskyLoss(
            from_logits=False,
            num_classes=1,
            alpha=0.5,
            beta=0.5,
            reduction="sum",
        )
        loss_mean = BinaryTverskyLoss(
            from_logits=False,
            num_classes=1,
            alpha=0.5,
            beta=0.5,
            reduction="mean",
        )

        # reduction="none" -> [[0.2], [0.4]]
        # reduction="sum"  -> 0.6
        # reduction="mean" -> 0.3""",
    raises="""    ValueError: If ``target_class_ids`` is provided with an unsupported
        type or contains invalid class IDs.
    ValueError: If ``ignore_class_ids`` is used while ``num_classes > 1``.""",
    default_name="binary_tversky_loss",
)

SparseTverskyLoss.__doc__ = SPARSE_LOSS_DOCSTRING
CategoricalTverskyLoss.__doc__ = CATEGORICAL_LOSS_DOCSTRING
BinaryTverskyLoss.__doc__ = BINARY_LOSS_DOCSTRING
