from keras import ops

from .base import BaseCenterlineDiceLoss


class BinaryCenterlineDiceLoss(BaseCenterlineDiceLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        iters=50,
        target_class_ids=None,
        ignore_class_ids=None,
        memory_efficient_skeleton=True,
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
            iters=iters,
            memory_efficient_skeleton=memory_efficient_skeleton,
            smooth=smooth,
            reduction=reduction,
            name=name or "binary_cldice",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.sigmoid(y_pred)
        else:
            return y_pred


class SparseCenterlineDiceLoss(BaseCenterlineDiceLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        iters=50,
        target_class_ids=None,
        ignore_class_ids=None,
        memory_efficient_skeleton=True,
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
            iters=iters,
            memory_efficient_skeleton=memory_efficient_skeleton,
            smooth=smooth,
            reduction=reduction,
            name=name or "sparse_cldice",
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

        y = ops.cast(y_true, "int32")
        fg_masks = []
        for c in range(self.num_classes):
            fg = ops.cast(ops.equal(y, c), "float32")
            fg_masks.append(fg)

        return ops.stack(fg_masks, axis=-1)


class CategoricalineDiceLoss(BaseCenterlineDiceLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        iters=50,
        target_class_ids=None,
        ignore_class_ids=None,
        memory_efficient_skeleton=True,
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
            iters=iters,
            memory_efficient_skeleton=memory_efficient_skeleton,
            smooth=smooth,
            reduction=reduction,
            name=name or "categorical_cldice",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        else:
            return y_pred
