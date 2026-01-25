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

        y_true = ops.cast(y_true, "int32")
        return ops.one_hot(y_true, self.num_classes, dtype="float32")


class CategoricalCenterlineDiceLoss(BaseCenterlineDiceLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        iters=50,
        target_class_ids=None,
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
            ignore_class_ids=None,
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


_SPARSE_CLDICE_EXTRA_DOC = """

Sparse input notes
------------------
This loss expects **sparse integer labels** as ground truth, where each voxel
contains a single class index:

    y_true shape: (batch, [depth], height, width, 1)
    y_true values: {0, 1, ..., num_classes - 1}

The sparse labels are internally converted to one-hot encoding before loss
computation.

Important usage guidelines
--------------------------
- **Background exclusion**:
  Background is typically encoded as class `0`. To exclude background from the
  loss, explicitly specify:

      target_class_ids=1  (or a list of foreground classes)

- **Ignoring invalid regions**:
  If your dataset contains an "invalid" or "ignore" label (e.g., class `2`),
  you must explicitly provide:

      ignore_class_ids=2

  This ensures that voxels belonging to the ignored class are masked out
  *before* skeletonization and topology computation.

- **Why explicit ignore is important**:
  Even when optimizing only selected target classes, ignored class labels
  must be masked spatially to prevent them from contributing to skeleton
  extraction, topology precision, or sensitivity.

Example
-------
For sparse labels with:
    0 = background
    1 = foreground
    2 = ignore / invalid

Use:
    SparseCenterlineDiceLoss(
        from_logits=True,
        num_classes=3,
        target_class_ids=1,
        ignore_class_ids=2,
    )
"""


SparseCenterlineDiceLoss.__doc__ = BaseCenterlineDiceLoss.__doc__ + _SPARSE_CLDICE_EXTRA_DOC
BinaryCenterlineDiceLoss.__doc__ = BaseCenterlineDiceLoss.__doc__
CategoricalCenterlineDiceLoss.__doc__ = BaseCenterlineDiceLoss.__doc__
