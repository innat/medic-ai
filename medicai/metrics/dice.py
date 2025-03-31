from medicai.utils.general import hide_warnings

hide_warnings()

from keras import ops
from keras.metrics import Metric
from .base import BaseDiceMetric


class SparseDiceMetric(BaseDiceMetric):
    def __init__(
        self,
        num_classes,
        class_id=None,
        from_logits=True,
        smooth=1e-6,
        name="sparse_categorical_dice",
        **kwargs
    ):
        super().__init__(
            num_classes=num_classes,
            class_id=class_id,
            smooth=smooth,
            name=name,
            **kwargs
        )
        self.from_logits = from_logits

    def compute_dice_components(self, y_true, y_pred):
        y_true_processed = ops.one_hot(
            ops.squeeze(ops.cast(y_true, 'int32')), 
            num_classes=self.num_classes
        )
        y_pred_processed = ops.nn.softmax(y_pred) if self.from_logits else y_pred
        return y_true_processed, y_pred_processed

class DiceMetric(Metric):

    reduction_map = {
        "mean": ops.mean,
        "sum": ops.sum,
        "none": lambda x: x,
    }

    def __init__(
        self,
        num_classes,
        include_background=True,
        reduction="mean",
        ignore_empty=True,
        smooth=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.include_background = include_background
        self.reduction = reduction
        self.ignore_empty = ignore_empty
        self.smooth = smooth
        self.intersection = self.add_weight(
            name="intersection", shape=(num_classes,), initializer="zeros"
        )
        self.union = self.add_weight(name="union", shape=(num_classes,), initializer="zeros")
        self.not_nans = self.add_weight(name="not_nans", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.one_hot(
            ops.squeeze(ops.cast(y_true, "int32"), axis=-1), num_classes=self.num_classes
        )
        y_true_reshaped = ops.reshape(y_true, [-1, self.num_classes])

        y_pred = ops.cast(y_pred, y_true.dtype)
        y_pred = ops.nn.softmax(y_pred)
        y_pred_reshaped = ops.reshape(y_pred, [-1, self.num_classes])

        intersection = ops.sum(y_true_reshaped * y_pred_reshaped, axis=0)
        union = ops.sum(y_true_reshaped, axis=0) + ops.sum(y_pred_reshaped, axis=0)

        if self.ignore_empty:
            empty_gt = ops.sum(y_true_reshaped, axis=0) == 0
            intersection = ops.where(empty_gt, ops.zeros_like(intersection), intersection)
            union = ops.where(empty_gt, ops.zeros_like(union), union)

        self.intersection.assign_add(intersection)
        self.union.assign_add(union)
        self.not_nans.assign_add(ops.sum(ops.cast(union > 0, "float32")))

    def result(self):
        dice = (2.0 * self.intersection + self.smooth) / (self.union + self.smooth)
        if not self.include_background:
            dice = dice[1:]

        if self.reduction not in self.reduction_map:
            raise ValueError(f"Unsupported reduction mode: {self.reduction}")

        return self.reduction_map.get(self.reduction)(dice)

    def reset_states(self):
        self.intersection.assign(ops.zeros_like(self.intersection))
        self.union.assign(ops.zeros_like(self.union))
        self.not_nans.assign(0.0)
