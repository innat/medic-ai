from medicai.utils.general import hide_warnings

hide_warnings()

from keras import ops

from .base import BaseDiceMetric


class BinaryDiceMetric(BaseDiceMetric):
    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.cast(ops.sigmoid(y_pred) > self.threshold, dtype="float32")
        else:
            return ops.cast(y_pred > self.threshold, dtype="float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, dtype="float32")
        y_pred = ops.cast(y_pred, dtype="float32")
        super().update_state(y_true, y_pred, sample_weight)


class CategoricalDiceMetric(BaseDiceMetric):
    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.one_hot(ops.argmax(y_pred, axis=-1), num_classes=self.num_classes)
        else:
            return y_pred

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, y_pred, sample_weight)


class SparseDiceMetric(BaseDiceMetric):
    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.one_hot(ops.argmax(y_pred, axis=-1), num_classes=self.num_classes)
        else:
            return y_pred

    def update_state(self, y_true, y_pred, sample_weight=None):
        if y_true.shape[-1] == 1:
            y_true = ops.squeeze(y_true, axis=-1)
        y_true_one_hot = ops.one_hot(ops.cast(y_true, "int32"), num_classes=self.num_classes)
        super().update_state(y_true_one_hot, y_pred, sample_weight)
