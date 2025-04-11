from medicai.utils.general import hide_warnings

hide_warnings()

from keras import ops

from .base import BaseDiceLoss


class SparseDiceLoss(BaseDiceLoss):
    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.one_hot(ops.argmax(y_pred, axis=-1), num_classes=self.num_classes)
        else:
            return y_pred

    def call(self, y_true, y_pred):
        if y_true.shape[-1] == 1:
            y_true = ops.squeeze(y_true, axis=-1)

        y_true = ops.one_hot(y_true, ops.shape(y_pred)[-1])
        return super().call(y_true, y_pred)


class CategoricalDiceLoss(BaseDiceLoss):
    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.one_hot(ops.argmax(y_pred, axis=-1), num_classes=self.num_classes)
        else:
            return y_pred


class BinaryDiceLoss(BaseDiceLoss):
    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.sigmoid(y_pred)
        else:
            return y_pred
