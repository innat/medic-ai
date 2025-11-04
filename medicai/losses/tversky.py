from keras import ops

from medicai.utils import DescribeMixin

from .base import BaseTverskyLoss


class SparseTverskyLoss(BaseTverskyLoss, DescribeMixin):
    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        else:
            return y_pred

    def _process_inputs(self, y_true):
        if y_true.shape[-1] == 1:
            y_true = ops.squeeze(y_true, axis=-1)

        y_true = ops.one_hot(y_true, num_classes=self.num_classes)
        return y_true


class CategoricalTverskyLoss(BaseTverskyLoss, DescribeMixin):
    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        else:
            return y_pred


class BinaryTverskyLoss(BaseTverskyLoss, DescribeMixin):
    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.sigmoid(y_pred)
        else:
            return y_pred
