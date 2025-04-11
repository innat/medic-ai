from medicai.utils.general import hide_warnings

hide_warnings()

import keras
from keras import losses, ops

from .dice import BinaryDiceLoss, CategoricalDiceLoss, SparseDiceLoss


class SparseDiceCELoss(SparseDiceLoss):

    def _process_inputs(self, y_true, y_pred):
        if y_true.shape[-1] == 1:
            y_true = ops.squeeze(y_true, axis=-1)

        y_true = ops.one_hot(y_true, ops.shape(y_pred)[-1])
        # TODO: fix for ce loss!
        # y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)
        y_true = ops.argmax(y_true, axis=-1)
        return y_true, y_pred

    def call(self, y_true, y_pred):
        dice_loss = super().call(y_true, y_pred)

        if self.class_id is not None:
            y_true, y_pred = self._process_inputs(y_true, y_pred)

        ce_loss = losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        return dice_loss + ops.mean(ce_loss)


class CategoricalDiceCELoss(CategoricalDiceLoss):

    def _process_inputs(self, y_true, y_pred):
        # TODO: fix for ce loss!
        # y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)
        return y_true, y_pred

    def call(self, y_true, y_pred):
        dice_loss = super().call(y_true, y_pred)

        if self.class_id is not None:
            y_true, y_pred = self._process_inputs(y_true, y_pred)

        ce_loss = losses.categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        return dice_loss + ops.mean(ce_loss)


class BinaryDiceCELoss(BinaryDiceLoss):

    def _process_inputs(self, y_true, y_pred):
        # TODO: fix for ce loss!
        # y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)
        return y_true, y_pred

    def call(self, y_true, y_pred):
        dice_loss = super().call(y_true, y_pred)

        if self.class_id is not None:
            y_true, y_pred = self._process_inputs(y_true, y_pred)

        if y_pred.shape[-1] == 1:
            ce_loss = losses.binary_crossentropy(
                y_true,
                y_pred,
                from_logits=self.from_logits,
            )
        else:
            ce_loss = losses.categorical_crossentropy(
                y_true,
                y_pred,
                from_logits=self.from_logits,
            )

        return dice_loss + ops.mean(ce_loss)
