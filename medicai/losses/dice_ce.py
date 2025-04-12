from medicai.utils.general import hide_warnings

hide_warnings()

import keras
from keras import ops

from .dice import BinaryDiceLoss, CategoricalDiceLoss, SparseDiceLoss

class SparseDiceCELoss(SparseDiceLoss):
    def call(self, y_true, y_pred):
        dice_loss = super().call(y_true, y_pred)

        if self.class_id is not None:
            y_true = super()._process_inputs(y_true)
            y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)
            y_true = ops.argmax(y_true, axis=-1, keepdims=True)

        ce_loss = keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        return dice_loss + ops.mean(ce_loss)


class CategoricalDiceCELoss(CategoricalDiceLoss):
    def call(self, y_true, y_pred):
        dice_loss = super().call(y_true, y_pred)

        if self.class_id is not None:
            y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)

        ce_loss = keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        return dice_loss + ops.mean(ce_loss)


class BinaryDiceCELoss(BinaryDiceLoss):
    def call(self, y_true, y_pred):
        dice_loss = super().call(y_true, y_pred)

        if self.class_id is not None:
            y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)

        if y_pred.shape[-1] == 1:
            ce_loss = keras.losses.binary_crossentropy(
                y_true,
                y_pred,
                from_logits=self.from_logits,
            )
        else:
            ce_loss = keras.losses.categorical_crossentropy(
                y_true,
                y_pred,
                from_logits=self.from_logits,
            )

        return dice_loss + ops.mean(ce_loss)
    