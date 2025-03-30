from medicai.utils.general import hide_warnings

hide_warnings()

import keras
from keras import losses, ops


class SparseDiceCELoss(keras.losses.Loss):
    def __init__(
        self, from_logits=False, smooth=1e-5, reduction="sum_over_batch_size", name="sparse_dice_ce"
    ):
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits
        self.smooth = smooth

    def dice_loss(self, y_true, y_pred):
        intersection = ops.sum(y_true * y_pred, axis=[1, 2, 3])
        union = ops.sum(y_true, axis=[1, 2, 3]) + ops.sum(y_pred, axis=[1, 2, 3])
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - ops.mean(dice_score)

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = ops.softmax(y_pred, axis=-1)

        # Convert sparse labels to one-hot for Dice, keep sparse for CE
        y_true_onehot = ops.one_hot(ops.squeeze(y_true, axis=-1), ops.shape(y_pred)[-1])

        # Clip predictions for numerical stability
        y_pred = ops.clip(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())

        dice = self.dice_loss(y_true_onehot, y_pred)
        ce = losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)

        return dice + ops.mean(ce)


class CategoricalDiceCELoss(keras.losses.Loss):
    def __init__(
        self,
        from_logits=False,
        smooth=1e-5,
        reduction="sum_over_batch_size",
        name="categorical_dice_ce",
    ):
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits
        self.smooth = smooth

    def dice_loss(self, y_true, y_pred):
        intersection = ops.sum(y_true * y_pred, axis=[1, 2, 3])
        union = ops.sum(y_true, axis=[1, 2, 3]) + ops.sum(y_pred, axis=[1, 2, 3])
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - ops.mean(dice_score)

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = ops.softmax(y_pred, axis=-1)

        # Clip predictions for numerical stability
        y_pred = ops.clip(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())

        dice = self.dice_loss(y_true, y_pred)
        ce = losses.categorical_crossentropy(y_true, y_pred, from_logits=False)

        return dice + ops.mean(ce)


class BinaryDiceCELoss(keras.losses.Loss):
    def __init__(
        self, from_logits=False, smooth=1e-5, reduction="sum_over_batch_size", name="binary_dice_ce"
    ):
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits
        self.smooth = smooth

    def dice_loss(self, y_true, y_pred):
        intersection = ops.sum(y_true * y_pred, axis=[1, 2, 3])
        union = ops.sum(y_true, axis=[1, 2, 3]) + ops.sum(y_pred, axis=[1, 2, 3])
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - ops.mean(dice_score)

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = ops.sigmoid(y_pred)

        # Ensure y_true is float32 (matches y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)

        # Clip predictions for numerical stability
        y_pred = ops.clip(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())

        dice = self.dice_loss(y_true, y_pred)
        ce = losses.binary_crossentropy(y_true, y_pred, from_logits=False)

        return dice + ops.mean(ce)
