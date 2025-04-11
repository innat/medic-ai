import keras
from keras import ops


class BaseDiceLoss(keras.losses.Loss):
    def __init__(
        self,
        from_logits,
        num_classes,
        class_id=None,
        smooth=1e-5,
        squared_pred=False,
        name="base_dice_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Handle class_id specification
        if class_id is None:
            self.class_id = list(range(num_classes))
        elif isinstance(class_id, int):
            self.class_id = [class_id]
        else:
            self.class_id = class_id

        self.num_classes = num_classes
        self.from_logits = from_logits
        self.squared_pred = squared_pred
        self.smooth = smooth or keras.backend.epsilon()

    def _validate_and_get_class_id(self, class_id, num_classes):
        if class_id is None:
            return list(range(num_classes))
        elif isinstance(class_id, int):
            return [class_id]
        elif isinstance(class_id, list):
            for cid in class_id:
                if not 0 <= cid < num_classes:
                    raise ValueError(
                        f"Class ID {cid} is out of the valid range [0, {num_classes - 1}]."
                    )
            return class_id
        else:
            raise ValueError(
                "class_id must be an integer, a list of integers, or None to consider all classes."
            )

    def _get_desired_class_channels(self, y_true, y_pred):
        if self.class_id is None:
            return y_true, y_pred

        # for single binary case
        if self.num_classes == 1:
            return y_true, y_pred

        y_true = ops.take(y_true, self.class_id, axis=-1)
        y_pred = ops.take(y_pred, self.class_id, axis=-1)

        return y_true, y_pred

    def _process_predictions(self, y_pred):
        return y_pred

    def dice_loss(self, y_true, y_pred):
        y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)
        intersection = ops.sum(y_true * y_pred, axis=[1, 2, 3])
        union = ops.sum(y_true, axis=[1, 2, 3]) + ops.sum(
            ops.square(y_pred) if self.squared_pred else y_pred, axis=[1, 2, 3]
        )
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - ops.mean(dice_score)

    def call(self, y_true, y_pred):
        y_pred_processed = self._process_predictions(y_pred)
        y_true_processed = y_true

        y_pred_processed = ops.clip(y_pred_processed, self.smooth, 1.0 - self.smooth)
        dice = self.dice_loss(y_true_processed, y_pred_processed)
        return dice
