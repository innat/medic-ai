from medicai.utils.general import hide_warnings

hide_warnings()

from keras import ops

from .base import BaseDiceLoss


class SparseDiceLoss(BaseDiceLoss):
    """Dice loss for sparse categorical segmentation labels.

    This loss function adapts the Dice loss to work with sparse labels
    (integer class indices) by one-hot encoding them before calculating
    the Dice coefficient.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a softmax activation.
        num_classes (int): The total number of classes in the segmentation task.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the Dice loss will be calculated only for the specified class(es).
            If None, the Dice loss will be calculated for all classes and averaged.
        smooth (float, optional): A small smoothing factor to prevent division by zero.
            Defaults to 1e-7.
        squared_pred (bool, optional): If True, the predictions in the denominator
            of the Dice coefficient will be squared. Defaults to False.
        name (str, optional): Name of the loss function. Defaults to "sparse_dice_loss".
        **kwargs: Additional keyword arguments passed to `BaseDiceLoss`.
    """

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


class CategoricalDiceLoss(BaseDiceLoss):
    """Dice loss for categorical (one-hot encoded) segmentation labels.

    This loss function calculates the Dice loss directly using the provided
    one-hot encoded labels and prediction probabilities.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a softmax activation.
        num_classes (int): The total number of classes in the segmentation task.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the Dice loss will be calculated only for the specified class(es).
            If None, the Dice loss will be calculated for all classes and averaged.
        smooth (float, optional): A small smoothing factor to prevent division by zero.
            Defaults to 1e-7.
        squared_pred (bool, optional): If True, the predictions in the denominator
            of the Dice coefficient will be squared. Defaults to False.
        name (str, optional): Name of the loss function. Defaults to "categorical_dice_loss".
        **kwargs: Additional keyword arguments passed to `BaseDiceLoss`.
    """

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        else:
            return y_pred


class BinaryDiceLoss(BaseDiceLoss):
    """Dice loss for binary segmentation tasks.

    This loss function is specifically designed for binary segmentation where
    the labels typically have a single channel (representing the foreground).
    It applies a sigmoid activation to logits if provided.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a sigmoid activation.
        num_classes (int): For binary tasks, this is typically 1, but can be 2
            if the background is explicitly represented. The `class_id` parameter
            can be used to target the foreground channel if `num_classes` is 2.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the Dice loss will be calculated only for the specified class(es).
            For binary tasks, you might specify `class_id=0` or `class_id=[0]`
            to target the foreground channel if `num_classes=2`. If None, and
            `num_classes=1`, it will calculate for the single channel.
        smooth (float, optional): A small smoothing factor to prevent division by zero.
            Defaults to 1e-7.
        squared_pred (bool, optional): If True, the predictions in the denominator
            of the Dice coefficient will be squared. Defaults to False.
        name (str, optional): Name of the loss function. Defaults to "binary_dice_loss".
        **kwargs: Additional keyword arguments passed to `BaseDiceLoss`.
    """

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.sigmoid(y_pred)
        else:
            return y_pred
