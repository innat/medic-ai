from keras import ops

from medicai.utils import DescribeMixin

from .base import BaseDiceMetric


class BinaryDiceMetric(BaseDiceMetric, DescribeMixin):
    """Dice metric for binary segmentation tasks.

    Calculates the Dice coefficient for binary segmentation by thresholding
    the predictions.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a sigmoid activation before
            thresholding.
        num_classes (int): For binary tasks, typically 1.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the Dice metric will be calculated only for the specified class(es).
            Defaults to None (calculates for all classes, usually just one).
        ignore_empty (bool, optional): If True, samples where the ground truth
            is entirely empty for the selected classes will be ignored.
            Defaults to True.
        smooth (float, optional): A small smoothing factor to prevent division by zero.
            Defaults to 1e-6.
        name (str, optional): Name of the metric. Defaults to "binary_dice".
        threshold (float, optional): Threshold value (between 0 and 1) used to
            convert probabilities to binary predictions. Defaults to 0.5.
        **kwargs: Additional keyword arguments passed to `BaseDiceMetric`.
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        ignore_empty=True,
        smooth=1e-6,
        threshold=0.5,
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
            ignore_empty=ignore_empty,
            smooth=smooth,
            threshold=threshold,
            name=name or "binary_dice_score",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.cast(ops.sigmoid(y_pred) > self.threshold, dtype="float32")
        else:
            return y_pred


class CategoricalDiceMetric(BaseDiceMetric):
    """Dice metric for categorical segmentation tasks.

    Calculates the Dice coefficient for multi-class segmentation where the
    ground truth is one-hot encoded. The predictions are converted to
    one-hot format by taking the argmax along the class axis.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a softmax activation before
            taking the argmax.
        num_classes (int): The total number of classes in the segmentation task.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the Dice metric will be calculated only for the specified class(es).
            Defaults to None (calculates for all classes).
        ignore_empty (bool, optional): If True, samples where the ground truth
            is entirely empty for the selected classes will be ignored.
            Defaults to True.
        smooth (float, optional): A small smoothing factor to prevent division by zero.
            Defaults to 1e-6.
        name (str, optional): Name of the metric. Defaults to "categorical_dice".
        **kwargs: Additional keyword arguments passed to `BaseDiceMetric`.
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_empty=True,
        smooth=1e-6,
        name=None,
        **kwargs,
    ):
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=None,
            ignore_empty=ignore_empty,
            smooth=smooth,
            name=name or "categorical_dice_score",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.one_hot(ops.argmax(y_pred, axis=-1), num_classes=self.num_classes)
        else:
            return y_pred


class SparseDiceMetric(BaseDiceMetric):
    """Dice metric for sparse categorical segmentation tasks.

    Calculates the Dice coefficient for multi-class segmentation where the
    ground truth labels are sparse (integer class indices). The predictions
    are converted to one-hot format by taking the argmax along the class axis,
    and the ground truth is also converted to one-hot format.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a softmax activation before
            taking the argmax.
        num_classes (int): The total number of classes in the segmentation task.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the Dice metric will be calculated only for the specified class(es).
            Defaults to None (calculates for all classes).
        ignore_empty (bool, optional): If True, samples where the ground truth
            is entirely empty for the selected classes will be ignored.
            Defaults to True.
        smooth (float, optional): A small smoothing factor to prevent division by zero.
            Defaults to 1e-6.
        name (str, optional): Name of the metric. Defaults to "sparse_dice".
        **kwargs: Additional keyword arguments passed to `BaseDiceMetric`.
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        ignore_empty=True,
        smooth=1e-6,
        name=None,
        **kwargs,
    ):
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
            ignore_empty=ignore_empty,
            smooth=smooth,
            name=name or "sparse_categorical_dice_score",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.one_hot(ops.argmax(y_pred, axis=-1), num_classes=self.num_classes)
        else:
            return y_pred

    def _process_targets(self, y_true):
        if y_true.shape[-1] == 1:
            y_true = ops.squeeze(y_true, axis=-1)
        y_true = ops.one_hot(ops.cast(y_true, "int32"), num_classes=self.num_classes)
        return y_true
