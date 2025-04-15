import keras
from keras import ops


class BaseDiceLoss(keras.losses.Loss):
    """Base class for Dice-based loss functions.

    This class provides a foundation for calculating Dice loss, a common metric
    for evaluating the overlap between predicted and ground truth segmentation masks.
    It handles class ID selection, smoothing, and prediction processing.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a sigmoid activation.
        num_classes (int): The total number of classes in the segmentation task.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the Dice loss will be calculated only for the specified class(es).
            If None, the Dice loss will be calculated for all classes and averaged.
        smooth (float, optional): A small smoothing factor to prevent division by zero.
            Defaults to 1e-7.
        squared_pred (bool, optional): If True, the predictions in the denominator
            of the Dice coefficient will be squared. This is used in some variations
            like the F-beta loss. Defaults to False.
        dice_weight (float): The trade-off weight for the Dice loss component.
            Must be >= 0.0. A higher value gives more importance to Dice loss.
            Defaults to 1.0.
        ce_weight (float): The trade-off weight for the Cross-Entropy loss component.
            Must be >= 0.0. A higher value gives more importance to Cross-Entropy loss.
            Defaults to 1.0.
        name (str, optional): Name of the loss function. Defaults to "dice_loss".
        **kwargs: Additional keyword arguments passed to `keras.losses.Loss`.
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        class_ids=None,
        smooth=1e-7,
        squared_pred=False,
        dice_weight=1.0,
        ce_weight=1.0,
        name="dice_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if dice_weight < 0.0:
            raise ValueError("dice_weight should be not less than 0.0.")
        if ce_weight < 0.0:
            raise ValueError("ce_weight should be not less than 0.0.")

        self.class_ids = self._validate_and_get_class_ids(class_ids, num_classes)
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.squared_pred = squared_pred
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth or keras.backend.epsilon()

    def _validate_and_get_class_ids(self, class_ids, num_classes):
        if class_ids is None:
            return list(range(num_classes))
        elif isinstance(class_ids, int):
            return [class_ids]
        elif isinstance(class_ids, list):
            for cid in class_ids:
                if not 0 <= cid < num_classes:
                    raise ValueError(
                        f"Class ID {cid} is out of the valid range [0, {num_classes - 1}]."
                    )
            return class_ids
        else:
            raise ValueError(
                "class_ids must be an integer, a list of integers, or None to consider all classes."
            )

    def _get_desired_class_channels(self, y_true, y_pred):
        """Selects the channels corresponding to the desired class IDs.

        Args:
            y_true (Tensor): Ground truth tensor.
            y_pred (Tensor): Prediction tensor.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the ground truth and
                prediction tensors with only the desired class channels.
        """

        if self.num_classes == 1:
            return y_true, y_pred

        selected_y_true = []
        selected_y_pred = []

        for class_index in self.class_ids:
            selected_y_true.append(y_true[..., class_index : class_index + 1])
            selected_y_pred.append(y_pred[..., class_index : class_index + 1])

        y_true = ops.concatenate(selected_y_true, axis=-1)
        y_pred = ops.concatenate(selected_y_pred, axis=-1)

        return y_true, y_pred

    def _process_predictions(self, y_pred):
        return y_pred

    def _process_inputs(self, y_true):
        return y_true

    def dice_loss(self, y_true, y_pred):
        """Calculates the Dice loss.

        Args:
            y_true (Tensor): Ground truth tensor.
            y_pred (Tensor): Processed prediction tensor (after sigmoid if from logits).

        Returns:
            Tensor: The Dice loss.
        """
        y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)
        intersection = ops.sum(y_true * y_pred, axis=[1, 2, 3])
        union = ops.sum(y_true, axis=[1, 2, 3]) + ops.sum(
            ops.square(y_pred) if self.squared_pred else y_pred, axis=[1, 2, 3]
        )
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - ops.mean(dice_score)

    def call(self, y_true, y_pred):
        """Computes the Dice loss.

        Args:
            y_true (Tensor): Ground truth tensor.
            y_pred (Tensor): Prediction tensor.

        Returns:
            Tensor: The computed Dice loss.
        """
        y_pred_processed = self._process_predictions(y_pred)
        y_true_processed = self._process_inputs(y_true)

        y_pred_processed = ops.clip(y_pred_processed, self.smooth, 1.0 - self.smooth)
        dice = self.dice_loss(y_true_processed, y_pred_processed)
        return dice
