import keras
from keras import ops
from keras.metrics import Metric


class BaseDiceMetric(Metric):
    """Base class for Dice-based metrics.

    This class provides a foundation for calculating the Dice coefficient, a common
    metric for evaluating the overlap between predicted and ground truth
    segmentation masks. It handles class ID selection, ignoring empty samples,
    smoothing, and thresholding for binary cases.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a sigmoid activation for binary
            tasks or softmax for multi-class tasks in subclasses.
        num_classes (int): The total number of classes in the segmentation task.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the Dice metric will be calculated only for the specified class(es).
            If None, the Dice metric will be calculated for all classes and averaged.
        ignore_empty (bool, optional): If True, samples where the ground truth
            is entirely empty for the selected classes will be ignored during
            metric calculation. Defaults to True.
        smooth (float, optional): A small smoothing factor to prevent division by zero.
            Defaults to 1e-6.
        name (str, optional): Name of the metric. Defaults to "base_dice".
        threshold (float, optional): Threshold value used to convert probabilities
            to binary predictions (for binary or multi-label cases). Defaults to 0.5.
        **kwargs: Additional keyword arguments passed to `keras.metrics.Metric`.
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        class_ids=None,
        ignore_empty=True,
        smooth=1e-6,
        name="base_dice",
        threshold=0.5,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.class_ids = self._validate_and_get_class_ids(class_ids, num_classes)
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.ignore_empty = ignore_empty
        self.threshold = threshold
        self.smooth = smooth or keras.backend.epsilon()

        # State variables
        self.total_intersection = self.add_variable(
            name="total_intersection", shape=(len(self.class_ids),), initializer="zeros"
        )
        self.total_union = self.add_variable(
            name="total_union", shape=(len(self.class_ids),), initializer="zeros"
        )
        self.valid_counts = self.add_variable(
            name="valid_counts", shape=(len(self.class_ids),), initializer="zeros"
        )

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
                "class_id must be an integer, a list of integers, or None to consider all classes."
            )

    def _process_predictions(self, y_pred):
        return y_pred

    def _process_inputs(self, y_true):
        return y_true

    def _get_desired_class_channels(self, y_true, y_pred):
        if self.class_ids is None:
            return y_true, y_pred

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

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates the metric's state based on new ground truth and predictions.

        Args:
            y_true (Tensor): Ground truth tensor.
            y_pred (Tensor): Prediction tensor.
            sample_weight (Tensor, optional): Optional weighting of the samples.
                Not currently used in this base implementation.
        """
        y_pred = ops.cast(y_pred, y_true.dtype)

        y_pred_processed = self._process_predictions(y_pred)
        y_true_processed = self._process_inputs(y_true)

        # Select only the classes we want to evaluate
        y_true_processed, y_pred_processed = self._get_desired_class_channels(
            y_true_processed, y_pred_processed
        )

        # Calculate metrics
        intersection = y_true_processed * y_pred_processed  # [B, D, H, W, C]
        union = y_true_processed + y_pred_processed  # [B, D, H, W, C]
        gt_sum = ops.sum(y_true_processed, axis=[1, 2, 3])  # [B, C]
        pred_sum = ops.sum(y_pred_processed, axis=[1, 2, 3])  # [B, C]

        # Valid samples mask
        if self.ignore_empty:
            # Invalid when GT is empty AND prediction is NOT empty
            invalid_mask = ops.logical_and(
                gt_sum == 0, pred_sum != 0  # Empty GT  # Non-empty prediction
            )
            valid_mask = ops.logical_not(invalid_mask)  # [B, C]
        else:
            valid_mask = ops.ones_like(gt_sum, dtype="bool")

        # Convert mask to float and expand dimensions for broadcasting
        valid_mask_float = ops.cast(valid_mask, "float32")  # [B, C]

        # Apply mask to metrics
        masked_intersection = ops.sum(intersection, axis=[1, 2, 3]) * valid_mask_float  # [B, C]
        masked_union = ops.sum(union, axis=[1, 2, 3]) * valid_mask_float  # [B, C]

        # Update state variables
        self.total_intersection.assign_add(ops.sum(masked_intersection, axis=0))  # [C]
        self.total_union.assign_add(ops.sum(masked_union, axis=0))  # [C]
        self.valid_counts.assign_add(ops.sum(valid_mask_float, axis=0))  # [C]

    def result(self):
        """Computes the Dice coefficient.

        Returns:
            Tensor: The mean Dice coefficient across the selected classes,
                handling cases where no valid samples exist for a class.
        """
        # Calculate Dice per class
        dice_per_class = (2.0 * self.total_intersection + self.smooth) / (
            self.total_union + self.smooth
        )

        # Only average over classes with valid counts > 0
        valid_classes = self.valid_counts > 0
        dice_per_class = ops.where(valid_classes, dice_per_class, ops.zeros_like(dice_per_class))
        num_valid_classes = ops.sum(ops.cast(valid_classes, "float32"))
        return ops.cond(
            num_valid_classes > 0,
            lambda: ops.sum(dice_per_class) / num_valid_classes,
            lambda: ops.cast(0.0, self.dtype),  # Return 0 if no valid classes
        )

    def reset_states(self):
        self.total_intersection.assign(ops.zeros(len(self.class_ids)))
        self.total_union.assign(ops.zeros(len(self.class_ids)))
        self.valid_counts.assign(ops.zeros(len(self.class_ids)))
