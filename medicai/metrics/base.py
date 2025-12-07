import keras
from keras import ops
from keras.metrics import Metric

from medicai.utils import camel_to_snake


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
        target_class_ids=None,
        ignore_class_ids=None,
        ignore_empty=True,
        smooth=1e-6,
        name=None,
        threshold=0.5,
        **kwargs,
    ):
        if name is None:
            name = (
                "dice_metric"
                if self.__class__ is BaseDiceMetric
                else camel_to_snake(self.__class__.__name__)
            )

        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.from_logits = from_logits
        self.ignore_empty = ignore_empty
        self.threshold = threshold
        self.smooth = smooth or keras.backend.epsilon()
        self.target_class_ids = self._validate_class_ids(target_class_ids, allow_none=False)
        self.ignore_class_ids = self._validate_class_ids(ignore_class_ids, allow_none=True)

        # State variables
        self.total_intersection = self.add_variable(
            name="total_intersection", shape=(len(self.target_class_ids),), initializer="zeros"
        )
        self.total_union = self.add_variable(
            name="total_union", shape=(len(self.target_class_ids),), initializer="zeros"
        )
        self.valid_counts = self.add_variable(
            name="valid_counts", shape=(len(self.target_class_ids),), initializer="zeros"
        )

    def _validate_class_ids(self, class_ids, allow_none=False):
        if class_ids is None:
            if allow_none:
                # Used for ignore_class_ids: None means don't ignore any.
                return None
            else:
                # Used for target_class_ids: None means target all classes.
                return list(range(self.num_classes))

        elif isinstance(class_ids, int):
            if not allow_none and not 0 <= class_ids < self.num_classes:
                raise ValueError(
                    f"Class ID {class_ids} is out of the valid range [0, {self.num_classes - 1}]."
                )
            return [class_ids]

        # Handle list case
        elif isinstance(class_ids, list):
            for cid in class_ids:
                if not allow_none and not 0 <= cid < self.num_classes:
                    raise ValueError(
                        f"Class ID {cid} is out of the valid range [0, {self.num_classes - 1}]."
                    )
            return class_ids

        # When invalid type
        else:
            valid_types = "an integer, or a list of integers"
            if not allow_none:
                valid_types += ", or None to consider all classes"

            raise ValueError(f"class_ids must be {valid_types}.")

    def _process_predictions(self, y_pred):
        return y_pred

    def _process_targets(self, y_true):
        return y_true

    def _process_inputs(self, y_true, y_pred):
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        y_pred = ops.cast(y_pred, y_true.dtype)

        if self.ignore_class_ids:
            ignore_classes = ops.convert_to_tensor(self.ignore_class_ids, dtype="int32")
            is_ignored_mask = ops.any(
                ops.equal(y_true, ignore_classes),
                axis=-1,
            )
            valid_mask = ops.cast(~is_ignored_mask, y_pred.dtype)
        else:
            valid_mask = ops.ones_like(y_pred[..., 0], dtype=y_pred.dtype)

        y_pred_processed = self._process_predictions(y_pred)
        y_true_processed = self._process_targets(y_true)

        # Select only the classes we want to evaluate
        y_true_processed, y_pred_processed = self._get_desired_class_channels(
            y_true_processed, y_pred_processed
        )

        # reshape valid mask
        valid_mask = ops.expand_dims(valid_mask, axis=-1)
        valid_mask = ops.broadcast_to(valid_mask, ops.shape(y_pred_processed))
        return y_true_processed, y_pred_processed, valid_mask

    def _get_desired_class_channels(self, y_true, y_pred):
        if self.target_class_ids is None:
            return y_true, y_pred

        if self.num_classes == 1:
            return y_true, y_pred

        selected_y_true = []
        selected_y_pred = []

        for class_index in self.target_class_ids:
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
        y_true_processed, y_pred_processed, valid_mask = self._process_inputs(y_true, y_pred)

        # Dynamically determine the spatial dimensions to sum over.
        spatial_dims = list(range(1, len(y_pred_processed.shape) - 1))

        # Calculate metrics
        intersection = valid_mask * y_true_processed * y_pred_processed
        union = valid_mask * y_true_processed + valid_mask * y_pred_processed
        gt_sum = ops.sum(y_true_processed * valid_mask, axis=spatial_dims)  # [B, C]
        pred_sum = ops.sum(y_pred_processed * valid_mask, axis=spatial_dims)  # [B, C]

        # Check if ALL pixels are ignored (all zeros in y_true_processed and y_pred_processed)
        # And if all pixels are ignored in a sample, we should ignore that sample entirely
        total_pixels_per_sample = ops.prod(ops.shape(y_true_processed)[1:-1])
        total_pixels_per_sample = ops.cast(total_pixels_per_sample, y_true.dtype)
        all_ignored_mask = ops.all(
            ops.all(y_true_processed == 0, axis=spatial_dims), axis=-1
        )  # [B]
        all_ignored_mask = ops.expand_dims(all_ignored_mask, -1)  # [B, 1]
        all_ignored_mask = ops.tile(all_ignored_mask, [1, gt_sum.shape[-1]])  # [B, C]

        if self.ignore_empty:
            # Invalid when GT is empty AND prediction is NOT empty
            invalid_mask = ops.logical_or(
                ops.logical_and(gt_sum == 0, pred_sum != 0),  # Empty GT but non-empty prediction
                all_ignored_mask,  # All pixels ignored
            )
            valid_mask = ops.logical_not(invalid_mask)  # [B, C]
        else:
            valid_mask = ops.logical_not(all_ignored_mask)  # [B, C]

        # Convert mask to float and expand dimensions for broadcasting
        valid_mask_float = ops.cast(valid_mask, "float32")  # [B, C]

        # Apply mask to metrics
        masked_intersection = ops.sum(intersection, axis=spatial_dims) * valid_mask_float  # [B, C]
        masked_union = ops.sum(union, axis=spatial_dims) * valid_mask_float  # [B, C]

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
        self.total_intersection.assign(ops.zeros(len(self.target_class_ids)))
        self.total_union.assign(ops.zeros(len(self.target_class_ids)))
        self.valid_counts.assign(ops.zeros(len(self.target_class_ids)))
