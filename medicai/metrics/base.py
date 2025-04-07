from keras import ops
from keras.metrics import Metric


class BaseDiceMetric(Metric):
    def __init__(
        self,
        from_logits,
        num_classes,
        class_id=None,
        ignore_empty=True,
        smooth=1e-6,
        name="base_dice",
        threshold=0.5,
        **kwargs
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
        self.ignore_empty = ignore_empty
        self.threshold = threshold
        self.smooth = smooth

        # State variables
        self.total_intersection = self.add_variable(
            name="total_intersection", shape=(len(self.class_id),), initializer="zeros"
        )
        self.total_union = self.add_variable(
            name="total_union", shape=(len(self.class_id),), initializer="zeros"
        )
        self.valid_counts = self.add_variable(
            name="valid_counts", shape=(len(self.class_id),), initializer="zeros"
        )

    def _process_predictions(self, y_pred):
        return y_pred

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = ops.cast(y_pred, y_true.dtype)

        y_pred_processed = self._process_predictions(y_pred)
        y_true_processed = y_true

        # Select only the classes we want to evaluate
        y_true_processed = ops.take(y_true_processed, self.class_id, axis=-1)
        y_pred_processed = ops.take(y_pred_processed, self.class_id, axis=-1)

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
        self.total_intersection.assign(ops.zeros(len(self.class_id)))
        self.total_union.assign(ops.zeros(len(self.class_id)))
        self.valid_counts.assign(ops.zeros(len(self.class_id)))
