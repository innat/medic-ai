import keras.ops as ops
from keras.metrics import Metric


class BaseDiceMetric(Metric):
    def __init__(self, num_classes, class_id=None, smooth=1e-6, name="dice_metric", **kwargs):
        super().__init__(name=name, **kwargs)

        # Handle class_id specification
        if class_id is None:
            self.class_id = list(range(num_classes))
        elif isinstance(class_id, int):
            self.class_id = [class_id]
        else:
            self.class_id = class_id

        self.num_classes = num_classes
        self.smooth = smooth

        # Track intersection and union
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def compute_dice_components(self, y_true, y_pred):
        """To be implemented by subclasses"""
        raise NotImplementedError("Must be implemented by subclasses")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = ops.cast(y_pred, y_true.dtype)
        y_true_processed, y_pred_processed = self.compute_dice_components(y_true, y_pred)

        # Select only the classes we want to evaluate
        y_true_selected = ops.take(y_true_processed, self.class_id, axis=-1)
        y_pred_selected = ops.take(y_pred_processed, self.class_id, axis=-1)

        # Flatten the tensors
        y_true_flat = ops.reshape(y_true_selected, [-1])
        y_pred_flat = ops.reshape(y_pred_selected, [-1])

        # Compute intersection and union
        intersection = ops.sum(y_true_flat * y_pred_flat)
        union = ops.sum(y_true_flat) + ops.sum(y_pred_flat)

        # Update state
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        dice = (2.0 * self.intersection + self.smooth) / (self.union + self.smooth)
        return dice

    def reset_states(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)
