from keras.metrics import Metric
import keras.ops as ops

class BaseDiceMetric(Metric):
    def __init__(
        self,
        from_logits,
        num_classes,
        class_id=None,
        smooth=1e-6,
        name="dice_metric",
        **kwargs
    ):
        if class_id is None:
            class_id = list(range(num_classes))
        elif isinstance(class_id, int):
            class_id = [class_id]
        else:
            class_id = class_id
    
        super().__init__(name=name, **kwargs)

        self.class_id = class_id
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.smooth = smooth

        self.intersection = self.add_weight(name=f"intersection_{name}", initializer="zeros")
        self.union = self.add_weight(name=f"union_{name}", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        raise NotImplementedError("Must be implemented in subclasses.")

    def result(self):
        dice = (2.0 * self.intersection + self.smooth) / (self.union + self.smooth)
        return ops.mean(dice)

    def reset_states(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

    def get_target_class(self, *inputs, indices=None):
        if indices is None:
            return inputs
        elif isinstance(indices, (int)):
            indices = [indices]

        outputs = []
        for sample in inputs:
            x = ops.take(sample, indices, axis=-1)
            outputs.append(x)
    
        return outputs