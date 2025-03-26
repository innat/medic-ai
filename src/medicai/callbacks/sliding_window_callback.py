import os

from medicai.utils import hide_warnings

hide_warnings()

import keras

if keras.backend.backend() == "tensorflow":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras import callbacks

from medicai.metrics import DiceMetric
from medicai.utils.inference import SlidingWindowInference


class SlidingWindowInferenceCallback(callbacks.Callback):
    def __init__(
        self,
        model,
        dataset,
        num_classes,
        overlap=0.5,
        roi_size=(96, 96, 96),
        sw_batch_size=4,
        interval=5,
        mode="constant",
        padding_mode="constant",
        sigma_scale=0.125,
        cval=0.0,
        roi_weight_map=0.8,
        save_path="model.weights.h5",
    ):
        """
        Custom Keras callback to perform inference on a dataset periodically and save best model.

        Args:
            dataset (tf.data.Dataset or tuple): Dataset to perform inference on. If tuple, should be (X, y).
            interval (int): Number of epochs between each inference run.
            save_path (str): File path to save the best model weights.
        """
        super().__init__()
        self._model = model
        self.dataset = dataset
        self.num_classes = num_classes
        self.overlap = overlap
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.interval = interval
        self.save_path = save_path
        self.mode = mode
        self.padding_mode = padding_mode
        self.sigma_scale = sigma_scale
        self.cval = cval
        self.roi_weight_map = roi_weight_map
        self.best_score = -float("inf")  # Initialize best score

        self.swi = SlidingWindowInference(
            self._model,
            num_classes=self.num_classes,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            overlap=self.overlap,
            mode=self.mode,
            sigma_scale=self.sigma_scale,
            padding_mode=self.padding_mode,
            cval=self.cval,
            roi_weight_map=self.roi_weight_map,
        )

        self.metric = DiceMetric(
            num_classes=4,
            include_background=True,
            reduction="mean",
            ignore_empty=True,
            smooth=1e-6,
            name="dice_score",
        )

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            print(f"\nEpoch {epoch+1}: Running inference...")

            self.metric.reset_state()  # Reset metric before evaluation

            for x, y in self.dataset:  # (bs, d, h, w, channel)
                y_pred = self.swi(x)
                self.metric.update_state(y, y_pred)

            score = self.metric.result().numpy()
            print(f"Epoch {epoch+1}: Score = {score:.4f}")

            # Save model if Dice score improves
            if score > self.best_score:
                self.best_score = score
                self.model.save_weights(self.save_path)
                print(f"New best score! Model saved to {self.save_path}")
