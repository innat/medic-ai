from keras.callbacks import Callback

from medicai.metrics import DiceMetric
from medicai.utils.inference import SlidingWindowInference


class SlidingWindowInferenceCallback(Callback):
    def __init__(self, dataset, num_classes, interval=5, save_path="best_model.weights.h5"):
        """
        Custom Keras callback to perform inference on a dataset periodically and save best model.

        Args:
            dataset (tf.data.Dataset or tuple): Dataset to perform inference on. If tuple, should be (X, y).
            interval (int): Number of epochs between each inference run.
            save_path (str): File path to save the best model weights.
        """
        super().__init__()
        self.dataset = dataset
        self.interval = interval
        self.save_path = save_path
        self.best_dice_score = -float("inf")  # Initialize best score

        self.dice_metric = DiceMetric(
            num_classes=4,
            include_background=True,
            reduction="mean",
            ignore_empty=True,
            smooth=1e-6,
            name="dice_score",
        )
        self.roi_size = (96, 96, 96)
        self.sw_batch_size = 4
        self.num_classes = 4

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            print(f"\nEpoch {epoch+1}: Running inference...")

            self.dice_metric.reset_state()  # Reset metric before evaluation

            for x, y in self.dataset:  # (bs, d, h, w, channel)
                y_pred = output = sliding_window_inference(
                    x, self.roi_size, self.sw_batch_size, self.model, overlap=0.8
                )
                self.dice_metric.update_state(y, y_pred)

            dice_score = self.dice_metric.result().numpy()
            print(f"Epoch {epoch+1}: Dice Score = {dice_score:.4f}")

            # Save model if Dice score improves
            if dice_score > self.best_dice_score:
                self.best_dice_score = dice_score
                self.model.save_weights(self.save_path)
                print(f"New best Dice score! Model saved to {self.save_path}")

    def sliding_window_inference(self, x):
        """Apply sliding window inference (modify as needed)"""
        return self.model.predict(x, batch_size=1)  # Change batch size as needed
