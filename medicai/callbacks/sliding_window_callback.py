from medicai.utils import hide_warnings

hide_warnings()


from keras import callbacks

from medicai.utils.inference import SlidingWindowInference


class SlidingWindowInferenceCallback(callbacks.Callback):
    def __init__(
        self,
        model,
        dataset,
        metrics,
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
        """Initializes the SlidingWindowInferenceCallback.

        Args:
            model (keras.Model): The Keras model to perform inference with.
            dataset (tf.data.Dataset or tuple): Dataset to perform inference on.
                If a tuple, it should be (X, y). Each element of the dataset
                should yield a tuple of (input_tensor, ground_truth_tensor).
            metrics (keras.metrics.Metric or list of keras.metrics.Metric):
                Keras metric(s) to evaluate the inference results.
            num_classes (int): The number of classes in the segmentation task.
            overlap (float): Amount of overlap between adjacent ROIs in each
                dimension. A value between 0.0 and 1.0. Default is 0.5.
            roi_size (tuple of int): The size of the Region Of Interest (ROI)
                to process with the sliding window. Should have the same
                dimensionality as the input of the model (e.g., (96, 96, 96) for
                3D). Default is (96, 96, 96).
            sw_batch_size (int): Batch size for processing individual ROIs.
                Default is 4.
            interval (int): Number of epochs between each inference run.
                Inference will be performed at the end of every `interval` epochs.
                Default is 5.
            mode (str): How to combine overlapping predictions. Options are:
                "gaussian", "constant", "max". Default is "constant".
            padding_mode (str): How to pad the input if its dimensions are not
                divisible by the `roi_size` with the given `overlap`. Options
                are: "constant", "reflect", "replicate", "circular".
                Default is "constant".
            sigma_scale (float): Standard deviation for the Gaussian weighting
                window, as a fraction of the `roi_size`. Only used if `mode` is
                "gaussian". Default is 0.125.
            cval (float): Constant value used for padding if `padding_mode` is
                "constant". Default is 0.0.
            roi_weight_map (tensor or float): Optional ROI weight map. If a
                tensor, it should have the same spatial dimensions as the input
                and will be used to weight the contribution of each ROI during
                aggregation. If a float, a constant weight map will be used.
                Default is 0.8.
            save_path (str): File path to save the best model weights.
                The model weights will be saved if the evaluated metric on the
                inference dataset improves. Default is "model.weights.h5".
        """
        super().__init__()
        self._model = model
        self._metrics = metrics
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

    def on_epoch_end(self, epoch, logs=None):
        """Performs inference at the end of each epoch if the epoch number
        is a multiple of the specified interval. Evaluates the model on the
        provided dataset using sliding window inference and saves the model
        weights if the evaluated metric improves.

        Args:
            epoch (int): The current epoch number (0-indexed).
            logs (dict, optional): Dictionary of logs passed by the Keras
                training loop. Defaults to None.
        """
        if (epoch + 1) % self.interval == 0:
            print(f"\nEpoch {epoch+1}: Running inference...")

            self._metrics.reset_state()  # Reset metric before evaluation

            for x, y in self.dataset:  # (bs, d, h, w, channel)
                y_pred = self.swi(x)
                self._metrics.update_state(y, y_pred)

            score = self._metrics.result().numpy()
            print(f"Epoch {epoch+1}: Score = {score:.4f}")

            # Save model if Dice score improves
            if score > self.best_score:
                self.best_score = score
                self.model.save_weights(self.save_path)
                print(f"New best score! Model saved to {self.save_path}")
