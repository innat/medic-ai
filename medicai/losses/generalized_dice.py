from keras import ops

from medicai.utils import DescribeMixin

from .base import BaseGeneralizedDiceLoss


class SparseGeneralizedDiceLoss(BaseGeneralizedDiceLoss, DescribeMixin):
    """Generalized Dice loss for sparse categorical segmentation labels.

    This loss function adapts the Generalized Dice Loss (GDL) to work with sparse labels
    (integer class indices) by one-hot encoding them before calculating the GDL coefficient,
    and applies a softmax activation if predictions are logits.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a softmax activation.
        num_classes (int): The total number of classes in the segmentation task.
        weight_type (str, optional): The weighting scheme to balance class contributions.
            Options include: 'square' (inverse square of class volume), 'simple' (inverse
            of class volume), or 'uniform' (no weighting).
            Defaults to 'square'.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the loss will be calculated only for the specified class(es).
            If None, the loss will be calculated for all classes and averaged.
        smooth (float, optional): A small smoothing factor to prevent division by zero.
            Defaults to 1e-7.
        reduction (str, optional): Type of reduction to apply to the loss.
            The output of `call()` is the loss value per batch element/class,
            and this parameter controls how it is aggregated.

            * **'sum'**: Sum the loss tensor over all batch elements and classes.
            * **'mean'**: Compute the **mean of the loss tensor over all elements**
                (Batch Size x Number of Classes).
            * **'sum_over_batch_size'**: Compute the **sum of the loss tensor over
                all elements, then divide by the Batch Size**.
            * **'none'**: Return the loss tensor without aggregation, preserving the
                shape `(Batch Size, Num Classes)`.

            Example:
                # After spatial reduction (output of `compute_loss`):
                per_sample_per_class_loss = [
                    [0.2, 0.8, 0.4],  # Sample 1: class0, class1, class2 losses
                    [0.3, 0.7, 0.5]   # Sample 2: class0, class1, class2 losses
                ]

                # reduction="sum": 2.9
                # reduction="mean": 2.9 / 6 = 0.483
                # reduction="sum_over_batch_size": 2.9 / 2 = 1.45
                # reduction=None: returns the original [[0.2, 0.8, 0.4], [0.3, 0.7, 0.5]]

            Defaults to 'mean'.
        name (str, optional): Name of the loss function. Defaults to "sparse_generalized_dice_loss".
        **kwargs: Additional keyword arguments passed to `keras.losses.Loss`.
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        weight_type="square",
        class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "sparse_generalized_dice_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            weight_type=weight_type,
            class_ids=class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name,
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        else:
            return y_pred

    def _process_inputs(self, y_true):
        if y_true.shape[-1] == 1:
            y_true = ops.squeeze(y_true, axis=-1)

        y_true = ops.one_hot(y_true, num_classes=self.num_classes)
        return y_true


class CategoricalGeneralizedDiceLoss(BaseGeneralizedDiceLoss, DescribeMixin):
    """Generalized Dice loss for categorical (one-hot encoded) segmentation labels.

    This loss function calculates the Generalized Dice Loss (GDL) directly using the provided
    one-hot encoded labels and prediction probabilities, applying a softmax activation
    if predictions are logits.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a softmax activation.
        num_classes (int): The total number of classes in the segmentation task.
        weight_type (str, optional): The weighting scheme to balance class contributions.
            Options include: 'square' (inverse square of class volume), 'simple' (inverse
            of class volume), or 'uniform' (no weighting).
            Defaults to 'square'.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the loss will be calculated only for the specified class(es).
            If None, the loss will be calculated for all classes and averaged.
        smooth (float, optional): A small smoothing factor to prevent division by zero.
            Defaults to 1e-7.
        reduction (str, optional): Type of reduction to apply to the loss.
            The output of `call()` is the loss value per batch element/class,
            and this parameter controls how it is aggregated.

            * **'sum'**: Sum the loss tensor over all batch elements and classes.
            * **'mean'**: Compute the **mean of the loss tensor over all elements**
                (Batch Size x Number of Classes).
            * **'sum_over_batch_size'**: Compute the **sum of the loss tensor over
                all elements, then divide by the Batch Size**.
            * **'none'**: Return the loss tensor without aggregation, preserving the
                shape `(Batch Size, Num Classes)`.

            Example:
                # After spatial reduction (output of `compute_loss`):
                per_sample_per_class_loss = [
                    [0.2, 0.8, 0.4],  # Sample 1: class0, class1, class2 losses
                    [0.3, 0.7, 0.5]   # Sample 2: class0, class1, class2 losses
                ]

                # reduction="sum": 2.9
                # reduction="mean": 2.9 / 6 = 0.483
                # reduction="sum_over_batch_size": 2.9 / 2 = 1.45
                # reduction=None: returns the original [[0.2, 0.8, 0.4], [0.3, 0.7, 0.5]]

            Defaults to 'mean'.
        name (str, optional): Name of the loss function. Defaults to "categorical_generalized_dice_loss".
        **kwargs: Additional keyword arguments passed to `keras.losses.Loss`.
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        weight_type="square",
        class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "categorical_generalized_dice_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            weight_type=weight_type,
            class_ids=class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name,
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        else:
            return y_pred


class BinaryGeneralizedDiceLoss(BaseGeneralizedDiceLoss, DescribeMixin):
    """Generalized Dice loss for binary or multi-label segmentation tasks.

    This loss function is specifically designed for binary or multi-label
    segmentation where the labels typically have a single or multi-label channel
    (representing the foreground). It uses a **Sigmoid** activation on logits.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a sigmoid activation.
        num_classes (int): Must be set to **1** for true binary segmentation,
            or **2** if the input/output explicitly contains two channels (e.g.,
            foreground and background).
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the Dice loss will be calculated only for the specified class(es).
            If None and `num_classes=1`, the loss is calculated on the single channel.
        smooth (float, optional): A small smoothing factor to prevent division by zero.
            Defaults to 1e-7.
        weight_type (str, optional): The weighting scheme to balance class contributions.
            Options include: 'square' (inverse square of class volume), 'simple' (inverse
            of class volume), or 'uniform' (no weighting).
            Defaults to 'square'.
        reduction (str, optional): Type of reduction to apply to the loss.
            The output of `call()` is the loss value per batch element/class,
            and this parameter controls how it is aggregated.

            * **'sum'**: Sum the loss tensor over all batch elements and classes.
            * **'mean'**: Compute the **mean of the loss tensor over all elements**
                (Batch Size x Number of Classes).
            * **'sum_over_batch_size'**: Compute the **sum of the loss tensor over
                all elements, then divide by the Batch Size**.
            * **'none'**: Return the loss tensor without aggregation, preserving the
                shape `(Batch Size, Num Classes)`.

            Example:
                # After spatial reduction (output of `compute_loss`):
                per_sample_per_class_loss = [
                    [0.2, 0.8, 0.4],  # Sample 1: class0, class1, class2 losses
                    [0.3, 0.7, 0.5]   # Sample 2: class0, class1, class2 losses
                ]

                # reduction="sum": 2.9
                # reduction="mean": 2.9 / 6 = 0.483
                # reduction="sum_over_batch_size": 2.9 / 2 = 1.45
                # reduction=None: returns the original [[0.2, 0.8, 0.4], [0.3, 0.7, 0.5]]

            Defaults to 'mean'.
        name (str, optional): Name of the loss function.
            Defaults to "binary_dice_loss".
        **kwargs: Additional keyword arguments passed to `keras.losses.Loss`.
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        class_ids=None,
        smooth=1e-7,
        weight_type="square",
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "binary_generalized_dice_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            class_ids=class_ids,
            smooth=smooth,
            weight_type=weight_type,
            reduction=reduction,
            name=name,
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.sigmoid(y_pred)
        else:
            return y_pred
