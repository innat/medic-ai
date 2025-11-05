import keras
from keras import ops

from medicai.utils import DescribeMixin

from .dice import BinaryDiceLoss, CategoricalDiceLoss, SparseDiceLoss


class SparseDiceCELoss(SparseDiceLoss, DescribeMixin):
    """Combined Sparse Dice and Categorical Cross-Entropy Loss.

    This loss function combines the Sparse Dice loss with the Categorical
    Cross-Entropy loss. It is often used in multi-class segmentation tasks
    to leverage the strengths of both loss functions, encouraging both region
    overlap (Dice) and per-pixel classification accuracy (Cross-Entropy).

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a softmax activation for Dice
            loss and used directly for Cross-Entropy.
        num_classes (int): The total number of classes in the segmentation task.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            both the Dice and Cross-Entropy losses will be calculated only for the
            specified class(es). If None, both losses will be calculated for all
            classes and averaged.
        smooth (float, optional): A small smoothing factor for the Dice loss to
            prevent division by zero. Defaults to 1e-7.
        dice_weight (float): The trade-off weight for the Dice loss component.
            Must be >= 0.0. A higher value gives more importance to Dice loss.
            Defaults to 1.0.
        ce_weight (float): The trade-off weight for the Cross-Entropy loss component.
            Must be >= 0.0. A higher value gives more importance to Cross-Entropy loss.
            Defaults to 1.0.
        reduction (str, optional): Type of reduction to apply to the loss.
            The output of `call()` is the loss value per batch element/class,
            and this parameter controls how it is aggregated.

            * **'sum'**: Sum the loss tensor over all batch elements and classes.
            * **'mean'**: Compute the **mean of the loss tensor over all elements** (Batch Size x Number of Classes).
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
        name (str, optional): Name of the loss function. Defaults to "sparse_dice_crossentropy".
        **kwargs: Additional keyword arguments passed to `SparseDiceLoss`.
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        class_ids=None,
        smooth=1e-7,
        dice_weight=1.0,
        ce_weight=1.0,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "sparse_dice_crossentropy"
        if dice_weight < 0.0:
            raise ValueError("dice_weight should be not less than 0.0.")
        if ce_weight < 0.0:
            raise ValueError("ce_weight should be not less than 0.0.")

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            class_ids=class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name,
            **kwargs,
        )

    def call(self, y_true, y_pred):
        """Computes the combined Sparse Dice and Categorical Cross-Entropy loss.

        Args:
            y_true (Tensor): Sparse ground truth tensor (integer class indices).
            y_pred (Tensor): Prediction tensor (logits or probabilities).

        Returns:
            Tensor: The combined loss.
        """
        dice_loss = super().call(y_true, y_pred)

        if self.class_ids is not None:
            y_true = super()._process_inputs(y_true)
            y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)

        ce_loss = keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        return (self.dice_weight * dice_loss) + (self.ce_weight * ops.mean(ce_loss))


class CategoricalDiceCELoss(CategoricalDiceLoss, DescribeMixin):
    """Combined Categorical Dice and Categorical Cross-Entropy Loss.

    This loss function combines the Categorical Dice loss with the standard
    Categorical Cross-Entropy loss. It is suitable for multi-class segmentation
    tasks where the ground truth labels are one-hot encoded.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a softmax activation for both
            Dice and Cross-Entropy losses.
        num_classes (int): The total number of classes in the segmentation task.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            both the Dice and Cross-Entropy losses will be calculated only for the
            specified class(es). If None, both losses will be calculated for all
            classes and averaged.
        smooth (float, optional): A small smoothing factor for the Dice loss to
            prevent division by zero. Defaults to 1e-7.
        dice_weight (float): The trade-off weight for the Dice loss component.
            Must be >= 0.0. A higher value gives more importance to Dice loss.
            Defaults to 1.0.
        ce_weight (float): The trade-off weight for the Cross-Entropy loss component.
            Must be >= 0.0. A higher value gives more importance to Cross-Entropy loss.
            Defaults to 1.0.
        reduction (str, optional): Type of reduction to apply to the loss.
            The output of `call()` is the loss value per batch element/class,
            and this parameter controls how it is aggregated.

            * **'sum'**: Sum the loss tensor over all batch elements and classes.
            * **'mean'**: Compute the **mean of the loss tensor over all elements** (Batch Size x Number of Classes).
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
        name (str, optional): Name of the loss function. Defaults to "categorical_dice_crossentropy".
        **kwargs: Additional keyword arguments passed to `CategoricalDiceLoss`.
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        class_ids=None,
        smooth=1e-7,
        dice_weight=1.0,
        ce_weight=1.0,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "categorical_dice_crossentropy"
        if dice_weight < 0.0:
            raise ValueError("dice_weight should be not less than 0.0.")
        if ce_weight < 0.0:
            raise ValueError("ce_weight should be not less than 0.0.")

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            class_ids=class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name,
            **kwargs,
        )

    def call(self, y_true, y_pred):
        """Computes the combined Categorical Dice and Categorical Cross-Entropy loss.

        Args:
            y_true (Tensor): One-hot encoded ground truth tensor.
            y_pred (Tensor): Prediction tensor (logits or probabilities).

        Returns:
            Tensor: The combined loss.
        """
        dice_loss = super().call(y_true, y_pred)

        if self.class_ids is not None:
            y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)

        ce_loss = keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        return (self.dice_weight * dice_loss) + (self.ce_weight * ops.mean(ce_loss))


class BinaryDiceCELoss(BinaryDiceLoss, DescribeMixin):
    """Combined Binary Dice and Binary/Categorical Cross-Entropy Loss.

    This loss function combines the Binary Dice loss with either Binary or
    Categorical Cross-Entropy, depending on the number of channels in the
    prediction tensor (`y_pred`):

    - If `y_pred` has one channel (shape `(..., 1)`), it's treated as a
      single-class binary segmentation problem (e.g., foreground vs. background).
      Binary Cross-Entropy is used in this case.

    - If `y_pred` has more than one channel (shape `(..., C)` where C > 1),
      it's treated as a multi-label binary segmentation problem, where each
      channel represents the probability of a specific binary class being present.
      Categorical Cross-Entropy is used in this case, with the ground truth
      (`y_true`) expected to be one-hot encoded or have a compatible shape.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a sigmoid activation for Dice
            loss and used directly for Cross-Entropy.
        num_classes (int): For single-class binary tasks, this is typically 1.
            For multi-label binary tasks, this should match the number of channels
            in `y_pred`.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the Dice and Cross-Entropy losses will be calculated for the specified
            channel(s). For single-class binary tasks, you might use `class_id=0`
            or `[0]`. For multi-label, you can target specific labels. If None,
            all channels are considered.
        smooth (float, optional): A small smoothing factor for the Dice loss to
            prevent division by zero. Defaults to 1e-7.
        dice_weight (float): The trade-off weight for the Dice loss component.
            Must be >= 0.0. A higher value gives more importance to Dice loss.
            Defaults to 1.0.
        ce_weight (float): The trade-off weight for the Cross-Entropy loss component.
            Must be >= 0.0. A higher value gives more importance to Cross-Entropy loss.
            Defaults to 1.0.
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
        name (str, optional): Name of the loss function. Defaults to "binary_dice_crossentropy".
        **kwargs: Additional keyword arguments passed to `BinaryDiceLoss`.
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        class_ids=None,
        smooth=1e-7,
        dice_weight=1.0,
        ce_weight=1.0,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        name = name or "binary_dice_crossentropy"
        if dice_weight < 0.0:
            raise ValueError("dice_weight should be not less than 0.0.")
        if ce_weight < 0.0:
            raise ValueError("ce_weight should be not less than 0.0.")

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            class_ids=class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name,
            **kwargs,
        )

    def call(self, y_true, y_pred):
        """Computes the combined Binary Dice and Binary/Categorical Cross-Entropy loss.

        Args:
            y_true (Tensor): Ground truth tensor.
                - For single-class binary: Typically has one channel (values 0 or 1).
                - For multi-label binary: Can be one-hot encoded or have a shape
                  compatible with Categorical Cross-Entropy.
            y_pred (Tensor): Prediction tensor.
                - For single-class binary: Shape `(..., 1)`.
                - For multi-label binary: Shape `(..., C)` where C > 1.

        Returns:
            Tensor: The combined loss.
        """
        dice_loss = super().call(y_true, y_pred)

        if self.class_ids is not None:
            y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)

        ce_loss = keras.losses.binary_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
        )

        return (self.dice_weight * dice_loss) + (self.ce_weight * ops.mean(ce_loss))
