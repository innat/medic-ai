from medicai.utils.general import hide_warnings

hide_warnings()

import keras
from keras import ops

from .dice import BinaryDiceLoss, CategoricalDiceLoss, SparseDiceLoss


class SparseDiceCELoss(SparseDiceLoss):
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
        squared_pred (bool, optional): If True, the predictions in the denominator
            of the Dice coefficient will be squared. Defaults to False.
        name (str, optional): Name of the loss function. Defaults to "sparse_dice_crossentropy".
        **kwargs: Additional keyword arguments passed to `SparseDiceLoss`.
    """

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
        return dice_loss + ops.mean(ce_loss)


class CategoricalDiceCELoss(CategoricalDiceLoss):
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
        squared_pred (bool, optional): If True, the predictions in the denominator
            of the Dice coefficient will be squared. Defaults to False.
        name (str, optional): Name of the loss function. Defaults to "categorical_dice_crossentropy".
        **kwargs: Additional keyword arguments passed to `CategoricalDiceLoss`.
    """

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
        return dice_loss + ops.mean(ce_loss)


class BinaryDiceCELoss(BinaryDiceLoss):
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
        squared_pred (bool, optional): If True, the predictions in the denominator
            of the Dice coefficient will be squared. Defaults to False.
        name (str, optional): Name of the loss function. Defaults to "binary_dice_crossentropy".
        **kwargs: Additional keyword arguments passed to `BinaryDiceLoss`.
    """

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

        if y_pred.shape[-1] == 1:
            ce_loss = keras.losses.binary_crossentropy(
                y_true,
                y_pred,
                from_logits=self.from_logits,
            )
        else:
            ce_loss = keras.losses.categorical_crossentropy(
                y_true,
                y_pred,
                from_logits=self.from_logits,
            )

        return dice_loss + ops.mean(ce_loss)
