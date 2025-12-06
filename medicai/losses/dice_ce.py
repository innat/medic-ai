from keras import ops

from medicai.utils import DescribeMixin

from .base import BASE_COMMON_ARGS
from .dice import BinaryDiceLoss, CategoricalDiceLoss, SparseDiceLoss


def cross_entropy(y_true, y_pred, smooth=1e-7, mask=None):
    spatial_dims = list(range(1, len(y_pred.shape) - 1))
    y_pred = ops.clip(y_pred, smooth, 1.0 - smooth)
    ce_loss = -y_true * ops.log(y_pred)

    if mask is not None:
        masked_ce_loss = ce_loss * mask
        masked_ce_loss = ops.sum(masked_ce_loss, axis=spatial_dims)
        valid_pixels = ops.sum(mask, axis=spatial_dims)
        valid_pixels = ops.maximum(valid_pixels, 1.0)
        mean_ce_loss = masked_ce_loss / valid_pixels
        return mean_ce_loss
    else:
        ce_loss = ops.mean(ce_loss, axis=spatial_dims)
        return ce_loss


def binary_cross_entropy(y_true, y_pred, smooth=1e-7, mask=None):
    spatial_dims = list(range(1, len(y_pred.shape) - 1))
    y_pred = ops.clip(y_pred, smooth, 1.0 - smooth)
    ce_loss = -(y_true * ops.log(y_pred) + (1.0 - y_true) * ops.log(1.0 - y_pred))

    if mask is not None:
        masked_ce_loss = ce_loss * mask
        masked_ce_loss = ops.sum(masked_ce_loss, axis=spatial_dims)
        valid_pixels = ops.sum(mask, axis=spatial_dims)
        valid_pixels = ops.maximum(valid_pixels, 1.0)
        mean_ce_loss = masked_ce_loss / valid_pixels
        return mean_ce_loss
    else:
        ce_loss = ops.mean(ce_loss, axis=spatial_dims)
        return ce_loss


def apply_reduction(loss, reduction):
    if reduction == "none":
        return loss
    elif reduction == "sum":
        return ops.sum(loss)
    elif reduction == "mean":
        return ops.mean(loss)
    elif reduction == "sum_over_batch_size":
        batch_size = ops.cast(ops.shape(loss)[0], loss.dtype)
        return ops.sum(loss) / batch_size
    else:
        raise ValueError(
            f"Unsupported reduction type: {reduction} "
            "Supported methods are none, mean, sum, sum_over_batch_size"
        )


class SparseDiceCELoss(SparseDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
        dice_weight=1.0,
        ce_weight=1.0,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
            smooth=smooth,
            reduction="none",
            name=name or "sparse_dice_crossentropy",
            **kwargs,
        )
        if dice_weight < 0.0:
            raise ValueError("dice_weight must be >= 0.0")
        if ce_weight < 0.0:
            raise ValueError("ce_weight must be >= 0.0")

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.hybrid_reduction = reduction

    def call(self, y_true, y_pred):
        """Computes the combined Sparse Dice and Categorical Cross-Entropy loss.

        Args:
            y_true (Tensor): Sparse ground truth tensor (integer class indices).
            y_pred (Tensor): Prediction tensor (logits or probabilities).

        Returns:
            Tensor: The combined loss.
        """
        # Prepare inputs
        y_true_processed, y_pred_processed, valid_mask = self._process_inputs(y_true, y_pred)

        # Compute dice loss using parent's compute_loss
        dice_loss = self.compute_loss(y_true_processed, y_pred_processed, valid_mask)

        # Compute CE loss
        ce_loss = cross_entropy(
            y_true_processed, y_pred_processed, smooth=self.smooth, mask=valid_mask
        )

        combined_loss = (self.dice_weight * dice_loss) + (self.ce_weight * ce_loss)
        combined_loss = apply_reduction(combined_loss, self.hybrid_reduction)
        return combined_loss


class CategoricalDiceCELoss(CategoricalDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        smooth=1e-7,
        dice_weight=1.0,
        ce_weight=1.0,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=None,
            smooth=smooth,
            reduction="none",
            name=name or "categorical_dice_crossentropy",
            **kwargs,
        )
        if dice_weight < 0.0:
            raise ValueError("dice_weight must be >= 0.0")
        if ce_weight < 0.0:
            raise ValueError("ce_weight must be >= 0.0")

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.hybrid_reduction = reduction

    def call(self, y_true, y_pred):
        """Computes the combined Categorical Dice and Categorical Cross-Entropy loss.

        Args:
            y_true (Tensor): One-hot encoded ground truth tensor.
            y_pred (Tensor): Prediction tensor (logits or probabilities).

        Returns:
            Tensor: The combined loss.
        """
        # Prepare inputs
        y_true_processed, y_pred_processed, valid_mask = self._process_inputs(y_true, y_pred)

        # Compute dice loss using parent's compute_loss
        dice_loss = self.compute_loss(y_true_processed, y_pred_processed, valid_mask)

        # Compute CE loss
        ce_loss = cross_entropy(
            y_true_processed, y_pred_processed, smooth=self.smooth, mask=valid_mask
        )

        combined_loss = (self.dice_weight * dice_loss) + (self.ce_weight * ce_loss)
        combined_loss = apply_reduction(combined_loss, self.hybrid_reduction)
        return combined_loss


class BinaryDiceCELoss(BinaryDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
        dice_weight=1.0,
        ce_weight=1.0,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
            smooth=smooth,
            reduction="none",
            name=name or "binary_dice_crossentropy",
            **kwargs,
        )
        if dice_weight < 0.0:
            raise ValueError("dice_weight must be >= 0.0")
        if ce_weight < 0.0:
            raise ValueError("ce_weight must be >= 0.0")

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.hybrid_reduction = reduction

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
        # Prepare inputs
        y_true_processed, y_pred_processed, valid_mask = self._process_inputs(y_true, y_pred)

        # Compute dice loss using parent's compute_loss
        dice_loss = self.compute_loss(y_true_processed, y_pred_processed, valid_mask)

        # Compute BCE loss
        bce_loss = binary_cross_entropy(
            y_true_processed, y_pred_processed, smooth=self.smooth, mask=valid_mask
        )

        combined_loss = (self.dice_weight * dice_loss) + (self.ce_weight * bce_loss)
        combined_loss = apply_reduction(combined_loss, self.hybrid_reduction)
        return combined_loss


DICE_CE_DOCSTRING = """    dice_weight (float): The trade-off weight for the Dice loss 
    component. Must be >= 0.0. A higher value gives more importance to Dice loss.
        Defaults to 1.0.
    ce_weight (float): The trade-off weight for the Cross-Entropy loss component.
        Must be >= 0.0. A higher value gives more importance to Cross-Entropy loss.
        Defaults to 1.0.
"""

CATEGORICAL_LOSS_DOCSTRING = """Combined Categorical Dice and Categorical Cross-Entropy Loss.

This loss function combines the Categorical Dice loss with the standard
Categorical Cross-Entropy loss. It is suitable for multi-class segmentation
tasks where the ground truth labels are one-hot encoded.

""" + BASE_COMMON_ARGS.format(
    specific_args=DICE_CE_DOCSTRING, default_name="categorical_dice_crossentropy"
)


SPARSE_LOSS_DOCSTRING = """Combined Sparse Dice and Categorical Cross-Entropy Loss.

This loss function combines the Sparse Dice loss with the Categorical
Cross-Entropy loss. It is often used in multi-class segmentation tasks
to leverage the strengths of both loss functions, encouraging both region
overlap (Dice) and per-pixel classification accuracy (Cross-Entropy).

""" + BASE_COMMON_ARGS.format(
    specific_args=DICE_CE_DOCSTRING, default_name="sparse_dice_crossentropy"
)

BINARY_LOSS_DOCSTRING = """Combined Binary Dice and Binary Cross-Entropy Loss.

This loss function combines the Binary Dice loss with Binary Cross-Entropy, 
regardless of the number of channels in the prediction tensor (`y_pred`):

- If `y_pred` has one channel (shape `(..., 1)`), it's treated as a
    single-class binary segmentation problem (e.g., foreground vs. background).
    Binary Cross-Entropy is used in this case.

- If `y_pred` has more than one channel (shape `(..., C)` where C > 1),
    it's treated as a multi-label binary segmentation problem, where each
    channel represents the probability of a specific binary class being present.

""" + BASE_COMMON_ARGS.format(
    specific_args=DICE_CE_DOCSTRING, default_name="binary_dice_crossentropy"
)

CategoricalDiceCELoss.__doc__ = CATEGORICAL_LOSS_DOCSTRING
SparseDiceCELoss.__doc__ = SPARSE_LOSS_DOCSTRING
BinaryDiceCELoss.__doc__ = BINARY_LOSS_DOCSTRING
