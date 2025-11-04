import keras
from keras import ops

from medicai.utils import camel_to_snake


class BaseDiceLoss(keras.losses.Loss):
    """Base class for Dice-based loss functions.

    This class provides a foundation for calculating Dice loss, a common metric
    for evaluating the overlap between predicted and ground truth segmentation masks.
    It handles class ID selection, smoothing, and prediction processing.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits. If True,
            the predictions will be passed through a sigmoid activation.
        num_classes (int): The total number of classes in the segmentation task.
        class_ids (int, list of int, or None): If an integer or a list of integers,
            the Dice loss will be calculated only for the specified class(es).
            If None, the Dice loss will be calculated for all classes and averaged.
        smooth (float, optional): A small smoothing factor to prevent division by zero.
            Defaults to 1e-7.
        dice_weight (float): The trade-off weight for the Dice loss component.
            Must be >= 0.0. A higher value gives more importance to Dice loss.
            Defaults to 1.0.
        ce_weight (float): The trade-off weight for the Cross-Entropy loss component.
            Must be >= 0.0. A higher value gives more importance to Cross-Entropy loss.
            Defaults to 1.0.
        name (str, optional): Name of the loss function. Defaults to "dice_loss".
        **kwargs: Additional keyword arguments passed to `keras.losses.Loss`.
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        class_ids=None,
        smooth=1e-7,
        dice_weight=1.0,
        ce_weight=1.0,
        name=None,
        **kwargs,
    ):
        if name is None:
            name = (
                "dice_loss"
                if self.__class__ is BaseDiceLoss
                else camel_to_snake(self.__class__.__name__)
            )

        super().__init__(name=name, **kwargs)

        if dice_weight < 0.0:
            raise ValueError("dice_weight should be not less than 0.0.")
        if ce_weight < 0.0:
            raise ValueError("ce_weight should be not less than 0.0.")

        self.class_ids = self._validate_and_get_class_ids(class_ids, num_classes)
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth or keras.backend.epsilon()

    def _validate_and_get_class_ids(self, class_ids, num_classes):
        if class_ids is None:
            return list(range(num_classes))
        elif isinstance(class_ids, int):
            return [class_ids]
        elif isinstance(class_ids, list):
            for cid in class_ids:
                if not 0 <= cid < num_classes:
                    raise ValueError(
                        f"Class ID {cid} is out of the valid range [0, {num_classes - 1}]."
                    )
            return class_ids
        else:
            raise ValueError(
                "class_ids must be an integer, a list of integers, or None to consider all classes."
            )

    def _get_desired_class_channels(self, y_true, y_pred):
        """Selects the channels corresponding to the desired class IDs.

        Args:
            y_true (Tensor): Ground truth tensor.
            y_pred (Tensor): Prediction tensor.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the ground truth and
                prediction tensors with only the desired class channels.
        """

        if self.num_classes == 1:
            return y_true, y_pred

        selected_y_true = []
        selected_y_pred = []

        for class_index in self.class_ids:
            selected_y_true.append(y_true[..., class_index : class_index + 1])
            selected_y_pred.append(y_pred[..., class_index : class_index + 1])

        y_true = ops.concatenate(selected_y_true, axis=-1)
        y_pred = ops.concatenate(selected_y_pred, axis=-1)

        return y_true, y_pred

    def _process_predictions(self, y_pred):
        return y_pred

    def _process_inputs(self, y_true):
        return y_true

    def dice_loss(self, y_true, y_pred):
        """Calculates the Dice loss.

        Args:
            y_true (Tensor): Ground truth tensor.
            y_pred (Tensor): Processed prediction tensor (after sigmoid if from logits).

        Returns:
            Tensor: The Dice loss.
        """
        y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)

        # Dynamically determine the spatial dimensions to sum over.
        # This works for both 2D (batch, H, W, C) and 3D (batch, D, H, W, C) inputs.
        spatial_dims = list(range(1, len(y_pred.shape) - 1))

        intersection = ops.sum(y_true * y_pred, axis=spatial_dims)
        union = ops.sum(y_true, axis=spatial_dims) + ops.sum(y_pred, axis=spatial_dims)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - ops.mean(dice_score)

    def call(self, y_true, y_pred):
        """Computes the Dice loss.

        Args:
            y_true (Tensor): Ground truth tensor.
            y_pred (Tensor): Prediction tensor.

        Returns:
            Tensor: The computed Dice loss.
        """
        y_pred_processed = self._process_predictions(y_pred)
        y_true_processed = self._process_inputs(y_true)

        y_pred_processed = ops.clip(y_pred_processed, self.smooth, 1.0 - self.smooth)
        dice = self.dice_loss(y_true_processed, y_pred_processed)
        return dice


class BaseIoULoss(BaseDiceLoss):
    """Base class for IoU/Jaccard-based loss functions.

    Implements IoU loss for evaluating set overlap between predicted and
    ground truth segmentation masks.

    Args:
        from_logits (bool): Whether `y_pred` is expected to be logits.
        num_classes (int): Number of segmentation classes.
        class_ids (int, list of int, or None): Classes to include in loss.
        smooth (float): Smoothing constant to avoid division by zero.
        name (str): Optional name, defaults to "iou_loss".
    """

    def __init__(
        self,
        from_logits,
        num_classes,
        class_ids=None,
        smooth=1e-7,
        name=None,
        **kwargs,
    ):
        name = name or "iou_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            class_ids=class_ids,
            smooth=smooth,
            name=name,
            **kwargs,
        )

    def iou_loss(self, y_true, y_pred):
        y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)

        # Determine spatial dimensions dynamically (e.g., [1, 2] for 4D input)
        # We exclude batch dim (0) and channel/class dim (-1)
        spatial_dims = list(range(1, len(y_pred.shape) - 1))

        # Intersection: (B, C) tensor
        intersection = ops.sum(y_true * y_pred, axis=spatial_dims)

        # Total area (Union): (B, C) tensor
        # IoU Denominator = Area(A) + Area(B) - Area(A intersect B)
        total = (
            ops.sum(y_true, axis=spatial_dims) + ops.sum(y_pred, axis=spatial_dims) - intersection
        )

        # IoU Score per batch element and per class: (B, C) tensor
        # Add smooth factor to avoid division by zero
        iou_score = (intersection + self.smooth) / (total + self.smooth)

        # Jaccard Loss: 1.0 - mean IoU score
        # Averaged over the entire batch
        return 1.0 - ops.mean(iou_score)

    def call(self, y_true, y_pred):
        y_pred_processed = self._process_predictions(y_pred)
        y_true_processed = self._process_inputs(y_true)

        y_pred_processed = ops.clip(y_pred_processed, self.smooth, 1.0 - self.smooth)
        return self.iou_loss(y_true_processed, y_pred_processed)


class BaseTverskyLoss(BaseDiceLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        alpha=0.5,
        beta=0.5,
        class_ids=None,
        smooth=1e-7,
        name=None,
        **kwargs,
    ):
        name = name or "tversky_loss"
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            class_ids=class_ids,
            smooth=smooth,
            name=name,
            **kwargs,
        )
        self.alpha = alpha
        self.beta = beta

    def tversky_loss(self, y_true, y_pred):
        y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)
        spatial_dims = list(range(1, len(y_pred.shape) - 1))

        tp = ops.sum(y_true * y_pred, axis=spatial_dims)
        fp = ops.sum(y_pred * (1 - y_true), axis=spatial_dims)
        fn = ops.sum((1 - y_pred) * y_true, axis=spatial_dims)

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - ops.mean(tversky_index)

    def call(self, y_true, y_pred):
        y_pred_processed = self._process_predictions(y_pred)
        y_true_processed = self._process_inputs(y_true)

        y_pred_processed = ops.clip(y_pred_processed, self.smooth, 1.0 - self.smooth)
        return self.tversky_loss(y_true_processed, y_pred_processed)


class BaseGeneralizedDiceLoss(BaseDiceLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        type_weight="square",
        class_ids=None,
        smooth=1e-7,
        name=None,
        **kwargs,
    ):
        name = name or "generalized_dice_loss"
        self.type_weight = type_weight.lower()
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            class_ids=class_ids,
            smooth=smooth,
            name=name,
            **kwargs,
        )

    def generalized_dice_loss(self, y_true, y_pred):
        y_true, y_pred = self._get_desired_class_channels(y_true, y_pred)

        # Get spatial dimensions (all except batch and channel)
        spatial_dims = list(range(1, len(y_pred.shape) - 1))

        # Calculate reference volumes (sum over spatial dimensions)
        ref_vol = ops.sum(y_true, axis=spatial_dims)

        # Calculate intersection and segmentation volumes
        intersection = ops.sum(y_true * y_pred, axis=spatial_dims)
        seg_vol = ops.sum(y_pred, axis=spatial_dims)

        # Calculate weights based on type_weight (using the original ref_vol)
        if self.type_weight == "square":
            weights = 1.0 / (ref_vol**2 + self.smooth)
        elif self.type_weight == "simple":
            weights = 1.0 / (ref_vol + self.smooth)
        elif self.type_weight == "uniform":
            weights = ops.ones_like(ref_vol)
        else:
            raise ValueError(f'The variable type_weight "{self.type_weight}" is not defined.')

        # Mask weights to zero where the reference volume is near-zero (i.e., class is absent)
        weights = ops.where(ref_vol < self.smooth, ops.zeros_like(weights), weights)

        # Calculate generalized dice score components
        generalised_dice_numerator = 2.0 * ops.sum(weights * intersection, axis=-1)  # [batch]
        generalised_dice_denominator = ops.sum(weights * (seg_vol + ref_vol), axis=-1)  # [batch]

        # Apply smoothing to the overall fraction denominator
        generalised_dice_score = generalised_dice_numerator / (
            generalised_dice_denominator + self.smooth
        )

        # Handle NaN scores (set score to 1.0 -> loss to 0.0)
        generalised_dice_score = ops.where(
            ops.isnan(generalised_dice_score),
            ops.ones_like(generalised_dice_score),
            generalised_dice_score,
        )

        return 1.0 - generalised_dice_score

    def call(self, y_true, y_pred):
        y_pred_processed = self._process_predictions(y_pred)
        y_true_processed = self._process_inputs(y_true)
        y_pred_processed = ops.clip(y_pred_processed, self.smooth, 1.0 - self.smooth)
        return self.generalized_dice_loss(
            y_true_processed,
            y_pred_processed,
        )
