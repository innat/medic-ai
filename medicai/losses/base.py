import keras
from keras import ops

from medicai.utils import camel_to_snake


class BaseLoss(keras.losses.Loss):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        if name is None:
            name = (
                "base_loss"
                if self.__class__ is BaseLoss
                else camel_to_snake(self.__class__.__name__)
            )

        super().__init__(name=name, reduction=reduction, **kwargs)

        self.num_classes = num_classes
        self.from_logits = from_logits
        self.smooth = smooth or keras.backend.epsilon()
        self.target_class_ids = self._validate_class_ids(target_class_ids, allow_none=False)
        self.ignore_class_ids = self._validate_class_ids(ignore_class_ids, allow_none=True)

    def _validate_class_ids(self, class_ids, allow_none=False):
        if class_ids is None:
            if allow_none:
                # Used for ignore_class_ids: None means don't ignore any.
                return None
            else:
                # Used for target_class_ids: None means target all classes.
                return list(range(self.num_classes))

        elif isinstance(class_ids, int):
            if not allow_none and not 0 <= class_ids < self.num_classes:
                raise ValueError(
                    f"Class ID {class_ids} is out of the valid range [0, {self.num_classes - 1}]."
                )
            return [class_ids]

        # Handle list case
        elif isinstance(class_ids, list):
            for cid in class_ids:
                if not allow_none and not 0 <= cid < self.num_classes:
                    raise ValueError(
                        f"Class ID {cid} is out of the valid range [0, {self.num_classes - 1}]."
                    )
            return class_ids

        # When invalid type
        else:
            valid_types = "an integer, or a list of integers"
            if not allow_none:
                valid_types += ", or None to consider all classes"

            raise ValueError(f"class_ids must be {valid_types}.")

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

        for class_index in self.target_class_ids:
            selected_y_true.append(y_true[..., class_index : class_index + 1])
            selected_y_pred.append(y_pred[..., class_index : class_index + 1])

        y_true = ops.concatenate(selected_y_true, axis=-1)
        y_pred = ops.concatenate(selected_y_pred, axis=-1)

        return y_true, y_pred

    def _process_predictions(self, y_pred):
        return y_pred

    def _process_targets(self, y_true):
        return y_true

    def _process_inputs(self, y_true, y_pred):
        y_true_processed = self._process_targets(y_true)
        y_pred_processed = self._process_predictions(y_pred)

        if self.ignore_class_ids:
            ignore_classes = ops.convert_to_tensor(self.ignore_class_ids, dtype=y_true.dtype)
            is_ignored_mask = ops.any(
                ops.equal(ops.argmax(y_true_processed, axis=-1, keepdims=True), ignore_classes),
                axis=-1,
            )
            valid_mask = ops.cast(~is_ignored_mask, y_pred_processed.dtype)
        else:
            valid_mask = ops.ones_like(y_pred_processed[..., 0], dtype=y_pred_processed.dtype)

        y_pred_processed = ops.clip(y_pred_processed, self.smooth, 1.0 - self.smooth)
        y_true_processed, y_pred_processed = self._get_desired_class_channels(
            y_true_processed, y_pred_processed
        )
        valid_mask = ops.expand_dims(valid_mask, axis=-1)
        valid_mask = ops.broadcast_to(valid_mask, ops.shape(y_pred_processed))

        return y_true_processed, y_pred_processed, valid_mask

    def compute_loss(self, y_true, y_pred, mask):
        """
        Abstract method to compute the core loss (Dice, IoU, Tversky, etc.).

        Must be implemented in subclasses.

        Args:
            y_true: Ground truth tensor.
            y_pred: Prediction tensor.

        Returns:
            Tensor: The computed loss value.
        """
        raise NotImplementedError(
            "Subclasses must implement the `compute_loss` method "
            "to define the core loss calculation logic."
        )

    def call(self, y_true, y_pred):
        """Computes the loss.

        Args:
            y_true (Tensor): Ground truth tensor.
            y_pred (Tensor): Prediction tensor.

        Returns:
            Tensor: The computed loss.
        """
        y_true_processed, y_pred_processed, valid_mask = self._process_inputs(y_true, y_pred)
        loss = self.compute_loss(y_true_processed, y_pred_processed, valid_mask)
        return loss


class BaseDiceLoss(BaseLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
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
            reduction=reduction,
            name=name or "dice_loss",
            **kwargs,
        )

    def compute_loss(self, y_true, y_pred, mask):
        """Calculates the Dice loss.

        Args:
            y_true (Tensor): Ground truth tensor.
            y_pred (Tensor): Processed prediction tensor.

        Returns:
            Tensor: The Dice loss.
        """
        spatial_dims = list(range(1, len(y_pred.shape) - 1))
        intersection = ops.sum(mask * y_true * y_pred, axis=spatial_dims)
        union = ops.sum(mask * y_true, axis=spatial_dims) + ops.sum(
            mask * y_pred, axis=spatial_dims
        )
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score


class BaseIoULoss(BaseLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
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
            reduction=reduction,
            name=name or "iou_loss",
            **kwargs,
        )

    def compute_loss(self, y_true, y_pred, mask):
        # Exclude batch dim (0) and channel/class dim (-1)
        spatial_dims = list(range(1, len(y_pred.shape) - 1))

        # Intersection: (B, C) tensor
        intersection = ops.sum(mask * y_true * y_pred, axis=spatial_dims)

        # Total area (Union): (B, C) tensor
        # IoU Denominator = Area(A) + Area(B) - Area(A intersect B)
        total = (
            ops.sum(mask * y_true, axis=spatial_dims)
            + ops.sum(mask * y_pred, axis=spatial_dims)
            - intersection
        )

        # IoU Score per batch element and per class: (B, C) tensor
        # Add smooth factor to avoid division by zero
        iou_score = (intersection + self.smooth) / (total + self.smooth)
        return 1.0 - iou_score


class BaseTverskyLoss(BaseLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        alpha=0.5,
        beta=0.5,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
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
            reduction=reduction,
            name=name or "tversky_loss",
            **kwargs,
        )
        self.alpha = alpha
        self.beta = beta

    def compute_loss(self, y_true, y_pred, mask):
        # Exclude batch dim (0) and channel/class dim (-1)
        spatial_dims = list(range(1, len(y_pred.shape) - 1))

        # True Positives (TP): correctly predicted positive pixels
        tp = ops.sum(mask * y_true * y_pred, axis=spatial_dims)

        # False Positives (FP): predicted as positive but actually negative
        fp = ops.sum(mask * y_pred * (1 - y_true), axis=spatial_dims)

        # False Negatives (FN): predicted as negative but actually positive
        fn = ops.sum(mask * (1 - y_pred) * y_true, axis=spatial_dims)

        # Tversky index: weighted ratio of TP over TP + alpha*FP + beta*FN
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky_index


class BaseGeneralizedDiceLoss(BaseLoss):
    WEIGHT_TYPE = [
        "square",
        "simple",
        "uniform",
    ]

    def __init__(
        self,
        from_logits,
        num_classes,
        weight_type="square",
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
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
            reduction=reduction,
            name=name or "generalized_dice_loss",
            **kwargs,
        )
        self.weight_type = weight_type.lower()
        if self.weight_type not in self.WEIGHT_TYPE:
            raise ValueError(
                f'Invalid weight_type "{weight_type}". '
                f'Supported values are: {", ".join(self.WEIGHT_TYPE)}.'
            )

    def compute_loss(self, y_true, y_pred, mask):
        # Exclude batch dim (0) and channel/class dim (-1)
        spatial_dims = list(range(1, len(y_pred.shape) - 1))

        # Reference volumes (sum over spatial dimensions)
        ref_vol = ops.sum(mask * y_true, axis=spatial_dims)

        # Intersection and prediction volumes
        intersection = ops.sum(mask * y_true * y_pred, axis=spatial_dims)
        seg_vol = ops.sum(mask * y_pred, axis=spatial_dims)

        # Compute weights according to weight_type
        if self.weight_type == "square":
            weights = 1.0 / (ref_vol**2 + self.smooth)
        elif self.weight_type == "simple":
            weights = 1.0 / (ref_vol + self.smooth)
        else:  # uniform
            weights = ops.ones_like(ref_vol)

        # Mask weights where the class is absent
        weights = ops.where(ref_vol < self.smooth, ops.zeros_like(weights), weights)

        # Generalized Dice score
        weighted_intersection = ops.sum(weights * intersection, axis=-1)
        weighted_total = ops.sum(weights * (seg_vol + ref_vol), axis=-1)
        gld_component = (2.0 * weighted_intersection) / (weighted_total + self.smooth)

        # Handle empty foreground (no class present)
        no_foreground = ops.all(
            ops.less(ref_vol + seg_vol, self.smooth),
            axis=-1,
        )
        gld_component = ops.where(
            no_foreground,
            ops.ones_like(gld_component),
            gld_component,
        )

        # Handle NaN values safely
        gld_component = ops.where(
            ops.isnan(gld_component),
            ops.ones_like(gld_component),
            gld_component,
        )
        return 1.0 - gld_component


BASE_COMMON_ARGS = """
Args:
    from_logits (bool): Whether `y_pred` is expected to be logits. If True,
        the predictions will be passed through the appropriate activation 
        (sigmoid/softmax).
    num_classes (int): The total number of classes in the segmentation task.
{specific_args}
    target_class_ids (int, list of int, or None): If an integer or a list of 
        integers, the loss will be calculated only for the specified class(es).
        If None, the loss will be calculated for all classes and averaged.
        Default: None.
    ignore_class_ids (int, list of int, or None): The ID of a class to be ignored 
        during computation. This is useful, for example, in segmentation
        problems featuring a "void" or un-label class (usually -1 or 255) in
        segmentation maps. If None, the loss will be calculated for all classes 
        and averaged. Default: None.
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
                [0.2, 0.8, 0.4],  # Sample 1: class0, class1, class2 losses (3 classes)
                [0.3, 0.7, 0.5]   # Sample 2: class0, class1, class2 losses (2 samples)
            ]

            # reduction="sum": 2.9
            # reduction="mean": 2.9 / 6 = 0.483
            # reduction="sum_over_batch_size": 2.9 / 2 = 1.45  
            # reduction=None: returns the original [[0.2, 0.8, 0.4], [0.3, 0.7, 0.5]]
        
        Defaults to 'mean'.
    name (str, optional): Name of the loss function. Defaults to "{default_name}".
    **kwargs: Additional keyword arguments passed to `keras.losses.Loss`.

"""

BASE_LOSS_DOCSTRING = """Base class for segmentation loss functions.

This class provides a foundation for calculating overlap-based losses (Dice, IoU, Tversky, etc.)
It handles class ID selection, smoothing, and prediction processing before computing 
the core metric in `compute_loss`.
""" + BASE_COMMON_ARGS.format(
    specific_args="", default_name="base_loss"
)


DICE_LOSS_DOCSTRING = """Base class for Dice-based loss functions.

This class implements the core `1.0 - Dice Score` logic.
""" + BASE_COMMON_ARGS.format(
    specific_args="", default_name="dice_loss"
)

IOU_LOSS_DOCSTRING = """Base class for IoU/Jaccard-based loss functions.

This class implements the core `1.0 - IoU/Jaccard Score` logic.
""" + BASE_COMMON_ARGS.format(
    specific_args="", default_name="iou_loss"
)

TVERSKY_SPECIFIC_ARGS = """    alpha (float, optional): Weight for **False Positives (FP)**. 
        Controls the penalty for predicting positive when the ground truth is negative. 
        Defaults to 0.5 (Tversky becomes Dice with alpha=0.5, beta=0.5).
    beta (float, optional): Weight for **False Negatives (FN)**. Controls the penalty 
        for predicting negative when the ground truth is positive.
        Defaults to 0.5. Note: alpha + beta should typically equal 1.0.
"""
TVERSKY_LOSS_DOCSTRING = """Base class for Tversky-based loss functions.

This class implements the core `1.0 - Tversky Index` logic, generalizing Dice and Jaccard.
""" + BASE_COMMON_ARGS.format(
    specific_args=TVERSKY_SPECIFIC_ARGS, default_name="tversky_loss"
)

GDL_SPECIFIC_ARGS = """    weight_type (str, optional): The weighting scheme to balance 
        class contributions. Options include: 'square' (inverse square of class volume),
        'simple' (inverse of class volume), or 'uniform' (no weighting).
        Defaults to 'square'.
"""
GDL_LOSS_DOCSTRING = """Base class for Generalized Dice Loss (GDL) functions.

This class implements the core 1.0 - GDL logic, designed to address class imbalance.

Note:
    Unlike other losses, Generalized Dice Loss aggregates across all classes before
    applying reduction. Therefore, with `reduction='none'`, it returns shape `[batch]`
    instead of `[batch, num_classes]`.

""" + BASE_COMMON_ARGS.format(
    specific_args=GDL_SPECIFIC_ARGS, default_name="generalized_dice_loss"
)


BaseLoss.__doc__ = BASE_LOSS_DOCSTRING
BaseDiceLoss.__doc__ = DICE_LOSS_DOCSTRING
BaseIoULoss.__doc__ = IOU_LOSS_DOCSTRING
BaseTverskyLoss.__doc__ = TVERSKY_LOSS_DOCSTRING
BaseGeneralizedDiceLoss.__doc__ = GDL_LOSS_DOCSTRING
