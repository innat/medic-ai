"""
nnunet_keras/training/metrics.py
=================================
Evaluation metrics for segmentation:
  - dice_coefficient  : per-class Dice score
  - mean_dice         : mean over foreground classes
  - hausdorff_distance: 95th-percentile HD (optional, scipy-based)

All differentiable metrics use keras.ops; HD uses scipy and is only computed
at validation time (not in the training graph).
"""

from __future__ import annotations



import keras
import numpy as np
from keras import ops

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import directed_hausdorff

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Keras-compatible Dice metrics
# ---------------------------------------------------------------------------


def dice_coefficient(
    y_true: "keras.Tensor",
    y_pred: "keras.Tensor",
    smooth: float = 1e-5,
    n_classes: Optional[int] = None,
) -> "keras.Tensor":
    """
    Compute per-class Dice coefficient.

    Parameters
    ----------
    y_true   : one-hot [B, *spatial, C] OR int [B, *spatial]
    y_pred   : softmax [B, *spatial, C]
    smooth   : Laplace smoothing
    n_classes: required when y_true is an integer label map

    Returns
    -------
    Tensor [C] — mean Dice per class over the batch
    """
    if n_classes is None:
        n_classes = y_pred.shape[-1]

    # Convert y_true to one-hot if needed
    if len(y_true.shape) == len(y_pred.shape) - 1:
        y_true = ops.one_hot(ops.cast(y_true, "int32"), n_classes)

    y_true = ops.cast(y_true, "float32")

    ndim = len(y_pred.shape)
    spatial_axes = list(range(1, ndim - 1))

    intersection = ops.sum(y_true * y_pred, axis=spatial_axes)  # [B, C]
    sum_true = ops.sum(y_true, axis=spatial_axes)  # [B, C]
    sum_pred = ops.sum(y_pred, axis=spatial_axes)  # [B, C]

    dice = (2.0 * intersection + smooth) / (sum_true + sum_pred + smooth)  # [B, C]
    return ops.mean(dice, axis=0)  # [C]


def mean_dice(
    y_true: "keras.Tensor",
    y_pred: "keras.Tensor",
    ignore_background: bool = True,
    smooth: float = 1e-5,
    n_classes: Optional[int] = None,
) -> "keras.Tensor":
    """
    Mean Dice score over foreground classes.

    Parameters
    ----------
    y_true             : label tensor
    y_pred             : softmax predictions
    ignore_background  : if True, exclude class 0 (default: True)

    Returns
    -------
    Scalar
    """
    per_class = dice_coefficient(y_true, y_pred, smooth=smooth, n_classes=n_classes)
    if ignore_background:
        per_class = per_class[1:]  # drop background
    return ops.mean(per_class)


# ---------------------------------------------------------------------------
# Keras Metric class wrappers
# ---------------------------------------------------------------------------


class MeanDiceMetric(keras.metrics.Metric):
    """
    Stateful Keras metric for mean Dice (foreground classes).

    Compatible with model.compile(metrics=[MeanDiceMetric(n_classes=2)]).
    """

    def __init__(self, n_classes: int = 2, name: str = "mean_dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.dice_sum = self.add_weight(name="dice_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle deep supervision: take highest-res output
        if isinstance(y_pred, dict):
            y_pred = y_pred["final"]
        elif isinstance(y_pred, (list, tuple)):
            y_pred = y_pred[0]

        score = mean_dice(y_true, y_pred, n_classes=self.n_classes)
        self.dice_sum.assign_add(score)
        self.count.assign_add(1.0)

    def result(self):
        return self.dice_sum / (self.count + 1e-8)

    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config["n_classes"] = self.n_classes
        return config


class PerClassDiceMetric(keras.metrics.Metric):
    """
    Per-class Dice score metric (returns mean, logs per-class values separately).
    """

    def __init__(self, n_classes: int = 2, name: str = "per_class_dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.dice_sums = [
            self.add_weight(name=f"dice_class_{i}", initializer="zeros") for i in range(n_classes)
        ]
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if isinstance(y_pred, dict):
            y_pred = y_pred["final"]
        elif isinstance(y_pred, (list, tuple)):
            y_pred = y_pred[0]

        per_class = dice_coefficient(y_true, y_pred, n_classes=self.n_classes)
        for i in range(self.n_classes):
            self.dice_sums[i].assign_add(per_class[i])
        self.count.assign_add(1.0)

    def result(self):
        return ops.stack([s / (self.count + 1e-8) for s in self.dice_sums])

    def reset_state(self):
        for s in self.dice_sums:
            s.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config["n_classes"] = self.n_classes
        return config


# ---------------------------------------------------------------------------
# Hausdorff Distance (NumPy / SciPy — validation only)
# ---------------------------------------------------------------------------


def hausdorff_distance_95(
    y_true_np: np.ndarray,
    y_pred_np: np.ndarray,
    class_idx: int = 1,
) -> float:
    """
    Compute 95th-percentile Hausdorff Distance for a given class.

    Parameters
    ----------
    y_true_np : integer numpy array [D, H, W]
    y_pred_np : integer numpy array [D, H, W]
    class_idx : foreground class to evaluate

    Returns
    -------
    float  (inf if either mask is empty)
    """
    if not _SCIPY_AVAILABLE:
        return float("nan")

    mask_true = (y_true_np == class_idx).astype(np.uint8)
    mask_pred = (y_pred_np == class_idx).astype(np.uint8)

    if mask_true.sum() == 0 or mask_pred.sum() == 0:
        return float("inf")

    # Build surface point sets
    surface_true = mask_true ^ binary_erosion(mask_true)
    surface_pred = mask_pred ^ binary_erosion(mask_pred)

    pts_true = np.column_stack(np.where(surface_true))
    pts_pred = np.column_stack(np.where(surface_pred))

    if len(pts_true) == 0 or len(pts_pred) == 0:
        return float("inf")

    # Distance from each surface point in A to nearest in B
    tree_true = cKDTree(pts_true)
    tree_pred = cKDTree(pts_pred)

    d_pred_to_true, _ = tree_true.query(pts_pred)
    d_true_to_pred, _ = tree_pred.query(pts_true)

    all_distances = np.concatenate([d_pred_to_true, d_true_to_pred])
    return float(np.percentile(all_distances, 95))
