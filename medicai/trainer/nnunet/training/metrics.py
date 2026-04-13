import numpy as np

try:
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


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
