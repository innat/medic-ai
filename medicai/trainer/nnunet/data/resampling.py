"""
medicai/trainer/nnunet/data/resampling.py
================================
Spacing-based resampling for medical images using scipy.ndimage.

Key decisions
-------------
* Data arrays are resampled with order=3 (cubic) by default.
* Label / segmentation arrays are resampled with order=0 (nearest-neighbour).
* Accepts and returns float32 arrays with shape [D, H, W] (single modality)
  or [C, D, H, W] (multi-channel).
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

# ---------------------------------------------------------------------------
# Core resampling function
# ---------------------------------------------------------------------------


def compute_zoom_factors(
    original_spacing,
    target_spacing,
):
    """
    Compute per-axis zoom factors from spacing values.

    zoom = original_spacing / target_spacing

    A factor > 1 means upsampling (finer spacing), < 1 means downsampling.
    """
    assert len(original_spacing) == len(
        target_spacing
    ), f"Spacing length mismatch: {len(original_spacing)} vs {len(target_spacing)}"
    return tuple(float(o) / float(t) for o, t in zip(original_spacing, target_spacing, strict=True))


def resample_image(
    image,
    original_spacing,
    target_spacing,
    order=3,
    anti_aliasing=True,
    ensure_channel_last=True,
):
    """
    Resample *image* to *target_spacing*.

    Parameters
    ----------
    image              : ndarray shape [D, H, W], [D, H, W, C], or [C, D, H, W]
    original_spacing   : sequence of float, length 3  (z, y, x) in mm
    target_spacing     : sequence of float, length 3  (z, y, x) in mm
    order              : interpolation order (3 = cubic, 0 = nearest)
    anti_aliasing      : apply Gaussian pre-filter when downsampling (order >= 1)
    ensure_channel_last: if True, 4D is [D,H,W,C]; if False, [C,D,H,W]

    Returns
    -------
    resampled ndarray with new spatial shape
    """
    factors = compute_zoom_factors(original_spacing, target_spacing)

    if image.ndim == 3:
        # [D, H, W]
        return _zoom_3d(image, factors, order, anti_aliasing)
    elif image.ndim == 4:
        # Multi-channel — resample each channel independently
        if ensure_channel_last:
            # [D, H, W, C]
            resampled_channels = [
                _zoom_3d(image[..., c], factors, order, anti_aliasing)
                for c in range(image.shape[-1])
            ]
            return np.stack(resampled_channels, axis=-1)
        else:
            # [C, D, H, W]
            resampled_channels = [
                _zoom_3d(image[c], factors, order, anti_aliasing)
                for c in range(image.shape[0])
            ]
            return np.stack(resampled_channels, axis=0)
    else:
        raise ValueError(f"Expected 3-D or 4-D array, got shape {image.shape}")


def resample_label(
    label,
    original_spacing,
    target_spacing,
):
    """
    Resample a segmentation label map using nearest-neighbour interpolation.

    Label values are integers; nearest-neighbour preserves them exactly.
    """
    return resample_image(
        label.astype(np.float32),
        original_spacing,
        target_spacing,
        order=0,
        anti_aliasing=False,
    ).astype(np.int64)


# ---------------------------------------------------------------------------
# 2-D resampling (for 2D U-Net slices)
# ---------------------------------------------------------------------------


def resample_image_2d(
    image,
    original_spacing,
    target_spacing,
    order=3,
    ensure_channel_last=True,
):
    """
    Resample a 2-D image [H, W], [H, W, C], or [C, H, W].

    Parameters
    ----------
    image              : ndarray shape [H, W], [H, W, C], or [C, H, W]
    original_spacing   : (dy, dx) in mm
    target_spacing     : (dy, dx) in mm
    ensure_channel_last: if True, 3D is [H,W,C]; if False, [C,H,W]
    """
    assert len(original_spacing) == 2 and len(target_spacing) == 2
    factors = compute_zoom_factors(original_spacing, target_spacing)

    if image.ndim == 2:
        return zoom(image.astype(np.float32), factors, order=order, mode="nearest")
    elif image.ndim == 3:
        if ensure_channel_last:
            # [H, W, C]
            resampled = [
                zoom(image[..., c].astype(np.float32), factors, order=order, mode="nearest")
                for c in range(image.shape[-1])
            ]
            return np.stack(resampled, axis=-1)
        else:
            # [C, H, W]
            resampled = [
                zoom(image[c].astype(np.float32), factors, order=order, mode="nearest")
                for c in range(image.shape[0])
            ]
            return np.stack(resampled, axis=0)
    else:
        raise ValueError(f"Expected 2-D or 3-D array for 2D resampling, got {image.shape}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zoom_3d(
    vol,
    factors,
    order,
    anti_aliasing,
):
    """Apply scipy.ndimage.zoom to a 3-D volume with optional pre-filtering."""
    vol = vol.astype(np.float32)

    # Gaussian pre-smoothing for downsampling to reduce aliasing
    if anti_aliasing and order >= 1:
        sigma = [max(0.0, (1.0 / f - 1.0) * 0.5) if f < 1.0 else 0.0 for f in factors]
        if any(s > 0 for s in sigma):
            vol = gaussian_filter(vol, sigma=sigma)

    return zoom(vol, factors, order=order, mode="nearest")


# ---------------------------------------------------------------------------
# Utility: compute new shape after resampling
# ---------------------------------------------------------------------------


def compute_new_shape(
    original_shape,
    original_spacing,
    target_spacing,
):
    """Return the spatial shape after resampling (rounded to int)."""
    factors = compute_zoom_factors(original_spacing, target_spacing)
    return tuple(max(1, round(s * f)) for s, f in zip(original_shape, factors, strict=True))
