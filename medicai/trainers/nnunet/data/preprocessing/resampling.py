import numpy as np
from scipy.ndimage import gaussian_filter, zoom

# Core resampling function


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


def get_anisotropy_axis(spacing, threshold=3.0):
    """
    Detect if volume is anisotropic and return the axis index of lowest resolution.
    Returns None if isotropic.
    """
    if len(spacing) != 3:
        return None
    ratio = max(spacing) / max(min(spacing), 1e-8)
    if ratio > threshold:
        return int(np.argmax(spacing))
    return None


def resample_image(
    image,
    original_spacing,
    target_spacing,
    order=3,
    anti_aliasing=True,
):
    """
    Resample *image* to *target_spacing*.

    Parameters
    ----------
    image              : ndarray shape [D, H, W] or [D, H, W, C]
    original_spacing   : sequence of float, length 3  (z, y, x) in mm
    target_spacing     : sequence of float, length 3  (z, y, x) in mm
    order              : interpolation order (3 = cubic, 0 = nearest)
    anti_aliasing      : apply Gaussian pre-filter when downsampling (order >= 1)

    Returns
    -------
    resampled ndarray with new spatial shape
    """
    factors = compute_zoom_factors(original_spacing, target_spacing)
    lowres_axis = get_anisotropy_axis(original_spacing)

    if image.ndim == 3:
        # [D, H, W]
        return _zoom_3d(image, factors, order, anti_aliasing, lowres_axis=lowres_axis)
    elif image.ndim == 4:
        # Multi-channel [D, H, W, C] — resample each channel independently
        resampled_channels = [
            _zoom_3d(image[..., c], factors, order, anti_aliasing, lowres_axis=lowres_axis)
            for c in range(image.shape[-1])
        ]
        return np.stack(resampled_channels, axis=-1)
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


# 2-D resampling (for 2D U-Net slices)


def resample_image_2d(
    image,
    original_spacing,
    target_spacing,
    order=3,
):
    """
    Resample a 2-D image [H, W] or [H, W, C].

    Parameters
    ----------
    image              : ndarray shape [H, W] or [H, W, C]
    original_spacing   : (dy, dx) in mm
    target_spacing     : (dy, dx) in mm
    """
    assert len(original_spacing) == 2 and len(target_spacing) == 2
    factors = compute_zoom_factors(original_spacing, target_spacing)

    if image.ndim == 2:
        return zoom(image.astype(np.float32), factors, order=order, mode="nearest")
    elif image.ndim == 3:
        # Multi-channel [H, W, C]
        resampled = [
            zoom(image[..., c].astype(np.float32), factors, order=order, mode="nearest")
            for c in range(image.shape[-1])
        ]
        return np.stack(resampled, axis=-1)
    else:
        raise ValueError(f"Expected 2-D or 3-D array for 2D resampling, got {image.shape}")


# Internal helpers


def _zoom_3d(
    vol,
    factors,
    order,
    anti_aliasing,
    lowres_axis=None,
):
    """Apply scipy.ndimage.zoom to a 3-D volume with optional pre-filtering."""
    # TODO: Scipy's zoom is single-threaded and can be a bottleneck for large datasets.
    # Consider using a faster backend (e.g. cupy or specialized kernels) if performance becomes an issue.
    vol = vol.astype(np.float32)

    # Gaussian pre-smoothing for downsampling to reduce aliasing
    if anti_aliasing and order >= 1:
        sigma = [max(0.0, (1.0 / f - 1.0) * 0.5) if f < 1.0 else 0.0 for f in factors]
        if any(s > 0 for s in sigma):
            vol = gaussian_filter(vol, sigma=sigma)

    if lowres_axis is not None and order > 0:
        # Separate axis resampling for anisotropic data (Official nnU-Net style)
        # Resample high-res axes first
        other_axes = [i for i in range(3) if i != lowres_axis]
        factors_highres = [factors[i] for i in other_axes]

        # New shape after high-res zoom
        intermediate_shape = list(vol.shape)
        for i, f in zip(other_axes, factors_highres):
            intermediate_shape[i] = max(1, round(intermediate_shape[i] * f))

        # We use a 2D-like zoom for the high-res planes
        # Scipy zoom doesn't support partial axis zoom easily without slicing or multiple calls
        # Here we follow the official strategy: Resample in-plane then out-of-plane

        # 1. Resample in-plane (using order=order)
        # To simplify, we'll do two calls to zoom if anisotropic
        in_plane_factors = [1.0, 1.0, 1.0]
        for i in other_axes:
            in_plane_factors[i] = factors[i]

        vol = zoom(vol, in_plane_factors, order=order, mode="nearest")

        # 2. Resample low-res axis (usually using order=0 or 3 depending on settings)
        low_res_factors = [1.0, 1.0, 1.0]
        low_res_factors[lowres_axis] = factors[lowres_axis]

        return zoom(vol, low_res_factors, order=order, mode="nearest")

    return zoom(vol, factors, order=order, mode="nearest")


# Utility: compute new shape after resampling


def compute_new_shape(
    original_shape,
    original_spacing,
    target_spacing,
):
    """Return the spatial shape after resampling (rounded to int)."""
    factors = compute_zoom_factors(original_spacing, target_spacing)
    return tuple(max(1, round(s * f)) for s, f in zip(original_shape, factors, strict=True))
