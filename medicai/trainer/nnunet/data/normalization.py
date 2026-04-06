"""
Per-modality intensity normalization strategies:
  - CTNormalizer  : clip to [p0.5, p99.5] percentile range, then z-score
  - MRINormalizer : z-score over non-zero foreground voxels per modality
  - NoNormalizer  : pass-through (for already-normalised data)

All normalizers accept numpy arrays and return float32.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseNormalizer(ABC):
    """Abstract base for all intensity normalizers."""

    @abstractmethod
    def normalize(self, image, **kwargs):  # pragma: no cover
        """Normalize a single-modality image array [D, H, W] or [H, W]."""

    def __call__(self, image, **kwargs):
        return self.normalize(image, **kwargs)

    @property
    @abstractmethod
    def name(self):  # pragma: no cover
        ...


class CTNormalizer(BaseNormalizer):
    """
    CT-specific normalization following nnU-Net:
      1. Clip intensities to [global_p00_5, global_p99_5] computed from the
         entire training dataset (provided via *stats*).
      2. Z-score using the global mean and std (within the clipped range).

    Parameters
    ----------
    stats : dict with keys 'mean', 'std', 'percentile_00_5', 'percentile_99_5'
    """

    def __init__(self, stats):
        self.clip_min = float(stats["percentile_00_5"])
        self.clip_max = float(stats["percentile_99_5"])
        self.mean = float(stats["mean"])
        self.std = max(float(stats["std"]), 1e-8)

    @property
    def name(self):
        return "ct"

    def normalize(self, image, **kwargs):
        image = image.astype(np.float32)
        image = np.clip(image, self.clip_min, self.clip_max)
        image = (image - self.mean) / self.std
        return image


class MRINormalizer(BaseNormalizer):
    """
    MRI normalization following nnU-Net:
      Z-score over non-zero voxels of the *individual* case (not global stats).
      This accounts for the high variability in MRI intensities between scanners.

    Parameters
    ----------
    nonzero_only : if True (default), compute mean/std only over non-zero voxels.
    """

    def __init__(self, nonzero_only=True):
        self.nonzero_only = nonzero_only

    @property
    def name(self):
        return "z_score"

    def normalize(self, image, mask=None, **kwargs):
        """
        Parameters
        ----------
        image : float32 array [D, H, W] or [H, W]
        mask  : optional boolean mask; if None, non-zero pixels are used

        Notes
        -----
        When ``nonzero_only=True`` and all foreground voxels are zero,
        the image is returned unchanged (un-normalized).
        """
        image = image.astype(np.float32)

        if self.nonzero_only:
            if mask is not None:
                fg = image[mask > 0]
            else:
                fg = image[image != 0]

            if fg.size == 0:
                return image  # all-zero image: leave as-is

            mean = float(fg.mean())
            std = max(float(fg.std()), 1e-8)
        else:
            mean = float(image.mean())
            std = max(float(image.std()), 1e-8)

        image = (image - mean) / std
        return image


class NoNormalizer(BaseNormalizer):
    """Pass-through normalizer for data that is already normalised."""

    @property
    def name(self):
        return "none"

    def normalize(self, image, **kwargs):
        return image.astype(np.float32)


NORMALIZER_REGISTRY = {
    "ct": CTNormalizer,
    "z_score": MRINormalizer,
    "none": NoNormalizer,
}


def get_normalizer(scheme, stats=None):
    """
    Instantiate a normalizer by name.

    Parameters
    ----------
    scheme : one of 'ct', 'z_score', 'none'
    stats  : required for 'ct' scheme (global dataset statistics)
    """
    scheme = scheme.lower()
    if scheme not in NORMALIZER_REGISTRY:
        raise ValueError(
            f"Unknown normalization scheme '{scheme}'. " f"Choose from: {list(NORMALIZER_REGISTRY)}"
        )
    cls = NORMALIZER_REGISTRY[scheme]
    if scheme == "ct":
        if stats is None:
            raise ValueError("CT normalizer requires 'stats' dict.")
        return cls(stats)
    return cls()


def compute_intensity_stats(
    images,
    percentile_lo=0.5,
    percentile_hi=99.5,
    nonzero_only=False,
):
    """
    Compute pooled intensity statistics from a list of images.

    Returns keys: mean, std, min, max, percentile_00_5, percentile_99_5
    """
    voxels_list = []
    for img in images:
        flat = img.astype(np.float32).ravel()
        if nonzero_only:
            flat = flat[flat != 0]
        voxels_list.append(flat)

    if not voxels_list:
        return {}

    all_voxels = np.concatenate(voxels_list, axis=0)

    if all_voxels.size == 0:
        return {}

    return {
        "mean": float(all_voxels.mean()),
        "std": float(all_voxels.std()),
        "min": float(all_voxels.min()),
        "max": float(all_voxels.max()),
        "percentile_00_5": float(np.percentile(all_voxels, percentile_lo)),
        "percentile_99_5": float(np.percentile(all_voxels, percentile_hi)),
    }
