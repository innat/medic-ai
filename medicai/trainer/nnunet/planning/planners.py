from collections.abc import Sequence

"""
nnunet_keras/planning/patch_size_planner.py
============================================
Heuristic patch size computation following nnU-Net rules:

Given the median image shape (after resampling to target spacing), find the
largest patch that:
  - Keeps total voxel count ≤ ``max_patch_voxels``
  - Has each dimension divisible by 2^n_pooling (needed for U-Net)
  - Handles anisotropy: the lowest resolution axis is compressed first

This mirrors the original nnU-Net ``ExperimentPlanner`` behaviour without
hardcoding any per-dataset constants.
"""

import math



def compute_patch_size(
    median_shape: Sequence[int],
    spacing: Sequence[float],
    n_pooling: int,
    max_patch_voxels: int = 128 * 128 * 128,
    min_patch_size: int = 32,
    anisotropy_threshold: float = 3.0,
) -> list[int]:
    """
    Compute the patch size for a U-Net given median image shape.

    Algorithm
    ---------
    1. Start from *median_shape* rounded to the nearest multiple of 2^n_pooling.
    2. While total_voxels > max_patch_voxels, shrink the **largest** dimension
       by one factor of 2 (halve it), skipping anisotropic axes first.
    3. Enforce minimum size per dimension.

    Parameters
    ----------
    median_shape       : [D, H, W] — median spatial size after resampling
    spacing            : [dz, dy, dx] — voxel spacing (mm)
    n_pooling          : number of pooling layers → patch dims must be divisible
                         by ``2^n_pooling``
    max_patch_voxels   : hard upper bound on patch volume (default 128³ ≈ 2 M)
    min_patch_size     : minimum size per spatial dimension
    anisotropy_threshold: axis is considered anisotropic if its spacing is
                          > threshold × min_spacing

    Returns
    -------
    list[int] patch size [D, H, W]
    """
    factor = 2**n_pooling
    median_shape = list(median_shape)
    spacing = list(spacing)

    assert len(median_shape) == len(
        spacing
    ), f"median_shape length ({len(median_shape)}) must match spacing length ({len(spacing)})"
    ndim = len(median_shape)

    # 1. Round UP to nearest multiple of factor
    patch = [_round_to_multiple(s, factor) for s in median_shape]
    patch = [max(p, factor) for p in patch]  # ensure at least one pool step

    # 2. Identify anisotropic axes (those with much larger spacing)
    min_spacing = min(spacing)
    is_anisotropic = [sp / min_spacing > anisotropy_threshold for sp in spacing]

    # 3. Shrink until fits in memory — prefer shrinking anisotropic axes first
    for _ in range(100):  # safety max iterations
        total = _volume(patch)
        if total <= max_patch_voxels:
            break

        # Try shrinking the largest dimension
        # Prefer anisotropic (low-res) axes
        dim_order = _get_shrink_order(patch, spacing, is_anisotropic)

        for ax in dim_order:
            new_size = patch[ax] - factor
            if new_size >= max(min_patch_size, factor):
                patch[ax] = new_size
                break
        else:
            # Can't shrink further — accept current patch
            break

    # 4. Clamp to median shape (don't exceed actual image)
    patch = [min(p, _round_to_multiple(s, factor)) for p, s in zip(patch, median_shape, strict=True)]
    patch = [max(p, factor) for p in patch]

    return patch


def compute_n_pooling(
    patch_size: Sequence[int],
    min_feature_map_size: int = 4,
    max_pooling: int = 6,
) -> int:
    """
    Determine how many pooling layers can be applied given the patch size.

    The smallest spatial dimension after n_pooling halves must be ≥ min_feature_map_size.
    """
    min_dim = min(patch_size)
    n = 0
    while n < max_pooling:
        if min_dim // (2 ** (n + 1)) < min_feature_map_size:
            break
        n += 1
    return max(n, 1)  # at least 1 pooling layer


def compute_pool_op_kernel_sizes(
    patch_size: Sequence[int],
    n_pooling: int,
    spacing: Sequence[float],
    anisotropy_threshold: float = 3.0,
) -> list[list[int]]:
    """
    Compute per-stage pooling kernel sizes.

    For isotropic data: 2×2×2 at every level.
    For anisotropic data: pool in-plane (2×2×1) until the anisotropic axis
    catches up, then switch to 2×2×2.

    Returns list of length n_pooling, each element [kD, kH, kW].
    """
    spacing = list(spacing)
    ndim = len(spacing)
    min_spacing = min(spacing)
    is_anisotropic = [sp / min_spacing > anisotropy_threshold for sp in spacing]

    current_spacing = spacing[:]
    kernels = []

    for _ in range(n_pooling):
        kernel = []
        for ax in range(ndim):
            if is_anisotropic[ax]:
                # Don't pool this axis yet
                kernel.append(1)
            else:
                kernel.append(2)
        kernels.append(kernel)

        # Update current spacing after pooling
        current_spacing = [current_spacing[ax] * kernel[ax] for ax in range(ndim)]
        # Re-evaluate anisotropy with updated spacings
        min_sp = min(current_spacing)
        is_anisotropic = [sp / min_sp > anisotropy_threshold for sp in current_spacing]

    return kernels


def compute_anisotropic_kernel_sizes(
    n_pooling: int,
    spacing: Sequence[float],
    anisotropy_threshold: float = 3.0,
) -> list[list[int]]:
    """
    Compute convolution kernel sizes per stage, using 1×3×3 for anisotropic
    axes (thick slices) and 3×3×3 otherwise.
    """
    spacing = list(spacing)
    min_spacing = min(spacing)

    kernels = []
    for _ in range(n_pooling + 1):  # +1 for bottleneck
        is_aniso = [sp / min_spacing > anisotropy_threshold for sp in spacing]
        kernel = [1 if iso else 3 for iso in is_aniso]
        kernels.append(kernel)

    return kernels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_to_multiple(value: int, multiple: int) -> int:
    """Round *value* up to nearest *multiple*."""
    return math.ceil(value / multiple) * multiple


def _volume(patch: list[int]) -> int:
    result = 1
    for d in patch:
        result *= d
    return result


def _get_shrink_order(
    patch: list[int],
    spacing: list[float],
    is_anisotropic: list[bool],
) -> list[int]:
    """
    Return axis indices in the order we should try to shrink.

    Priority:
      1. Anisotropic axes (low-res, large spacing)
      2. Then by largest patch dimension (shrink biggest first)
    """
    # Score: (is_isotropic, -patch_size)  → sort ascending
    # anisotropic axes get is_isotropic=0, isotropic get 1
    scored = [(0 if is_anisotropic[i] else 1, -patch[i], i) for i in range(len(patch))]
    scored.sort()
    return [item[2] for item in scored]


"""
nnunet_keras/planning/batch_size_estimator.py
==============================================
GPU-aware batch size estimation.

Instead of hardcoding batch sizes, this module estimates the maximum batch
size that fits in a given GPU memory budget, based on the number of voxels
per patch and a rough per-voxel memory cost model.

Memory model
------------
bytes_per_sample ≈ n_voxels × n_feature_maps_avg × bytes_per_fp
                   × activation_memory_factor

For a U-Net with base_filters=32, max_filters=320, depth=5:
  average feature maps ≈ ~96  (rough average across all levels)
Memory for a full forward+backward pass is typically ~6–8× the parameter count.

We use a conservative empirical constant calibrated to nnU-Net defaults.
"""

import math


# ---------------------------------------------------------------------------
# Constants (tunable)
# ---------------------------------------------------------------------------

# Bytes per float32 element
BYTES_PER_FP32 = 4
BYTES_PER_FP16 = 2

# Empirical factor: ratio of peak memory to raw voxel count × feature maps
# Accounts for gradients, optimizer states, activations stored for backward
_MEMORY_FACTOR = 8.0

# Conservative average feature maps across all U-Net levels
# For base_filters=32, depth=5: 32,64,128,256,320 → avg ~156 but encoder+decoder
# doubles it at bottleneck; use 100 as sensible default
_AVG_FEATURE_MAPS = 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_batch_size(
    patch_size: Sequence[int],
    n_modalities: int = 1,
    base_filters: int = 32,
    n_pooling: int = 5,
    target_gpu_memory_gb: float = 8.0,
    mixed_precision: bool = True,
    min_batch_size: int = 2,
    max_batch_size: int = 64,
) -> int:
    """
    Estimate the largest batch size that fits in the target GPU memory.

    Parameters
    ----------
    patch_size            : [D, H, W] or [H, W]
    n_modalities          : number of input channels
    base_filters          : starting feature map count in the U-Net
    n_pooling             : number of pooling stages
    target_gpu_memory_gb  : VRAM budget (GB)
    mixed_precision       : use FP16 for activations if True
    min_batch_size        : never go below this
    max_batch_size        : hard ceiling

    Returns
    -------
    int batch size
    """
    target_bytes = target_gpu_memory_gb * (1024**3)

    bytes_per_act = BYTES_PER_FP16 if mixed_precision else BYTES_PER_FP32

    # Voxels per patch (input)
    n_voxels = 1
    for d in patch_size:
        n_voxels *= d

    # Rough peak feature maps averaged across encoder+decoder
    avg_maps = _average_feature_maps(base_filters, n_pooling)

    # Memory per sample (forward + backward activations)
    mem_per_sample = (
        n_voxels * avg_maps * bytes_per_act * _MEMORY_FACTOR
        + n_voxels * n_modalities * BYTES_PER_FP32  # input data
    )

    batch = int(target_bytes // mem_per_sample)
    batch = max(min_batch_size, min(batch, max_batch_size))

    # nnU-Net typically uses batch=2 for 3D and larger for 2D
    return batch


def _average_feature_maps(base_filters: int, n_pooling: int, max_filters: int = 320) -> float:
    """Rough average feature map count across all U-Net levels (encoder + decoder)."""
    maps = []
    f = base_filters
    for _ in range(n_pooling + 1):
        maps.append(f)
        f = min(f * 2, max_filters)
    # Decoder mirrors encoder
    maps.extend(reversed(maps[:-1]))
    return sum(maps) / len(maps)


def estimate_2d_batch_size(
    patch_size: Sequence[int],
    n_modalities: int = 1,
    base_filters: int = 32,
    n_pooling: int = 5,
    target_gpu_memory_gb: float = 8.0,
    mixed_precision: bool = True,
) -> int:
    """
    Batch size for 2D U-Net — similar logic but 2D patches are much smaller
    so we can afford bigger batches.

    2D patches are [H, W]; add a minimum of 12 for 2D training.
    """
    bs = estimate_batch_size(
        patch_size=patch_size,
        n_modalities=n_modalities,
        base_filters=base_filters,
        n_pooling=n_pooling,
        target_gpu_memory_gb=target_gpu_memory_gb,
        mixed_precision=mixed_precision,
        min_batch_size=12,
        max_batch_size=256,
    )
    return bs


"""
nnunet_keras/planning/experiment_planner.py
============================================
Automatic experiment planning following nnU-Net heuristics.

Given a DatasetFingerprint, produces a NNUNetPlan with:
  - Target spacing (median of all cases, anisotropy-aware)
  - 2D plan  (median slice shape)
  - 3D full-res plan  (median volume at target spacing)
  - 3D low-res plan  (if cascade is triggered)
  - Cascade decision (volume too large → low-res + full-res)
  - Patch sizes, batch sizes, pooling depths
  - Normalization schemes per modality

Design follows the original nnU-Net ExperimentPlanner logic without
hardcoding per-dataset constants.
"""

from pathlib import Path

import numpy as np

from medicai.trainer.nnunet.data.resampling import compute_zoom_factors
from medicai.trainer.nnunet.utils.config import DatasetFingerprint, NetworkConfig, nnUNetPlan

# ---------------------------------------------------------------------------
# nnU-Net heuristic constants
# ---------------------------------------------------------------------------

# Maximum number of voxels per 3D patch (≈ GPU memory governor)
MAX_VOXELS_3D = 128 * 128 * 128  # 2 097 152
MAX_VOXELS_3D_LOWRES = 64 * 64 * 64  # 262 144

# Cascade threshold: if resampled median volume exceeds this we run cascade
CASCADE_VOLUME_THRESHOLD = 2 * MAX_VOXELS_3D

# 2D patch size target
MAX_VOXELS_2D = 512 * 512

# Pooling / architecture
DEFAULT_BASE_FILTERS = 32
DEFAULT_MAX_FILTERS = 320
MIN_FEATURE_MAP_SIZE = 4

# GPU budget (GB) — conservative default; override via plan if known
DEFAULT_GPU_GB = 8.0


def _resolve_output_activation(task_type: str) -> str:
    if task_type in {"binary", "multi-label"}:
        return "sigmoid"
    return "softmax"


# ---------------------------------------------------------------------------
# Target spacing heuristics
# ---------------------------------------------------------------------------


def _compute_target_spacing(fingerprint: DatasetFingerprint) -> list[float]:
    """
    Compute target spacing following nnU-Net rules:
      - For isotropic data: median spacing across all cases
      - For anisotropic data: keep the in-plane (high-res) spacing at the
        median, but raise the anisotropic axis to the 10th percentile of
        that axis's spacing (avoid resampling to very thin slices)

    Returns [dz, dy, dx] in mm.
    """
    spacings = np.array(fingerprint.spacings, dtype=np.float32)  # [N, 3]
    median_spacing = np.median(spacings, axis=0).tolist()

    if not fingerprint.is_anisotropic:
        return [float(x) for x in median_spacing]

    # Anisotropic: find the thick axis (largest median spacing)
    thick_ax = int(np.argmax(median_spacing))

    # Use 10th percentile for the thick axis so we don't over-upsample
    p10 = float(np.percentile(spacings[:, thick_ax], 10))
    target = list(float(x) for x in median_spacing)
    target[thick_ax] = p10  # target is FINER than median for thick axis

    return target


# ---------------------------------------------------------------------------
# 3D network planning
# ---------------------------------------------------------------------------


def _plan_3d(
    fingerprint: DatasetFingerprint,
    target_spacing: list[float],
    max_voxels: int = MAX_VOXELS_3D,
    gpu_gb: float = DEFAULT_GPU_GB,
    mixed_precision: bool = True,
    configuration: str = "3d_fullres",
) -> NetworkConfig:
    """Build a NetworkConfig for a 3D U-Net."""

    # Compute expected median shape at target_spacing
    median_shape = _resampled_median_shape(fingerprint, target_spacing)

    # Determine pooling depth
    n_pooling = compute_n_pooling(median_shape, min_feature_map_size=MIN_FEATURE_MAP_SIZE)

    # Compute patch size
    patch_size = compute_patch_size(
        median_shape=median_shape,
        spacing=target_spacing,
        n_pooling=n_pooling,
        max_patch_voxels=max_voxels,
    )

    # Re-derive n_pooling based on actual patch (might be smaller)
    n_pooling = compute_n_pooling(patch_size)

    # Pool op kernels per stage
    pool_kernels = compute_pool_op_kernel_sizes(
        patch_size=patch_size,
        n_pooling=n_pooling,
        spacing=target_spacing,
    )

    # Convolution kernel sizes (anisotropy-aware)
    conv_kernels = compute_anisotropic_kernel_sizes(n_pooling=n_pooling, spacing=target_spacing)
    # Use first conv kernel as the representative kernel size.
    # NOTE: This is informational only. The actual model in dynamic_unet.py
    # derives per-stage kernels independently from pool_op_kernel_sizes.
    kernel_size = conv_kernels[0] if len(conv_kernels) > 0 else [3, 3, 3]

    # Batch size
    batch_size = estimate_batch_size(
        patch_size=patch_size,
        n_modalities=len(fingerprint.modalities),
        base_filters=DEFAULT_BASE_FILTERS,
        n_pooling=n_pooling,
        target_gpu_memory_gb=gpu_gb,
        mixed_precision=mixed_precision,
    )

    return NetworkConfig(
        spatial_dims=3,
        patch_size=patch_size,
        batch_size=batch_size,
        n_pooling=n_pooling,
        base_filters=DEFAULT_BASE_FILTERS,
        max_filters=DEFAULT_MAX_FILTERS,
        kernel_size=kernel_size,
        pool_op_kernel_sizes=pool_kernels,
        conv_per_stage=2,
        deep_supervision=True,
        n_classes=fingerprint.output_channels,
        n_modalities=len(fingerprint.modalities),
        output_activation=_resolve_output_activation(fingerprint.task_type),
    )


# ---------------------------------------------------------------------------
# 2D network planning
# ---------------------------------------------------------------------------


def _plan_2d(
    fingerprint: DatasetFingerprint,
    target_spacing: list[float],
    gpu_gb: float = DEFAULT_GPU_GB,
    mixed_precision: bool = True,
) -> NetworkConfig:
    """Build a NetworkConfig for a 2D U-Net (slices)."""

    if fingerprint.spatial_dims == 2:
        median_shape_2d = list(fingerprint.median_size)
        target_spacing_2d = list(target_spacing)
        original_spacing_2d = list(fingerprint.median_spacing)
    else:
        median_shape_2d = fingerprint.median_size[1:]
        target_spacing_2d = target_spacing[1:]
        original_spacing_2d = fingerprint.median_spacing[1:]

    # Recompute 2D shape at target in-plane spacing
    factors_2d = compute_zoom_factors(original_spacing_2d, target_spacing_2d)
    median_shape_2d = [max(1, int(round(s * f))) for s, f in zip(median_shape_2d, factors_2d, strict=True)]

    n_pooling = compute_n_pooling(median_shape_2d)
    patch_2d = compute_patch_size(
        median_shape=median_shape_2d,
        spacing=target_spacing_2d,
        n_pooling=n_pooling,
        max_patch_voxels=MAX_VOXELS_2D,
    )

    pool_kernels = []
    for _ in range(n_pooling):
        pool_kernels.append([2, 2])

    batch_size = estimate_2d_batch_size(
        patch_size=patch_2d,
        n_modalities=len(fingerprint.modalities),
        base_filters=DEFAULT_BASE_FILTERS,
        n_pooling=n_pooling,
        target_gpu_memory_gb=gpu_gb,
        mixed_precision=mixed_precision,
    )

    return NetworkConfig(
        spatial_dims=2,
        patch_size=patch_2d,
        batch_size=batch_size,
        n_pooling=n_pooling,
        base_filters=DEFAULT_BASE_FILTERS,
        max_filters=DEFAULT_MAX_FILTERS,
        kernel_size=[3, 3],
        pool_op_kernel_sizes=pool_kernels,
        conv_per_stage=2,
        deep_supervision=True,
        n_classes=fingerprint.output_channels,
        n_modalities=len(fingerprint.modalities),
        output_activation=_resolve_output_activation(fingerprint.task_type),
    )


# ---------------------------------------------------------------------------
# Normalization scheme selection
# ---------------------------------------------------------------------------


def _select_normalization_schemes(fingerprint: DatasetFingerprint) -> list[str]:
    """
    Choose the normalization scheme per modality.
      - CT modalities → 'ct' (global clip + z-score)
      - Everything else → 'z_score' (per-case foreground z-score)
    """
    schemes = []
    for mod in fingerprint.modalities:
        if "ct" in mod.lower():
            schemes.append("ct")
        else:
            schemes.append("z_score")
    return schemes


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------


class nnUNetPlanner:
    """
    Produces a :class:`~nnunet_keras.utils.config.nnUNetPlan` from a
    :class:`~nnunet_keras.utils.config.DatasetFingerprint`.

    Usage
    -----
    ::

        planner = nnUNetPlanner(fingerprint)
        plan = planner.plan()
        plan.to_json("nnunet_plans.json")
    """

    def __init__(
        self,
        fingerprint: DatasetFingerprint,
        gpu_memory_gb: float = DEFAULT_GPU_GB,
        mixed_precision: bool = True,
    ) -> None:
        self.fp = fingerprint
        self.gpu_gb = gpu_memory_gb
        self.mixed_precision = mixed_precision

    def plan(self, output_path: str | Path | None = None) -> nnUNetPlan:
        """Run the full planning pipeline."""
        fp = self.fp

        # Target spacing
        target_spacing = _compute_target_spacing(fp)

        plan_3d_fullres = None
        plan_3d_lowres = None
        plan_2d = None
        use_cascade = False

        # ---- Cascade decision
        if fp.spatial_dims == 2:
            plan_2d = _plan_2d(
                fp,
                target_spacing,
                gpu_gb=self.gpu_gb,
                mixed_precision=self.mixed_precision,
            )
            network_type = "2d"
        else:
            plan_3d_fullres = _plan_3d(
                fp,
                target_spacing,
                max_voxels=MAX_VOXELS_3D,
                gpu_gb=self.gpu_gb,
                mixed_precision=self.mixed_precision,
                configuration="3d_fullres",
            )
            median_shape_3d = _resampled_median_shape(fp, target_spacing)
            resampled_volume = int(np.prod(median_shape_3d))
            use_cascade = resampled_volume > CASCADE_VOLUME_THRESHOLD

        # ---- 3D low-res plan  (only if cascade)
        plan_3d_lowres = None
        if fp.spatial_dims != 2 and use_cascade:
            lowres_spacing = [s * 2.0 for s in target_spacing]  # 2× coarser
            plan_3d_lowres = _plan_3d(
                fp,
                lowres_spacing,
                max_voxels=MAX_VOXELS_3D_LOWRES,
                gpu_gb=self.gpu_gb,
                mixed_precision=self.mixed_precision,
                configuration="3d_lowres",
            )

        # ---- 2D plan
        if fp.spatial_dims != 2:
            plan_2d = _plan_2d(
                fp,
                target_spacing,
                gpu_gb=self.gpu_gb,
                mixed_precision=self.mixed_precision,
            )

        # ---- Normalization schemes
        norm_schemes = _select_normalization_schemes(fp)

        # ---- Determine default network type
        if fp.spatial_dims == 2:
            network_type = "2d"
        elif use_cascade:
            network_type = "3d_cascade"
        else:
            network_type = "3d_fullres"

        plan = nnUNetPlan(
            dataset_name=fp.dataset_name,
            network_type=network_type,
            target_spacing=target_spacing,
            plan_3d_fullres=plan_3d_fullres,
            plan_3d_lowres=plan_3d_lowres,
            plan_2d=plan_2d,
            use_cascade=use_cascade,
            normalization_schemes=norm_schemes,
            task_type=fp.task_type,
            ignore_class_ids=fp.ignore_class_ids,
            target_class_ids=fp.target_class_ids,
            output_channels=fp.output_channels,
        )

        if output_path is not None:
            plan.to_json(output_path)

        return plan


class nnUNetPlannerResEncM(nnUNetPlanner):
    """
    Planner configured for Medium-scale Residual Encoders.
    Follows nnU-Net official naming convention for scaling up standard configurations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Implement ResEncM-specific architecture modifications
        # (e.g., increased base_filters, different max_filters)


class nnUNetPlannerResEncL(nnUNetPlanner):
    """
    Planner configured for Large-scale Residual Encoders.
    Follows nnU-Net official naming convention for scaling up standard configurations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Implement ResEncL-specific architecture modifications
        # (e.g., larger base_filters, more decoder heads)


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _resampled_median_shape(
    fingerprint: DatasetFingerprint,
    target_spacing: list[float],
) -> list[int]:
    """Compute the median image shape after resampling to *target_spacing*."""
    factors = compute_zoom_factors(fingerprint.median_spacing, target_spacing)
    new_shape = [max(1, int(round(s * f))) for s, f in zip(fingerprint.median_size, factors, strict=True)]
    return new_shape


# Expose helper for patch_size_planner (avoid circular import)
def compute_new_shape_after_resample(
    original_shape: Sequence[int],
    original_spacing: Sequence[float],
    target_spacing: Sequence[float],
) -> list[int]:
    factors = compute_zoom_factors(original_spacing, target_spacing)
    return [max(1, int(round(s * f))) for s, f in zip(original_shape, factors, strict=True)]
