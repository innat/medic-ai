"""Transform-specific benchmark definitions."""

from dataclasses import dataclass
from typing import Callable

from medicai.transforms import (
    CropForeground,
    Flip,
    NormalizeIntensity,
    Orientation,
    RandomCutOut,
    RandomFlip,
    RandomRotate,
    RandomRotate90,
    RandomShiftIntensity,
    RandomSpatialCrop,
    Resize,
    Rotate90,
    ScaleIntensityRange,
    ShiftIntensity,
    SignalFillEmpty,
    Spacing,
    SpatialCrop,
)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Describe one transform benchmark case."""

    name: str
    group: str
    factory: Callable[[str, int], object]
    inverse: bool = False


def transform_specs(layout: str, spatial_size: int) -> list[BenchmarkSpec]:
    """Return representative CPU-only and tensor-only transform cases."""
    is_3d = layout in ("DHWC", "BDHWC")
    axis = 1 if layout.startswith("B") else 0
    crop_extent = max(8, spatial_size - spatial_size // 8)
    crop_shape = (
        (crop_extent, crop_extent, crop_extent)
        if is_3d
        else (crop_extent, crop_extent)
    )
    interpolation = ("trilinear", "nearest") if is_3d else ("bilinear", "nearest")
    specs = [
        BenchmarkSpec("NormalizeIntensity", "cpu+gpu", lambda l, s: NormalizeIntensity(keys=["image"], channel_wise=True, input_layout=l)),
        BenchmarkSpec("ScaleIntensityRange", "cpu+gpu", lambda l, s: ScaleIntensityRange(keys=["image"], source_value_range=(-1.0, 1.0), target_value_range=(0.0, 1.0), clip=True, input_layout=l), True),
        BenchmarkSpec("ShiftIntensity", "cpu+gpu", lambda l, s: ShiftIntensity(keys=["image"], offset=0.1, input_layout=l), True),
        BenchmarkSpec("SignalFillEmpty", "cpu+gpu", lambda l, s: SignalFillEmpty(keys=["image"], fill_value=0.0, input_layout=l)),
        BenchmarkSpec("Flip", "cpu+gpu", lambda l, s: Flip(keys=["image", "label"], spatial_axis=axis, input_layout=l), True),
        BenchmarkSpec("Rotate90", "cpu+gpu", lambda l, s: Rotate90(keys=["image", "label"], k=1, input_layout=l), True),
        BenchmarkSpec("Resize", "cpu+gpu", lambda l, s: Resize(keys=["image", "label"], interpolation=interpolation, target_shape=crop_shape, input_layout=l), True),
        BenchmarkSpec("SpatialCrop", "cpu+gpu", lambda l, s: SpatialCrop(keys=["image", "label"], crop_size=crop_shape, input_layout=l), True),
        BenchmarkSpec("RandomFlip", "cpu+gpu", lambda l, s: RandomFlip(keys=["image", "label"], spatial_axis=axis, prob=1.0, seed=s, input_layout=l), True),
        BenchmarkSpec("RandomRotate90", "cpu+gpu", lambda l, s: RandomRotate90(keys=["image", "label"], max_k=3, prob=1.0, seed=s, input_layout=l), True),
        BenchmarkSpec("RandomRotate", "cpu+gpu", lambda l, s: RandomRotate(keys=["image", "label"], factor=0.1, prob=1.0, seed=s, input_layout=l), True),
        BenchmarkSpec("RandomShiftIntensity", "cpu+gpu", lambda l, s: RandomShiftIntensity(keys=["image"], offset=0.1, prob=1.0, seed=s, input_layout=l), True),
        BenchmarkSpec("RandomSpatialCrop", "cpu+gpu", lambda l, s: RandomSpatialCrop(keys=["image", "label"], crop_size=crop_shape, input_layout=l, seed=s), True),
        BenchmarkSpec("RandomCutOut", "cpu+gpu", lambda l, s: RandomCutOut(keys=["image", "label"], mask_size=(4, 4), num_cuts=1, prob=1.0, input_layout=l, seed=s)),
    ]
    if is_3d and layout == "DHWC":
        specs.extend([
            BenchmarkSpec("CropForeground", "cpu", lambda l, s: CropForeground(keys=["image", "label"], source_key="image", k_divisible=(4, 4, 4), input_layout=l), True),
            BenchmarkSpec("Orientation", "cpu", lambda l, s: Orientation(keys=["image", "label"], axcodes="RAS", input_layout=l), True),
            BenchmarkSpec("Spacing", "cpu", lambda l, s: Spacing(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), input_layout=l), True),
        ])
    return specs
