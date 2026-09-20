"""Transform-specific benchmark definitions."""

from dataclasses import dataclass
from typing import Callable

from medicai.transforms import (
    CropForeground,
    Flip,
    NormalizeIntensity,
    Orientation,
    RandomCropByPosNegLabel,
    RandomCutOut,
    RandomAffine,
    RandomElasticTransform,
    RandomFlip,
    RandomRotate,
    RandomRotate90,
    RandomShiftIntensity,
    RandomShear,
    RandomSpatialCrop,
    RandomTranslate,
    RandomZoom,
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
    factory: Callable[[str, int], object]
    inverse: bool = False


def transform_specs(layout: str, spatial_size: int) -> list[BenchmarkSpec]:
    """Return representative sample-level and batch-level transform cases."""
    is_3d = layout in ("DHWC", "BDHWC")
    axis = 1 if layout.startswith("B") else 0
    crop_extent = max(8, spatial_size - spatial_size // 8)
    crop_shape = (crop_extent, crop_extent, crop_extent) if is_3d else (crop_extent, crop_extent)
    interpolation = ("trilinear", "nearest") if is_3d else ("bilinear", "nearest")
    specs = [
        BenchmarkSpec(
            "NormalizeIntensity",
            lambda layout, s: NormalizeIntensity(
                keys=["image"], channel_wise=True, input_layout=layout
            ),
        ),
        BenchmarkSpec(
            "ScaleIntensityRange",
            lambda layout, s: ScaleIntensityRange(
                keys=["image"],
                source_value_range=(-1.0, 1.0),
                target_value_range=(0.0, 1.0),
                clip=True,
                input_layout=layout,
            ),
            True,
        ),
        BenchmarkSpec(
            "ShiftIntensity",
            lambda layout, s: ShiftIntensity(keys=["image"], offset=0.1, input_layout=layout),
            True,
        ),
        BenchmarkSpec(
            "SignalFillEmpty",
            lambda layout, s: SignalFillEmpty(keys=["image"], fill_value=0.0, input_layout=layout),
        ),
        BenchmarkSpec(
            "Flip",
            lambda layout, s: Flip(keys=["image", "label"], spatial_axis=axis, input_layout=layout),
            True,
        ),
        BenchmarkSpec(
            "Rotate90",
            lambda layout, s: Rotate90(keys=["image", "label"], k=1, input_layout=layout),
            True,
        ),
        BenchmarkSpec(
            "Resize",
            lambda layout, s: Resize(
                keys=["image", "label"],
                interpolation=interpolation,
                target_shape=crop_shape,
                input_layout=layout,
            ),
            True,
        ),
        BenchmarkSpec(
            "SpatialCrop",
            lambda layout, s: SpatialCrop(
                keys=["image", "label"], crop_size=crop_shape, input_layout=layout
            ),
            True,
        ),
        BenchmarkSpec(
            "RandomFlip",
            lambda layout, s: RandomFlip(
                keys=["image", "label"], spatial_axis=axis, prob=1.0, seed=s, input_layout=layout
            ),
            True,
        ),
        BenchmarkSpec(
            "RandomRotate90",
            lambda layout, s: RandomRotate90(
                keys=["image", "label"], max_k=3, prob=1.0, seed=s, input_layout=layout
            ),
            True,
        ),
        BenchmarkSpec(
            "RandomRotate",
            lambda layout, s: RandomRotate(
                keys=["image", "label"], factor=0.1, prob=1.0, seed=s, input_layout=layout
            ),
            True,
        ),
        BenchmarkSpec(
            "RandomShiftIntensity",
            lambda layout, s: RandomShiftIntensity(
                keys=["image"], offset=0.1, prob=1.0, seed=s, input_layout=layout
            ),
            True,
        ),
        BenchmarkSpec(
            "RandomSpatialCrop",
            lambda layout, s: RandomSpatialCrop(
                keys=["image", "label"], crop_size=crop_shape, input_layout=layout, seed=s
            ),
            True,
        ),
        BenchmarkSpec(
            "RandomTranslate",
            lambda layout, s: RandomTranslate(
                keys=["image", "label"],
                factor=0.1,
                prob=1.0,
                seed=s,
                input_layout=layout,
            ),
            True,
        ),
        BenchmarkSpec(
            "RandomZoom",
            lambda layout, s: RandomZoom(
                keys=["image", "label"],
                zoom_factor=0.1,
                prob=1.0,
                seed=s,
                input_layout=layout,
            ),
            True,
        ),
        BenchmarkSpec(
            "RandomShear",
            lambda layout, s: RandomShear(
                keys=["image", "label"],
                shear_factor=0.1,
                prob=1.0,
                seed=s,
                input_layout=layout,
            ),
            True,
        ),
        BenchmarkSpec(
            "RandomCutOut",
            lambda layout, s: RandomCutOut(
                keys=["image"],
                mask_size=(4, 4, 4) if is_3d else (4, 4),
                num_cuts=1,
                prob=1.0,
                input_layout=layout,
                seed=s,
            ),
        ),
        BenchmarkSpec(
            "RandomElasticTransform",
            lambda layout, s: RandomElasticTransform(
                keys=["image", "label"],
                input_layout=layout,
                interpolation={
                    "image": "trilinear" if is_3d else "bilinear",
                    "label": "nearest",
                },
                control_grid_spacing=(8,) * (3 if is_3d else 2),
                alpha=3.0,
                sigma=5.0,
                prob=1.0,
                seed=s,
            ),
        ),
        BenchmarkSpec(
            "RandomAffine",
            lambda layout, s: RandomAffine(
                keys=["image", "label"],
                rotation_factor=0.1,
                zoom_factor=0.1,
                translation_factor=0.1,
                shear_factor=0.1,
                prob=1.0,
                seed=s,
                input_layout=layout,
            ),
            True,
        ),
    ]
    if layout in ("HWC", "DHWC"):
        specs.insert(
            8,
            BenchmarkSpec(
                "RandomCropByPosNegLabel",
                lambda layout, s: RandomCropByPosNegLabel(
                    keys=["image", "label"],
                    target_shape=crop_shape,
                    pos=1,
                    neg=1,
                    input_layout=layout,
                    seed=s,
                ),
                True,
            ),
        )
    if layout in ("HWC", "DHWC"):
        specs.append(
            BenchmarkSpec(
                "CropForeground",
                lambda layout, s: CropForeground(
                    keys=["image", "label"],
                    source_key="image",
                    k_divisible=(4, 4, 4) if layout == "DHWC" else (4, 4),
                    input_layout=layout,
                ),
                True,
            )
        )
    if is_3d and layout == "DHWC":
        specs.extend(
            [
                BenchmarkSpec(
                    "Orientation",
                    lambda layout, s: Orientation(
                        keys=["image", "label"], axcodes="RAS", input_layout=layout
                    ),
                    True,
                ),
                BenchmarkSpec(
                    "Spacing",
                    lambda layout, s: Spacing(
                        keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), input_layout=layout
                    ),
                    True,
                ),
            ]
        )
    return specs
