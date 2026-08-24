"""Benchmark MedicAI transforms on CPU and, when available, on GPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import keras
import numpy as np
from keras import ops

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
    TensorBundle,
)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    group: str
    factory: Callable[[str, int], object]
    inverse: bool = False


def _sync(value) -> None:
    """Materialize backend work, including tensors in a TensorBundle."""
    if isinstance(value, TensorBundle):
        for tensor in value.data.values():
            _sync(tensor)
        return
    if isinstance(value, dict):
        for item in value.values():
            _sync(item)
        return
    try:
        ops.convert_to_numpy(value)
    except (TypeError, ValueError, AttributeError):
        pass


def _devices(requested: str) -> list[str]:
    try:
        available = list(keras.distribution.list_devices())
    except AttributeError:
        available = ["cpu:0"]
    cpus = [device for device in available if "cpu" in device.lower()] or ["cpu:0"]
    gpus = [
        device
        for device in available
        if any(token in device.lower() for token in ("gpu", "cuda", "rocm"))
    ]
    if requested == "cpu":
        return [cpus[0]]
    if requested == "gpu":
        if not gpus:
            raise RuntimeError("--device gpu was requested, but no GPU was detected.")
        return [gpus[0]]
    if requested == "both":
        return [cpus[0], *gpus[:1]]
    raise ValueError(f"Unknown device selection: {requested!r}")


def _make_case(layout: str, device: str) -> TensorBundle:
    """Create a small aligned image/label case on the selected device."""
    shapes = {
        "HWC": (64, 64, 1),
        "DHWC": (12, 32, 32, 1),
        "BHWC": (4, 64, 64, 1),
        "BDHWC": (2, 12, 32, 32, 1),
    }
    if layout not in shapes:
        raise ValueError(f"Unsupported benchmark layout: {layout!r}")
    with keras.device(device):
        image = ops.convert_to_tensor(
            np.random.default_rng(7).normal(size=shapes[layout]), dtype="float32"
        )
        label = ops.cast(image > 0.0, "int32")
        meta = {}
        if layout in ("DHWC", "BDHWC"):
            meta["affine"] = ops.eye(4, dtype="float32")
        return TensorBundle({"image": image, "label": label}, meta)


def _factory_table(layout: str) -> list[BenchmarkSpec]:
    """Return representative CPU-only and tensor-only transform cases."""
    is_3d = layout in ("DHWC", "BDHWC")
    axis = 1 if layout.startswith("B") else 0
    crop_shape = (8, 24, 24) if is_3d else (48, 48)
    specs = [
        BenchmarkSpec(
            "NormalizeIntensity", "cpu+gpu",
            lambda l, s: NormalizeIntensity(keys=["image"], channel_wise=True, input_layout=l),
        ),
        BenchmarkSpec(
            "ScaleIntensityRange", "cpu+gpu",
            lambda l, s: ScaleIntensityRange(
                keys=["image"], source_value_range=(-1.0, 1.0),
                target_value_range=(0.0, 1.0), clip=True, input_layout=l
            ),
            True,
        ),
        BenchmarkSpec(
            "ShiftIntensity", "cpu+gpu",
            lambda l, s: ShiftIntensity(keys=["image"], offset=0.1, input_layout=l),
            True,
        ),
        BenchmarkSpec(
            "SignalFillEmpty", "cpu+gpu",
            lambda l, s: SignalFillEmpty(keys=["image"], fill_value=0.0, input_layout=l),
        ),
        BenchmarkSpec(
            "Flip", "cpu+gpu",
            lambda l, s: Flip(keys=["image", "label"], spatial_axis=axis, input_layout=l),
            True,
        ),
        BenchmarkSpec(
            "Rotate90", "cpu+gpu",
            lambda l, s: Rotate90(keys=["image", "label"], k=1, input_layout=l),
            True,
        ),
        BenchmarkSpec(
            "Resize", "cpu+gpu",
            lambda l, s: Resize(
                keys=["image", "label"],
                interpolation=("trilinear", "nearest") if is_3d else ("bilinear", "nearest"),
                target_shape=crop_shape,
                input_layout=l,
            ),
            True,
        ),
        BenchmarkSpec(
            "SpatialCrop", "cpu+gpu",
            lambda l, s: SpatialCrop(keys=["image", "label"], crop_size=crop_shape, input_layout=l),
            True,
        ),
        BenchmarkSpec(
            "RandomFlip", "cpu+gpu",
            lambda l, s: RandomFlip(keys=["image", "label"], spatial_axis=axis, prob=1.0, seed=s, input_layout=l),
            True,
        ),
        BenchmarkSpec(
            "RandomRotate90", "cpu+gpu",
            lambda l, s: RandomRotate90(keys=["image", "label"], max_k=3, prob=1.0, seed=s, input_layout=l),
            True,
        ),
        BenchmarkSpec(
            "RandomRotate", "cpu+gpu",
            lambda l, s: RandomRotate(keys=["image", "label"], factor=0.1, prob=1.0, seed=s, input_layout=l),
            True,
        ),
        BenchmarkSpec(
            "RandomShiftIntensity", "cpu+gpu",
            lambda l, s: RandomShiftIntensity(keys=["image"], offset=0.1, prob=1.0, seed=s, input_layout=l),
            True,
        ),
        BenchmarkSpec(
            "RandomSpatialCrop", "cpu+gpu",
            lambda l, s: RandomSpatialCrop(keys=["image", "label"], crop_size=crop_shape, input_layout=l, seed=s),
            True,
        ),
        BenchmarkSpec(
            "RandomCutOut", "cpu+gpu",
            lambda l, s: RandomCutOut(keys=["image", "label"], mask_size=(4, 4), num_cuts=1, prob=1.0, input_layout=l, seed=s),
        ),
    ]
    if is_3d and layout == "DHWC":
        specs.extend(
            [
                BenchmarkSpec(
                    "CropForeground", "cpu",
                    lambda l, s: CropForeground(keys=["image", "label"], source_key="image", k_divisible=(4, 4, 4), input_layout=l),
                    True,
                ),
                BenchmarkSpec(
                    "Orientation", "cpu",
                    lambda l, s: Orientation(keys=["image", "label"], axcodes="RAS", input_layout=l),
                    True,
                ),
                BenchmarkSpec(
                    "Spacing", "cpu",
                    lambda l, s: Spacing(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), input_layout=l),
                    True,
                ),
            ]
        )
    return specs


def _profile(spec, layout, device, iterations, warmup, seed):
    transform = spec.factory(layout, seed)
    for _ in range(warmup):
        _sync(transform(_make_case(layout, device)))

    forward_times = []
    inverse_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = transform(_make_case(layout, device))
        _sync(result)
        forward_times.append((time.perf_counter() - start) * 1000.0)
        if spec.inverse:
            start = time.perf_counter()
            _sync(transform.inverse(result))
            inverse_times.append((time.perf_counter() - start) * 1000.0)
    return {
        "backend": keras.config.backend(),
        "device": device,
        "layout": layout,
        "forward_median_ms": statistics.median(forward_times),
        "forward_p95_ms": float(np.percentile(forward_times, 95)),
        "inverse_median_ms": statistics.median(inverse_times) if inverse_times else None,
        "iterations": iterations,
        "warmup": warmup,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="cpu")
    parser.add_argument("--group", choices=("cpu", "cpu+gpu", "all"), default="all")
    parser.add_argument("--layout", choices=("HWC", "DHWC", "BHWC", "BDHWC"), default="BDHWC")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    results = []
    for spec in _factory_table(args.layout):
        if args.group != "all" and spec.group != args.group:
            continue
        for device in _devices(args.device):
            result = _profile(spec, args.layout, device, args.iterations, args.warmup, args.seed)
            result.update(transform=spec.name, group=spec.group)
            results.append(result)
            inverse = result["inverse_median_ms"] or 0.0
            print(
                f"{spec.name:24} {device:10} {args.layout:6} "
                f"forward={result['forward_median_ms']:.2f} ms inverse={inverse:.2f} ms"
            )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
