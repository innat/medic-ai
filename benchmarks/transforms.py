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


def _compile_forward(transform, backend: str):
    """Compile a tensor-only transform adapter for the active backend."""
    def forward(image, label):
        result = transform(TensorBundle({"image": image, "label": label}, {}))
        return result["image"], result["label"]

    if backend == "tensorflow":
        import tensorflow as tf

        return tf.function(forward, jit_compile=True)
    if backend == "jax":
        import jax

        return jax.jit(forward)
    if backend == "torch":
        import importlib.util
        import torch

        if importlib.util.find_spec("torch_xla") is None:
            raise RuntimeError(
                "Torch XLA compilation requires the optional `torch_xla` package."
            )
        return torch.compile(forward, backend="openxla", fullgraph=True)
    raise RuntimeError(f"Unsupported Keras backend for compilation: {backend!r}")


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
    """Return benchmark device names using the active backend runtime.

    ``keras.distribution.list_devices`` is currently not a portable device
    discovery API across all Keras backends, so backend runtimes are imported
    locally and only by this benchmark module.
    """
    backend = keras.config.backend()
    if backend == "tensorflow":
        import tensorflow as tf

        logical_devices = tf.config.list_logical_devices()
        cpus = ["CPU:0"] if any(device.device_type == "CPU" for device in logical_devices) else []
        gpus = [
            f"GPU:{index}"
            for index, device in enumerate(
                device for device in logical_devices if device.device_type == "GPU"
            )
        ]
    elif backend == "torch":
        import torch

        cpus = ["cpu:0"]
        gpus = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    elif backend == "jax":
        import jax

        cpus = ["cpu:0"] if any(device.platform == "cpu" for device in jax.devices()) else []
        gpus = [
            f"gpu:{index}"
            for index, device in enumerate(
                device for device in jax.devices() if device.platform in ("gpu", "cuda")
            )
        ]
    else:
        raise RuntimeError(f"Unsupported Keras backend for device discovery: {backend!r}")

    cpus = cpus or ["cpu:0"]
    if requested == "cpu":
        return [cpus[0]]
    if requested == "gpu":
        if not gpus:
            raise RuntimeError("--device gpu was requested, but no GPU was detected.")
        return [gpus[0]]
    if requested == "both":
        return [cpus[0], *gpus[:1]]
    raise ValueError(f"Unknown device selection: {requested!r}")


def _make_case(
    layout: str,
    device: str,
    spatial_size: int,
    batch_size: int,
    channels: int,
    seed: int,
) -> TensorBundle:
    """Create a small aligned image/label case on the selected device."""
    if layout in ("HWC", "BHWC"):
        shape = (spatial_size, spatial_size, channels)
        if layout == "BHWC":
            shape = (batch_size, *shape)
    elif layout in ("DHWC", "BDHWC"):
        shape = (spatial_size, spatial_size, spatial_size, channels)
        if layout == "BDHWC":
            shape = (batch_size, *shape)
    else:
        raise ValueError(f"Unsupported benchmark layout: {layout!r}")
    with keras.device(device):
        image = ops.convert_to_tensor(
            np.random.default_rng(seed).normal(size=shape), dtype="float32"
        )
        label = ops.cast(image > 0.0, "int32")
        meta = {}
        if layout in ("DHWC", "BDHWC"):
            meta["affine"] = ops.eye(4, dtype="float32")
        return TensorBundle({"image": image, "label": label}, meta)


def _factory_table(layout: str, spatial_size: int) -> list[BenchmarkSpec]:
    """Return representative CPU-only and tensor-only transform cases."""
    is_3d = layout in ("DHWC", "BDHWC")
    axis = 1 if layout.startswith("B") else 0
    crop_extent = max(8, spatial_size - spatial_size // 8)
    crop_shape = (crop_extent, crop_extent, crop_extent) if is_3d else (crop_extent, crop_extent)
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


def _profile(
    spec,
    layout,
    device,
    spatial_size,
    batch_size,
    channels,
    iterations,
    warmup,
    seed,
    compile_mode,
):
    transform = spec.factory(layout, seed)
    setup_start = time.perf_counter()
    template = _make_case(layout, device, spatial_size, batch_size, channels, seed)
    case_setup_ms = (time.perf_counter() - setup_start) * 1000.0

    def fresh_case():
        return TensorBundle(dict(template.data), dict(template.meta))

    compiled_forward = None
    compile_time_ms = None
    if compile_mode == "xla":
        if spec.group == "cpu":
            raise RuntimeError("Metadata-dependent transforms are not supported by --compile xla.")
        compiled_forward = _compile_forward(transform, keras.config.backend())
        compile_start = time.perf_counter()
        case = fresh_case()
        _sync(compiled_forward(case["image"], case["label"]))
        compile_time_ms = (time.perf_counter() - compile_start) * 1000.0

    for _ in range(warmup):
        if compiled_forward is None:
            _sync(transform(fresh_case()))
        else:
            case = fresh_case()
            _sync(compiled_forward(case["image"], case["label"]))

    if spec.inverse and compiled_forward is None:
        for _ in range(warmup):
            forward = transform(fresh_case())
            _sync(forward)
            _sync(transform.inverse(forward))

    forward_times = []
    inverse_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        if compiled_forward is None:
            result = transform(fresh_case())
            _sync(result)
        else:
            case = fresh_case()
            result = compiled_forward(case["image"], case["label"])
            _sync(result)
        forward_times.append((time.perf_counter() - start) * 1000.0)
        if spec.inverse and compiled_forward is None:
            start = time.perf_counter()
            _sync(transform.inverse(result))
            inverse_times.append((time.perf_counter() - start) * 1000.0)
    output_image = result[0] if compiled_forward is not None else result["image"]
    return {
        "backend": keras.config.backend(),
        "device": device,
        "layout": layout,
        "spatial_size": spatial_size,
        "batch_size": batch_size,
        "channels": channels,
        "input_shape": list(output_image.shape),
        "forward_median_ms": statistics.median(forward_times),
        "forward_p95_ms": float(np.percentile(forward_times, 95)),
        "inverse_median_ms": statistics.median(inverse_times) if inverse_times else None,
        "case_setup_ms": case_setup_ms,
        "case_reused": True,
        "compile_mode": compile_mode,
        "compile_time_ms": compile_time_ms,
        "inverse_status": (
            "not-compiled"
            if spec.inverse and compile_mode == "xla"
            else "measured" if spec.inverse else "non-invertible"
        ),
        "iterations": iterations,
        "warmup": warmup,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="cpu")
    parser.add_argument("--group", choices=("cpu", "cpu+gpu", "all"), default="all")
    parser.add_argument("--layout", choices=("HWC", "DHWC", "BHWC", "BDHWC"), default="BDHWC")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="Square 2D or cubic 3D spatial sizes. Defaults to 224 for 2D and 96 for 3D.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--compile",
        choices=("none", "xla"),
        default="none",
        help="Compilation mode. `xla` uses the active backend's XLA compiler.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.batch_size < 1 or args.channels < 1:
        parser.error("--batch-size and --channels must be positive.")
    default_size = 224 if args.layout in ("HWC", "BHWC") else 96
    sizes = args.sizes or [default_size]
    results = []
    for spatial_size in sizes:
        if spatial_size < 1:
            parser.error("--sizes must contain positive integers.")
        for spec in _factory_table(args.layout, spatial_size):
            if args.group != "all" and spec.group != args.group:
                continue
            for device in _devices(args.device):
                try:
                    result = _profile(
                        spec,
                        args.layout,
                        device,
                        spatial_size,
                        args.batch_size,
                        args.channels,
                        args.iterations,
                        args.warmup,
                        args.seed,
                        args.compile,
                    )
                except RuntimeError as error:
                    print(f"SKIP {spec.name:24} {device:10}: {error}")
                    continue
                result.update(transform=spec.name, group=spec.group)
                results.append(result)
                inverse = result["inverse_median_ms"]
                inverse_display = (
                    result["inverse_status"]
                    if inverse is None
                    else f"{inverse:.2f} ms"
                )
                compile_display = (
                    "-" if result["compile_time_ms"] is None
                    else f"{result['compile_time_ms']:.2f} ms"
                )
                print(
                    f"{spec.name:24} {device:10} {args.layout:6} size={spatial_size:<4} "
                    f"forward={result['forward_median_ms']:.2f} ms "
                    f"inverse={inverse_display} compile={compile_display}"
                )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
