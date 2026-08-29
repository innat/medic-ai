"""Execution, compilation, and timing logic for transform benchmarks."""

import statistics
import time

import keras
import numpy as np

from medicai.transforms import TensorBundle

from .cases import make_case
from .specs import BenchmarkSpec


def sync(value) -> None:
    """Materialize backend work, including tensors in a ``TensorBundle``."""
    if isinstance(value, TensorBundle):
        for tensor in value.data.values():
            sync(tensor)
        return
    if isinstance(value, (dict, tuple, list)):
        for item in value:
            sync(item)
        return
    try:
        keras.ops.convert_to_numpy(value)
    except (TypeError, ValueError, AttributeError):
        pass


def compile_forward(transform, backend: str):
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
                "XLA unavailable: Torch compilation requires the optional `torch_xla` package."
            )
        return torch.compile(forward, backend="openxla", fullgraph=True)
    raise RuntimeError(f"Unsupported Keras backend for compilation: {backend!r}")


def _failed_result(spec, layout, device, spatial_size, batch_size, channels, iterations, warmup, case_setup_ms, compile_mode, compile_time_ms, error):
    status = "xla-unavailable" if str(error).startswith("XLA unavailable:") else "not-xla-compatible"
    return {
        "backend": keras.config.backend(), "device": device, "layout": layout,
        "spatial_size": spatial_size, "batch_size": batch_size, "channels": channels,
        "input_shape": None, "forward_median_ms": None, "forward_p95_ms": None,
        "inverse_median_ms": None, "case_setup_ms": case_setup_ms, "case_reused": True,
        "compile_mode": compile_mode, "compile_time_ms": compile_time_ms,
        "compile_status": status, "compile_error": f"{type(error).__name__}: {error}",
        "inverse_status": status, "iterations": iterations, "warmup": warmup,
        "transform": spec.name, "group": spec.group,
    }


def profile(spec: BenchmarkSpec, layout: str, device: str, spatial_size: int, batch_size: int, channels: int, iterations: int, warmup: int, seed: int, compile_mode: str) -> dict:
    """Profile one transform case and return a JSON-serializable result."""
    transform = spec.factory(layout, seed)
    setup_start = time.perf_counter()
    template = make_case(layout, device, spatial_size, batch_size, channels, seed)
    case_setup_ms = (time.perf_counter() - setup_start) * 1000.0

    def fresh_case():
        return TensorBundle(dict(template.data), dict(template.meta))

    compiled_forward = None
    compile_time_ms = None
    compile_status = "not-requested"
    compile_error = None
    if compile_mode == "xla":
        if spec.group == "cpu":
            raise RuntimeError("Metadata-dependent transforms are not supported by --compile xla.")
        compile_start = time.perf_counter()
        try:
            compiled_forward = compile_forward(transform, keras.config.backend())
            case = fresh_case()
            sync(compiled_forward(case["image"], case["label"]))
            compile_time_ms = (time.perf_counter() - compile_start) * 1000.0
            compile_status = "compiled"
        except Exception as error:
            compile_time_ms = (time.perf_counter() - compile_start) * 1000.0
            return _failed_result(spec, layout, device, spatial_size, batch_size, channels, iterations, warmup, case_setup_ms, compile_mode, compile_time_ms, error)

    for _ in range(warmup):
        if compiled_forward is None:
            sync(transform(fresh_case()))
        else:
            case = fresh_case()
            sync(compiled_forward(case["image"], case["label"]))

    inverse_times = []
    if spec.inverse and compiled_forward is None:
        for _ in range(warmup):
            forward = transform(fresh_case())
            sync(forward)
            sync(transform.inverse(forward))

    forward_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        if compiled_forward is None:
            result = transform(fresh_case())
            sync(result)
        else:
            case = fresh_case()
            result = compiled_forward(case["image"], case["label"])
            sync(result)
        forward_times.append((time.perf_counter() - start) * 1000.0)
        if spec.inverse and compiled_forward is None:
            start = time.perf_counter()
            sync(transform.inverse(result))
            inverse_times.append((time.perf_counter() - start) * 1000.0)

    output_image = result[0] if compiled_forward is not None else result["image"]
    return {
        "backend": keras.config.backend(), "device": device, "layout": layout,
        "spatial_size": spatial_size, "batch_size": batch_size, "channels": channels,
        "input_shape": list(output_image.shape), "forward_median_ms": statistics.median(forward_times),
        "forward_p95_ms": float(np.percentile(forward_times, 95)),
        "inverse_median_ms": statistics.median(inverse_times) if inverse_times else None,
        "case_setup_ms": case_setup_ms, "case_reused": True, "compile_mode": compile_mode,
        "compile_time_ms": compile_time_ms, "compile_status": compile_status,
        "compile_error": compile_error,
        "inverse_status": "not-compiled" if spec.inverse and compile_mode == "xla" else "measured" if spec.inverse else "non-invertible",
        "iterations": iterations, "warmup": warmup, "transform": spec.name, "group": spec.group,
    }
