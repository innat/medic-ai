"""Backend-specific device discovery for transform benchmarks."""

import keras


def devices(requested: str) -> list[str]:
    """Return CPU/GPU device names for the active Keras backend."""
    backend = keras.config.backend()
    if backend == "tensorflow":
        import tensorflow as tf

        logical_devices = tf.config.list_logical_devices()
        cpus = ["CPU:0"] if any(d.device_type == "CPU" for d in logical_devices) else []
        gpus = [
            f"GPU:{index}"
            for index, device in enumerate(d for d in logical_devices if d.device_type == "GPU")
        ]
    elif backend == "torch":
        import torch

        cpus = ["cpu:0"]
        gpus = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    elif backend == "jax":
        import jax

        available = jax.devices()
        cpus = ["cpu:0"] if any(d.platform == "cpu" for d in available) else []
        gpus = [
            f"gpu:{index}"
            for index, device in enumerate(d for d in available if d.platform in ("gpu", "cuda"))
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
