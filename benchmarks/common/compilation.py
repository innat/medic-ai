"""Backend compilation adapters for benchmarks."""

import keras

from medicai.transforms import TensorBundle


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
