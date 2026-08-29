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
        import torch

        # This is the compiler used by Keras when Torch models are compiled.
        return torch.compile(forward, backend="inductor", fullgraph=True)
    raise RuntimeError(f"Unsupported Keras backend for compilation: {backend!r}")
