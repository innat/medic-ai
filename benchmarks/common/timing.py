"""Timing and backend synchronization helpers."""

import keras

from medicai.transforms import TensorBundle


def sync(value) -> None:
    """Materialize backend work, including tensors in a ``TensorBundle``."""
    if isinstance(value, TensorBundle):
        for tensor in value.data.values():
            sync(tensor)
        return
    if isinstance(value, dict):
        for item in value.values():
            sync(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            sync(item)
        return
    try:
        keras.ops.convert_to_numpy(value)
    except (TypeError, ValueError, AttributeError):
        pass
