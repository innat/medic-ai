"""Synthetic input cases used by transform benchmarks."""

import keras
import numpy as np

from medicai.transforms import TensorBundle


def make_case(
    layout: str,
    device: str,
    spatial_size: int,
    batch_size: int,
    channels: int,
    seed: int,
) -> TensorBundle:
    """Create an aligned image/label case on ``device``."""
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
        image = keras.ops.convert_to_tensor(
            np.random.default_rng(seed).normal(size=shape), dtype="float32"
        )
        label = keras.ops.cast(image > 0.0, "int32")
        meta = {}
        if layout in ("DHWC", "BDHWC"):
            meta["affine"] = keras.ops.eye(4, dtype="float32")
        return TensorBundle({"image": image, "label": label}, meta)
