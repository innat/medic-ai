import os
import subprocess
import sys

import pytest

BACKENDS = ("tensorflow", "torch", "jax")


def _run_backend_snippet(backend: str, snippet: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["KERAS_BACKEND"] = backend
    return subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _is_missing_backend(result: subprocess.CompletedProcess) -> bool:
    combined = f"{result.stdout}\n{result.stderr}"
    return any(
        token in combined
        for token in ["ModuleNotFoundError", "No module named", "ImportError", "cannot import name"]
    )


@pytest.mark.integration
@pytest.mark.parametrize("backend", BACKENDS)
def test_intensity_transforms_smoke_on_each_backend(backend):
    script = (
        """
import numpy as np
import keras
from keras import ops

from medicai.transforms import (
    NormalizeIntensity,
    RandomShiftIntensity,
    ScaleIntensityRange,
    ShiftIntensity,
    SignalFillEmpty,
    TensorBundle,
)

def as_tensor(x, dtype=None):
    return ops.convert_to_tensor(np.asarray(x), dtype=dtype)

image_hwc = as_tensor(
    np.array(
        [
            [[0.0], [1.0], [np.nan]],
            [[3.0], [0.0], [0.5]],
        ],
        dtype=np.float32,
    )
)

filled = SignalFillEmpty(
    keys=["image"],
    fill_value=0.0,
    input_layout="HWC",
)(TensorBundle({"image": image_hwc}))
filled_image = filled["image"]
assert ops.dtype(filled_image) == "float32"
assert np.isfinite(ops.convert_to_numpy(filled_image)).all()

normalized = NormalizeIntensity(
    keys=["image"],
    nonzero=True,
    channel_wise=True,
    input_layout="HWC",
)(TensorBundle({"image": filled_image}))
normalized_image = normalized["image"]
assert tuple(ops.shape(normalized_image)) == (2, 3, 1)
assert np.isfinite(ops.convert_to_numpy(normalized_image)).all()

scaled = ScaleIntensityRange(
    keys=["image"],
    source_value_range=(0.0, 3.0),
    target_value_range=(-1.0, 1.0),
    input_layout="HWC",
)(TensorBundle({"image": as_tensor(np.array([[[0.0], [1.5], [3.0]]], dtype=np.float32))}))
scaled_image = ops.convert_to_numpy(scaled["image"])
np.testing.assert_allclose(
    scaled_image,
    np.array([[[-1.0], [0.0], [1.0]]], dtype=np.float32),
    rtol=1e-6,
)

shift = ShiftIntensity(keys=["image"], offset=2.0, input_layout="HWC")
shifted = shift(TensorBundle({"image": as_tensor(np.ones((3, 4, 1), dtype=np.float32))}))
restored = shift.inverse(TensorBundle({"image": shifted["image"]}, shifted.meta))
np.testing.assert_allclose(
    ops.convert_to_numpy(restored["image"]),
    np.ones((3, 4, 1), dtype=np.float32),
    rtol=1e-6,
)

random_shift_a = RandomShiftIntensity(
    keys=["image"],
    offset=0.5,
    prob=1.0,
    channel_wise=True,
    seed=17,
    input_layout="HWC",
)(TensorBundle({"image": as_tensor(np.ones((3, 4, 2), dtype=np.float32))}))
random_shift_b = RandomShiftIntensity(
    keys=["image"],
    offset=0.5,
    prob=1.0,
    channel_wise=True,
    seed=17,
    input_layout="HWC",
)(TensorBundle({"image": as_tensor(np.ones((3, 4, 2), dtype=np.float32))}))

np.testing.assert_allclose(
    ops.convert_to_numpy(random_shift_a["image"]),
    ops.convert_to_numpy(random_shift_b["image"]),
    rtol=1e-6,
)

random_restored = RandomShiftIntensity(
    keys=["image"],
    offset=0.5,
    prob=1.0,
    channel_wise=True,
    seed=19,
    input_layout="HWC",
)
forward = random_restored(TensorBundle({"image": as_tensor(np.ones((3, 4, 2), dtype=np.float32))}))
inverse = random_restored.inverse(TensorBundle({"image": forward["image"]}, forward.meta))
np.testing.assert_allclose(
    ops.convert_to_numpy(inverse["image"]),
    np.ones((3, 4, 2), dtype=np.float32),
    rtol=1e-6,
)

assert keras.backend.backend() == '"""
        + backend
        + """'
"""
    )

    result = _run_backend_snippet(backend, script)
    if result.returncode != 0 and _is_missing_backend(result):
        pytest.skip(f"{backend} backend runtime not installed in this environment.")

    assert (
        result.returncode == 0
    ), f"{backend} intensity transform smoke failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
