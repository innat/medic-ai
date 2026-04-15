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
def test_metrics_transforms_models_smoke_on_each_backend(backend):
    script = (
        """
import numpy as np
import keras
from keras import ops

from medicai.metrics import BinaryDiceMetric
from medicai.transforms import Compose
from medicai.models import DenseNet121

def as_tensor(x, dtype=None):
    return ops.convert_to_tensor(np.asarray(x), dtype=dtype)

# metric smoke
y_true = as_tensor(np.array([[[[1], [0]], [[0], [1]]]], dtype=np.float32))
y_pred = as_tensor(np.array([[[[0.9], [0.1]], [[0.2], [0.8]]]], dtype=np.float32))
metric = BinaryDiceMetric(from_logits=False, num_classes=1, ignore_empty=False)
metric.update_state(y_true, y_pred)
score = float(ops.convert_to_numpy(metric.result()))
assert np.isfinite(score)

# transform smoke
def add_one(bundle):
    bundle["image"] = bundle["image"] + 1.0
    return bundle

image = np.zeros((1, 2, 2), dtype=np.float32)
out = Compose([add_one])({"image": image})
shifted = ops.convert_to_numpy(out["image"])
assert shifted.shape == (1, 2, 2)
assert np.allclose(shifted, 1.0)

# model smoke
model = DenseNet121(input_shape=(32, 32, 1), num_classes=2)
x = as_tensor(np.random.randn(1, 32, 32, 1).astype(np.float32))
y = model(x)
assert tuple(ops.shape(y)) == (1, 2)

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
    ), f"{backend} smoke failed.\\nSTDOUT:\\n{result.stdout}\\nSTDERR:\\n{result.stderr}"
