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
    missing_signals = [
        "ModuleNotFoundError",
        "No module named",
        "cannot import name",
        "ImportError",
    ]
    combined = f"{result.stdout}\n{result.stderr}"
    return any(signal in combined for signal in missing_signals)


@pytest.mark.integration
@pytest.mark.parametrize("backend", BACKENDS)
def test_losses_and_metrics_smoke_on_each_backend(backend):
    smoke_script = """
import numpy as np
import keras
from keras import ops
from medicai.losses import BinaryDiceLoss, SparseDiceCELoss
from medicai.metrics import SparseDiceMetric

y_true_sparse = ops.convert_to_tensor(
    np.array([[[[0], [1]], [[1], [2]]]], dtype=np.int32), dtype="int32"
)
y_pred_probs = ops.convert_to_tensor(
    np.array(
        [[
            [[0.97, 0.02, 0.01], [0.02, 0.97, 0.01]],
            [[0.01, 0.98, 0.01], [0.01, 0.01, 0.98]],
        ]],
        dtype=np.float32,
    ),
    dtype="float32",
)

y_true_binary = ops.convert_to_tensor(
    np.array([[[[1], [0]], [[0], [1]]]], dtype=np.float32), dtype="float32"
)
y_pred_binary = ops.convert_to_tensor(
    np.array([[[[0.9], [0.1]], [[0.2], [0.8]]]], dtype=np.float32), dtype="float32"
)

loss_sparse = SparseDiceCELoss(from_logits=False, num_classes=3)(y_true_sparse, y_pred_probs)
loss_binary = BinaryDiceLoss(from_logits=False, num_classes=1)(y_true_binary, y_pred_binary)

metric = SparseDiceMetric(from_logits=False, num_classes=3, ignore_empty=False)
metric.update_state(y_true_sparse, y_pred_probs)
score = metric.result()

for value in [loss_sparse, loss_binary, score]:
    val = float(ops.convert_to_numpy(value))
    assert np.isfinite(val)

assert keras.backend.backend() == '""" + backend + """'
"""

    result = _run_backend_snippet(backend, smoke_script)
    if result.returncode != 0 and _is_missing_backend(result):
        pytest.skip(f"{backend} backend runtime not installed in this environment.")

    assert result.returncode == 0, (
        f"{backend} backend smoke failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

