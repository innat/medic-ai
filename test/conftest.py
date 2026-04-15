import numpy as np
import pytest
import keras


@pytest.fixture(autouse=True)
def deterministic_seed():
    np.random.seed(7)
    keras.utils.set_random_seed(7)


def _gpu_available_for_backend() -> bool:
    backend = keras.backend.backend()
    try:
        if backend == "tensorflow":
            import tensorflow as tf

            return len(tf.config.list_physical_devices("GPU")) > 0
        if backend == "torch":
            import torch

            return bool(torch.cuda.is_available())
        if backend == "jax":
            import jax

            return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False
    return False


def pytest_runtest_setup(item):
    if "gpu" in item.keywords and not _gpu_available_for_backend():
        pytest.skip("GPU not available for current Keras backend.")
