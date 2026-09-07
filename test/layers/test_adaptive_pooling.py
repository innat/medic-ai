import numpy as np
import pytest
from keras import ops

from medicai.layers import (
    AdaptiveAveragePooling2D,
    AdaptiveAveragePooling3D,
    AdaptiveMaxPooling2D,
    AdaptiveMaxPooling3D,
)


def require_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("Torch is required for the reference comparison.")
    return torch


@pytest.mark.unit
def test_adaptive_pooling_2d_matches_torch_reference():
    torch = require_torch()
    rng = np.random.default_rng(7)
    torch_input = torch.from_numpy(rng.standard_normal((1, 64, 8, 9)).astype(np.float32))
    keras_input = torch_input.numpy().transpose(0, 2, 3, 1)
    pool_size = (5, 7)

    torch_max = torch.nn.AdaptiveMaxPool2d(pool_size)(torch_input)
    keras_max = AdaptiveMaxPooling2D(output_size=pool_size)(keras_input)
    np.testing.assert_allclose(
        ops.convert_to_numpy(keras_max),
        torch_max.numpy().transpose(0, 2, 3, 1),
        atol=1e-6,
        rtol=1e-6,
    )

    torch_average = torch.nn.AdaptiveAvgPool2d(pool_size)(torch_input)
    keras_average = AdaptiveAveragePooling2D(output_size=pool_size)(keras_input)
    np.testing.assert_allclose(
        ops.convert_to_numpy(keras_average),
        torch_average.numpy().transpose(0, 2, 3, 1),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.unit
def test_adaptive_pooling_3d_matches_torch_reference():
    torch = require_torch()
    rng = np.random.default_rng(7)
    torch_input = torch.from_numpy(rng.standard_normal((1, 64, 8, 9, 10)).astype(np.float32))
    keras_input = torch_input.numpy().transpose(0, 2, 3, 4, 1)
    pool_size = (5, 7, 9)

    torch_max = torch.nn.AdaptiveMaxPool3d(pool_size)(torch_input)
    keras_max = AdaptiveMaxPooling3D(output_size=pool_size)(keras_input)
    np.testing.assert_allclose(
        ops.convert_to_numpy(keras_max),
        torch_max.numpy().transpose(0, 2, 3, 4, 1),
        atol=1e-6,
        rtol=1e-6,
    )

    torch_average = torch.nn.AdaptiveAvgPool3d(pool_size)(torch_input)
    keras_average = AdaptiveAveragePooling3D(output_size=pool_size)(keras_input)
    np.testing.assert_allclose(
        ops.convert_to_numpy(keras_average),
        torch_average.numpy().transpose(0, 2, 3, 4, 1),
        atol=1e-6,
        rtol=1e-6,
    )
