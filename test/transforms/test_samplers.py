import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from medicai.transforms.utils import (
    sample_nearest,
    sample_trilinear,
    sample_volume,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_sample_nearest_returns_exact_integer_samples():
    volume = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    coords = as_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype="float32")

    out = sample_nearest(volume, coords)

    np.testing.assert_allclose(
        ops.convert_to_numpy(out),
        np.array([[0.0], [7.0]], dtype=np.float32),
    )


@pytest.mark.unit
def test_sample_trilinear_matches_integer_corner_samples():
    volume = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    coords = as_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype="float32")

    out = sample_trilinear(volume, coords)

    np.testing.assert_allclose(
        ops.convert_to_numpy(out),
        np.array([[0.0], [7.0]], dtype=np.float32),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_sample_trilinear_interpolates_midpoint():
    volume = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    coords = as_tensor([[0.5, 0.5, 0.5]], dtype="float32")

    out = sample_trilinear(volume, coords)

    np.testing.assert_allclose(
        ops.convert_to_numpy(out), np.array([[3.5]], dtype=np.float32), rtol=1e-6
    )


@pytest.mark.unit
def test_sample_volume_uses_constant_fill_outside_bounds():
    volume = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    coords = as_tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype="float32")

    nearest = sample_volume(volume, coords, interpolation="nearest", fill_value=-5.0)
    trilinear = sample_volume(volume, coords, interpolation="trilinear", fill_value=-7.0)

    np.testing.assert_allclose(
        ops.convert_to_numpy(nearest),
        np.array([[-5.0], [-5.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(trilinear),
        np.array([[-7.0], [-7.0]], dtype=np.float32),
    )


@pytest.mark.unit
def test_sample_volume_supports_multi_channel_volumes():
    volume = as_tensor(
        np.stack(
            [
                np.arange(8, dtype=np.float32).reshape(2, 2, 2),
                np.arange(8, 16, dtype=np.float32).reshape(2, 2, 2),
            ],
            axis=-1,
        )
    )
    coords = as_tensor([[1.0, 1.0, 1.0]], dtype="float32")

    out = sample_nearest(volume, coords)

    np.testing.assert_allclose(
        ops.convert_to_numpy(out),
        np.array([[7.0, 15.0]], dtype=np.float32),
    )


@pytest.mark.unit
def test_sample_nearest_preserves_integer_dtype():
    volume = as_tensor(np.arange(8, dtype=np.int32).reshape(2, 2, 2, 1))
    coords = as_tensor([[1.0, 1.0, 1.0]], dtype="float32")

    out = sample_nearest(volume, coords)

    assert out.dtype == volume.dtype
    np.testing.assert_array_equal(ops.convert_to_numpy(out), np.array([[7]], dtype=np.int32))


@pytest.mark.unit
def test_sample_trilinear_preserves_float_dtype():
    volume = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    coords = as_tensor([[0.5, 0.5, 0.5]], dtype="float32")

    out = sample_trilinear(volume, coords)

    assert out.dtype == volume.dtype
    np.testing.assert_allclose(
        ops.convert_to_numpy(out), np.array([[3.5]], dtype=np.float32), rtol=1e-6
    )


@pytest.mark.unit
def test_sample_volume_runs_under_tf_function():
    volume = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    coords = as_tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype="float32")

    @tf.function
    def apply(volume_tensor, coord_tensor):
        return sample_volume(volume_tensor, coord_tensor, interpolation="trilinear")

    out = apply(volume, coords)

    np.testing.assert_allclose(
        ops.convert_to_numpy(out),
        np.array([[3.5], [7.0]], dtype=np.float32),
        rtol=1e-6,
    )
