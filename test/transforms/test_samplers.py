import numpy as np
import pytest
from keras import ops

from medicai.transforms.utils import (
    sample_nearest,
    sample_trilinear,
    sample_volume,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


def as_numpy(tensor, dtype=None):
    return np.asarray(ops.convert_to_numpy(tensor), dtype=dtype)


@pytest.mark.unit
def test_sample_nearest_returns_exact_integer_samples():
    volume = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    coords = as_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype="float32")

    out = sample_nearest(volume, coords)

    np.testing.assert_allclose(
        as_numpy(out),
        np.array([[0.0], [7.0]], dtype=np.float32),
    )


@pytest.mark.unit
def test_sample_trilinear_matches_integer_corner_samples():
    volume = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    coords = as_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype="float32")

    out = sample_trilinear(volume, coords)

    np.testing.assert_allclose(
        as_numpy(out),
        np.array([[0.0], [7.0]], dtype=np.float32),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_sample_trilinear_interpolates_midpoint():
    volume = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    coords = as_tensor([[0.5, 0.5, 0.5]], dtype="float32")

    out = sample_trilinear(volume, coords)

    np.testing.assert_allclose(as_numpy(out), np.array([[3.5]], dtype=np.float32), rtol=1e-6)


@pytest.mark.unit
def test_sample_volume_uses_constant_fill_outside_bounds():
    volume = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    coords = as_tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype="float32")

    nearest = sample_volume(volume, coords, interpolation="nearest", fill_value=-5.0)
    trilinear = sample_volume(volume, coords, interpolation="trilinear", fill_value=-7.0)

    np.testing.assert_allclose(
        as_numpy(nearest),
        np.array([[-5.0], [-5.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        as_numpy(trilinear),
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
        as_numpy(out),
        np.array([[7.0, 15.0]], dtype=np.float32),
    )


@pytest.mark.unit
def test_sample_nearest_preserves_integer_dtype():
    volume = as_tensor(np.arange(8, dtype=np.int32).reshape(2, 2, 2, 1))
    coords = as_tensor([[1.0, 1.0, 1.0]], dtype="float32")

    out = sample_nearest(volume, coords)

    assert out.dtype == volume.dtype
    np.testing.assert_array_equal(as_numpy(out), np.array([[7]], dtype=np.int32))


@pytest.mark.unit
def test_sample_trilinear_preserves_float_dtype():
    volume = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    coords = as_tensor([[0.5, 0.5, 0.5]], dtype="float32")

    out = sample_trilinear(volume, coords)

    assert out.dtype == volume.dtype
    np.testing.assert_allclose(as_numpy(out), np.array([[3.5]], dtype=np.float32), rtol=1e-6)


@pytest.mark.unit
def test_sample_volume_rejects_unsupported_interpolation_and_padding():
    volume = as_tensor(np.zeros((2, 2, 2, 1), dtype=np.float32))
    coords = as_tensor([[0.0, 0.0, 0.0]], dtype="float32")

    with pytest.raises(ValueError, match="Unsupported interpolation"):
        sample_volume(volume, coords, interpolation="bilinear")

    with pytest.raises(ValueError, match="Unsupported padding_mode"):
        sample_volume(volume, coords, interpolation="nearest", padding_mode="border")


@pytest.mark.unit
def test_sample_nearest_validates_volume_and_coordinate_shapes():
    coords = as_tensor([[0.0, 0.0, 0.0]], dtype="float32")

    with pytest.raises(ValueError, match="Expected a 4D channel-last volume"):
        sample_nearest(ops.zeros((2, 2, 1), dtype="float32"), coords)

    volume = ops.zeros((2, 2, 2, 1), dtype="float32")
    with pytest.raises(ValueError, match=r"Expected coords shaped \(N, 3\)"):
        sample_nearest(volume, ops.zeros((1, 2), dtype="float32"))
