import numpy as np
import pytest
from keras import ops

from medicai.transforms.utils import (
    compute_destination_affine,
    compute_output_shape,
    direction_from_affine,
    make_output_grid,
    origin_from_affine,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_compute_destination_affine_preserves_direction_and_origin_by_default():
    src_affine = as_tensor(
        np.array(
            [
                [0.0, 0.0, -4.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [2.0, 0.0, 0.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    dst_affine = compute_destination_affine(src_affine, pixdim=as_tensor([1.0, 1.5, 2.0]))

    np.testing.assert_allclose(
        ops.convert_to_numpy(direction_from_affine(dst_affine)),
        ops.convert_to_numpy(direction_from_affine(src_affine)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(origin_from_affine(dst_affine)),
        ops.convert_to_numpy(origin_from_affine(src_affine)),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_compute_destination_affine_supports_diagonal_output():
    src_affine = as_tensor(
        np.array(
            [
                [0.0, 0.0, -4.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [2.0, 0.0, 0.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    dst_affine = compute_destination_affine(
        src_affine,
        pixdim=as_tensor([1.0, 1.5, 2.0]),
        diagonal=True,
    )

    np.testing.assert_allclose(
        ops.convert_to_numpy(dst_affine),
        np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.5, 0.0, 20.0],
                [0.0, 0.0, 2.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_compute_output_shape_matches_identity_destination_affine():
    src_affine = as_tensor(np.diag([2.0, 3.0, 4.0, 1.0]).astype(np.float32))
    input_shape = as_tensor([4, 5, 6], dtype="int32")

    output_shape = compute_output_shape(
        input_shape=input_shape,
        src_affine=src_affine,
        dst_affine=src_affine,
    )

    np.testing.assert_array_equal(ops.convert_to_numpy(output_shape), np.array([4, 5, 6]))


@pytest.mark.unit
def test_compute_output_shape_scales_isotropic_spacing_change():
    src_affine = as_tensor(np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float32))
    dst_affine = as_tensor(np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32))
    input_shape = as_tensor([4, 5, 6], dtype="int32")

    output_shape = compute_output_shape(
        input_shape=input_shape,
        src_affine=src_affine,
        dst_affine=dst_affine,
    )

    np.testing.assert_array_equal(ops.convert_to_numpy(output_shape), np.array([8, 10, 12]))


@pytest.mark.unit
def test_compute_output_shape_scales_anisotropic_spacing_change():
    src_affine = as_tensor(np.diag([2.0, 3.0, 4.0, 1.0]).astype(np.float32))
    dst_affine = as_tensor(np.diag([1.0, 1.5, 2.0, 1.0]).astype(np.float32))
    input_shape = as_tensor([4, 5, 6], dtype="int32")

    output_shape = compute_output_shape(
        input_shape=input_shape,
        src_affine=src_affine,
        dst_affine=dst_affine,
    )

    np.testing.assert_array_equal(ops.convert_to_numpy(output_shape), np.array([8, 10, 12]))


@pytest.mark.unit
def test_compute_output_shape_uses_half_up_rounding_for_half_voxels():
    src_affine = as_tensor(np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32))
    dst_affine = as_tensor(np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float32))
    input_shape = as_tensor([301, 5, 6], dtype="int32")

    output_shape = compute_output_shape(
        input_shape=input_shape,
        src_affine=src_affine,
        dst_affine=dst_affine,
    )

    np.testing.assert_array_equal(ops.convert_to_numpy(output_shape), np.array([151, 3, 3]))


@pytest.mark.unit
def test_compute_output_shape_align_corners_uses_shape_minus_one_extent():
    src_affine = as_tensor(np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float32))
    dst_affine = as_tensor(np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32))
    input_shape = as_tensor([4, 5, 6], dtype="int32")

    output_shape = compute_output_shape(
        input_shape=input_shape,
        src_affine=src_affine,
        dst_affine=dst_affine,
        align_corners=True,
    )

    np.testing.assert_array_equal(ops.convert_to_numpy(output_shape), np.array([7, 9, 11]))


@pytest.mark.unit
def test_make_output_grid_returns_expected_shape_and_axis_order():
    grid = make_output_grid(as_tensor([2, 3, 4], dtype="int32"))

    assert tuple(ops.shape(grid)) == (24, 3)
    np.testing.assert_allclose(
        ops.convert_to_numpy(grid[:6]),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
    )
