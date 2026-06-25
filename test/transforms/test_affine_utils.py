import numpy as np
import pytest
from keras import ops

from medicai.transforms.utils import (
    affine_apply,
    build_affine,
    direction_from_affine,
    invert_affine,
    is_axis_aligned_affine,
    origin_from_affine,
    spacing_from_affine,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_spacing_from_affine_extracts_diagonal_spacing():
    affine = as_tensor(np.diag([2.0, 3.0, 4.0, 1.0]).astype(np.float32))

    spacing = spacing_from_affine(affine)

    np.testing.assert_allclose(ops.convert_to_numpy(spacing), np.array([2.0, 3.0, 4.0]))


@pytest.mark.unit
def test_direction_from_affine_extracts_normalized_columns():
    affine = as_tensor(
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

    direction = direction_from_affine(affine)

    np.testing.assert_allclose(
        ops.convert_to_numpy(direction),
        np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_origin_from_affine_extracts_translation():
    affine = as_tensor(
        np.array(
            [
                [1.0, 0.0, 0.0, 5.0],
                [0.0, 1.0, 0.0, 6.0],
                [0.0, 0.0, 1.0, 7.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    origin = origin_from_affine(affine)

    np.testing.assert_allclose(ops.convert_to_numpy(origin), np.array([5.0, 6.0, 7.0]))


@pytest.mark.unit
def test_build_affine_round_trips_spacing_direction_and_origin():
    spacing = as_tensor(np.array([2.0, 3.0, 4.0], dtype=np.float32))
    direction = as_tensor(
        np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    origin = as_tensor(np.array([10.0, 20.0, 30.0], dtype=np.float32))

    affine = build_affine(spacing, direction, origin)

    np.testing.assert_allclose(ops.convert_to_numpy(spacing_from_affine(affine)), [2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        ops.convert_to_numpy(direction_from_affine(affine)), ops.convert_to_numpy(direction)
    )
    np.testing.assert_allclose(ops.convert_to_numpy(origin_from_affine(affine)), [10.0, 20.0, 30.0])


@pytest.mark.unit
def test_affine_apply_and_invert_affine_round_trip_points():
    affine = as_tensor(
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
    points = as_tensor(np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32))

    world = affine_apply(affine, points)
    restored = affine_apply(invert_affine(affine), world)

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored), ops.convert_to_numpy(points), rtol=1e-6
    )


@pytest.mark.unit
def test_is_axis_aligned_affine_accepts_diagonal_and_rejects_permuted_axes():
    diagonal_affine = as_tensor(np.diag([2.0, -3.0, 4.0, 1.0]).astype(np.float32))
    permuted_affine = as_tensor(
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

    assert bool(ops.convert_to_numpy(is_axis_aligned_affine(diagonal_affine)))
    assert not bool(ops.convert_to_numpy(is_axis_aligned_affine(permuted_affine)))
