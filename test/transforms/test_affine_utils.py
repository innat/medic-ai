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
    validate_affine_matrix,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_spacing_from_affine_extracts_diagonal_spacing():
    affine = as_tensor(np.diag([2.0, 3.0, 4.0, 1.0]).astype(np.float32))

    spacing = spacing_from_affine(affine)

    np.testing.assert_allclose(ops.convert_to_numpy(spacing), np.array([2.0, 3.0, 4.0]))


@pytest.mark.unit
def test_spacing_from_affine_extracts_column_norms_from_non_diagonal_affine():
    direction = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    spacing = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    affine = np.eye(4, dtype=np.float32)
    affine[:3, :3] = direction * spacing[None, :]

    extracted = spacing_from_affine(as_tensor(affine))

    np.testing.assert_allclose(ops.convert_to_numpy(extracted), spacing, rtol=1e-6)


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


@pytest.mark.unit
@pytest.mark.parametrize(
    "off_diagonal, expected",
    [(2e-6, True), (1e-3, False)],
    ids=["below_atol", "above_atol"],
)
def test_is_axis_aligned_affine_respects_atol(off_diagonal, expected):
    affine = np.diag([2.0, 3.0, 4.0, 1.0]).astype(np.float32)
    affine[0, 1] = off_diagonal

    result = is_axis_aligned_affine(as_tensor(affine), atol=1e-5)

    assert bool(ops.convert_to_numpy(result)) is expected


@pytest.mark.unit
def test_spacing_from_affine_rejects_non_4x4_matrices():
    affine = as_tensor(np.eye(3, dtype=np.float32))

    with pytest.raises(ValueError, match="Expected a 4x4 affine matrix"):
        spacing_from_affine(affine)


@pytest.mark.unit
@pytest.mark.parametrize(
    "shape",
    [(3, 3), (4, 3), (3, 4), (4,), ()],
    ids=["3x3", "4x3", "3x4", "rank_1", "scalar"],
)
def test_validate_affine_matrix_rejects_malformed_shapes(shape):
    affine = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="Expected a 4x4 affine matrix"):
        validate_affine_matrix(affine)


@pytest.mark.unit
def test_validate_affine_matrix_accepts_plain_numpy_input():
    affine = np.eye(4, dtype=np.float32)

    validated = validate_affine_matrix(affine)

    np.testing.assert_allclose(ops.convert_to_numpy(validated), affine)


@pytest.mark.unit
def test_validate_affine_matrix_normalizes_dtype_to_float32():
    affine = np.eye(4, dtype=np.float64)

    validated = validate_affine_matrix(affine)

    assert ops.convert_to_numpy(validated).dtype == np.float32


@pytest.mark.unit
def test_affine_apply_supports_batched_points():
    affine = as_tensor(
        np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    point_values = np.array(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        dtype=np.float32,
    )
    points = as_tensor(point_values)

    transformed = affine_apply(affine, points)
    expected = point_values + np.array([10.0, 20.0, 30.0], dtype=np.float32)

    np.testing.assert_allclose(ops.convert_to_numpy(transformed), expected, rtol=1e-6)
