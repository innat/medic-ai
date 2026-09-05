import numpy as np
import pytest
from keras import ops
from scipy.interpolate import BSpline

from medicai.utils.image import resample_displacement_field, resize_volumes


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


def scipy_bspline_reference(field, target_shape):
    """Evaluate the same uniform cubic B-spline control-point convention."""
    field = np.asarray(field)
    rank = len(target_shape)
    input_shape = field.shape[1 : rank + 1]
    basis = BSpline.basis_element(np.arange(-2, 3, dtype=np.float64))
    output = np.zeros((field.shape[0], *target_shape, field.shape[-1]), dtype=field.dtype)

    for batch in range(field.shape[0]):
        for output_index in np.ndindex(*target_shape):
            neighbors = []
            for axis, output_position in enumerate(output_index):
                scale = max(input_shape[axis] - 1, 0) / max(target_shape[axis] - 1, 1)
                coordinate = output_position * scale
                base = int(np.floor(coordinate))
                fraction = coordinate - base
                axis_neighbors = []
                for offset in (-1, 0, 1, 2):
                    index = np.clip(base + offset, 0, input_shape[axis] - 1)
                    weight = float(basis(fraction - offset))
                    axis_neighbors.append((index, weight))
                neighbors.append(axis_neighbors)

            for choices in np.ndindex(*(4,) * rank):
                source = tuple(
                    neighbors[axis][choice][0] for axis, choice in enumerate(choices)
                )
                weight = np.prod(
                    [neighbors[axis][choice][1] for axis, choice in enumerate(choices)]
                )
                output[(batch, *output_index)] += field[(batch, *source)] * weight
    return output


@pytest.mark.unit
def test_resize_volumes_trilinear_shape_and_dtype():
    volumes = as_tensor(np.random.default_rng(7).random((2, 4, 6, 8, 1), dtype=np.float32))
    out = resize_volumes(volumes, depth=2, height=3, width=4, method="trilinear")

    assert out.shape == (2, 2, 3, 4, 1)
    assert out.dtype == volumes.dtype


@pytest.mark.unit
def test_resize_volumes_nearest_shape_and_dtype():
    volumes = as_tensor(np.random.default_rng(7).integers(0, 10, (1, 5, 5, 5, 2), dtype=np.int32))
    out = resize_volumes(volumes, depth=3, height=4, width=2, method="nearest")

    assert out.shape == (1, 3, 4, 2, 2)
    assert out.dtype == volumes.dtype


@pytest.mark.unit
def test_resize_volumes_unsupported_method_raises():
    volumes = ops.ones((1, 2, 2, 2, 1), dtype="float32")
    with pytest.raises(ValueError, match="Unsupported resize method"):
        resize_volumes(volumes, depth=2, height=2, width=2, method="bicubic")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_shape", "target_shape"),
    [((1, 4, 5, 2), (6, 8)), ((1, 4, 5, 6, 3), (6, 8, 9))],
    ids=["2d", "3d"],
)
def test_resample_displacement_field_bspline_matches_scipy(input_shape, target_shape):
    field_np = np.random.default_rng(7).normal(size=input_shape).astype(np.float32)
    expected = scipy_bspline_reference(field_np, target_shape)

    actual = ops.convert_to_numpy(
        resample_displacement_field(
            as_tensor(field_np),
            target_shape=target_shape,
            method="bspline",
        )
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_resample_displacement_field_bilinear_supports_2d_fields():
    field = as_tensor(np.zeros((2, 4, 5, 2), dtype=np.float32))

    output = resample_displacement_field(
        field,
        target_shape=(7, 8),
        method="bilinear",
    )

    assert tuple(ops.shape(output)) == (2, 7, 8, 2)
    assert output.dtype == field.dtype
