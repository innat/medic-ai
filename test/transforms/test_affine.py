import numpy as np
import pytest
from keras import ops

from medicai.transforms.random.affine import (
    AFFINE_AXES_2D,
    AFFINE_AXES_3D,
    centered_affine_matrix,
    compose_affine_matrices,
    invert_affine_matrix,
    normalize_resampling_options,
    resample_affine_keys,
    _sample_batched_coordinates,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_affine_axis_contract_matches_channel_last_medical_layouts():
    assert AFFINE_AXES_2D == ("y", "x")
    assert AFFINE_AXES_3D == ("z", "y", "x")


@pytest.mark.unit
def test_centered_affine_identity_and_known_scale():
    identity = as_tensor(np.eye(2, dtype=np.float32))
    np.testing.assert_allclose(
        ops.convert_to_numpy(centered_affine_matrix(identity, (5, 7))),
        np.eye(3, dtype=np.float32),
    )

    scale = as_tensor([[2.0, 0.0], [0.0, 1.0]])
    expected = np.array(
        [[2.0, 0.0, -2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(centered_affine_matrix(scale, (5, 7))),
        expected,
    )


@pytest.mark.unit
def test_centered_affine_preserves_known_translation():
    identity = as_tensor(np.eye(2, dtype=np.float32))
    translation = as_tensor([2.0, -3.0])
    expected = np.array(
        [[1.0, 0.0, 2.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        ops.convert_to_numpy(centered_affine_matrix(identity, (5, 7), translation)),
        expected,
    )


@pytest.mark.unit
def test_centered_affine_preserves_known_3d_scale_and_translation():
    linear = as_tensor(
        np.diag([2.0, 0.5, 1.5]).astype(np.float32),
    )
    translation = as_tensor([1.0, -2.0, 3.0])
    expected = np.array(
        [
            [2.0, 0.0, 0.0, -1.0],
            [0.0, 0.5, 0.0, -0.5],
            [0.0, 0.0, 1.5, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        ops.convert_to_numpy(centered_affine_matrix(linear, (5, 7, 9), translation)),
        expected,
    )


@pytest.mark.unit
def test_centered_affine_preserves_known_2d_shear():
    shear = as_tensor([[1.0, 0.2], [-0.1, 1.0]])
    expected = np.array(
        [[1.0, 0.2, -0.6], [-0.1, 1.0, 0.2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        ops.convert_to_numpy(centered_affine_matrix(shear, (5, 7))),
        expected,
        atol=1e-6,
    )


@pytest.mark.unit
def test_affine_composition_and_inverse_restore_identity():
    translation = as_tensor([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    scale = as_tensor([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
    composed = compose_affine_matrices(translation, scale)
    restored = compose_affine_matrices(composed, invert_affine_matrix(composed))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored), np.eye(3, dtype=np.float32), atol=1e-6
    )


@pytest.mark.unit
def test_resample_affine_keys_shares_matrix_and_preserves_key_options():
    data = {"image": as_tensor([1.0]), "label": as_tensor([2.0])}
    matrix = as_tensor(np.eye(3, dtype=np.float32))
    calls = []

    def sampler(tensor, received_matrix, interpolation, fill_mode, fill_value):
        calls.append((tensor, received_matrix, interpolation, fill_mode, fill_value))
        return tensor

    result = resample_affine_keys(
        data,
        keys=("image", "label"),
        matrix=matrix,
        resample=sampler,
        interpolation={"image": "bilinear", "label": "nearest"},
        fill_mode={"image": "reflect", "label": "constant"},
        fill_value={"image": 0.0, "label": 1.0},
    )

    assert result is data
    assert len(calls) == 2
    assert all(call[1] is matrix for call in calls)
    assert [(call[2], call[3], call[4]) for call in calls] == [
        ("bilinear", "reflect", 0.0),
        ("nearest", "constant", 1.0),
    ]


@pytest.mark.unit
def test_normalize_resampling_options_lowercases_and_validates_per_key_values():
    interpolation, fill_mode = normalize_resampling_options(
        keys=("image", "label"),
        interpolation={"image": "BILINEAR", "label": "NEAREST"},
        fill_mode={"image": "Reflect", "label": "CONSTANT"},
        spatial_rank=2,
        interpolation_modes={2: {"bilinear", "nearest"}},
        fill_modes={"constant", "reflect"},
    )

    assert interpolation == {"image": "bilinear", "label": "nearest"}
    assert fill_mode == {"image": "reflect", "label": "constant"}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("interpolation", {"image": "trilinear"}, "Unsupported interpolation"),
        ("fill_mode", {"image": "wrap"}, "Unsupported fill_mode"),
    ],
)
def test_normalize_resampling_options_rejects_invalid_values(option, value, message):
    options = {
        "interpolation": {"image": "bilinear"},
        "fill_mode": {"image": "constant"},
    }
    options[option] = value

    with pytest.raises(ValueError, match=message):
        normalize_resampling_options(
            keys=("image",),
            interpolation=options["interpolation"],
            fill_mode=options["fill_mode"],
            spatial_rank=2,
            interpolation_modes={2: {"bilinear", "nearest"}},
            fill_modes={"constant", "reflect"},
        )


def _numpy_boundary_reference(volume, coordinates, fill_mode, fill_value):
    result = np.empty(coordinates.shape[:-1] + (volume.shape[-1],), dtype=np.float32)
    spatial_shape = volume.shape[1:-1]
    for batch in range(volume.shape[0]):
        for output_index in np.ndindex(coordinates.shape[1:-1]):
            coordinate = coordinates[(batch,) + output_index]
            indices = []
            valid = True
            for axis, size in enumerate(spatial_shape):
                value = coordinate[axis]
                if fill_mode == "constant":
                    valid &= 0 <= value <= size - 1
                    index = np.clip(value, 0, size - 1)
                elif fill_mode == "wrap":
                    index = np.mod(value, size)
                elif fill_mode == "reflect":
                    period = 2 * size
                    reflected = np.mod(value, period)
                    index = reflected if reflected < size else period - 1 - reflected
                elif fill_mode == "mirror":
                    if size == 1:
                        index = 0
                    else:
                        period = 2 * (size - 1)
                        reflected = np.mod(abs(value), period)
                        index = reflected if reflected <= size - 1 else period - reflected
                else:
                    index = np.clip(value, 0, size - 1)
                indices.append(int(np.rint(index)))
            result[(batch,) + output_index] = (
                fill_value if fill_mode == "constant" and not valid else volume[(batch, *indices)]
            )
    return result


@pytest.mark.unit
@pytest.mark.parametrize(
    ("spatial_shape", "coordinates"),
    [
        ((3, 4), np.array([[[[-1, -1], [-1, 1]], [[1, 4], [3, 2]]]], dtype=np.float32)),
        (
            (2, 3, 4),
            np.array(
                [
                    [
                        [[[-1, -1, -1], [-1, 1, 2]], [[0, 3, 1], [2, 1, 4]]],
                        [[[2, 0, 0], [1, -1, 2]], [[0, 1, 1], [1, 2, 2]]],
                    ]
                ],
                dtype=np.float32,
            ),
        ),
    ],
    ids=["2d", "3d"],
)
@pytest.mark.parametrize("fill_mode", ["nearest", "constant", "reflect", "mirror", "wrap"])
def test_batched_affine_sampler_boundary_modes_match_numpy(spatial_shape, coordinates, fill_mode):
    volume = np.arange(np.prod((1, *spatial_shape, 1)), dtype=np.float32).reshape(
        (1, *spatial_shape, 1)
    )
    fill_value = -7.5
    actual = _sample_batched_coordinates(
        as_tensor(volume),
        as_tensor(coordinates),
        interpolation="nearest",
        fill_mode=fill_mode,
        fill_value=fill_value,
    )
    expected = _numpy_boundary_reference(volume, coordinates, fill_mode, fill_value)
    np.testing.assert_allclose(ops.convert_to_numpy(actual), expected)
