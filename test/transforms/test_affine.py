import numpy as np
import pytest
from keras import ops

from medicai.transforms.random.affine import (
    AFFINE_AXES_2D,
    AFFINE_AXES_3D,
    centered_affine_matrix,
    compose_affine_matrices,
    invert_affine_matrix,
    resample_affine_keys,
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
