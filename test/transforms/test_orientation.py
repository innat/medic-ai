import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    Orientation,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_orientation_rejects_2d_inputs_with_clear_error():
    image = as_tensor(np.random.randn(8, 8, 1).astype(np.float32))
    affine = as_tensor(np.eye(4, dtype=np.float32))
    orientation = Orientation(keys=["image"], axcodes="RAS")

    with pytest.raises(
        ValueError,
        match="supports only 3D channel-last tensors shaped \\(D, H, W, C\\)",
    ):
        orientation(TensorBundle({"image": image}, {"affine": affine}))


@pytest.mark.unit
def test_orientation_requires_affine_and_valid_axcodes():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))

    with pytest.raises(ValueError, match="axcodes must be a 3-character string"):
        Orientation(keys=["image"], axcodes="RA")

    with pytest.raises(ValueError, match="R/L/A/P/S/I"):
        Orientation(keys=["image"], axcodes="XYZ")

    with pytest.raises(ValueError, match="exactly one code from each anatomical axis family"):
        Orientation(keys=["image"], axcodes="RRS")

    with pytest.raises(ValueError, match="Affine matrix is required"):
        Orientation(keys=["image"], axcodes="RAS")(TensorBundle({"image": image}))


@pytest.mark.unit
def test_orientation_rejects_non_4x4_affine():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    affine = as_tensor(np.eye(3, dtype=np.float32))
    orientation = Orientation(keys=["image"], axcodes="RAS")

    with pytest.raises(ValueError, match="Expected a 4x4 affine matrix"):
        orientation(TensorBundle({"image": image}, {"affine": affine}))


@pytest.mark.unit
def test_orientation_allow_missing_keys_returns_bundle_unchanged():
    affine = as_tensor(np.eye(4, dtype=np.float32))
    orientation = Orientation(keys=["image"], axcodes="RAS", allow_missing_keys=True)
    bundle = TensorBundle(
        {"other": as_tensor(np.ones((4, 5, 6, 1), dtype=np.float32))}, {"affine": affine}
    )

    out = orientation(bundle)

    assert out is bundle


@pytest.mark.unit
def test_orientation_records_trace_and_inverse_restores_shape():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))
    affine = as_tensor(np.eye(4, dtype=np.float32))

    orientation = Orientation(keys=["image", "label"], axcodes="RAS")
    forward = orientation(TensorBundle({"image": image, "label": label}, {"affine": affine}))
    trace = forward.get_applied_transforms()[-1]

    assert trace["name"] == "Orientation"
    assert trace["invertible"] is True
    assert trace["params"]["target_tensor_axcodes"] == "SAR"
    restored = orientation.inverse(
        TensorBundle({"image": forward["image"], "label": forward["label"]}, forward.meta)
    )
    assert tuple(ops.shape(restored["image"])) == (4, 5, 6, 1)
    assert tuple(ops.shape(restored["label"])) == (4, 5, 6, 1)


@pytest.mark.unit
def test_orientation_flip_only_restores_original_layout_and_affine():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    affine = as_tensor(np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32))

    orientation = Orientation(keys=["image"], axcodes="RAS")
    forward = orientation(TensorBundle({"image": image}, {"affine": affine}))

    trace = forward.get_applied_transforms()[-1]
    assert trace["params"]["target_tensor_axcodes"] == "SAR"
    np.testing.assert_allclose(
        ops.convert_to_numpy(trace["params"]["original_affine"]),
        ops.convert_to_numpy(affine),
        rtol=1e-6,
    )

    restored = orientation.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (4, 5, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["affine"]),
        np.diag([-1.0, 1.0, 1.0, 1.0]),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_orientation_supports_multiple_flip_axes():
    image = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))
    affine = as_tensor(np.eye(4, dtype=np.float32))

    orientation = Orientation(keys=["image"], axcodes="LPI")
    forward = orientation(TensorBundle({"image": image}, {"affine": affine}))

    # ``LPI`` maps to tensor-axis order ``IPL``, so the spatial axes are
    # permuted from ``(D, H, W)`` to ``(W, H, D)`` before flipping.
    assert tuple(ops.shape(forward["image"])) == (4, 3, 2, 1)
    restored = orientation.inverse(TensorBundle({"image": forward["image"]}, forward.meta))
    np.testing.assert_array_equal(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_orientation_permutation_changes_spatial_order_and_inverse_restores():
    image = as_tensor(np.random.randn(2, 3, 4, 1).astype(np.float32))
    affine = as_tensor(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    orientation = Orientation(keys=["image"], axcodes="RAS")
    forward = orientation(TensorBundle({"image": image}, {"affine": affine}))

    assert tuple(ops.shape(forward["image"])) == (2, 4, 3, 1)

    restored = orientation.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (2, 3, 4, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["affine"]),
        ops.convert_to_numpy(affine),
        rtol=1e-6,
    )
