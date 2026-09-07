import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    Orientation,
    Spacing,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_spacing_rejects_2d_inputs_with_clear_error():
    image = as_tensor(np.random.randn(8, 8, 1).astype(np.float32))
    spacing = Spacing(keys=["image"], pixdim=(1.0, 1.0, 1.0))

    with pytest.raises(
        ValueError, match="supports only 3D channel-last tensors shaped \\(D, H, W, C\\)"
    ):
        spacing(TensorBundle({"image": image}))


@pytest.mark.unit
def test_spacing_and_orientation_accept_dhwc_input_layout_and_reject_other_layouts():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))
    affine = as_tensor(np.eye(4, dtype=np.float32))

    spacing = Spacing(
        keys=["image", "label"],
        pixdim=(1.0, 1.0, 1.0),
        input_layout="dhwc",
    )
    spaced = spacing(TensorBundle({"image": image, "label": label}, {"affine": affine}))
    assert spaced.get_applied_transforms()[-1]["params"]["input_layout"] == "DHWC"

    orientation = Orientation(keys=["image", "label"], axcodes="RAS", input_layout="DHWC")
    oriented = orientation(TensorBundle({"image": image, "label": label}, {"affine": affine}))
    assert oriented.get_applied_transforms()[-1]["params"]["input_layout"] == "DHWC"

    with pytest.raises(ValueError, match="supports only input_layout values"):
        Spacing(keys=["image"], pixdim=(1.0, 1.0, 1.0), input_layout="BHWC")

    with pytest.raises(ValueError, match="supports only input_layout values"):
        Orientation(keys=["image"], axcodes="RAS", input_layout="HWC")


@pytest.mark.unit
def test_spacing_uses_default_spacing_when_affine_missing():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    spacing = Spacing(keys=["image"], pixdim=(2.0, 2.0, 2.0))

    with pytest.warns(UserWarning, match="Affine matrix is not provided"):
        out = spacing(TensorBundle({"image": image}))

    np.testing.assert_allclose(ops.convert_to_numpy(out.meta["pixdim"]), np.array([2.0, 2.0, 2.0]))


@pytest.mark.unit
def test_spacing_validates_pixdim_and_interpolation():
    with pytest.raises(ValueError, match="`pixdim` must be 3D"):
        Spacing(keys=["image"], pixdim=(1.0, 1.0))

    with pytest.raises(ValueError, match="strictly positive"):
        Spacing(keys=["image"], pixdim=(1.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="Invalid interpolation 'bilinear' for 3D input"):
        Spacing(keys=["image"], pixdim=(1.0, 1.0, 1.0), interpolation="bilinear")


@pytest.mark.unit
def test_spacing_rejects_mapping_without_all_requested_keys():
    with pytest.raises(ValueError, match="Missing interpolation mode for keys"):
        Spacing(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            interpolation={"image": "trilinear"},
        )


@pytest.mark.unit
def test_spacing_records_trace_and_inverse_restores_original_shape():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))
    affine = as_tensor(np.eye(4, dtype=np.float32))

    spacing = Spacing(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5))
    forward = spacing(TensorBundle({"image": image, "label": label}, {"affine": affine}))
    trace = forward.get_applied_transforms()[-1]

    assert trace["name"] == "Spacing"
    assert trace["invertible"] is True
    restored = spacing.inverse(
        TensorBundle({"image": forward["image"], "label": forward["label"]}, forward.meta)
    )
    assert tuple(ops.shape(restored["image"])) == (4, 5, 6, 1)
    assert tuple(ops.shape(restored["label"])) == (4, 5, 6, 1)


@pytest.mark.unit
def test_spacing_records_static_original_shapes_as_python_lists():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    affine = as_tensor(np.eye(4, dtype=np.float32))

    spacing = Spacing(keys=["image"], pixdim=(0.5, 0.5, 0.5))
    forward = spacing(TensorBundle({"image": image}, {"affine": affine}))

    trace = forward.get_applied_transforms()[-1]

    assert trace["params"]["original_shapes"]["image"] == [4, 5, 6]


@pytest.mark.unit
def test_spacing_updates_affine_metadata_and_inverse_restores_it():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    affine = as_tensor(np.diag([2.0, 3.0, 4.0, 1.0]).astype(np.float32))

    spacing = Spacing(keys=["image"], pixdim=(1.0, 1.5, 2.0))
    forward = spacing(TensorBundle({"image": image}, {"affine": affine}))

    trace = forward.get_applied_transforms()[-1]
    np.testing.assert_allclose(
        ops.convert_to_numpy(trace["params"]["original_affine"]),
        ops.convert_to_numpy(affine),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        ops.convert_to_numpy(forward.meta["pixdim"]),
        np.array([1.0, 1.5, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(forward.meta["affine"]),
        np.diag([1.0, 1.5, 2.0, 1.0]).astype(np.float32),
        rtol=1e-6,
    )

    restored = spacing.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored.meta["pixdim"]),
        np.array([2.0, 3.0, 4.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored.meta["affine"]),
        ops.convert_to_numpy(affine),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_spacing_uses_fast_resize_path_for_axis_aligned_affine():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    affine = as_tensor(np.diag([2.0, 3.0, 4.0, 1.0]).astype(np.float32))

    spacing = Spacing(keys=["image"], pixdim=(1.0, 1.5, 2.0))
    forward = spacing(TensorBundle({"image": image}, {"affine": affine}))
    trace = forward.get_applied_transforms()[-1]

    assert trace["params"]["used_fast_resize_path"] is True
