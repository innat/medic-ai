import numpy as np
import pytest
from keras import ops

from medicai.transforms import RandomAffine, TensorBundle


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
@pytest.mark.parametrize(
    "layout, shape",
    [("HWC", (5, 6, 1)), ("DHWC", (3, 5, 6, 1)), ("BHWC", (2, 5, 6, 1)),
     ("BDHWC", (2, 3, 5, 6, 1))],
)
def test_random_affine_identity_preserves_shape_and_alignment(layout, shape):
    image = as_tensor(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    label = image * 2.0
    transform = RandomAffine(
        keys=["image", "label"],
        rotation_factor=0.0,
        zoom_factor=0.0,
        translation_factor=0.0,
        shear_factor=0.0,
        prob=1.0,
        interpolation={"image": "nearest", "label": "nearest"},
        input_layout=layout,
    )

    output = transform(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(output["image"])) == shape
    np.testing.assert_allclose(
        ops.convert_to_numpy(output["label"]),
        ops.convert_to_numpy(output["image"]) * 2.0,
    )


@pytest.mark.unit
def test_random_affine_records_one_composed_geometry():
    image = as_tensor(np.zeros((2, 6, 6, 1), dtype=np.float32))
    transform = RandomAffine(
        keys=["image"],
        rotation_factor=0.1,
        zoom_factor=0.1,
        translation_factor=0.1,
        shear_factor=0.1,
        prob=1.0,
        input_layout="BHWC",
        seed=7,
    )

    output = transform(TensorBundle({"image": image}))
    params = output.get_applied_transforms()[-1]["params"]

    assert tuple(ops.shape(params["forward_matrix"])) == (2, 3, 3)
    assert tuple(ops.shape(params["inverse_matrix"])) == (2, 3, 3)


@pytest.mark.unit
def test_random_affine_inverse_reuses_recorded_matrix():
    image = as_tensor(np.arange(2 * 7 * 7, dtype=np.float32).reshape(2, 7, 7, 1))
    transform = RandomAffine(
        keys=["image"],
        rotation_factor=0.1,
        zoom_factor=0.1,
        translation_factor=0.1,
        shear_factor=0.1,
        prob=1.0,
        interpolation="nearest",
        input_layout="BHWC",
        seed=11,
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(
        TensorBundle({"image": forward["image"]}, forward.meta)
    )

    assert tuple(ops.shape(restored["image"])) == tuple(ops.shape(image))
    assert np.isfinite(ops.convert_to_numpy(restored["image"])).all()
    assert restored.get_applied_transforms() == []


@pytest.mark.unit
def test_random_affine_accepts_trilinear_for_3d_images_and_nearest_labels():
    transform = RandomAffine(
        keys=["image", "label"],
        interpolation={"image": "trilinear", "label": "nearest"},
        input_layout="DHWC",
    )

    assert transform.interpolation == {"image": "trilinear", "label": "nearest"}


@pytest.mark.unit
def test_random_affine_uses_rank_aware_default_interpolation():
    image_2d = RandomAffine(keys=["image", "label"], input_layout="BHWC")
    image_3d = RandomAffine(keys=["image", "label"], input_layout="BDHWC")

    assert image_2d.interpolation == {"image": "bilinear", "label": "nearest"}
    assert image_3d.interpolation == {"image": "trilinear", "label": "nearest"}


@pytest.mark.unit
def test_random_affine_accepts_disabled_and_axis_specific_components():
    transform = RandomAffine(
        keys=["image"],
        rotation_factor=None,
        zoom_factor={"z": 0.1, "x": (0.0, 0.2)},
        translation_factor={"y": 0.1},
        shear_factor={"xy": 0.05, "yx": (-0.1, 0.1)},
        input_layout="DHWC",
    )

    assert transform.rotation_ranges == {
        "z": (0.0, 0.0),
        "y": (0.0, 0.0),
        "x": (0.0, 0.0),
    }
    assert transform.zoom_ranges["z"] == (-0.1, 0.1)
    assert transform.zoom_ranges["x"] == (0.0, 0.2)
    assert transform.translation_ranges["y"] == (-0.1, 0.1)
    assert transform.shear_ranges["xy"] == (-0.05, 0.05)


@pytest.mark.unit
def test_random_affine_probability_zero_records_skip_without_changing_input():
    image = as_tensor(np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6, 1))
    transform = RandomAffine(
        keys=["image"],
        rotation_factor=0.2,
        zoom_factor=0.2,
        translation_factor=0.2,
        shear_factor=0.2,
        prob=0.0,
        interpolation="nearest",
        input_layout="BHWC",
        seed=13,
    )

    output = transform(TensorBundle({"image": image}))

    np.testing.assert_array_equal(
        ops.convert_to_numpy(output["image"]), ops.convert_to_numpy(image)
    )
    assert not bool(ops.convert_to_numpy(output.get_applied_transforms()[-1]["applied"]))


@pytest.mark.unit
def test_random_affine_rejects_wrong_rank_interpolation():
    with pytest.raises(ValueError, match="Unsupported interpolation"):
        RandomAffine(keys=["image"], interpolation="bilinear", input_layout="DHWC")

    with pytest.raises(ValueError, match="Unsupported interpolation"):
        RandomAffine(keys=["image"], interpolation="trilinear", input_layout="BHWC")


@pytest.mark.unit
def test_random_affine_resolves_per_key_interpolation_and_fill_options():
    transform = RandomAffine(
        keys=["image", "label"],
        interpolation={"image": "trilinear", "label": "nearest"},
        fill_mode={"image": "reflect", "label": "constant"},
        fill_value={"image": -1.0, "label": 2.0},
        input_layout="DHWC",
    )

    assert transform.interpolation == {"image": "trilinear", "label": "nearest"}
    assert transform.fill_mode == {"image": "reflect", "label": "constant"}
    assert transform.fill_value == {"image": -1.0, "label": 2.0}


@pytest.mark.unit
def test_random_affine_allows_missing_keys_when_requested():
    transform = RandomAffine(
        keys=["image", "label"],
        input_layout="HWC",
        allow_missing_keys=True,
    )

    output = transform(TensorBundle({"image": as_tensor(np.zeros((4, 5, 1)))}))

    assert "image" in output.data
