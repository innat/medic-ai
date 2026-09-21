import numpy as np
import pytest
import keras
from keras import ops

from medicai.transforms import RandomScale, TensorBundle


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_random_scale_preserves_shape_and_aligns_selected_keys():
    image = as_tensor(np.arange(2 * 8 * 9, dtype=np.float32).reshape(2, 8, 9, 1))
    label = image * 2.0
    transform = RandomScale(
        keys=["image", "label"],
        factor=0.0,
        prob=1.0,
        input_layout="BHWC",
        interpolation={"image": "nearest", "label": "nearest"},
    )

    output = transform(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(output["image"])) == (2, 8, 9, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(output["label"]),
        ops.convert_to_numpy(output["image"]) * 2.0,
    )


@pytest.mark.unit
def test_random_scale_accepts_3d_axis_factors():
    image = as_tensor(np.zeros((2, 3, 4, 5, 1), dtype=np.float32))
    transform = RandomScale(
        keys=["image"],
        factor={"z": 0.1, "x": 0.2, "y": 0.3},
        prob=1.0,
        input_layout="BDHWC",
    )

    output = transform(TensorBundle({"image": image}))

    assert tuple(ops.shape(output["image"])) == (2, 3, 4, 5, 1)


@pytest.mark.unit
def test_random_scale_resolves_rank_aware_defaults_and_per_key_options():
    transform = RandomScale(
        keys=["image", "label"],
        factor=(0.0, 0.2),
        interpolation={"image": "trilinear", "label": "nearest"},
        fill_mode={"image": "reflect", "label": "constant"},
        fill_value={"image": -1.0, "label": 2.0},
        input_layout="DHWC",
    )

    assert transform.interpolation == {"image": "trilinear", "label": "nearest"}
    assert transform.fill_mode == {"image": "reflect", "label": "constant"}
    assert transform.fill_value == {"image": -1.0, "label": 2.0}
    assert transform.ranges["z"] == (0.0, 0.2)


@pytest.mark.unit
def test_random_scale_rejects_wrong_rank_interpolation_and_unknown_axis():
    with pytest.raises(ValueError, match="Unsupported interpolation"):
        RandomScale(keys=["image"], factor=0.1, interpolation="bilinear", input_layout="DHWC")

    with pytest.raises(ValueError, match="Scale factor axes"):
        RandomScale(keys=["image"], factor={"invalid": 0.1}, input_layout="HWC")


@pytest.mark.unit
def test_random_scale_allows_missing_keys():
    transform = RandomScale(
        keys=["image", "label"],
        factor=0.0,
        input_layout="HWC",
        allow_missing_keys=True,
    )
    output = transform(TensorBundle({"image": as_tensor(np.zeros((4, 5, 1)))}))

    assert "image" in output.data


@pytest.mark.unit
def test_random_scale_uses_plane_path_for_xy_only_3d_scale(monkeypatch):
    if keras.config.backend() == "torch":
        pytest.skip("Torch uses the general 3D sampler for this path.")

    image = as_tensor(np.zeros((2, 3, 5, 6, 1), dtype=np.float32))
    label = image + 1.0
    transform = RandomScale(
        keys=["image", "label"],
        factor={"x": 0.1, "y": 0.1},
        prob=1.0,
        input_layout="BDHWC",
        seed=7,
    )

    monkeypatch.setattr(
        "medicai.transforms.random.random_scale.sample_affine_volumes",
        lambda *args, **kwargs: pytest.fail("general 3D sampler was used"),
    )
    output = transform(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(output["image"])) == (2, 3, 5, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(output["label"]),
        ops.convert_to_numpy(output["image"]) + 1.0,
    )
