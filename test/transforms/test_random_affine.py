import numpy as np
import pytest
import keras
from keras import ops

from medicai.transforms import RandomAffine, TensorBundle


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
@pytest.mark.parametrize(
    "layout, shape",
    [
        ("HWC", (5, 6, 1)),
        ("DHWC", (3, 5, 6, 1)),
        ("BHWC", (2, 5, 6, 1)),
        ("BDHWC", (2, 3, 5, 6, 1)),
    ],
)
def test_random_affine_identity_preserves_shape_and_alignment(layout, shape):
    image = as_tensor(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    label = image * 2.0
    transform = RandomAffine(
        keys=["image", "label"],
        rotation_factor=0.0,
        scale_factor=0.0,
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
        scale_factor=0.1,
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
@pytest.mark.parametrize(
    "layout, spatial_shape, batch_size, matrix_size",
    [
        ("HWC", (5, 6), 1, 3),
        ("DHWC", (3, 5, 6), 1, 4),
        ("BHWC", (5, 6), 2, 3),
        ("BDHWC", (3, 5, 6), 2, 4),
    ],
)
def test_random_affine_composed_matrix_inverse_restores_identity(
    layout, spatial_shape, batch_size, matrix_size
):
    transform = RandomAffine(
        keys=["image"],
        rotation_factor=0.1,
        scale_factor=0.1,
        translation_factor=0.1,
        shear_factor=0.05,
        input_layout=layout,
        prob=1.0,
        seed=17,
    )

    forward, inverse, _ = transform._matrices(spatial_shape, batch_size)
    product = ops.matmul(forward, inverse)
    expected = np.broadcast_to(
        np.eye(matrix_size, dtype=np.float32),
        (batch_size, matrix_size, matrix_size),
    )

    np.testing.assert_allclose(ops.convert_to_numpy(product), expected, atol=1e-5)


@pytest.mark.unit
def test_random_affine_probability_can_skip_individual_batch_items(monkeypatch):
    image = as_tensor(np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6, 1))
    transform = RandomAffine(
        keys=["image"],
        translation_factor=0.1,
        input_layout="BHWC",
        interpolation="nearest",
        fill_mode="wrap",
        prob=0.5,
    )

    def sample_uniform(*, shape, minval=0.0, maxval=1.0, dtype="float32"):
        if minval == 0.0 and maxval == 1.0:
            values = [0.0, 0.9]
        else:
            values = [minval, minval]
        return as_tensor(values[: shape[0]], dtype=dtype)

    monkeypatch.setattr(transform, "random_uniform", sample_uniform)
    output = transform(TensorBundle({"image": image}))

    np.testing.assert_array_equal(
        ops.convert_to_numpy(output["image"])[1],
        ops.convert_to_numpy(image)[1],
    )


@pytest.mark.unit
def test_random_affine_samples_distinct_composed_matrices_per_batch_item():
    transform = RandomAffine(
        keys=["image"],
        rotation_factor=0.2,
        scale_factor=0.2,
        translation_factor=0.2,
        shear_factor=0.1,
        input_layout="BHWC",
        prob=1.0,
        seed=17,
    )

    forward, _, _ = transform._matrices((6, 6), 2)

    assert not np.allclose(
        ops.convert_to_numpy(forward)[0],
        ops.convert_to_numpy(forward)[1],
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"translation_factor": 0.1},
            [[1.0, 0.0, -0.5], [0.0, 1.0, -0.7], [0.0, 0.0, 1.0]],
        ),
        (
            {"scale_factor": 0.1},
            [[0.9, 0.0, 0.2], [0.0, 0.9, 0.3], [0.0, 0.0, 1.0]],
        ),
        (
            {"shear_factor": 0.1},
            [[1.0, -0.1, 0.3], [-0.1, 1.0, 0.2], [0.0, 0.0, 1.0]],
        ),
    ],
    ids=["translation", "scale", "shear"],
)
def test_random_affine_records_expected_single_component_geometry(monkeypatch, kwargs, expected):
    transform = RandomAffine(
        keys=["image"],
        input_layout="HWC",
        prob=1.0,
        **kwargs,
    )

    def sample_uniform(*, shape, minval=0.0, maxval=1.0, dtype="float32"):
        value = 0.5 if (minval == 0.0 and maxval == 1.0) else minval
        return as_tensor(np.full(tuple(shape), value, dtype=np.float32), dtype=dtype)

    monkeypatch.setattr(transform, "random_uniform", sample_uniform)
    forward, _, applied = transform._matrices((5, 7), 1)

    assert bool(ops.convert_to_numpy(applied))
    np.testing.assert_allclose(
        ops.convert_to_numpy(forward[0]),
        np.asarray(expected, dtype=np.float32),
        atol=1e-6,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"translation_factor": {"z": 0.1}},
            [
                [1.0, 0.0, 0.0, -0.3],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),
        (
            {"scale_factor": {"z": 0.1}},
            [
                [0.9, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),
        (
            {"shear_factor": {"zy": 0.1}},
            [
                [1.0, -0.1, 0.0, 0.2],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),
    ],
    ids=["translation-z", "scale-z", "shear-zy"],
)
def test_random_affine_records_expected_3d_single_axis_geometry(monkeypatch, kwargs, expected):
    transform = RandomAffine(
        keys=["image"],
        input_layout="DHWC",
        prob=1.0,
        **kwargs,
    )

    def sample_uniform(*, shape, minval=0.0, maxval=1.0, dtype="float32"):
        value = 0.5 if (minval == 0.0 and maxval == 1.0) else minval
        return as_tensor(np.full(tuple(shape), value, dtype=np.float32), dtype=dtype)

    monkeypatch.setattr(transform, "random_uniform", sample_uniform)
    forward, _, applied = transform._matrices((3, 5, 7), 1)

    assert bool(ops.convert_to_numpy(applied))
    np.testing.assert_allclose(
        ops.convert_to_numpy(forward[0]),
        np.asarray(expected, dtype=np.float32),
        atol=1e-6,
    )


@pytest.mark.unit
def test_random_affine_inverse_reuses_recorded_matrix():
    image = as_tensor(np.arange(2 * 7 * 7, dtype=np.float32).reshape(2, 7, 7, 1))
    transform = RandomAffine(
        keys=["image"],
        rotation_factor=0.1,
        scale_factor=0.1,
        translation_factor=0.1,
        shear_factor=0.1,
        prob=1.0,
        interpolation="nearest",
        input_layout="BHWC",
        seed=11,
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

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
def test_random_affine_accepts_explicit_2d_z_rotation_mapping():
    transform = RandomAffine(
        keys=["image"],
        rotation_factor={"z": 0.1},
        input_layout="HWC",
    )

    assert transform.rotation_ranges == {"z": (-0.1, 0.1)}


@pytest.mark.unit
@pytest.mark.parametrize(
    "transform_type",
    [RandomAffine],
)
def test_random_affine_restores_integer_label_dtype(transform_type):
    label = as_tensor(np.arange(4 * 5, dtype=np.int32).reshape(4, 5, 1))
    transform = transform_type(
        keys=["label"],
        rotation_factor=0.0,
        scale_factor=0.0,
        translation_factor=0.0,
        shear_factor=0.0,
        interpolation="nearest",
        input_layout="HWC",
        prob=1.0,
    )

    output = transform(TensorBundle({"label": label}))

    assert output["label"].dtype == label.dtype


@pytest.mark.unit
def test_random_affine_accepts_disabled_and_axis_specific_components():
    transform = RandomAffine(
        keys=["image"],
        rotation_factor=None,
        scale_factor={"z": 0.1, "x": (0.0, 0.2)},
        translation_factor={"y": 0.1},
        shear_factor={"xy": 0.05, "yx": (-0.1, 0.1)},
        input_layout="DHWC",
    )

    assert transform.rotation_ranges == {
        "z": (0.0, 0.0),
        "y": (0.0, 0.0),
        "x": (0.0, 0.0),
    }
    assert transform.scale_ranges["z"] == (-0.1, 0.1)
    assert transform.scale_ranges["x"] == (0.0, 0.2)
    assert transform.translation_ranges["y"] == (-0.1, 0.1)
    assert transform.shear_ranges["xy"] == (-0.05, 0.05)


@pytest.mark.unit
def test_random_affine_probability_zero_records_skip_without_changing_input():
    image = as_tensor(np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6, 1))
    transform = RandomAffine(
        keys=["image"],
        rotation_factor=0.2,
        scale_factor=0.2,
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


@pytest.mark.unit
def test_random_affine_uses_plane_path_for_hw_separable_3d_geometry(monkeypatch):
    if keras.config.backend() == "torch":
        pytest.skip("Torch uses the general 3D sampler for this path.")

    image = as_tensor(np.zeros((2, 3, 5, 6, 1), dtype=np.float32))
    transform = RandomAffine(
        keys=["image"],
        rotation_factor={"z": 0.1},
        scale_factor={"x": 0.1, "y": 0.1},
        translation_factor={"x": 0.1, "y": 0.1},
        shear_factor={"xy": 0.05, "yx": 0.05},
        interpolation="trilinear",
        input_layout="BDHWC",
        seed=7,
    )

    monkeypatch.setattr(
        "medicai.transforms.random.random_affine.sample_affine_volumes",
        lambda *args, **kwargs: pytest.fail("general 3D sampler was used"),
    )
    output = transform(TensorBundle({"image": image}))

    assert tuple(ops.shape(output["image"])) == (2, 3, 5, 6, 1)


@pytest.mark.unit
def test_random_affine_general_3d_path_does_not_use_vectorized_map(monkeypatch):
    image = as_tensor(np.zeros((2, 3, 5, 6, 1), dtype=np.float32))
    transform = RandomAffine(
        keys=["image"],
        rotation_factor={"x": 0.1},
        scale_factor={"z": 0.1},
        translation_factor={"z": 0.1},
        shear_factor={"zx": 0.05},
        interpolation="trilinear",
        input_layout="BDHWC",
        seed=7,
    )

    monkeypatch.setattr(
        ops,
        "vectorized_map",
        lambda *args, **kwargs: pytest.fail("vectorized_map was used"),
    )
    output = transform(TensorBundle({"image": image}))

    assert tuple(ops.shape(output["image"])) == (2, 3, 5, 6, 1)
