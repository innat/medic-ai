import numpy as np
import pytest
from keras import ops
from medicai.transforms.random import random_elastic_transform as elastic_module

from medicai.transforms import (
    RandomElasticTransform,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_random_elastic_transform_supports_2d_sample_and_batch_layouts():
    image = as_tensor(np.arange(20, dtype=np.float32).reshape(4, 5, 1))
    mask = as_tensor((np.arange(20).reshape(4, 5, 1) > 10).astype(np.int32))
    transform = RandomElasticTransform(
        keys=["image", "mask"],
        alpha=1.0,
        sigma=1.0,
        interpolation={"image": "bilinear", "mask": "nearest"},
        prob=1.0,
        input_layout="HWC",
        seed=7,
    )

    sample = transform(TensorBundle({"image": image, "mask": mask}))
    assert tuple(ops.shape(sample["image"])) == (4, 5, 1)
    assert tuple(ops.shape(sample["mask"])) == (4, 5, 1)
    assert set(np.unique(ops.convert_to_numpy(sample["mask"]))).issubset({0, 1})

    batch_transform = RandomElasticTransform(
        keys=["image", "mask"],
        alpha=1.0,
        sigma=1.0,
        interpolation={"image": "bilinear", "mask": "nearest"},
        prob=1.0,
        input_layout="BHWC",
        seed=7,
    )
    batch = batch_transform(
        TensorBundle({"image": ops.stack([image, image]), "mask": ops.stack([mask, mask])})
    )
    assert tuple(ops.shape(batch["image"])) == (2, 4, 5, 1)
    assert tuple(ops.shape(batch["mask"])) == (2, 4, 5, 1)


@pytest.mark.unit
def test_random_elastic_transform_supports_3d_sample_and_batch_layouts():
    image = as_tensor(np.arange(60, dtype=np.float32).reshape(3, 4, 5, 1))
    mask = as_tensor((np.arange(60).reshape(3, 4, 5, 1) > 30).astype(np.int32))
    config = dict(
        keys=["image", "mask"],
        alpha=0.5,
        sigma=1.0,
        control_grid_spacing=(2, 2, 2),
        interpolation={"image": "trilinear", "mask": "nearest"},
        prob=1.0,
        seed=7,
    )

    sample = RandomElasticTransform(input_layout="DHWC", **config)(
        TensorBundle({"image": image, "mask": mask})
    )
    assert tuple(ops.shape(sample["image"])) == (3, 4, 5, 1)
    assert tuple(ops.shape(sample["mask"])) == (3, 4, 5, 1)
    assert set(np.unique(ops.convert_to_numpy(sample["mask"]))).issubset({0, 1})

    batch = RandomElasticTransform(input_layout="BDHWC", **config)(
        TensorBundle({"image": ops.stack([image, image]), "mask": ops.stack([mask, mask])})
    )
    assert tuple(ops.shape(batch["image"])) == (2, 3, 4, 5, 1)
    assert tuple(ops.shape(batch["mask"])) == (2, 3, 4, 5, 1)


@pytest.mark.unit
def test_random_elastic_transform_bspline_coarse_field_keeps_aligned_keys():
    image_np = np.arange(64, dtype=np.float32).reshape(4, 4, 4, 1)
    transform = RandomElasticTransform(
        keys=["image", "label"],
        input_layout="DHWC",
        alpha=1.0,
        sigma=1.0,
        control_grid_spacing=(2, 2, 2),
        field_interpolation="bspline",
        interpolation={"image": "nearest", "label": "nearest"},
        prob=1.0,
        seed=7,
    )

    result = transform(TensorBundle({"image": as_tensor(image_np), "label": as_tensor(image_np)}))

    assert tuple(ops.shape(result["image"])) == (4, 4, 4, 1)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(result["image"]),
        ops.convert_to_numpy(result["label"]),
    )


@pytest.mark.unit
def test_random_elastic_transform_bspline_coarse_field_respects_locked_borders():
    image = as_tensor(np.arange(125, dtype=np.float32).reshape(5, 5, 5, 1))
    transform = RandomElasticTransform(
        keys=["image"],
        input_layout="DHWC",
        alpha=2.0,
        sigma=1.0,
        control_grid_spacing=(2, 2, 2),
        field_interpolation="bspline",
        interpolation="nearest",
        locked_borders=1,
        prob=1.0,
        seed=7,
    )

    output = ops.convert_to_numpy(transform(TensorBundle({"image": image}))["image"])
    original = ops.convert_to_numpy(image)

    np.testing.assert_array_equal(output[0], original[0])
    np.testing.assert_array_equal(output[-1], original[-1])
    np.testing.assert_array_equal(output[:, 0], original[:, 0])
    np.testing.assert_array_equal(output[:, -1], original[:, -1])
    np.testing.assert_array_equal(output[:, :, 0], original[:, :, 0])
    np.testing.assert_array_equal(output[:, :, -1], original[:, :, -1])


@pytest.mark.unit
def test_random_elastic_transform_bspline_coarse_field_replays_seed_sequence():
    image = as_tensor(np.arange(64, dtype=np.float32).reshape(4, 4, 4, 1))
    config = dict(
        keys=["image"],
        input_layout="DHWC",
        alpha=1.0,
        sigma=1.0,
        control_grid_spacing=(2, 2, 2),
        field_interpolation="bspline",
        interpolation="nearest",
        prob=1.0,
        seed=101,
    )

    first = RandomElasticTransform(**config)(TensorBundle({"image": image}))
    second = RandomElasticTransform(**config)(TensorBundle({"image": image}))

    np.testing.assert_array_equal(
        ops.convert_to_numpy(first["image"]),
        ops.convert_to_numpy(second["image"]),
    )


@pytest.mark.unit
def test_random_elastic_transform_identity_matches_numpy_reference():
    image_np = np.arange(27, dtype=np.float32).reshape(3, 3, 3, 1)
    label_np = (image_np > 13).astype(np.int32)
    transform = RandomElasticTransform(
        keys=["image", "label"],
        alpha=0.0,
        input_layout="DHWC",
        prob=1.0,
        seed=7,
    )

    result = transform(
        TensorBundle(
            {
                "image": as_tensor(image_np),
                "label": as_tensor(label_np),
            }
        )
    )

    np.testing.assert_array_equal(ops.convert_to_numpy(result["image"]), image_np)
    np.testing.assert_array_equal(ops.convert_to_numpy(result["label"]), label_np)
    assert transform.interpolation == {"image": "trilinear", "label": "nearest"}


@pytest.mark.unit
def test_random_elastic_transform_constant_displacement_matches_numpy_reference():
    image_np = np.arange(9, dtype=np.float32).reshape(3, 3, 1)
    transform = RandomElasticTransform(
        keys=["image"],
        input_layout="HWC",
        interpolation="nearest",
        prob=1.0,
        seed=7,
    )
    field = ops.concatenate(
        [
            ops.zeros((1, 3, 3, 1), dtype="float32"),
            ops.ones((1, 3, 3, 1), dtype="float32"),
        ],
        axis=-1,
    )

    result = transform._warp_tensor(ops.expand_dims(as_tensor(image_np), axis=0), field, "nearest")
    expected = np.array([[1, 2, 2], [4, 5, 5], [7, 8, 8]], dtype=np.float32)[..., None]

    np.testing.assert_array_equal(ops.convert_to_numpy(result[0]), expected)


@pytest.mark.unit
def test_random_elastic_transform_keeps_image_and_label_aligned():
    pattern = np.zeros((3, 3, 1), dtype=np.float32)
    pattern[1, 1, 0] = 1.0
    image = as_tensor(pattern)
    label = as_tensor(pattern.astype(np.int32))
    transform = RandomElasticTransform(
        keys=["image", "label"],
        input_layout="HWC",
        interpolation={"image": "nearest", "label": "nearest"},
        prob=1.0,
        seed=7,
    )
    field = ops.concatenate(
        [
            ops.zeros((1, 3, 3, 1), dtype="float32"),
            ops.ones((1, 3, 3, 1), dtype="float32"),
        ],
        axis=-1,
    )
    transform._sample_or_zero_field = lambda tensor, should_apply, affine=None: field

    result = transform(TensorBundle({"image": image, "label": label}))

    np.testing.assert_array_equal(
        ops.convert_to_numpy(result["image"]),
        ops.convert_to_numpy(result["label"]),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_layout", "shape", "interpolation"),
    [
        ("HWC", (5, 6, 1), "bilinear"),
        ("DHWC", (3, 5, 6, 1), "trilinear"),
    ],
)
def test_random_elastic_transform_accepts_alpha_sigma_ranges(input_layout, shape, interpolation):
    image = as_tensor(np.zeros(shape, dtype=np.float32))
    transform = RandomElasticTransform(
        keys=["image"],
        alpha=(1.0, 3.0),
        sigma=(1.0, 2.0),
        interpolation=interpolation,
        prob=1.0,
        input_layout=input_layout,
        seed=7,
    )

    result = transform(TensorBundle({"image": image}))

    assert tuple(ops.shape(result["image"])) == shape
    assert transform.alpha == (1.0, 3.0)
    assert transform.sigma == (1.0, 2.0)


@pytest.mark.unit
def test_random_elastic_transform_supports_coarse_grid_for_2d():
    image = as_tensor(np.arange(36, dtype=np.float32).reshape(6, 6, 1))
    transform = RandomElasticTransform(
        keys=["image"],
        input_layout="HWC",
        control_grid_spacing=(2, 2),
        field_interpolation="bspline",
        prob=1.0,
        seed=7,
    )

    result = transform(TensorBundle({"image": image}))

    assert tuple(ops.shape(result["image"])) == (6, 6, 1)

    batch_transform = RandomElasticTransform(
        keys=["image"],
        input_layout="BHWC",
        control_grid_spacing=(2, 2),
        prob=1.0,
        seed=7,
    )
    batch_result = batch_transform(TensorBundle({"image": ops.stack([image, image])}))

    assert tuple(ops.shape(batch_result["image"])) == (2, 6, 6, 1)


@pytest.mark.unit
def test_random_elastic_transform_uses_rank_aware_2d_field_interpolation():
    transform = RandomElasticTransform(keys=["image"], input_layout="HWC")

    assert transform.field_interpolation == "bilinear"


@pytest.mark.unit
def test_random_elastic_transform_locks_3d_volume_borders():
    image = as_tensor(np.arange(64, dtype=np.float32).reshape(4, 4, 4, 1))
    transform = RandomElasticTransform(
        keys=["image"],
        alpha=20.0,
        sigma=1.0,
        interpolation="nearest",
        prob=1.0,
        input_layout="DHWC",
        locked_borders=1,
        seed=7,
    )

    output = ops.convert_to_numpy(transform(TensorBundle({"image": image}))["image"])
    original = ops.convert_to_numpy(image)

    np.testing.assert_array_equal(output[0], original[0])
    np.testing.assert_array_equal(output[-1], original[-1])
    np.testing.assert_array_equal(output[:, 0], original[:, 0])
    np.testing.assert_array_equal(output[:, -1], original[:, -1])
    np.testing.assert_array_equal(output[:, :, 0], original[:, :, 0])
    np.testing.assert_array_equal(output[:, :, -1], original[:, :, -1])


@pytest.mark.unit
def test_random_elastic_transform_locks_2d_image_borders():
    image = as_tensor(np.arange(36, dtype=np.float32).reshape(6, 6, 1))
    transform = RandomElasticTransform(
        keys=["image"],
        input_layout="HWC",
        alpha=2.0,
        sigma=1.0,
        control_grid_spacing=(2, 2),
        locked_borders=1,
        interpolation="nearest",
        prob=1.0,
        seed=7,
    )

    output = ops.convert_to_numpy(transform(TensorBundle({"image": image}))["image"])
    original = ops.convert_to_numpy(image)

    np.testing.assert_array_equal(output[0], original[0])
    np.testing.assert_array_equal(output[-1], original[-1])
    np.testing.assert_array_equal(output[:, 0], original[:, 0])
    np.testing.assert_array_equal(output[:, -1], original[:, -1])


@pytest.mark.unit
@pytest.mark.parametrize("fill_mode", ["nearest", "reflect", "wrap"])
def test_random_elastic_transform_accepts_boundary_modes(fill_mode):
    image = as_tensor(np.arange(4, dtype=np.float32).reshape(2, 2, 1))
    transform = RandomElasticTransform(
        keys=["image"],
        alpha=1.0,
        sigma=1.0,
        interpolation="bilinear",
        fill_mode=fill_mode,
        prob=1.0,
        input_layout="HWC",
        seed=7,
    )

    output = transform(TensorBundle({"image": image}))
    assert tuple(ops.shape(output["image"])) == (2, 2, 1)


@pytest.mark.unit
def test_random_elastic_transform_constant_boundary_uses_fill_value(monkeypatch):
    image = as_tensor(np.arange(4, dtype=np.float32).reshape(2, 2, 1))
    transform = RandomElasticTransform(
        keys=["image"],
        alpha=2.0,
        sigma=1.0,
        interpolation="nearest",
        fill_mode="constant",
        fill_value=-5.0,
        prob=1.0,
        input_layout="HWC",
        seed=7,
    )
    monkeypatch.setattr(
        transform,
        "random_normal",
        lambda shape, dtype: ops.ones(shape, dtype=dtype),
    )

    output = transform(TensorBundle({"image": image}))

    np.testing.assert_array_equal(
        ops.convert_to_numpy(output["image"]),
        np.full((2, 2, 1), -5.0, dtype=np.float32),
    )


@pytest.mark.unit
def test_random_elastic_transform_rejects_unknown_boundary_mode():
    with pytest.raises(ValueError, match="fill_mode"):
        RandomElasticTransform(
            keys=["image"],
            input_layout="HWC",
            fill_mode="mirror",
        )


@pytest.mark.unit
def test_random_elastic_transform_defaults_to_voxel_trilinear_field_units():
    transform = RandomElasticTransform(keys=["image"], input_layout="DHWC")

    assert transform.displacement_units == "voxel"
    assert transform.field_interpolation == "trilinear"


@pytest.mark.unit
def test_random_elastic_transform_rejects_invalid_field_configuration():
    with pytest.raises(ValueError, match="displacement_units"):
        RandomElasticTransform(
            keys=["image"],
            input_layout="DHWC",
            displacement_units="world",
        )
    with pytest.raises(ValueError, match="field_interpolation"):
        RandomElasticTransform(
            keys=["image"],
            input_layout="DHWC",
            field_interpolation="cubic",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_layout", "spacing"),
    [("HWC", (2, 2, 2)), ("DHWC", (2, 2))],
    ids=["2d-spacing-for-3d", "3d-spacing-for-2d"],
)
def test_random_elastic_transform_validates_control_grid_rank(input_layout, spacing):
    with pytest.raises(ValueError, match="one value per spatial axis"):
        RandomElasticTransform(
            keys=["image"],
            input_layout=input_layout,
            control_grid_spacing=spacing,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_layout", "field_interpolation"),
    [("HWC", "trilinear"), ("DHWC", "bilinear")],
    ids=["trilinear-for-2d", "bilinear-for-3d"],
)
def test_random_elastic_transform_validates_field_interpolation_rank(
    input_layout, field_interpolation
):
    with pytest.raises(ValueError, match="invalid for"):
        RandomElasticTransform(
            keys=["image"],
            input_layout=input_layout,
            field_interpolation=field_interpolation,
        )


@pytest.mark.unit
def test_random_elastic_transform_mm_requires_minimum_physical_spacing():
    with pytest.raises(ValueError, match="minimum_physical_spacing"):
        RandomElasticTransform(
            keys=["image"],
            input_layout="DHWC",
            displacement_units="mm",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "spacing",
    [float("nan"), float("inf"), float("-inf"), (1.0, 2.0, float("nan"))],
)
def test_random_elastic_transform_rejects_non_finite_minimum_physical_spacing(spacing):
    with pytest.raises(ValueError, match="finite and positive"):
        RandomElasticTransform(
            keys=["image"],
            input_layout="DHWC",
            displacement_units="mm",
            minimum_physical_spacing=spacing,
        )


@pytest.mark.unit
def test_random_elastic_transform_mm_requires_affine_metadata():
    image = as_tensor(np.zeros((3, 3, 3, 1), dtype=np.float32))
    transform = RandomElasticTransform(
        keys=["image"],
        input_layout="DHWC",
        displacement_units="mm",
        minimum_physical_spacing=1.0,
    )

    with pytest.raises(ValueError, match=r"bundle\.meta\['affine'\]"):
        transform(TensorBundle({"image": image}))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_layout", "tensor_shape", "affine_diagonal", "expected"),
    [
        (
            "HWC",
            (3, 4, 1),
            (2.0, 4.0, 1.0, 1.0),
            (1.0, 0.5),
        ),
        (
            "DHWC",
            (3, 4, 5, 1),
            (2.0, 4.0, 8.0, 1.0),
            (1.0, 0.5, 0.25),
        ),
    ],
    ids=["2d-mm-to-pixels", "3d-mm-to-voxels"],
)
def test_random_elastic_transform_converts_mm_to_tensor_axis_units(
    monkeypatch,
    input_layout,
    tensor_shape,
    affine_diagonal,
    expected,
):
    image = as_tensor(np.zeros(tensor_shape, dtype=np.float32))
    transform = RandomElasticTransform(
        keys=["image"],
        input_layout=input_layout,
        alpha=2.0,
        displacement_units="mm",
        minimum_physical_spacing=affine_diagonal[: len(expected)],
        prob=1.0,
    )
    monkeypatch.setattr(
        transform,
        "random_normal",
        lambda shape, dtype: ops.ones(shape, dtype=dtype),
    )
    affine = ops.diag(ops.convert_to_tensor(affine_diagonal, dtype="float32"))
    batched = ops.expand_dims(image, axis=0)
    field = transform._sample_or_zero_field(batched, True, affine=affine)

    field_np = ops.convert_to_numpy(field)
    expected_field = np.broadcast_to(
        np.asarray(expected, dtype=np.float32),
        field_np[0, ..., : len(expected)].shape,
    )
    np.testing.assert_allclose(field_np[0, ..., : len(expected)], expected_field, atol=1e-5)


@pytest.mark.unit
def test_random_elastic_transform_mm_rejects_invalid_affine_shape():
    image = as_tensor(np.zeros((3, 3, 3, 1), dtype=np.float32))
    transform = RandomElasticTransform(
        keys=["image"],
        input_layout="DHWC",
        displacement_units="mm",
        minimum_physical_spacing=1.0,
    )

    with pytest.raises(ValueError, match="Expected a 4x4 affine matrix"):
        transform(TensorBundle({"image": image}, {"affine": ops.eye(3)}))


@pytest.mark.unit
def test_random_elastic_transform_mm_uses_per_axis_smoothing(monkeypatch):
    image = as_tensor(np.zeros((4, 5, 6, 1), dtype=np.float32))
    transform = RandomElasticTransform(
        keys=["image"],
        input_layout="DHWC",
        alpha=2.0,
        sigma=4.0,
        displacement_units="mm",
        minimum_physical_spacing=(1.0, 2.0, 4.0),
        control_grid_spacing=(2, 3, 4),
        prob=1.0,
    )
    captured = {}

    def capture_smoothing(field, sigma, spatial_rank, *, max_sigma=None):
        captured["sigma"] = ops.convert_to_numpy(sigma)
        captured["max_sigma"] = max_sigma
        return field

    monkeypatch.setattr(elastic_module, "_gaussian_smooth_nd", capture_smoothing)
    affine = ops.diag(ops.convert_to_tensor((2.0, 4.0, 8.0, 1.0), dtype="float32"))
    transform._sample_or_zero_field(
        ops.expand_dims(image, axis=0),
        True,
        affine=affine,
    )

    np.testing.assert_allclose(captured["sigma"], (1.0, 1.0 / 3.0, 1.0 / 8.0))
    np.testing.assert_allclose(captured["max_sigma"], (2.0, 2.0 / 3.0, 0.25))


@pytest.mark.unit
def test_random_elastic_transform_supports_bspline_coarse_field():
    image = as_tensor(np.zeros((4, 4, 4, 1), dtype=np.float32))
    transform = RandomElasticTransform(
        keys=["image"],
        input_layout="DHWC",
        field_interpolation="bspline",
        control_grid_spacing=(2, 2, 2),
        prob=1.0,
        seed=7,
    )

    result = transform(TensorBundle({"image": image}))

    assert tuple(ops.shape(result["image"])) == (4, 4, 4, 1)


@pytest.mark.unit
def test_random_elastic_transform_limits_sampled_displacement(monkeypatch):
    image = as_tensor(np.zeros((3, 3, 3, 1), dtype=np.float32))
    transform = RandomElasticTransform(
        keys=["image"],
        alpha=2.0,
        sigma=1.0,
        interpolation="trilinear",
        prob=1.0,
        input_layout="DHWC",
        seed=7,
    )
    monkeypatch.setattr(
        transform,
        "random_normal",
        lambda shape, dtype: ops.full(shape, 100.0, dtype=dtype),
    )
    batched = ops.expand_dims(image, axis=0)

    field = transform._sample_or_zero_field(batched, True)

    assert float(ops.convert_to_numpy(ops.max(field))) == 2.0
    assert float(ops.convert_to_numpy(ops.min(field))) == 2.0


@pytest.mark.unit
def test_random_elastic_transform_coarse_field_respects_grid_and_borders():
    image = as_tensor(np.zeros((9, 10, 11, 1), dtype=np.float32))
    transform = RandomElasticTransform(
        keys=["image"],
        alpha=3.0,
        sigma=2.0,
        control_grid_spacing=(2, 2, 2),
        locked_borders=1,
        interpolation="trilinear",
        prob=1.0,
        input_layout="DHWC",
        seed=7,
    )

    field = transform._sample_or_zero_field(ops.expand_dims(image, axis=0), True)
    field_np = ops.convert_to_numpy(field)

    assert field_np.shape == (1, 9, 10, 11, 3)
    assert np.max(np.abs(field_np)) <= 3.0 + 1e-6
    np.testing.assert_array_equal(field_np[:, 0], 0.0)
    np.testing.assert_array_equal(field_np[:, -1], 0.0)
    np.testing.assert_array_equal(field_np[:, :, 0], 0.0)
    np.testing.assert_array_equal(field_np[:, :, -1], 0.0)
    np.testing.assert_array_equal(field_np[:, :, :, 0], 0.0)
    np.testing.assert_array_equal(field_np[:, :, :, -1], 0.0)


@pytest.mark.unit
def test_random_elastic_transform_probability_zero_is_noop():
    image = as_tensor(np.arange(20, dtype=np.float32).reshape(4, 5, 1))
    transform = RandomElasticTransform(
        keys=["image"],
        alpha=1.0,
        sigma=1.0,
        prob=0.0,
        input_layout="HWC",
        seed=7,
    )

    output = transform(TensorBundle({"image": image}))

    np.testing.assert_array_equal(
        ops.convert_to_numpy(output["image"]),
        ops.convert_to_numpy(image),
    )
    assert not bool(ops.convert_to_numpy(output.get_applied_transforms()[-1]["applied"]))


@pytest.mark.unit
def test_random_elastic_transform_applies_per_sample_mask():
    image = as_tensor(np.arange(18, dtype=np.float32).reshape(2, 3, 3, 1))
    transform = RandomElasticTransform(
        keys=["image"],
        input_layout="BHWC",
        prob=0.5,
        seed=7,
    )
    transform._sample_apply_mask = lambda batch_size: ops.convert_to_tensor(
        [True, False], dtype="bool"
    )
    transform._sample_or_zero_field = lambda tensor, should_apply, affine=None: ops.zeros(
        (2, 3, 3, 2), dtype="float32"
    )
    transform._warp_tensor = lambda tensor, field, interpolation: tensor + 1.0

    output = transform(TensorBundle({"image": image}))
    output_np = ops.convert_to_numpy(output["image"])
    image_np = ops.convert_to_numpy(image)

    np.testing.assert_array_equal(output_np[0], image_np[0] + 1.0)
    np.testing.assert_array_equal(output_np[1], image_np[1])


@pytest.mark.unit
def test_random_elastic_transform_samples_distinct_fields_per_batch_item():
    image = as_tensor(np.zeros((2, 8, 8, 1), dtype=np.float32))
    config = dict(
        keys=["image"],
        input_layout="BHWC",
        alpha=(1.0, 3.0),
        sigma=(1.0, 2.0),
        prob=1.0,
        seed=17,
    )
    first_transform = RandomElasticTransform(**config)
    second_transform = RandomElasticTransform(**config)
    apply_mask = ops.ones((2,), dtype="bool")

    first_field = first_transform._sample_or_zero_field(image, apply_mask)
    second_field = second_transform._sample_or_zero_field(image, apply_mask)
    first_field_np = ops.convert_to_numpy(first_field)
    second_field_np = ops.convert_to_numpy(second_field)

    assert not np.array_equal(first_field_np[0], first_field_np[1])
    np.testing.assert_array_equal(first_field_np, second_field_np)


@pytest.mark.unit
def test_random_elastic_transform_replays_seed_sequence():
    image = as_tensor(np.arange(20, dtype=np.float32).reshape(4, 5, 1))
    config = dict(keys=["image"], alpha=1.0, sigma=1.0, prob=1.0, input_layout="HWC", seed=7)
    first = RandomElasticTransform(**config)(TensorBundle({"image": image}))
    second = RandomElasticTransform(**config)(TensorBundle({"image": image}))

    np.testing.assert_array_equal(
        ops.convert_to_numpy(first["image"]),
        ops.convert_to_numpy(second["image"]),
    )
