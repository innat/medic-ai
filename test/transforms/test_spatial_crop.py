import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    RandomSpatialCrop,
    SpatialCrop,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_spatial_crop_accepts_input_layout():
    image = as_tensor(np.random.randn(2, 8, 8, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (2, 8, 8, 1)).astype(np.float32))

    out = SpatialCrop(
        keys=["image", "label"],
        crop_size=(3, 4),
        crop_start=(1, 1),
        input_layout="BHWC",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (2, 3, 4, 1)
    assert tuple(ops.shape(out["label"])) == (2, 3, 4, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_spatial_crop_supports_2d_and_3d_channel_last_tensors():
    image_2d = as_tensor(np.arange(30, dtype=np.float32).reshape(5, 6, 1))
    image_3d = as_tensor(np.arange(120, dtype=np.float32).reshape(4, 5, 6, 1))

    out_2d = SpatialCrop(keys=["image"], crop_size=(3, 4), crop_start=(1, 1), input_layout="HWC")(
        TensorBundle({"image": image_2d})
    )
    out_3d = SpatialCrop(
        keys=["image"],
        crop_size=(2, 3, 4),
        crop_center=(2, 2, 3),
        input_layout="DHWC",
    )(TensorBundle({"image": image_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (3, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 3, 4, 1)
    assert out_2d.get_applied_transforms()[-1]["name"] == "SpatialCrop"


@pytest.mark.unit
def test_spatial_crop_supports_batch_mode_with_shared_crop():
    image_2d = as_tensor(np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6, 1))
    image_3d = as_tensor(np.arange(2 * 4 * 5 * 6, dtype=np.float32).reshape(2, 4, 5, 6, 1))

    out_2d = SpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        crop_start=(1, 1),
        input_layout="BHWC",
    )(TensorBundle({"image": image_2d}))
    out_3d = SpatialCrop(
        keys=["image"],
        crop_size=(2, 3, 4),
        crop_start=(1, 1, 1),
        input_layout="BDHWC",
    )(TensorBundle({"image": image_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (2, 3, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 2, 3, 4, 1)
    assert out_2d.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_spatial_crop_inverse_restores_original_canvas_for_2d():
    image = as_tensor(np.zeros((5, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[1:4, 1:5, 0] = np.arange(12, dtype=np.float32).reshape(3, 4)
    image = as_tensor(image_np)

    transform = SpatialCrop(keys=["image"], crop_size=(3, 4), crop_start=(1, 1), input_layout="HWC")
    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (5, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_spatial_crop_inverse_restores_original_canvas_for_batch_mode():
    image = as_tensor(np.zeros((2, 5, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[:, 1:4, 1:5, 0] = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    image = as_tensor(image_np)

    transform = SpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        crop_start=(1, 1),
        input_layout="BHWC",
    )
    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (2, 5, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_spatial_crop_inverse_restores_original_canvas_for_3d():
    image = as_tensor(np.zeros((4, 5, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[1:3, 1:4, 1:5, 0] = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    image = as_tensor(image_np)

    transform = SpatialCrop(
        keys=["image"],
        crop_size=(2, 3, 4),
        crop_start=(1, 1, 1),
        input_layout="DHWC",
    )
    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (4, 5, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_spatial_crop_inverse_places_prediction_back_on_original_canvas():
    image = as_tensor(np.arange(30, dtype=np.float32).reshape(5, 6, 1))
    label = as_tensor(np.zeros((5, 6, 1), dtype=np.float32))
    transform = SpatialCrop(
        keys=["image", "label"],
        crop_size=(3, 4),
        crop_start=(1, 1),
        input_layout="HWC",
    )

    forward = transform(TensorBundle({"image": image, "label": label}))
    prediction = ops.ones_like(forward["label"])
    prediction_bundle = TensorBundle(
        {"image": forward["image"], "label": prediction},
        dict(forward.meta),
    )
    prediction_bundle.meta["applied_transforms"] = list(forward.get_applied_transforms())

    restored = transform.inverse(prediction_bundle)

    expected = np.zeros((5, 6, 1), dtype=np.float32)
    expected[1:4, 1:5, 0] = 1.0
    np.testing.assert_allclose(ops.convert_to_numpy(restored["label"]), expected)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"])[1:4, 1:5, :],
        ops.convert_to_numpy(forward["image"]),
    )


@pytest.mark.unit
def test_spatial_crop_inverse_without_trace_is_noop():
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 5, 1), dtype=np.float32))})
    transform = SpatialCrop(keys=["image"], crop_size=(2, 2), input_layout="HWC")

    restored = transform.inverse(bundle)

    assert restored is bundle


@pytest.mark.unit
def test_spatial_crop_validates_exclusive_start_and_center():
    with pytest.raises(
        ValueError, match="Only one of `crop_start` or `crop_center` may be provided"
    ):
        SpatialCrop(
            keys=["image"],
            crop_size=(2, 2),
            crop_start=(0, 0),
            crop_center=(1, 1),
            input_layout="HWC",
        )


@pytest.mark.unit
def test_spatial_crop_nonpositive_roi_uses_full_extent():
    image = as_tensor(np.arange(30, dtype=np.float32).reshape(5, 6, 1))
    out = SpatialCrop(keys=["image"], crop_size=(0, -1), input_layout="HWC")(
        TensorBundle({"image": image})
    )

    assert tuple(ops.shape(out["image"])) == (5, 6, 1)


@pytest.mark.unit
def test_spatial_crop_records_per_key_crop_bounds_for_mixed_shapes():
    image = as_tensor(np.arange(36, dtype=np.float32).reshape(6, 6, 1))
    label = as_tensor(np.arange(16, dtype=np.float32).reshape(4, 4, 1))
    transform = SpatialCrop(
        keys=["image", "label"],
        crop_size=(4, 4),
        crop_center=(4, 4),
        input_layout="HWC",
    )

    forward = transform(TensorBundle({"image": image, "label": label}))
    trace = forward.get_applied_transforms()[-1]

    image_start = ops.convert_to_numpy(trace["params"]["crop_start"]["image"])
    label_start = ops.convert_to_numpy(trace["params"]["crop_start"]["label"])
    assert image_start.tolist() == [2, 2]
    assert label_start.tolist() == [0, 0]


@pytest.mark.unit
def test_random_spatial_crop_supports_2d_and_3d_channel_last_tensors():
    image_2d = as_tensor(np.arange(30, dtype=np.float32).reshape(5, 6, 1))
    image_3d = as_tensor(np.arange(120, dtype=np.float32).reshape(4, 5, 6, 1))

    out_2d = RandomSpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        random_center=False,
        input_layout="HWC",
    )(TensorBundle({"image": image_2d}))
    out_3d = RandomSpatialCrop(
        keys=["image"],
        crop_size=(2, 3, 4),
        random_center=False,
        input_layout="DHWC",
    )(TensorBundle({"image": image_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (3, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 3, 4, 1)
    assert out_3d.get_applied_transforms()[-1]["kernel"] == "SpatialCrop"


@pytest.mark.unit
def test_random_spatial_crop_supports_batch_mode_with_shared_crop():
    image_2d = as_tensor(np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6, 1))
    image_3d = as_tensor(np.arange(2 * 4 * 5 * 6, dtype=np.float32).reshape(2, 4, 5, 6, 1))

    out_2d = RandomSpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        random_center=False,
        input_layout="BHWC",
    )(TensorBundle({"image": image_2d}))
    out_3d = RandomSpatialCrop(
        keys=["image"],
        crop_size=(2, 3, 4),
        random_center=False,
        input_layout="BDHWC",
    )(TensorBundle({"image": image_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (2, 3, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 2, 3, 4, 1)
    assert out_2d.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_random_spatial_crop_accepts_input_layout():
    image = as_tensor(np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6, 1))

    out = RandomSpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        random_center=False,
        input_layout="bhwc",
    )(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (2, 3, 4, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_random_spatial_crop_accepts_plain_numpy_inputs():
    image = np.arange(30, dtype=np.float32).reshape(5, 6, 1)

    out = RandomSpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        random_center=False,
        input_layout="HWC",
        seed=7,
    )({"image": image})

    assert tuple(ops.shape(out["image"])) == (3, 4, 1)
    assert ops.is_tensor(out["image"])


@pytest.mark.unit
def test_random_spatial_crop_shares_sampled_crop_across_batched_input():
    image = as_tensor(np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6, 1))
    transform = RandomSpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        random_center=True,
        input_layout="BHWC",
        seed=23,
    )

    out = transform(TensorBundle({"image": image}))
    crop_start = ops.convert_to_numpy(out.get_applied_transforms()[-1]["params"]["crop_start"])
    original = ops.convert_to_numpy(image)
    expected = original[:, crop_start[0] : crop_start[0] + 3, crop_start[1] : crop_start[1] + 4, :]

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), expected)


@pytest.mark.unit
def test_random_spatial_crop_samples_each_axis_with_its_own_valid_range(monkeypatch):
    transform = RandomSpatialCrop(
        keys=["image"],
        crop_size=(4, 3, 2),
        random_center=True,
        input_layout="DHWC",
    )
    spatial_shape = as_tensor([4, 5, 6], dtype="int32")
    crop_size = as_tensor([4, 3, 2], dtype="int32")

    monkeypatch.setattr(
        transform,
        "random_uniform",
        lambda **kwargs: as_tensor([0.0, 0.5, 1.0], dtype="float32"),
    )

    center = transform._get_random_center(spatial_shape, crop_size, spatial_rank=3)

    # max_start is [0, 2, 4], so the sampled starts are [0, 1, 4].
    np.testing.assert_array_equal(ops.convert_to_numpy(center), [2, 2, 5])


@pytest.mark.unit
def test_random_spatial_crop_inverse_restores_batched_input_canvas():
    image = as_tensor(np.zeros((2, 5, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[:, 1:4, 1:5, 0] = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    image = as_tensor(image_np)

    transform = RandomSpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        random_center=False,
        input_layout="BHWC",
        seed=23,
    )
    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_random_spatial_crop_inverse_restores_original_canvas_for_2d():
    image = as_tensor(np.zeros((5, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[1:4, 1:5, 0] = np.arange(12, dtype=np.float32).reshape(3, 4)
    image = as_tensor(image_np)

    transform = RandomSpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        random_center=False,
        input_layout="HWC",
    )
    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (5, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_random_spatial_crop_inverse_restores_original_canvas_for_batch_mode():
    image = as_tensor(np.zeros((2, 5, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[:, 1:4, 1:5, 0] = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    image = as_tensor(image_np)

    transform = RandomSpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        random_center=False,
        input_layout="BHWC",
    )
    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (2, 5, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_random_spatial_crop_inverse_restores_original_canvas_for_3d():
    image = as_tensor(np.zeros((4, 5, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[1:3, 1:4, 1:5, 0] = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    image = as_tensor(image_np)

    transform = RandomSpatialCrop(
        keys=["image"],
        crop_size=(2, 3, 4),
        random_center=False,
        input_layout="DHWC",
    )
    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (4, 5, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_random_spatial_crop_inverse_without_trace_is_noop():
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 5, 1), dtype=np.float32))})
    transform = RandomSpatialCrop(
        keys=["image"],
        crop_size=(2, 2),
        random_center=False,
        input_layout="HWC",
    )

    restored = transform.inverse(bundle)

    assert restored is bundle


@pytest.mark.unit
def test_random_spatial_crop_validates_configuration():
    with pytest.raises(ValueError, match="must contain at least one key"):
        RandomSpatialCrop(keys=[], crop_size=(2, 2), input_layout="HWC")

    with pytest.raises(ValueError, match="min_valid_ratio must be in range"):
        RandomSpatialCrop(keys=["image"], crop_size=(2, 2), min_valid_ratio=1.5, input_layout="HWC")

    with pytest.raises(ValueError, match="max_attempts must be a positive integer"):
        RandomSpatialCrop(keys=["image"], crop_size=(2, 2), max_attempts=0, input_layout="HWC")

    with pytest.raises(ValueError, match="must provide an invalid_label"):
        RandomSpatialCrop(keys=["image"], crop_size=(2, 2), min_valid_ratio=0.2, input_layout="HWC")

    with pytest.raises(ValueError, match="supports only input_layout values"):
        RandomSpatialCrop(keys=["image"], crop_size=(2, 2), input_layout="CHW")


@pytest.mark.unit
def test_random_spatial_crop_random_size_and_label_aware_modes():
    image = as_tensor(np.arange(120, dtype=np.float32).reshape(4, 5, 6, 1))
    label = as_tensor(np.zeros((4, 5, 6, 1), dtype=np.int32))
    label = ops.convert_to_tensor(
        np.pad(np.ones((2, 2, 2, 1), dtype=np.int32), ((1, 1), (1, 2), (2, 2), (0, 0)))
    )

    out = RandomSpatialCrop(
        keys=["image", "label"],
        crop_size=(1, 2, 2),
        max_crop_size=(2, 4, 4),
        random_shape=True,
        invalid_label=0,
        min_valid_ratio=0.0,
        input_layout="DHWC",
    )(TensorBundle({"image": image, "label": label}))

    trace = out.get_applied_transforms()[-1]
    crop_size = ops.convert_to_numpy(trace["params"]["crop_size"])
    assert np.all(crop_size >= np.array([1, 2, 2]))
    assert np.all(crop_size <= np.array([2, 4, 4]))


@pytest.mark.unit
def test_random_spatial_crop_requires_label_for_label_aware_mode():
    transform = RandomSpatialCrop(
        keys=["image"], crop_size=(2, 2), invalid_label=0, input_layout="HWC"
    )

    with pytest.raises(KeyError, match="`label` key is required"):
        transform(TensorBundle({"image": as_tensor(np.ones((4, 4, 1), dtype=np.float32))}))


@pytest.mark.unit
def test_random_spatial_crop_uses_second_key_for_label_aware_mode():
    image = as_tensor(np.arange(16, dtype=np.float32).reshape(4, 4, 1))
    mask = as_tensor(np.pad(np.ones((2, 2, 1), dtype=np.int32), ((1, 1), (1, 1), (0, 0))))

    out = RandomSpatialCrop(
        keys=["image", "mask"],
        crop_size=(2, 2),
        invalid_label=0,
        random_center=False,
        input_layout="HWC",
    )(TensorBundle({"image": image, "mask": mask}))

    assert tuple(ops.shape(out["image"])) == (2, 2, 1)
    assert tuple(ops.shape(out["mask"])) == (2, 2, 1)


@pytest.mark.unit
def test_random_spatial_crop_rejects_unsupported_spatial_rank():
    image_4d_spatial = as_tensor(np.ones((2, 3, 4, 5, 1), dtype=np.float32))
    transform = RandomSpatialCrop(keys=["image"], crop_size=(2, 2, 2, 2), input_layout="BDHWC")

    with pytest.raises(ValueError, match="Expected spatial rank in \\(2, 3\\)"):
        transform(TensorBundle({"image": image_4d_spatial}))


@pytest.mark.unit
def test_random_spatial_crop_label_aware_mode_keeps_thin_spatial_dimensions():
    image = as_tensor(np.arange(12, dtype=np.float32).reshape(1, 4, 3, 1))
    label = as_tensor(np.ones((1, 4, 3, 1), dtype=np.int32))

    out = RandomSpatialCrop(
        keys=["image", "label"],
        crop_size=(1, 2, 2),
        invalid_label=0,
        random_center=False,
        input_layout="DHWC",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (1, 2, 2, 1)


@pytest.mark.unit
def test_random_spatial_crop_label_aware_mode_supports_multi_channel_labels():
    image = as_tensor(np.arange(16, dtype=np.float32).reshape(4, 4, 1))
    label = as_tensor(
        np.stack(
            [
                np.pad(np.ones((2, 2), dtype=np.int32), ((1, 1), (1, 1))),
                np.zeros((4, 4), dtype=np.int32),
            ],
            axis=-1,
        )
    )

    out = RandomSpatialCrop(
        keys=["image", "label"],
        crop_size=(2, 2),
        invalid_label=0,
        random_center=False,
        input_layout="HWC",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (2, 2, 1)
    assert tuple(ops.shape(out["label"])) == (2, 2, 2)
