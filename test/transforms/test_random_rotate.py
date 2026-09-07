import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    RandomRotate,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_random_rotate_preserves_shape_and_records_trace():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))

    out = RandomRotate(keys=["image", "label"], factor=0.2, prob=1.0, input_layout="DHWC")(
        TensorBundle({"image": image, "label": label})
    )

    assert tuple(ops.shape(out["image"])) == (4, 5, 6, 1)
    assert tuple(ops.shape(out["label"])) == (4, 5, 6, 1)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandomRotate"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is True
    assert trace["kernel"] == "rotate_volume"


@pytest.mark.unit
def test_random_rotate_supports_2d_sample_and_batch_layouts():
    image = as_tensor(np.random.randn(8, 9, 1).astype(np.float32))
    batch = as_tensor(np.random.randn(2, 8, 9, 1).astype(np.float32))

    sample_out = RandomRotate(keys=["image"], factor=0.0, prob=1.0, input_layout="hwc", seed=7)(
        TensorBundle({"image": image})
    )
    batch_out = RandomRotate(keys=["image"], factor=0.0, prob=1.0, input_layout="BHWc", seed=7)(
        TensorBundle({"image": batch})
    )

    assert tuple(ops.shape(sample_out["image"])) == (8, 9, 1)
    assert tuple(ops.shape(batch_out["image"])) == (2, 8, 9, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(sample_out["image"]), ops.convert_to_numpy(image)
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(batch_out["image"]), ops.convert_to_numpy(batch)
    )


@pytest.mark.unit
def test_random_rotate_replays_with_same_integer_seed():
    image = as_tensor(np.random.randn(2, 8, 9, 1).astype(np.float32))

    first = RandomRotate(keys=["image"], factor=0.2, prob=1.0, input_layout="BHWC", seed=17)(
        TensorBundle({"image": image})
    )
    second = RandomRotate(keys=["image"], factor=0.2, prob=1.0, input_layout="BHWC", seed=17)(
        TensorBundle({"image": image})
    )

    np.testing.assert_allclose(
        ops.convert_to_numpy(first["image"]), ops.convert_to_numpy(second["image"])
    )


@pytest.mark.unit
def test_random_rotate_supports_axis_ranges_and_multi_axis_3d_rotation():
    image = as_tensor(np.random.randn(2, 4, 5, 6, 1).astype(np.float32))
    transform = RandomRotate(
        keys=["image"],
        factor={"h": (-0.1, 0.1), "W": 0.1},
        prob=1.0,
        input_layout="BDHWC",
        seed=3,
    )

    out = transform(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (2, 4, 5, 6, 1)
    assert set(out.get_applied_transforms()[-1]["params"]["angles"]) == {"H", "W"}


@pytest.mark.unit
def test_random_rotate_multi_axis_inverse_uses_recorded_geometry():
    image = as_tensor(np.random.randn(1, 4, 5, 6, 1).astype(np.float32))
    transform = RandomRotate(
        keys=["image"],
        factor={"h": 0.1, "w": 0.1},
        prob=1.0,
        input_layout="BDHWC",
        seed=11,
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == tuple(ops.shape(image))
    assert np.isfinite(ops.convert_to_numpy(restored["image"])).all()
    assert restored.get_applied_transforms() == []


@pytest.mark.unit
def test_random_rotate_inverse_preserves_mixed_probability_batch(monkeypatch):
    """Keep skipped items exact while restoring a rotated item in one batch."""
    image = as_tensor(np.arange(2 * 4 * 5 * 6, dtype=np.float32).reshape(2, 4, 5, 6, 1) / 255.0)
    transform = RandomRotate(
        keys=["image"],
        factor=0.2,
        prob=0.5,
        input_layout="BDHWC",
        fill_mode="reflect",
    )
    calls = 0

    def sample_uniform(*, shape, minval=0.0, maxval=1.0, dtype="float32"):
        nonlocal calls
        calls += 1
        # The first draw gates application. Keep item 0 skipped and item 1 active.
        values = [1.0, 0.0] if calls == 1 else [0.0, 1.0]
        return as_tensor(values, dtype=dtype)

    monkeypatch.setattr(transform, "random_uniform", sample_uniform)
    forward = transform(TensorBundle({"image": image}))
    trace = forward.get_applied_transforms()[-1]
    angles = ops.convert_to_numpy(trace["params"]["angles"]["D"])
    assert np.allclose(angles[0], 0.0)
    assert not np.isclose(angles[1], 0.0)
    forward_image = ops.convert_to_numpy(forward["image"])
    original_image = ops.convert_to_numpy(image)
    np.testing.assert_array_equal(forward_image[0], original_image[0])

    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))
    restored_image = ops.convert_to_numpy(restored["image"])
    np.testing.assert_array_equal(restored_image[0], original_image[0])
    np.testing.assert_allclose(restored_image[1], original_image[1], atol=0.5)


@pytest.mark.unit
def test_random_rotate_resolves_per_key_interpolation_fill_mode_and_fill_value():
    transform = RandomRotate(
        keys=["image", "label"],
        input_layout="DHWC",
        interpolation={"image": "BILINEAR", "label": "NEAREST"},
        fill_mode={"image": "reflect", "label": "constant"},
        fill_value={"image": -1.0, "label": 2.0},
    )

    assert transform.interpolation == {"image": "bilinear", "label": "nearest"}
    assert transform.fill_mode == {"image": "reflect", "label": "constant"}
    assert transform.fill_value == {"image": -1.0, "label": 2.0}


@pytest.mark.unit
def test_random_rotate_supports_batch_layout_and_records_input_layout():
    image = as_tensor(np.random.randn(2, 4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (2, 4, 5, 6, 1)).astype(np.float32))

    out = RandomRotate(keys=["image", "label"], factor=0.2, prob=1.0, input_layout="BDHWC")(
        TensorBundle({"image": image, "label": label})
    )

    assert tuple(ops.shape(out["image"])) == (2, 4, 5, 6, 1)
    assert tuple(ops.shape(out["label"])) == (2, 4, 5, 6, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BDHWC"


@pytest.mark.unit
def test_random_rotate_accepts_input_layout():
    image = as_tensor(np.random.randn(2, 4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (2, 4, 5, 6, 1)).astype(np.float32))

    out = RandomRotate(
        keys=["image", "label"],
        factor=0.2,
        prob=1.0,
        input_layout="bdhwc",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (2, 4, 5, 6, 1)
    assert tuple(ops.shape(out["label"])) == (2, 4, 5, 6, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BDHWC"


@pytest.mark.unit
def test_random_rotate_uses_same_batch_kernel_for_sample_and_batch_modes():
    sample = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    batch = ops.stack([sample, sample], axis=0)
    sample_angles = ops.ones((1,), dtype="float32") * 0.1
    batch_angles = ops.ones((2,), dtype="float32") * 0.1

    transform = RandomRotate(keys=["image"], factor=0.2, prob=1.0, input_layout="DHWC")

    sample_out = ops.convert_to_numpy(
        transform._apply_tensor(sample, "image", {"D": sample_angles})
    )
    batch_out = ops.convert_to_numpy(
        RandomRotate(keys=["image"], factor=0.2, prob=1.0, input_layout="BDHWC")._apply_tensor(
            batch, "image", {"D": batch_angles}
        )
    )

    assert sample_out.shape == (4, 5, 6, 1)
    assert batch_out.shape == (2, 4, 5, 6, 1)
    np.testing.assert_allclose(batch_out[0], sample_out)
    np.testing.assert_allclose(batch_out[1], sample_out)


@pytest.mark.unit
def test_random_rotate_inverse_restores_when_angle_is_zero():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))
    transform = RandomRotate(keys=["image", "label"], factor=0.0, prob=1.0, input_layout="DHWC")

    forward = transform(TensorBundle({"image": image, "label": label}))
    restored = transform.inverse(
        TensorBundle({"image": forward["image"], "label": forward["label"]}, forward.meta)
    )

    np.testing.assert_allclose(ops.convert_to_numpy(restored["image"]), ops.convert_to_numpy(image))
    np.testing.assert_allclose(ops.convert_to_numpy(restored["label"]), ops.convert_to_numpy(label))


@pytest.mark.unit
def test_random_rotate_inverse_is_noop_when_not_applied():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    transform = RandomRotate(keys=["image"], factor=0.2, prob=0.0, input_layout="DHWC")

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(ops.convert_to_numpy(restored["image"]), ops.convert_to_numpy(image))


@pytest.mark.unit
def test_random_rotate_supports_integer_label_tensors():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 3, (4, 5, 6, 1)).astype(np.int32))

    out = RandomRotate(keys=["image", "label"], factor=0.2, prob=1.0, input_layout="DHWC")(
        TensorBundle({"image": image, "label": label})
    )

    assert out["label"].dtype == label.dtype
    assert tuple(ops.shape(out["label"])) == (4, 5, 6, 1)
    assert set(np.unique(ops.convert_to_numpy(out["label"]))).issubset(
        set(np.unique(ops.convert_to_numpy(label)))
    )


@pytest.mark.unit
def test_random_rotate_validates_arguments_and_fill_modes():
    with pytest.raises(ValueError, match="must be non-negative"):
        RandomRotate(keys=["image"], factor=-0.1, input_layout="DHWC")

    with pytest.raises(ValueError, match="Unsupported fill_mode"):
        RandomRotate(keys=["image"], fill_mode="crop", input_layout="DHWC")

    for mode in ("constant", "nearest", "wrap", "mirror", "reflect"):
        transform = RandomRotate(keys=["image"], fill_mode=mode, input_layout="HWC")
        assert transform.fill_mode["image"] == mode

    with pytest.raises(ValueError, match="supports only the `D` rotation axis"):
        RandomRotate(keys=["image"], factor={"H": 0.1}, input_layout="HWC")

    image = as_tensor(np.random.randn(8, 8, 1).astype(np.float32))
    out = RandomRotate(keys=["image"], factor=0.0, prob=1.0, input_layout="HWC")(
        TensorBundle({"image": image})
    )
    assert tuple(ops.shape(out["image"])) == (8, 8, 1)


@pytest.mark.unit
def test_random_rotate_allow_missing_keys_and_prob_zero():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    transform = RandomRotate(
        keys=["image", "label"], prob=0.0, allow_missing_keys=True, input_layout="DHWC"
    )
    bundle = TensorBundle({"image": image})

    out = transform(bundle)

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), ops.convert_to_numpy(image))
    assert not bool(ops.convert_to_numpy(out.get_applied_transforms()[-1]["applied"]))

    with pytest.raises(ValueError, match="supports only input_layout values"):
        RandomRotate(keys=["image"], factor=0.2, input_layout="DCHW")
