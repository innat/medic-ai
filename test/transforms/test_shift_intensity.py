import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    Compose,
    RandomShiftIntensity,
    ShiftIntensity,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_shift_intensity_records_trace():
    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))
    out = ShiftIntensity(keys=["image"], offset=2.0, input_layout="HWC")(
        TensorBundle({"image": image})
    )

    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "ShiftIntensity"
    assert trace["params"]["keys"] == ["image"]
    assert trace["params"]["offset"] == 2.0
    assert trace["params"]["input_layout"] == "HWC"
    assert trace["invertible"] is True


@pytest.mark.unit
def test_shift_intensity_inverse_restores_scalar_offset():
    image = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    transform = ShiftIntensity(keys=["image"], offset=2.5, input_layout="HWC")

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_shift_intensity_supports_batch_mode():
    image_2d = as_tensor(np.ones((2, 3, 4, 1), dtype=np.float32))
    image_3d = as_tensor(np.ones((2, 3, 4, 5, 1), dtype=np.float32))

    out_2d = ShiftIntensity(keys=["image"], offset=0.5, input_layout="BHWC")(
        TensorBundle({"image": image_2d})
    )
    out_3d = ShiftIntensity(keys=["image"], offset=-0.25, input_layout="BDHWC")(
        TensorBundle({"image": image_3d})
    )

    np.testing.assert_allclose(ops.convert_to_numpy(out_2d["image"]), 1.5, rtol=1e-6)
    np.testing.assert_allclose(ops.convert_to_numpy(out_3d["image"]), 0.75, rtol=1e-6)


@pytest.mark.unit
def test_shift_intensity_accepts_input_layout():
    image = as_tensor(np.ones((2, 3, 4, 1), dtype=np.float32))

    out = ShiftIntensity(keys=["image"], offset=0.5, input_layout="BHWC")(
        TensorBundle({"image": image})
    )

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), 1.5, rtol=1e-6)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_shift_intensity_accepts_numpy_mapping_inputs():
    image = np.ones((3, 4, 1), dtype=np.float32)

    out = ShiftIntensity(keys=["image"], offset=0.5, input_layout="HWC")({"image": image})

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), 1.5, rtol=1e-6)


@pytest.mark.unit
def test_shift_intensity_uses_same_batch_kernel_for_sample_and_batch_modes():
    sample = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    batch = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))

    sample_out = ShiftIntensity(keys=["image"], offset=2.0, input_layout="HWC")(
        TensorBundle({"image": sample})
    )
    batch_out = ShiftIntensity(keys=["image"], offset=2.0, input_layout="BHWC")(
        TensorBundle({"image": batch})
    )

    np.testing.assert_allclose(
        ops.convert_to_numpy(sample_out["image"]),
        ops.convert_to_numpy(sample) + 2.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(batch_out["image"]),
        ops.convert_to_numpy(batch) + 2.0,
        rtol=1e-6,
    )


@pytest.mark.unit
def test_shift_intensity_inverse_restores_broadcast_channel_offsets():
    image = as_tensor(np.ones((3, 4, 2), dtype=np.float32))
    offsets = as_tensor(np.array([0.5, -0.25], dtype=np.float32))
    transform = ShiftIntensity(keys=["image"], offset=offsets, input_layout="HWC")

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_shift_intensity_inverse_without_trace_is_noop():
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 4, 1), dtype=np.float32))})
    transform = ShiftIntensity(keys=["image"], offset=1.0, input_layout="HWC")

    restored = transform.inverse(bundle)

    assert restored is bundle


@pytest.mark.unit
def test_shift_intensity_inverse_raises_for_missing_traced_key_when_strict():
    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))
    label = as_tensor(np.ones((4, 4, 1), dtype=np.float32))
    transform = ShiftIntensity(keys=["image", "label"], offset=1.0, input_layout="HWC")

    forward = transform(TensorBundle({"image": image, "label": label}))

    with pytest.raises(KeyError, match="label"):
        transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))


@pytest.mark.unit
def test_random_shift_intensity_preserves_shape_and_range():
    image = as_tensor(np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32))
    out = RandomShiftIntensity(keys=["image"], offset=(-0.2, 0.8), prob=1.0, input_layout="BHWC")(
        TensorBundle({"image": image})
    )
    shifted = ops.convert_to_numpy(out["image"])
    original = ops.convert_to_numpy(image)

    assert shifted.shape == (1, 2, 2, 1)
    assert np.all(shifted >= original - 0.8)
    assert np.all(shifted <= original + 0.8)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandomShiftIntensity"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is True
    assert trace["kernel"] == "ShiftIntensity"


@pytest.mark.unit
def test_random_shift_intensity_inverse_restores_scalar_sample():
    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))
    transform = RandomShiftIntensity(keys=["image"], offset=0.5, prob=1.0, input_layout="HWC")

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_random_shift_intensity_inverse_restores_channel_wise_sample():
    image = as_tensor(np.ones((3, 4, 2), dtype=np.float32))
    transform = RandomShiftIntensity(
        keys=["image"], offset=0.5, prob=1.0, channel_wise=True, input_layout="HWC"
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_random_shift_intensity_inverse_is_noop_when_not_applied():
    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))
    transform = RandomShiftIntensity(keys=["image"], offset=0.5, prob=0.0, input_layout="HWC")

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_random_shift_intensity_inverse_without_trace_is_noop():
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 4, 1), dtype=np.float32))})
    transform = RandomShiftIntensity(keys=["image"], offset=0.5, prob=1.0, input_layout="HWC")

    restored = transform.inverse(bundle)

    assert restored is bundle


@pytest.mark.unit
def test_random_shift_intensity_inverse_raises_for_missing_traced_key_when_strict():
    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))
    label = as_tensor(np.ones((4, 4, 1), dtype=np.float32))
    transform = RandomShiftIntensity(
        keys=["image", "label"], offset=0.5, prob=1.0, input_layout="HWC"
    )

    forward = transform(TensorBundle({"image": image, "label": label}))

    with pytest.raises(KeyError, match="label"):
        transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))


@pytest.mark.unit
def test_random_shift_intensity_channel_wise_records_per_channel_offsets():
    image = as_tensor(np.ones((4, 4, 2), dtype=np.float32))
    out = RandomShiftIntensity(
        keys=["image"], offset=0.5, prob=1.0, channel_wise=True, input_layout="HWC"
    )(TensorBundle({"image": image}))

    trace = out.get_applied_transforms()[-1]
    offsets = trace["params"]["sampled_offsets"]["image"]
    assert tuple(ops.shape(offsets)) == (1, 1, 2)


@pytest.mark.unit
def test_random_shift_intensity_prob_zero_is_noop():
    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))
    out = RandomShiftIntensity(keys=["image"], offset=0.5, prob=0.0, input_layout="HWC")(
        TensorBundle({"image": image})
    )

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), ops.convert_to_numpy(image))
    assert not bool(ops.convert_to_numpy(out.get_applied_transforms()[-1]["applied"]))


@pytest.mark.unit
def test_random_shift_intensity_supports_2d_and_3d_channel_last_tensors():
    image_2d = as_tensor(np.ones((6, 5, 1), dtype=np.float32))
    image_3d = as_tensor(np.ones((4, 6, 5, 1), dtype=np.float32))

    out_2d = RandomShiftIntensity(keys=["image"], offset=0.5, prob=1.0, input_layout="HWC")(
        TensorBundle({"image": image_2d})
    )
    out_3d = RandomShiftIntensity(keys=["image"], offset=0.25, prob=1.0, input_layout="DHWC")(
        TensorBundle({"image": image_3d})
    )

    shifted_2d = ops.convert_to_numpy(out_2d["image"])
    shifted_3d = ops.convert_to_numpy(out_3d["image"])

    assert shifted_2d.shape == (6, 5, 1)
    assert shifted_3d.shape == (4, 6, 5, 1)
    assert np.all(shifted_2d >= 0.5)
    assert np.all(shifted_2d <= 1.5)
    assert np.all(shifted_3d >= 0.75)
    assert np.all(shifted_3d <= 1.25)


@pytest.mark.unit
def test_random_shift_intensity_channel_wise_samples_per_channel_values():
    image = as_tensor(np.ones((3, 4, 2), dtype=np.float32))

    out = RandomShiftIntensity(
        keys=["image"], offset=0.5, prob=1.0, channel_wise=True, input_layout="HWC"
    )(TensorBundle({"image": image}))

    shifted = ops.convert_to_numpy(out["image"])
    trace = out.get_applied_transforms()[-1]
    offsets = ops.convert_to_numpy(trace["params"]["sampled_offsets"]["image"])

    assert shifted.shape == (3, 4, 2)
    assert offsets.shape == (1, 1, 2)
    assert np.all(shifted >= 0.5)
    assert np.all(shifted <= 1.5)


@pytest.mark.unit
def test_random_shift_intensity_replays_with_same_integer_seed():
    image = as_tensor(np.ones((3, 4, 2), dtype=np.float32))

    first = RandomShiftIntensity(
        keys=["image"],
        offset=0.5,
        prob=1.0,
        channel_wise=True,
        seed=11,
        input_layout="HWC",
    )(TensorBundle({"image": image}))
    second = RandomShiftIntensity(
        keys=["image"],
        offset=0.5,
        prob=1.0,
        channel_wise=True,
        seed=11,
        input_layout="HWC",
    )(TensorBundle({"image": image}))

    np.testing.assert_allclose(
        ops.convert_to_numpy(first["image"]),
        ops.convert_to_numpy(second["image"]),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_random_shift_intensity_shares_sampled_offsets_across_batched_input():
    image = as_tensor(np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2))
    transform = RandomShiftIntensity(
        keys=["image"],
        offset=0.5,
        prob=1.0,
        channel_wise=True,
        seed=17,
        input_layout="BHWC",
    )

    out = transform(TensorBundle({"image": image}))
    shifted = ops.convert_to_numpy(out["image"])
    original = ops.convert_to_numpy(image)
    delta = shifted - original
    trace = out.get_applied_transforms()[-1]
    sampled_offsets = np.squeeze(ops.convert_to_numpy(trace["params"]["sampled_offsets"]["image"]))

    np.testing.assert_allclose(
        shifted[0] - shifted[1],
        original[0] - original[1],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        delta[0],
        np.broadcast_to(sampled_offsets, delta[0].shape),
        rtol=1e-5,
        atol=1e-6,
    )
    assert trace["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_random_shift_intensity_accepts_batch_mode():
    image = as_tensor(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4, 1))
    transform = RandomShiftIntensity(
        keys=["image"],
        offset=0.5,
        prob=1.0,
        input_layout="BHWC",
        seed=17,
    )

    out = transform(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (2, 3, 4, 1)
    trace = out.get_applied_transforms()[-1]
    assert trace["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_random_shift_intensity_accepts_input_layout():
    image = as_tensor(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4, 1))
    transform = RandomShiftIntensity(
        keys=["image"],
        offset=0.5,
        prob=1.0,
        input_layout="BHWC",
        seed=17,
    )

    out = transform(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (2, 3, 4, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_random_shift_intensity_inverse_restores_batched_input():
    image = as_tensor(np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2))
    transform = RandomShiftIntensity(
        keys=["image"],
        offset=0.5,
        prob=1.0,
        channel_wise=True,
        input_layout="BHWC",
        seed=17,
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_random_shift_intensity_allow_missing_keys_records_empty_trace():
    transform = RandomShiftIntensity(
        keys=["image"],
        offset=0.1,
        prob=1.0,
        input_layout="HWC",
        allow_missing_keys=True,
    )
    bundle = TensorBundle({"label": as_tensor(np.ones((4, 4, 1), dtype=np.float32))})

    output = transform(bundle)

    assert output is bundle
    trace = output.get_applied_transforms()[-1]
    assert trace["params"]["keys"] == []
    assert not bool(ops.convert_to_numpy(trace["applied"]))

@pytest.mark.unit
def test_compose_inverse_restores_pipeline_with_multiple_shift_intensity_instances():
    image = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    pipeline = Compose(
        [
            ShiftIntensity(keys=["image"], offset=1.0, input_layout="HWC"),
            ShiftIntensity(keys=["image"], offset=-2.0, input_layout="HWC"),
        ]
    )

    forward = pipeline(TensorBundle({"image": image}))
    restored = pipeline.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )
    assert restored.get_applied_transforms() == []

