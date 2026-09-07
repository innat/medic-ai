import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    Compose,
    ScaleIntensityRange,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_scale_intensity_range_handles_flat_input():
    image = as_tensor(np.full((1, 2, 2), 5.0, dtype=np.float32))
    out = ScaleIntensityRange(
        keys=["image"],
        source_value_range=(5.0, 5.0),
        target_value_range=(0.0, 1.0),
        input_layout="HWC",
    )(TensorBundle({"image": image}))
    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), 0.0, rtol=1e-6)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "ScaleIntensityRange"
    assert trace["random"] is False
    assert trace["params"]["input_layout"] == "HWC"


@pytest.mark.unit
def test_scale_intensity_range_clips_and_preserves_dtype():
    image = as_tensor(np.array([[[-1.0], [0.5], [2.0]]], dtype=np.float32))
    out = ScaleIntensityRange(
        keys=["image"],
        source_value_range=(0.0, 1.0),
        target_value_range=(0.0, 10.0),
        clip=True,
        dtype=np.float32,
        input_layout="HWC",
    )(TensorBundle({"image": image}))

    np.testing.assert_allclose(
        ops.convert_to_numpy(out["image"]), np.array([[[0.0], [5.0], [10.0]]])
    )


@pytest.mark.unit
def test_scale_intensity_range_supports_batch_mode():
    image_2d = as_tensor(np.full((2, 3, 4, 1), 128.0, dtype=np.float32))
    image_3d = as_tensor(np.full((2, 3, 4, 5, 1), 0.5, dtype=np.float32))

    out_2d = ScaleIntensityRange(
        keys=["image"],
        source_value_range=(0.0, 255.0),
        target_value_range=(0.0, 1.0),
        input_layout="BHWC",
    )(TensorBundle({"image": image_2d}))
    out_3d = ScaleIntensityRange(
        keys=["image"],
        source_value_range=(0.0, 1.0),
        target_value_range=(-1.0, 1.0),
        input_layout="BDHWC",
    )(TensorBundle({"image": image_3d}))

    np.testing.assert_allclose(ops.convert_to_numpy(out_2d["image"]), 128.0 / 255.0, rtol=1e-6)
    np.testing.assert_allclose(ops.convert_to_numpy(out_3d["image"]), 0.0, rtol=1e-6)


@pytest.mark.unit
def test_scale_intensity_range_accepts_input_layout():
    image = as_tensor(np.full((2, 3, 4, 1), 128.0, dtype=np.float32))

    out = ScaleIntensityRange(
        keys=["image"],
        source_value_range=(0.0, 255.0),
        target_value_range=(0.0, 1.0),
        input_layout="BHWC",
    )(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (2, 3, 4, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_scale_intensity_range_uses_same_pixel_kernel_for_sample_and_batch_modes():
    sample = as_tensor(np.array([[[0.0], [127.5], [255.0]]], dtype=np.float32))
    batch = as_tensor(
        np.array(
            [
                [[[0.0], [127.5], [255.0]]],
                [[[255.0], [127.5], [0.0]]],
            ],
            dtype=np.float32,
        )
    )
    transform = ScaleIntensityRange(
        keys=["image"],
        source_value_range=(0.0, 255.0),
        target_value_range=(0.0, 1.0),
        input_layout="HWC",
    )

    sample_scaled = transform.scale_batch_tensor(sample)
    batch_scaled = transform.scale_batch_tensor(batch)

    np.testing.assert_allclose(
        ops.convert_to_numpy(sample_scaled),
        np.array([[[0.0], [0.5], [1.0]]], dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(batch_scaled),
        np.array(
            [
                [[[0.0], [0.5], [1.0]]],
                [[[1.0], [0.5], [0.0]]],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_scale_intensity_range_accepts_uint8_tensor_inputs():
    image = as_tensor(np.array([[[0], [128], [255]]], dtype=np.uint8))
    out = ScaleIntensityRange(
        keys=["image"],
        source_value_range=(0.0, 255.0),
        target_value_range=(0.0, 1.0),
        clip=True,
        input_layout="HWC",
    )(TensorBundle({"image": image}))

    assert ops.dtype(out["image"]) == "float32"
    np.testing.assert_allclose(
        ops.convert_to_numpy(out["image"]),
        np.array([[[0.0], [128.0 / 255.0], [1.0]]], dtype=np.float32),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_scale_intensity_range_accepts_numpy_mapping_inputs():
    image = np.array([[[0.0], [128.0], [255.0]]], dtype=np.float32)

    out = ScaleIntensityRange(
        keys=["image"],
        source_value_range=(0.0, 255.0),
        target_value_range=(0.0, 1.0),
        clip=True,
        input_layout="HWC",
    )({"image": image})

    np.testing.assert_allclose(
        ops.convert_to_numpy(out["image"]),
        np.array([[[0.0], [128.0 / 255.0], [1.0]]], dtype=np.float32),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_scale_intensity_range_inverse_restores_affine_mapping():
    image = as_tensor(np.array([[[0.0], [0.5], [1.0]]], dtype=np.float32))
    transform = ScaleIntensityRange(
        keys=["image"],
        source_value_range=(0.0, 1.0),
        target_value_range=(-1.0, 1.0),
        input_layout="HWC",
    )

    forward = transform(TensorBundle({"image": image}))
    trace = forward.get_applied_transforms()[-1]
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )
    assert trace["invertible"] is True


@pytest.mark.unit
def test_scale_intensity_range_inverse_restores_normalized_mapping():
    image = as_tensor(np.array([[[0.0], [127.5], [255.0]]], dtype=np.float32))
    transform = ScaleIntensityRange(
        keys=["image"],
        source_value_range=(0.0, 255.0),
        input_layout="HWC",
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_scale_intensity_range_inverse_uses_recorded_trace_parameters():
    image = as_tensor(np.array([[[0.0], [0.5], [1.0]]], dtype=np.float32))
    transform = ScaleIntensityRange(
        keys=["image"],
        source_value_range=(0.0, 1.0),
        target_value_range=(-1.0, 1.0),
        input_layout="HWC",
    )

    forward = transform(TensorBundle({"image": image}))

    transform.source_value_range = (-10.0, 10.0)
    transform.target_value_range = (5.0, 15.0)

    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_scale_intensity_range_inverse_is_noop_when_clipped():
    image = as_tensor(np.array([[[-1.0], [0.5], [2.0]]], dtype=np.float32))
    transform = ScaleIntensityRange(
        keys=["image"],
        source_value_range=(0.0, 1.0),
        target_value_range=(0.0, 10.0),
        clip=True,
        input_layout="HWC",
    )

    forward = transform(TensorBundle({"image": image}))
    trace = forward.get_applied_transforms()[-1]
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(forward["image"]),
    )
    assert trace["invertible"] is False


@pytest.mark.unit
def test_scale_intensity_range_inverse_raises_for_missing_traced_key_when_strict():
    image = as_tensor(np.array([[[0.0], [0.5], [1.0]]], dtype=np.float32))
    label = as_tensor(np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32))
    transform = ScaleIntensityRange(
        keys=["image", "label"],
        source_value_range=(0.0, 1.0),
        target_value_range=(-1.0, 1.0),
        input_layout="HWC",
    )

    forward = transform(TensorBundle({"image": image, "label": label}))

    with pytest.raises(KeyError, match="label"):
        transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))


@pytest.mark.unit
def test_scale_intensity_range_rejects_partial_target_range():
    with pytest.raises(ValueError, match="must contain exactly 2 values"):
        ScaleIntensityRange(
            keys=["image"],
            source_value_range=(0.0, 1.0),
            target_value_range=(0.0,),
            input_layout="HWC",
        )


@pytest.mark.unit
def test_compose_inverse_restores_pipeline_with_multiple_scale_intensity_range_instances():
    image = as_tensor(np.array([[[0.0], [0.5], [1.0]]], dtype=np.float32))
    pipeline = Compose(
        [
            ScaleIntensityRange(
                keys=["image"],
                source_value_range=(0.0, 1.0),
                target_value_range=(-1.0, 1.0),
                input_layout="HWC",
            ),
            ScaleIntensityRange(
                keys=["image"],
                source_value_range=(-1.0, 1.0),
                target_value_range=(0.0, 2.0),
                input_layout="HWC",
            ),
        ]
    )

    forward = pipeline(TensorBundle({"image": image}))
    restored = pipeline.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )
    assert restored.get_applied_transforms() == []
