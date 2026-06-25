import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    Compose,
    Flip,
    RandomShiftIntensity,
    ScaleIntensityRange,
    ShiftIntensity,
    TensorBundle,
)


@pytest.mark.unit
def test_compose_converts_numpy_for_data_and_meta():
    image = np.ones((4, 4, 1), dtype=np.float32)
    label = np.zeros((4, 4, 1), dtype=np.int32)
    meta = {"affine": np.eye(4, dtype=np.float32)}

    pipeline = Compose([])
    output = pipeline({"image": image, "label": label}, meta)

    assert isinstance(output, TensorBundle)
    assert hasattr(output["image"], "shape")
    assert hasattr(output["label"], "shape")
    assert hasattr(output["affine"], "shape")


@pytest.mark.unit
def test_compose_applies_transforms_in_order():
    def add_one(bundle):
        bundle["image"] = bundle["image"] + 1.0
        return bundle

    def multiply_by_two(bundle):
        bundle["image"] = bundle["image"] * 2.0
        return bundle

    pipeline = Compose([add_one, multiply_by_two])
    output = pipeline({"image": ops.convert_to_tensor(np.array([[[1.0]]], dtype=np.float32))})

    # (1 + 1) * 2 = 4
    assert float(output["image"].numpy().item()) == 4.0


@pytest.mark.unit
def test_tensorbundle_missing_key_raises():
    bundle = TensorBundle({"image": ops.zeros((2, 2, 1), dtype="float32")}, {"meta_key": "value"})
    with pytest.raises(KeyError):
        _ = bundle["unknown"]


@pytest.mark.unit
def test_tensorbundle_repr_contains_data_and_meta():
    bundle = TensorBundle({"image": ops.zeros((2, 2, 1), dtype="float32")}, {"spacing": [1.0, 1.0]})
    value = repr(bundle)
    assert "MetaTensor" in value
    assert "spacing" in value


@pytest.mark.unit
def test_compose_inverse_reverses_invertible_transforms_only():
    image = ops.convert_to_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    pipeline = Compose(
        [
            Flip(keys=["image"], spatial_axis=1),
            ShiftIntensity(keys=["image"], offset=3.0),
        ]
    )

    forward = pipeline({"image": image})
    restored = pipeline.inverse(TensorBundle({"image": forward["image"]}))

    expected = ops.convert_to_numpy(image) + 3.0
    np.testing.assert_allclose(ops.convert_to_numpy(restored["image"]), expected)


@pytest.mark.unit
def test_compose_inverse_accepts_mapping_inputs():
    image = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    pipeline = Compose([Flip(keys=["image"], spatial_axis=1)])

    forward = pipeline({"image": image})
    restored = pipeline.inverse({"image": forward["image"]})

    np.testing.assert_allclose(ops.convert_to_numpy(restored["image"]), image)


@pytest.mark.unit
def test_compose_inverse_traverses_nested_compose_blocks():
    image = ops.convert_to_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    inner = Compose([Flip(keys=["image"], spatial_axis=1)])
    outer = Compose([inner, ShiftIntensity(keys=["image"], offset=3.0)])

    forward = outer({"image": image})
    restored = outer.inverse(TensorBundle({"image": forward["image"]}))

    expected = ops.convert_to_numpy(image) + 3.0
    np.testing.assert_allclose(ops.convert_to_numpy(restored["image"]), expected)


@pytest.mark.unit
def test_compose_inverse_handles_repeated_shift_intensity_instances():
    image = ops.convert_to_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    pipeline = Compose(
        [
            ShiftIntensity(keys=["image"], offset=2.0),
            ShiftIntensity(keys=["image"], offset=-0.5),
        ]
    )

    forward = pipeline({"image": image})
    restored = pipeline.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )
    assert forward.get_applied_transforms() == []


@pytest.mark.unit
def test_compose_inverse_handles_repeated_scale_intensity_range_instances():
    image = ops.convert_to_tensor(np.array([[[0.0], [0.5], [1.0]]], dtype=np.float32))
    pipeline = Compose(
        [
            ScaleIntensityRange(
                keys=["image"],
                input_min=0.0,
                input_max=1.0,
                output_min=-1.0,
                output_max=1.0,
            ),
            ScaleIntensityRange(
                keys=["image"],
                input_min=-1.0,
                input_max=1.0,
                output_min=0.0,
                output_max=255.0,
            ),
        ]
    )

    forward = pipeline({"image": image})
    restored = pipeline.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )
    assert forward.get_applied_transforms() == []


@pytest.mark.unit
def test_compose_inverse_handles_repeated_random_shift_intensity_instances():
    image = ops.convert_to_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    pipeline = Compose(
        [
            RandomShiftIntensity(keys=["image"], offset=0.25, prob=1.0),
            RandomShiftIntensity(keys=["image"], offset=0.5, prob=1.0),
        ]
    )

    forward = pipeline({"image": image})
    restored = pipeline.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )
    assert forward.get_applied_transforms() == []
