import numpy as np
import pytest
from keras import ops

from medicai.transforms import RandomAffine, RandomScale, RandomShear, RandomTranslate, TensorBundle


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


def layout_cases():
    return [
        ("HWC", (4, 5, 1)),
        ("DHWC", (3, 4, 5, 1)),
        ("BHWC", (2, 4, 5, 1)),
        ("BDHWC", (2, 3, 4, 5, 1)),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("layout, shape", layout_cases())
@pytest.mark.parametrize(
    "transform_factory",
    [
        pytest.param(
            lambda layout: RandomTranslate(
                keys=["image"], factor=0.0, prob=1.0, input_layout=layout
            ),
            id="translate",
        ),
        pytest.param(
            lambda layout: RandomScale(keys=["image"], factor=0.0, prob=1.0, input_layout=layout),
            id="zoom",
        ),
        pytest.param(
            lambda layout: RandomShear(keys=["image"], factor=0.0, prob=1.0, input_layout=layout),
            id="shear",
        ),
    ],
)
def test_affine_components_support_all_channel_last_layouts(transform_factory, layout, shape):
    image = as_tensor(np.zeros(shape, dtype=np.float32))

    output = transform_factory(layout)(TensorBundle({"image": image}))

    assert tuple(ops.shape(output["image"])) == shape


@pytest.mark.unit
@pytest.mark.parametrize(
    "transform_factory",
    [
        pytest.param(
            lambda: RandomTranslate(
                keys=["image"], factor=0.1, prob=1.0, input_layout="BHWC", seed=17
            ),
            id="translate",
        ),
        pytest.param(
            lambda: RandomScale(
                keys=["image"],
                factor=0.1,
                prob=1.0,
                input_layout="BHWC",
                seed=17,
            ),
            id="scale",
        ),
        pytest.param(
            lambda: RandomShear(
                keys=["image"],
                factor=0.1,
                prob=1.0,
                input_layout="BHWC",
                seed=17,
            ),
            id="shear",
        ),
    ],
)
def test_affine_components_are_deterministic_and_invert_identity(transform_factory):
    image = as_tensor(np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6, 1))
    first = transform_factory()
    second = transform_factory()

    first_bundle = first(TensorBundle({"image": image}))
    second_bundle = second(TensorBundle({"image": image}))

    np.testing.assert_allclose(
        ops.convert_to_numpy(first_bundle["image"]),
        ops.convert_to_numpy(second_bundle["image"]),
    )

    restored = first.inverse(first_bundle)
    restored_image = ops.convert_to_numpy(restored["image"])
    assert restored_image.shape == ops.convert_to_numpy(image).shape
    assert np.isfinite(restored_image).all()


@pytest.mark.unit
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomScale, RandomShear])
@pytest.mark.parametrize(
    "layout, shape",
    [
        ("HWC", (5, 6, 1)),
        ("DHWC", (3, 5, 6, 1)),
        ("BHWC", (2, 5, 6, 1)),
        ("BDHWC", (2, 3, 5, 6, 1)),
    ],
)
def test_affine_components_inverse_supports_all_sample_and_batch_layouts(
    transform_type, layout, shape
):
    image = as_tensor(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    transform = transform_type(
        keys=["image"],
        factor=0.05,
        prob=1.0,
        interpolation="nearest",
        input_layout=layout,
        seed=17,
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(forward)
    restored_image = ops.convert_to_numpy(restored["image"])

    assert restored_image.shape == shape
    assert np.isfinite(restored_image).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "transform_type",
    [RandomTranslate, RandomScale, RandomShear],
)
@pytest.mark.parametrize(
    "layout, expected_image_interpolation",
    [("HWC", "bilinear"), ("DHWC", "trilinear")],
)
def test_affine_components_use_rank_aware_default_interpolation(
    transform_type, layout, expected_image_interpolation
):
    transform = transform_type(
        keys=["image", "label"],
        factor=0.1,
        input_layout=layout,
    )

    assert transform.interpolation == {
        "image": expected_image_interpolation,
        "label": "nearest",
    }


@pytest.mark.unit
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomScale, RandomShear])
def test_affine_components_accept_scalar_range_and_axis_mapping(transform_type):
    scalar = transform_type(keys=["image"], factor=0.1, input_layout="HWC")
    ranged = transform_type(keys=["image"], factor=(-0.1, 0.2), input_layout="HWC")
    mapped = transform_type(
        keys=["image"],
        factor=(
            {"x": 0.1, "y": (0.0, 0.2)}
            if transform_type is not RandomShear
            else {"xy": 0.1, "yx": (0.0, 0.2)}
        ),
        input_layout="HWC",
    )

    assert scalar.ranges
    assert ranged.ranges
    assert mapped.ranges


@pytest.mark.unit
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomScale])
def test_affine_components_use_channel_last_spatial_axis_order(transform_type):
    sample = transform_type(keys=["image"], factor=0.1, input_layout="HWC")
    volume = transform_type(keys=["image"], factor=0.1, input_layout="DHWC")

    assert tuple(sample.ranges) == ("y", "x")
    assert tuple(volume.ranges) == ("z", "y", "x")


@pytest.mark.unit
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomScale, RandomShear])
@pytest.mark.parametrize(
    "layout, interpolation",
    [("HWC", "trilinear"), ("DHWC", "bilinear")],
)
def test_affine_components_reject_wrong_rank_interpolation(transform_type, layout, interpolation):
    with pytest.raises(ValueError, match="Unsupported interpolation"):
        transform_type(
            keys=["image"],
            factor=0.1,
            interpolation=interpolation,
            input_layout=layout,
        )


@pytest.mark.unit
def test_affine_components_allow_missing_keys_when_requested():
    for transform_type in (RandomTranslate, RandomScale, RandomShear):
        transform = transform_type(
            keys=["image", "label"],
            factor=0.0,
            prob=1.0,
            input_layout="HWC",
            allow_missing_keys=True,
        )
        output = transform(TensorBundle({"image": as_tensor(np.zeros((4, 5, 1)))}))

        assert "image" in output.data


@pytest.mark.unit
@pytest.mark.parametrize(
    "transform_type", [RandomAffine, RandomTranslate, RandomScale, RandomShear]
)
def test_affine_inverse_validates_missing_recorded_keys_before_consuming_trace(transform_type):
    kwargs = {
        "keys": ["image", "label"],
        "factor": 0.0,
        "prob": 1.0,
        "input_layout": "HWC",
    }
    if transform_type is RandomAffine:
        kwargs = {
            "keys": ["image", "label"],
            "rotation_factor": 0.0,
            "scale_factor": 0.0,
            "translation_factor": 0.0,
            "shear_factor": 0.0,
            "prob": 1.0,
            "input_layout": "HWC",
        }
    transform = transform_type(**kwargs)
    bundle = TensorBundle(
        {
            "image": as_tensor(np.zeros((4, 5, 1), dtype=np.float32)),
            "label": as_tensor(np.zeros((4, 5, 1), dtype=np.float32)),
        }
    )
    transform(bundle)
    bundle.data.pop("label")

    with pytest.raises(KeyError, match="inverse"):
        transform.inverse(bundle)

    assert len(bundle.get_applied_transforms()) == 1


@pytest.mark.unit
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomScale, RandomShear])
def test_affine_components_resolve_per_key_interpolation_and_fill_options(transform_type):
    transform = transform_type(
        keys=["image", "label"],
        factor=0.1,
        interpolation={"image": "bilinear", "label": "nearest"},
        fill_mode={"image": "reflect", "label": "constant"},
        fill_value={"image": -1.0, "label": 2.0},
        input_layout="HWC",
    )

    assert transform.interpolation == {"image": "bilinear", "label": "nearest"}
    assert transform.fill_mode == {"image": "reflect", "label": "constant"}
    assert transform.fill_value == {"image": -1.0, "label": 2.0}


@pytest.mark.unit
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomScale, RandomShear])
def test_affine_components_reject_unknown_factor_axes(transform_type):
    with pytest.raises(ValueError, match="factor axes"):
        transform_type(
            keys=["image"],
            factor={"invalid": 0.1},
            input_layout="HWC",
        )


@pytest.mark.unit
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomScale, RandomShear])
def test_affine_components_restore_integer_label_dtype(transform_type):
    label = as_tensor(np.arange(4 * 5, dtype=np.int32).reshape(4, 5, 1))
    transform = transform_type(
        keys=["label"],
        factor=0.0,
        prob=1.0,
        interpolation="nearest",
        input_layout="HWC",
    )

    output = transform(TensorBundle({"label": label}))

    assert output["label"].dtype == label.dtype


@pytest.mark.unit
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomScale, RandomShear])
def test_affine_components_restore_float_image_dtype(transform_type):
    image = as_tensor(np.arange(4 * 5, dtype=np.float64).reshape(4, 5, 1))
    transform = transform_type(
        keys=["image"],
        factor=0.0,
        prob=1.0,
        interpolation="nearest",
        input_layout="HWC",
    )

    output = transform(TensorBundle({"image": image}))

    assert output["image"].dtype == image.dtype


@pytest.mark.unit
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomScale, RandomShear])
def test_affine_components_restore_integer_image_dtype_with_nearest(transform_type):
    image = as_tensor(np.arange(4 * 5, dtype=np.int32).reshape(4, 5, 1))
    transform = transform_type(
        keys=["image"],
        factor=0.0,
        prob=1.0,
        interpolation="nearest",
        input_layout="HWC",
    )

    output = transform(TensorBundle({"image": image}))

    assert output["image"].dtype == image.dtype


@pytest.mark.unit
def test_random_scale_samples_distinct_parameters_per_batch_item(monkeypatch):
    transform = RandomScale(
        keys=["image"],
        factor=0.2,
        input_layout="BHWC",
        prob=1.0,
    )

    def sample_uniform(*, shape, minval=0.0, maxval=1.0, dtype="float32"):
        del maxval
        values = [0.5, 0.5] if minval == 0.0 else [minval, minval + 0.2]
        return as_tensor(values[: shape[0]], dtype=dtype)

    monkeypatch.setattr(transform, "random_uniform", sample_uniform)
    scales, _ = transform._sample_scales(2)

    assert not np.allclose(
        ops.convert_to_numpy(scales["y"])[0],
        ops.convert_to_numpy(scales["y"])[1],
    )


@pytest.mark.unit
def test_random_shear_samples_distinct_parameters_per_batch_item(monkeypatch):
    transform = RandomShear(
        keys=["image"],
        factor=0.2,
        input_layout="BHWC",
        prob=1.0,
    )

    def sample_uniform(*, shape, minval=0.0, maxval=1.0, dtype="float32"):
        del maxval
        values = [0.5, 0.5] if minval == 0.0 else [minval, minval + 0.2]
        return as_tensor(values[: shape[0]], dtype=dtype)

    monkeypatch.setattr(transform, "random_uniform", sample_uniform)
    coefficients, _ = transform._sample_coefficients(2)

    assert not np.allclose(
        ops.convert_to_numpy(coefficients["xy"])[0],
        ops.convert_to_numpy(coefficients["xy"])[1],
    )


@pytest.mark.unit
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomScale, RandomShear])
@pytest.mark.parametrize(
    "layout, shape",
    [("HWC", (6, 7, 1)), ("DHWC", (3, 6, 7, 1))],
)
def test_affine_components_preserve_image_label_alignment(transform_type, layout, shape):
    image = as_tensor(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    label = image * 2.0
    transform = transform_type(
        keys=["image", "label"],
        factor=0.05,
        prob=1.0,
        interpolation={"image": "nearest", "label": "nearest"},
        input_layout=layout,
        seed=23,
    )

    output = transform(TensorBundle({"image": image, "label": label}))

    np.testing.assert_allclose(
        ops.convert_to_numpy(output["label"]),
        ops.convert_to_numpy(output["image"]) * 2.0,
    )
