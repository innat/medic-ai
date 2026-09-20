import numpy as np
import pytest
from keras import ops

from medicai.transforms import RandomShear, RandomTranslate, RandomZoom, TensorBundle


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
            lambda layout: RandomZoom(
                keys=["image"], factor=0.0, prob=1.0, input_layout=layout
            ),
            id="zoom",
        ),
        pytest.param(
            lambda layout: RandomShear(
                keys=["image"], factor=0.0, prob=1.0, input_layout=layout
            ),
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
            lambda: RandomZoom(
                keys=["image"],
                factor=0.1,
                prob=1.0,
                input_layout="BHWC",
                seed=17,
            ),
            id="zoom",
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
@pytest.mark.parametrize(
    "transform_type",
    [RandomTranslate, RandomZoom, RandomShear],
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
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomZoom, RandomShear])
def test_affine_components_accept_scalar_range_and_axis_mapping(transform_type):
    scalar = transform_type(keys=["image"], factor=0.1, input_layout="HWC")
    ranged = transform_type(keys=["image"], factor=(-0.1, 0.2), input_layout="HWC")
    mapped = transform_type(
        keys=["image"],
        factor={"x": 0.1, "y": (0.0, 0.2)}
        if transform_type is not RandomShear
        else {"xy": 0.1, "yx": (0.0, 0.2)},
        input_layout="HWC",
    )

    assert scalar.ranges
    assert ranged.ranges
    assert mapped.ranges


@pytest.mark.unit
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomZoom, RandomShear])
@pytest.mark.parametrize(
    "layout, interpolation",
    [("HWC", "trilinear"), ("DHWC", "bilinear")],
)
def test_affine_components_reject_wrong_rank_interpolation(
    transform_type, layout, interpolation
):
    with pytest.raises(ValueError, match="Unsupported interpolation"):
        transform_type(
            keys=["image"],
            factor=0.1,
            interpolation=interpolation,
            input_layout=layout,
        )


@pytest.mark.unit
def test_affine_components_allow_missing_keys_when_requested():
    for transform_type in (RandomTranslate, RandomZoom, RandomShear):
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
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomZoom, RandomShear])
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
@pytest.mark.parametrize("transform_type", [RandomTranslate, RandomZoom, RandomShear])
def test_affine_components_reject_unknown_factor_axes(transform_type):
    with pytest.raises(ValueError, match="factor axes"):
        transform_type(
            keys=["image"],
            factor={"invalid": 0.1},
            input_layout="HWC",
        )
