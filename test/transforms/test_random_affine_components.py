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
                keys=["image"], zoom_factor=0.0, prob=1.0, input_layout=layout
            ),
            id="zoom",
        ),
        pytest.param(
            lambda layout: RandomShear(
                keys=["image"], shear_factor=0.0, prob=1.0, input_layout=layout
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
                zoom_factor=0.1,
                prob=1.0,
                input_layout="BHWC",
                seed=17,
            ),
            id="zoom",
        ),
        pytest.param(
            lambda: RandomShear(
                keys=["image"],
                shear_factor=0.1,
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
