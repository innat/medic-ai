import numpy as np
import pytest
from keras import ops

from medicai.transforms import RandomAffine, TensorBundle


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
@pytest.mark.parametrize(
    "layout, shape",
    [("HWC", (5, 6, 1)), ("DHWC", (3, 5, 6, 1)), ("BHWC", (2, 5, 6, 1)),
     ("BDHWC", (2, 3, 5, 6, 1))],
)
def test_random_affine_identity_preserves_shape_and_alignment(layout, shape):
    image = as_tensor(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    label = image * 2.0
    transform = RandomAffine(
        keys=["image", "label"],
        rotation_factor=0.0,
        zoom_factor=0.0,
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
        zoom_factor=0.1,
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
def test_random_affine_inverse_reuses_recorded_matrix():
    image = as_tensor(np.arange(2 * 7 * 7, dtype=np.float32).reshape(2, 7, 7, 1))
    transform = RandomAffine(
        keys=["image"],
        rotation_factor=0.1,
        zoom_factor=0.1,
        translation_factor=0.1,
        shear_factor=0.1,
        prob=1.0,
        interpolation="nearest",
        input_layout="BHWC",
        seed=11,
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(
        TensorBundle({"image": forward["image"]}, forward.meta)
    )

    assert tuple(ops.shape(restored["image"])) == tuple(ops.shape(image))
    assert np.isfinite(ops.convert_to_numpy(restored["image"])).all()
    assert restored.get_applied_transforms() == []
