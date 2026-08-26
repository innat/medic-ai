import numpy as np
import pytest
from keras import ops

from medicai.utils.image import resize_volumes


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_resize_volumes_trilinear_shape_and_dtype():
    volumes = as_tensor(np.random.default_rng(7).random((2, 4, 6, 8, 1), dtype=np.float32))
    out = resize_volumes(volumes, depth=2, height=3, width=4, method="trilinear")

    assert out.shape == (2, 2, 3, 4, 1)
    assert out.dtype == volumes.dtype


@pytest.mark.unit
def test_resize_volumes_nearest_shape_and_dtype():
    volumes = as_tensor(np.random.default_rng(7).integers(0, 10, (1, 5, 5, 5, 2), dtype=np.int32))
    out = resize_volumes(volumes, depth=3, height=4, width=2, method="nearest")

    assert out.shape == (1, 3, 4, 2, 2)
    assert out.dtype == volumes.dtype


@pytest.mark.unit
def test_resize_volumes_unsupported_method_raises():
    volumes = ops.ones((1, 2, 2, 2, 1), dtype="float32")
    with pytest.raises(ValueError, match="Unsupported resize method"):
        resize_volumes(volumes, depth=2, height=2, width=2, method="bicubic")
