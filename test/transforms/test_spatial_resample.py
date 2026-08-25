import numpy as np
import pytest
from keras import ops

from medicai.transforms.utils import SpatialResample, build_affine


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


def as_numpy(tensor, dtype=None):
    return np.asarray(ops.convert_to_numpy(tensor), dtype=dtype)


@pytest.mark.unit
def test_spatial_resample_identity_nearest_returns_same_volume():
    tensor = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    affine = as_tensor(np.eye(4, dtype=np.float32))

    resampled = SpatialResample()(
        tensor=tensor,
        src_affine=affine,
        dst_affine=affine,
        output_shape=as_tensor([2, 2, 2], dtype="int32"),
        interpolation="nearest",
    )

    np.testing.assert_allclose(
        as_numpy(resampled),
        as_numpy(tensor),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_spatial_resample_identity_trilinear_returns_same_volume():
    tensor = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    affine = as_tensor(np.eye(4, dtype=np.float32))

    resampled = SpatialResample()(
        tensor=tensor,
        src_affine=affine,
        dst_affine=affine,
        output_shape=as_tensor([2, 2, 2], dtype="int32"),
        interpolation="trilinear",
    )

    np.testing.assert_allclose(
        as_numpy(resampled),
        as_numpy(tensor),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_spatial_resample_upscales_with_destination_spacing_change():
    tensor = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    src_affine = build_affine(
        as_tensor([2.0, 2.0, 2.0], dtype="float32"),
        as_tensor(np.eye(3, dtype=np.float32)),
        as_tensor([0.0, 0.0, 0.0], dtype="float32"),
    )
    dst_affine = build_affine(
        as_tensor([1.0, 1.0, 1.0], dtype="float32"),
        as_tensor(np.eye(3, dtype=np.float32)),
        as_tensor([0.0, 0.0, 0.0], dtype="float32"),
    )

    resampled = SpatialResample()(
        tensor=tensor,
        src_affine=src_affine,
        dst_affine=dst_affine,
        output_shape=as_tensor([4, 4, 4], dtype="int32"),
        interpolation="nearest",
        fill_value=-1.0,
    )

    assert tuple(ops.shape(resampled)) == (4, 4, 4, 1)
    resampled_np = as_numpy(resampled)
    np.testing.assert_allclose(resampled_np[0, 0, 0, 0], 0.0)
    np.testing.assert_allclose(resampled_np[2, 2, 2, 0], 7.0)
    np.testing.assert_allclose(resampled_np[3, 3, 3, 0], -1.0)


@pytest.mark.unit
def test_spatial_resample_chunked_matches_single_pass_output():
    tensor = as_tensor(np.arange(27, dtype=np.float32).reshape(3, 3, 3, 1))
    affine = as_tensor(np.eye(4, dtype=np.float32))

    full = SpatialResample(max_points_per_chunk=1024)(
        tensor=tensor,
        src_affine=affine,
        dst_affine=affine,
        output_shape=as_tensor([3, 3, 3], dtype="int32"),
        interpolation="trilinear",
    )
    chunked = SpatialResample(max_points_per_chunk=4)(
        tensor=tensor,
        src_affine=affine,
        dst_affine=affine,
        output_shape=as_tensor([3, 3, 3], dtype="int32"),
        interpolation="trilinear",
    )

    np.testing.assert_allclose(
        as_numpy(chunked),
        as_numpy(full),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_spatial_resample_many_matches_individual_calls():
    affine = as_tensor(np.eye(4, dtype=np.float32))
    image = as_tensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    label = as_tensor(np.arange(8, 16, dtype=np.float32).reshape(2, 2, 2, 1))
    resample = SpatialResample(max_points_per_chunk=4)
    output_shape = as_tensor([2, 2, 2], dtype="int32")

    image_out = resample(
        tensor=image,
        src_affine=affine,
        dst_affine=affine,
        output_shape=output_shape,
        interpolation="trilinear",
    )
    label_out = resample(
        tensor=label,
        src_affine=affine,
        dst_affine=affine,
        output_shape=output_shape,
        interpolation="nearest",
    )

    batched = resample.resample_many(
        tensors={"image": image, "label": label},
        src_affine=affine,
        dst_affine=affine,
        output_shape=output_shape,
        interpolation={"image": "trilinear", "label": "nearest"},
    )

    np.testing.assert_allclose(
        as_numpy(batched["image"]), as_numpy(image_out), rtol=1e-6
    )
    np.testing.assert_allclose(
        as_numpy(batched["label"]), as_numpy(label_out), rtol=1e-6
    )
