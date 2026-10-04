import numpy as np
import pytest

from medicai.trainer.nnunet.data.preprocessing import pipeline as preprocessing
from medicai.trainer.nnunet.data.preprocessing.pipeline import (
    _collect_class_locations,
    _load_labels,
)


def _loaded(data, affine=None, spacing=(1.0, 1.0, 1.0)):
    return data, affine if affine is not None else np.eye(4), None, spacing


def test_image_modalities_must_have_matching_spatial_shapes(monkeypatch):
    arrays = {
        "first.nii.gz": np.zeros((4, 5, 6), dtype=np.float32),
        "second.nii.gz": np.zeros((4, 5, 7), dtype=np.float32),
    }
    monkeypatch.setattr(
        preprocessing,
        "load_medical_image",
        lambda path: _loaded(arrays[str(path)]),
    )

    with pytest.raises(ValueError, match="does not match the other modalities"):
        preprocessing._load_image_channels(
            list(arrays), expected_spatial_dims=3, image_layout="DHW"
        )


def test_nifti_modalities_must_have_matching_affines(monkeypatch):
    arrays = {name: np.zeros((4, 5, 6), dtype=np.float32) for name in ("a.nii.gz", "b.nii.gz")}
    shifted = np.eye(4)
    shifted[0, 3] = 4
    affines = {"a.nii.gz": np.eye(4), "b.nii.gz": shifted}
    monkeypatch.setattr(
        preprocessing,
        "load_medical_image",
        lambda path: _loaded(arrays[str(path)], affines[str(path)]),
    )

    with pytest.raises(ValueError, match="does not match"):
        preprocessing._load_image_channels(
            list(arrays), expected_spatial_dims=3, image_layout="DHW"
        )


def test_modalities_must_have_matching_spacing(monkeypatch):
    arrays = {name: np.zeros((4, 5, 6), dtype=np.float32) for name in ("a.tif", "b.tif")}
    spacings = {"a.tif": (1.0, 1.0, 1.0), "b.tif": (2.0, 1.0, 1.0)}
    monkeypatch.setattr(
        preprocessing,
        "load_medical_image",
        lambda path: _loaded(arrays[str(path)], spacing=spacings[str(path)]),
    )

    with pytest.raises(ValueError, match="spacing .* expected .* other modalities"):
        preprocessing._load_image_channels(
            list(arrays), expected_spatial_dims=3, image_layout="DHW"
        )


def test_channel_free_image_layout_gets_channel_axis_when_stacked(monkeypatch):
    source = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    monkeypatch.setattr(
        preprocessing,
        "load_medical_image",
        lambda _: _loaded(source, spacing=(0.7, 0.8, 2.5)),
    )

    channels, spacing, _, spatial_dims, _ = preprocessing._load_image_channels(
        ["case.npy"], expected_spatial_dims=3, image_layout="HWD"
    )
    image = np.stack(channels, axis=-1)

    assert spatial_dims == 3
    assert spacing == [2.5, 0.7, 0.8]
    assert image.shape == (4, 2, 3, 1)


def test_image_loader_returns_source_nifti_affine(monkeypatch):
    affine = np.eye(4)
    monkeypatch.setattr(
        preprocessing,
        "load_medical_image",
        lambda _: _loaded(np.zeros((4, 5, 6), dtype=np.float32), affine),
    )

    *_, source_affine = preprocessing._load_image_channels(
        ["case.nii.gz"], expected_spatial_dims=3, image_layout="DHW"
    )

    np.testing.assert_array_equal(source_affine, affine)


@pytest.mark.parametrize(
    ("task_type", "expected"),
    [
        ("multi_class", np.asarray([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64)),
        ("binary", np.asarray([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64)),
    ],
)
def test_preprocessing_preserves_configured_ignore_label_id(monkeypatch, task_type, expected):
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int16).reshape(2, 2, 2)
    monkeypatch.setattr(
        "medicai.trainer.nnunet.data.preprocessing.pipeline.load_medical_image",
        lambda _: _loaded(labels),
    )

    result = _load_labels(
        "label.npy",
        spatial_dims=3,
        task_type=task_type,
        ignore_class_ids=[2],
        target_class_ids=[1],
        label_layout="DHW",
    )

    np.testing.assert_array_equal(result.reshape(-1), expected)


def test_foreground_locations_exclude_configured_ignore_label_id():
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int16).reshape(2, 2, 2)

    locations = _collect_class_locations(labels, ignore_class_ids=[2])

    assert set(locations) == {"1"}


def test_label_must_match_image_shape_and_nifti_affine(monkeypatch):
    image_affine = np.eye(4)
    label_affine = np.eye(4)
    label_affine[1, 3] = 2
    monkeypatch.setattr(
        preprocessing,
        "load_medical_image",
        lambda _: _loaded(np.zeros((4, 5, 6), dtype=np.float32), label_affine),
    )

    with pytest.raises(ValueError, match="does not share the image NIfTI affine"):
        preprocessing._validate_label_alignment(
            label_paths="label.nii.gz",
            image_shape=[4, 5, 6],
            image_affine=image_affine,
            spatial_dims=3,
            original_spacing=[1, 1, 1],
            label_layout="DHW",
        )
    with pytest.raises(ValueError, match="spatial shape"):
        monkeypatch.setattr(
            preprocessing,
            "load_medical_image",
            lambda _: _loaded(np.zeros((4, 5, 7), dtype=np.float32), image_affine),
        )
        preprocessing._validate_label_alignment(
            label_paths="label.nii.gz",
            image_shape=[4, 5, 6],
            image_affine=image_affine,
            spatial_dims=3,
            original_spacing=[1, 1, 1],
            label_layout="DHW",
        )


def test_label_spacing_uses_its_own_axis_order(monkeypatch):
    label = np.zeros((2, 3, 4), dtype=np.int16)  # HWD
    monkeypatch.setattr(
        preprocessing,
        "load_medical_image",
        lambda _: _loaded(label, spacing=None),
    )

    preprocessing._validate_label_alignment(
        label_paths="label.tif",
        image_shape=[4, 2, 3],  # canonical DHW
        image_affine=None,
        spatial_dims=3,
        original_spacing=[2.5, 0.7, 0.8],  # canonical DHW
        original_spacing_override=[2.5, 0.8, 0.7],  # source image DWH
        label_layout="HWD",
    )


def test_2d_configuration_resamples_3d_volumes_slice_wise():
    image = np.arange(3 * 8 * 10, dtype=np.float32).reshape(3, 8, 10)
    label = np.zeros((3, 8, 10), dtype=np.int64)
    label[:, 2:6, 3:7] = 2

    resampled_image = preprocessing._resample_channels(
        [image], original_spacing=[4.0, 1.0, 1.0], target_spacing=[0.5, 0.5], configuration="2d"
    )[0]
    resampled_label = preprocessing._resample_label_map(
        label,
        original_spacing=[4.0, 1.0, 1.0],
        target_spacing=[0.5, 0.5],
        configuration="2d",
    )

    assert resampled_image.shape == (3, 16, 20)
    assert resampled_label.shape == (3, 16, 20)
    assert set(np.unique(resampled_label)) <= {0, 2}
