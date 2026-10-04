import json

import numpy as np
import pytest

from medicai.trainer.nnunet.data.metadata.manifest import (
    CaseRecord,
    DatasetManifest,
    TaskSpec,
)
from medicai.trainer.nnunet.utils.io import normalize_layout_and_spacing


def _task(task_type="multi_class"):
    return TaskSpec(
        task_type=task_type,
        modalities=["image"],
        labels={"background": 0, "foreground": 1},
    )


def _case(path="case.npy", **kwargs):
    return CaseRecord(
        id="case_001",
        image=path,
        label="mask.npy",
        input_layout="DHWC",
        spacing=(1.0, 1.0, 1.0),
        **kwargs,
    )


def test_typed_manifest_adapts_to_existing_pipeline_contract():
    manifest = DatasetManifest(task=_task(), cases=[_case()])

    assert manifest.task_type == "multi-class"
    assert manifest.modalities == ["image"]
    assert manifest.items[0].images == ["case.npy"]
    assert manifest.items[0].labels == "mask.npy"
    assert manifest.items[0].image_layout == "DHWC"
    assert manifest.items[0].spacing == [1.0, 1.0, 1.0]


def test_case_record_uses_single_image_field_for_one_modality():
    case = CaseRecord(
        id="case_001",
        image="case.npy",
        label="mask.npy",
        input_layout="DHWC",
        spacing=(1.0, 1.0, 1.0),
    )
    manifest = DatasetManifest(task=_task(), cases=[case])

    assert case.image == "case.npy"
    assert manifest.items[0].images == ["case.npy"]


def test_case_record_image_list_follows_task_modality_order():
    task = TaskSpec(
        task_type="multi_class",
        modalities=["FLAIR", "T1", "T1CE"],
        labels={"background": 0, "tumor": 1},
    )
    case = CaseRecord(
        id="case_001",
        image=["flair.nii.gz", "t1.nii.gz", "t1ce.nii.gz"],
        label="mask.nii.gz",
    )
    manifest = DatasetManifest(task=task, cases=[case])

    assert manifest.items[0].images == ["flair.nii.gz", "t1.nii.gz", "t1ce.nii.gz"]
    assert manifest.to_dict()["cases"][0]["image"] == [
        "flair.nii.gz",
        "t1.nii.gz",
        "t1ce.nii.gz",
    ]


def test_case_record_accepts_keyed_input_layouts_for_image_and_label():
    case = CaseRecord(
        id="case_001",
        image="case.npy",
        label="mask.npy",
        input_layout={"image": "HWDC", "label": "HWD"},
        spacing=(0.7, 0.8, 2.5),
    )
    manifest = DatasetManifest(task=_task(), cases=[case])

    assert case.input_layout == {"image": "HWDC", "label": "HWD"}
    assert manifest.items[0].image_layout == "HWDC"
    assert manifest.items[0].label_layout == "HWD"
    assert "label_layout" not in case.to_dict()

    assert manifest.to_dict()["cases"][0]["input_layout"] == {
        "image": "HWDC",
        "label": "HWD",
    }


def test_legacy_label_layout_argument_is_not_supported():
    with pytest.raises(TypeError, match="label_layout"):
        CaseRecord(
            id="case_001",
            image="case.npy",
            label="mask.npy",
            input_layout="HWDC",
            label_layout="HWD",
        )


def test_typed_manifest_json_roundtrip(tmp_path):
    manifest = DatasetManifest(task=_task("binary"), cases=[_case()])
    path = tmp_path / "manifest.json"

    manifest.to_json(path)
    loaded = DatasetManifest.from_json(path)

    assert loaded.task.to_dict() == manifest.task.to_dict()
    assert loaded.cases[0].to_dict() == manifest.cases[0].to_dict()


def test_typed_manifest_preserves_dataset_input_layout(tmp_path):
    manifest = DatasetManifest(
        task=_task(),
        cases=[_case()],
        input_layout={"image": "DHWC", "label": "DHW"},
    )

    path = tmp_path / "manifest.json"
    manifest.to_json(path)
    restored = DatasetManifest.from_json(path)

    assert restored.to_dict()["input_layout"] == {
        "image": "DHWC",
        "label": "DHW",
    }


def test_legacy_manifest_format_is_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "modalities": ["image"],
                    "class_names": ["background", "foreground"],
                    "task_type": "binary",
                },
                "items": [
                    {
                        "id": "case_001",
                        "images": "case.npy",
                        "labels": "mask.npy",
                        "image_layout": "DHWC",
                        "spacing": [1, 1, 1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="typed 'task' and 'cases'"):
        DatasetManifest.from_json(path)


def test_dataset_manifest_requires_typed_task_and_cases():
    with pytest.raises(TypeError, match="task"):
        DatasetManifest()


@pytest.mark.parametrize(
    "task,case,error",
    [
        (
            _task(),
            CaseRecord(id="x", image={"wrong": "x.npy"}, label="y.npy", input_layout="DHWC"),
            "modalities mismatch",
        ),
        (
            _task(),
            CaseRecord(id="x", image={"image": "x.npy"}, label="y.npy", input_layout="DDHW"),
            "unique axis letters",
        ),
        (
            _task(),
            CaseRecord(
                id="x",
                image={"image": "x.npy"},
                label="y.npy",
                input_layout="HWDC",
                spacing=(1, 1),
            ),
            "spacing has",
        ),
        (
            _task(),
            CaseRecord(
                id="x",
                image="x.npy",
                label="y.npy",
                input_layout={"image": "DHWC", "label": "HW"},
                spacing=(1, 1, 1),
            ),
            "label layout declares 2D",
        ),
    ],
)
def test_invalid_case_contracts_fail_early(task, case, error):
    with pytest.raises(ValueError, match=error):
        DatasetManifest(task=task, cases=[case])


def test_dataset_rejects_mixed_spatial_ranks():
    first = _case("first.npy")
    second = CaseRecord(
        id="case_002",
        image="second.npy",
        label="second_mask.npy",
        input_layout="HWC",
        spacing=(1.0, 1.0),
    )

    with pytest.raises(ValueError, match="same spatial rank"):
        DatasetManifest(task=_task(), cases=[first, second])


def test_manifest_requires_labels_for_all_cases():
    first = _case("first.npy")
    with pytest.raises(TypeError, match="label"):
        CaseRecord(
            id="case_002",
            image="second.npy",
            input_layout="DHWC",
            spacing=(1.0, 1.0, 1.0),
        )


def test_binary_task_requires_two_categorical_labels():
    with pytest.raises(ValueError, match="exactly background=0 and foreground=1"):
        TaskSpec("binary", ["image"], {"background": 0, "a": 1, "b": 2})


def test_region_task_requires_complete_ordered_regions():
    with pytest.raises(ValueError, match="regions_class_order"):
        TaskSpec(
            "region_based",
            ["image"],
            {"background": 0, "outer": 1, "inner": 2},
            regions={"whole": [1, 2], "inner": [2]},
        )


def test_region_based_remains_the_public_task_type():
    task = TaskSpec(
        "region_based",
        ["image"],
        {"background": 0, "outer": 1, "inner": 2},
        regions={"outer_region": [1, 2], "inner_region": [2]},
        regions_class_order=[1, 2],
    )
    manifest = DatasetManifest(task=task, cases=[_case()])

    assert manifest.task_type == "region_based"
    assert manifest.items[0].task_type == "multi-label"  # Internal legacy region-channel mode.


def test_region_based_task_accepts_sparse_categorical_label_ids():
    task = TaskSpec(
        "region_based",
        ["T1", "T1ce", "T2", "FLAIR"],
        {"background": 0, "necrotic_core": 1, "edema": 2, "enhancing_tumor": 4},
        regions={"whole_tumor": [1, 2, 4], "tumor_core": [1, 4], "enhancing": [4]},
        regions_class_order=[2, 1, 4],
    )

    assert task.labels["enhancing_tumor"] == 4
    assert task.regions["whole_tumor"] == [1, 2, 4]


def test_multi_class_task_still_requires_consecutive_label_ids():
    with pytest.raises(ValueError, match="Multi-class label IDs must be consecutive"):
        TaskSpec(
            "multi_class",
            ["image"],
            {"background": 0, "class_a": 1, "class_b": 4},
        )


def test_multilabel_task_type_is_not_supported():
    with pytest.raises(ValueError, match="Unsupported task_type"):
        TaskSpec("multi_label", ["image"], {"background": 0, "organ": 1})


def test_named_independent_masks_are_not_supported():
    with pytest.raises(ValueError, match="one categorical label-map path"):
        CaseRecord(
            id="case_001",
            image="case.npy",
            label={"organ": "organ.npy"},
            input_layout="DHWC",
            spacing=(1, 1, 1),
        )


def test_tiff_spacing_required_or_read_from_official_sidecar(tmp_path):
    task = _task()
    image_path = tmp_path / "case_001_0000.tiff"
    case = CaseRecord(
        id="case_001",
        image=str(image_path),
        label="mask.tiff",
        input_layout="HWDC",
    )
    with pytest.raises(ValueError, match="requires positive per-case spacing"):
        DatasetManifest(task=task, cases=[case])

    (tmp_path / "case_001.json").write_text('{"spacing": [0.8, 0.8, 2.5]}', encoding="utf-8")
    with_sidecar = DatasetManifest(task=task, cases=[case])

    assert with_sidecar.cases[0].spacing == (0.8, 0.8, 2.5)


def test_hwdc_to_dhwc_reorders_array_and_spacing_together():
    source = np.arange(2 * 3 * 4).reshape(2, 3, 4, 1)

    normalized, spacing = normalize_layout_and_spacing(
        source,
        spatial_dims=3,
        spacing=(0.7, 0.8, 2.5),
        layout="HWDC",
    )

    np.testing.assert_array_equal(normalized, source.transpose(2, 0, 1, 3))
    assert normalized.shape == (4, 2, 3, 1)
    assert spacing == (2.5, 0.7, 0.8)
