import json

import numpy as np
import pytest

from medicai.dataloader.nnunet.manifest import CaseRecord, DatasetManifest, TaskSpec
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
        images={"image": path},
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


def test_typed_manifest_json_roundtrip(tmp_path):
    manifest = DatasetManifest(task=_task("binary"), cases=[_case(split="train")])
    path = tmp_path / "manifest.json"

    manifest.to_json(path)
    loaded = DatasetManifest.from_json(path)

    assert loaded.task.to_dict() == manifest.task.to_dict()
    assert loaded.cases[0].to_dict() == manifest.cases[0].to_dict()


def test_legacy_manifest_is_still_supported(tmp_path):
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

    loaded = DatasetManifest.from_json(path)

    assert loaded.task_type == "binary"
    assert loaded.items[0].image_layout == "DHWC"


@pytest.mark.parametrize(
    "task,case,error",
    [
        (
            _task(),
            CaseRecord(id="x", images={"wrong": "x.npy"}, label="y.npy", input_layout="DHWC"),
            "modalities mismatch",
        ),
        (
            _task(),
            CaseRecord(id="x", images={"image": "x.npy"}, label="y.npy", input_layout="DDHW"),
            "unique axis letters",
        ),
        (
            _task(),
            CaseRecord(
                id="x",
                images={"image": "x.npy"},
                label="y.npy",
                input_layout="HWDC",
                spacing=(1, 1),
            ),
            "spacing has",
        ),
    ],
)
def test_invalid_case_contracts_fail_early(task, case, error):
    with pytest.raises(ValueError, match=error):
        DatasetManifest(task=task, cases=[case])


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


def test_multilabel_named_masks_must_match_task_names():
    task = TaskSpec(
        "multi_label",
        ["image"],
        {"background": 0, "organ": 1, "lesion": 2},
    )
    case = CaseRecord(
        id="case_001",
        images={"image": "case.npy"},
        label={"organ": "organ.npy"},
        input_layout="DHWC",
        spacing=(1, 1, 1),
    )

    with pytest.raises(ValueError, match="mask names must match"):
        DatasetManifest(task=task, cases=[case])


def test_tiff_spacing_required_or_read_from_official_sidecar(tmp_path):
    task = _task()
    image_path = tmp_path / "case_001_0000.tiff"
    case = CaseRecord(
        id="case_001",
        images={"image": str(image_path)},
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
