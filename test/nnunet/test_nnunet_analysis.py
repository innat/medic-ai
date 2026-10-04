import numpy as np
import pytest

from medicai.trainer.nnunet.data.manifest import CaseRecord, DatasetManifest, TaskSpec
from medicai.trainer.nnunet import AnalysisReport, nnUNetPipeline


def _write_dataset(root, label_values):
    image_path = root / "case_001.npy"
    label_path = root / "case_001_label.npy"
    np.save(image_path, np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8))
    np.save(label_path, np.asarray(label_values, dtype=np.int16).reshape(8, 8, 8))

    manifest = DatasetManifest(
        task=TaskSpec(
            "binary",
            ["CT"],
            {"background": 0, "foreground": 1},
        ),
        cases=[
            CaseRecord(
                id="case_001",
                image=str(image_path),
                label=str(label_path),
                input_layout="DHW",
                spacing=(1.0, 1.0, 1.0),
            )
        ],
    )
    manifest.to_json(root / "manifest.json")
    return nnUNetPipeline(input_path=root)


def test_analyze_returns_fingerprint_without_writing_pipeline_artifacts(tmp_path):
    labels = np.zeros(8 * 8 * 8, dtype=np.int16)
    labels[100:140] = 1
    pipeline = _write_dataset(tmp_path, labels)

    report = pipeline.analyze()

    assert isinstance(report, AnalysisReport)
    assert report.is_valid
    assert not report.errors
    assert report.fingerprint.n_cases == 1
    assert not pipeline.fingerprint_path.exists()
    assert not pipeline.plan_path.exists()

    files_before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    repeated = pipeline.analyze()

    assert repeated.to_dict() == report.to_dict()
    files_after = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    assert files_after == files_before


def test_analyze_reports_undeclared_label_ids(tmp_path):
    labels = np.zeros(8 * 8 * 8, dtype=np.int16)
    labels[0] = 3
    pipeline = _write_dataset(tmp_path, labels)

    report = pipeline.analyze()

    assert report.fingerprint is None
    assert not report.is_valid
    assert any("undeclared label ID(s) [3]" in error for error in report.errors)


def test_analyze_reports_missing_image_files(tmp_path):
    pipeline = _write_dataset(tmp_path, np.zeros(8 * 8 * 8, dtype=np.int16))
    (tmp_path / "case_001.npy").unlink()

    report = pipeline.analyze()

    assert report.fingerprint is None
    assert not report.is_valid
    assert any("missing image file" in error for error in report.errors)


def test_analyze_reports_finite_image_values_with_case_context(tmp_path):
    labels = np.zeros(8 * 8 * 8, dtype=np.int16)
    pipeline = _write_dataset(tmp_path, labels)
    image_path = tmp_path / "case_001.npy"
    image = np.load(image_path)
    image[0, 0, 0] = np.nan
    np.save(image_path, image)

    report = pipeline.analyze()

    assert report.fingerprint is None
    assert any(
        "Case case_001" in error and "finite numeric values" in error
        for error in report.errors
    )


def test_analyze_reports_multiclass_prevalence_and_missing_classes(tmp_path):
    image_path = tmp_path / "multi_image.npy"
    label_path = tmp_path / "multi_label.npy"
    np.save(image_path, np.ones((8, 8, 8), dtype=np.float32))
    labels = np.zeros((8, 8, 8), dtype=np.int16)
    labels[0:2] = 1
    np.save(label_path, labels)
    manifest = DatasetManifest(
        task=TaskSpec(
            "multi_class",
            ["MR"],
            {"background": 0, "organ": 1, "lesion": 2},
        ),
        cases=[
            CaseRecord(
                id="multi_case",
                image=str(image_path),
                label=str(label_path),
                input_layout="DHW",
                spacing=(1.0, 1.0, 1.0),
            )
        ],
    )
    manifest.to_json(tmp_path / "manifest.json")
    report = nnUNetPipeline(input_path=tmp_path).analyze()

    assert report.is_valid
    assert report.class_prevalence["lesion"] == 0.0
    assert any("lesion" in warning for warning in report.warnings)


def test_region_analysis_preserves_public_task_type_and_overlap_prevalence(tmp_path):
    image_path = tmp_path / "region_image.npy"
    label_path = tmp_path / "region_label.npy"
    np.save(image_path, np.ones((4, 4, 4), dtype=np.float32))
    labels = np.zeros((4, 4, 4), dtype=np.int16)
    labels[0:2] = 1
    labels[1:3] = 2
    np.save(label_path, labels)
    manifest = DatasetManifest(
        task=TaskSpec(
            "region_based",
            ["MR"],
            {"background": 0, "core": 1, "enhancing": 2},
            regions={"whole": [1, 2], "core": [1]},
            regions_class_order=[1, 2],
        ),
        cases=[
            CaseRecord(
                id="region_case",
                image=str(image_path),
                label=str(label_path),
                input_layout="DHW",
                spacing=(1.0, 1.0, 1.0),
            )
        ],
    )
    manifest.to_json(tmp_path / "manifest.json")
    report = nnUNetPipeline(input_path=tmp_path).analyze()

    assert report.is_valid
    assert report.fingerprint.task_type == "region_based"
    assert report.region_prevalence["whole"] == 0.75
    assert report.region_prevalence["core"] == 0.25


def test_analyze_excludes_ignore_id_from_prevalence_denominator(tmp_path):
    image_path = tmp_path / "ignore_image.npy"
    label_path = tmp_path / "ignore_label.npy"
    np.save(image_path, np.ones((4, 4, 4), dtype=np.float32))
    labels = np.full((4, 4, 4), -1, dtype=np.int16)
    labels[:2] = 0
    labels[2:3] = 1
    np.save(label_path, labels)
    manifest = DatasetManifest(
        cases=[
            CaseRecord(
                id="ignore_case",
                image=str(image_path),
                label=str(label_path),
                input_layout="DHW",
                spacing=(1.0, 1.0, 1.0),
            )
        ],
        task=TaskSpec(
            "binary",
            ["CT"],
            {"background": 0, "foreground": 1},
            ignore_class_ids=[-1],
        ),
    )
    manifest.to_json(tmp_path / "manifest.json")
    report = nnUNetPipeline(input_path=tmp_path).analyze()

    assert report.is_valid
    assert report.class_prevalence["background"] == 2 / 3
    assert report.class_prevalence["foreground"] == 1 / 3


def test_manifest_rejects_unlabeled_inference_cases(tmp_path):
    inference_image = tmp_path / "inference_image.npy"
    np.save(inference_image, np.ones((4, 4, 4), dtype=np.float32))
    with pytest.raises(TypeError, match="label"):
        CaseRecord(
            id="inference_case",
            image=str(inference_image),
            input_layout="DHW",
            spacing=(1.0, 1.0, 1.0),
        )
