"""Dataset and geometry contracts for the nnU-Net trainer workflow.

The public dataclasses provide a typed MedicAI manifest while ``ManifestItem``
keeps the current preprocessing pipeline's legacy list-based interface alive.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_TASK_ALIASES = {
    "binary": "binary",
    "multi_class": "multi_class",
    "multi-class": "multi_class",
    "region_based": "region_based",
    "region-based": "region_based",
    "multi_label": "multi_label",
    "multi-label": "multi_label",
}
_VALID_SPLITS = {"train", "validation", "test", "inference"}
_SPATIAL_AXES = {"HW", "DHW"}
_ARRAY_SUFFIXES = (".npy", ".tif", ".tiff")


def _canonical_task_type(value: str) -> str:
    try:
        return _TASK_ALIASES[value]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unsupported task_type {value!r}; choose from "
            "'binary', 'multi_class', 'region_based', or 'multi_label'."
        ) from exc


def _layout_spatial_dims(layout: str | None, *, field_name: str) -> int | None:
    if layout is None:
        return None
    normalized = "".join(ch for ch in str(layout).upper() if ch.isalpha())
    if len(normalized) != len(layout) or len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique axis letters, got {layout!r}.")
    axes = normalized.replace("C", "")
    if axes not in _SPATIAL_AXES and axes[::-1] not in _SPATIAL_AXES:
        # Valid layouts may permute spatial axes arbitrarily, but contain either
        # exactly H/W or exactly D/H/W.
        if len(axes) not in (2, 3) or set(axes) != (set("HW") if len(axes) == 2 else set("DHW")):
            raise ValueError(
                f"{field_name} must be a permutation of HW/HWC or DHW/DHWC; got {layout!r}."
            )
    if "C" in normalized and normalized.count("C") != 1:
        raise ValueError(f"{field_name} may contain at most one channel axis, got {layout!r}.")
    return len(axes)


def _as_label_mapping(labels: dict[str, Any] | list[str] | tuple[str, ...]) -> dict[str, int]:
    if isinstance(labels, dict):
        try:
            return {str(name): int(value) for name, value in labels.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Task labels must map label names to integer IDs.") from exc
    if isinstance(labels, (list, tuple)):
        return {str(name): idx for idx, name in enumerate(labels)}
    raise TypeError("TaskSpec.labels must be a name-to-ID mapping or an ordered label-name list.")


@dataclass
class TaskSpec:
    """Describe a segmentation task and its categorical/region label contract.

    Args:
        task_type: ``binary``, ``multi_class``, ``region_based``, or
            ``multi_label``. Multi-label is a MedicAI convenience contract that
            is encoded through nnU-Net-style overlapping regions.
        modalities: Ordered modality/channel names, matching every case's image map.
        labels: Ordered label-name to integer-ID mapping. Categorical labels
            must include background ID 0; multi-label IDs name independent masks.
        regions: Optional named region to categorical label-ID mapping.
        regions_class_order: Integer labels used to collapse region probabilities
            to one map, in region order. Keep probabilities for overlapping output.
    """

    task_type: str
    modalities: list[str]
    labels: dict[str, int] | list[str]
    regions: dict[str, list[int]] = field(default_factory=dict)
    regions_class_order: list[int] | None = None

    def __post_init__(self) -> None:
        self.task_type = _canonical_task_type(self.task_type)
        self.modalities = [str(value) for value in self.modalities]
        self.labels = _as_label_mapping(self.labels)
        self.regions = {
            str(name): [int(label_id) for label_id in label_ids]
            for name, label_ids in self.regions.items()
        }
        if self.regions_class_order is not None:
            self.regions_class_order = [int(value) for value in self.regions_class_order]
        self.validate()

    def validate(self) -> None:
        if not self.modalities or any(not item for item in self.modalities):
            raise ValueError("TaskSpec.modalities must contain at least one non-empty name.")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("TaskSpec.modalities must be unique and ordered.")
        if not self.labels or len(set(self.labels.values())) != len(self.labels):
            raise ValueError("TaskSpec.labels must contain unique integer IDs.")

        ids = sorted(self.labels.values())
        if self.task_type != "multi_label":
            if self.labels.get("background") != 0:
                raise ValueError("Categorical/region labels must define 'background' as ID 0.")
            if ids != list(range(len(ids))):
                raise ValueError(
                    "Categorical label IDs must be consecutive integers beginning at 0."
                )
        elif 0 in ids and self.labels.get("background") != 0:
            raise ValueError(
                "If multi-label metadata includes ID 0, it must be named 'background'."
            )
        elif ids and ids != list(range(ids[0], ids[0] + len(ids))):
            raise ValueError("Multi-label IDs must be consecutive.")

        if self.task_type == "binary" and ids != [0, 1]:
            raise ValueError("Binary tasks require exactly background=0 and foreground=1.")
        if self.task_type == "region_based":
            if not self.regions:
                raise ValueError("Region-based tasks require at least one named region.")
            valid_ids = set(ids) - {0}
            if any(not values or not set(values) <= valid_ids for values in self.regions.values()):
                raise ValueError("Regions must contain declared, non-background label IDs.")
            if self.regions_class_order is None:
                raise ValueError("Region-based tasks require regions_class_order.")
            if len(self.regions_class_order) != len(self.regions):
                raise ValueError("regions_class_order must have one entry per declared region.")
            if any(value not in valid_ids for value in self.regions_class_order):
                raise ValueError(
                    "regions_class_order entries must be declared foreground label IDs."
                )
        elif self.regions and self.task_type != "multi_label":
            raise ValueError("Declare regions only for region_based or multi_label tasks.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseRecord:
    """Describe one case with named modalities, optional labels, and source geometry.

    ``input_layout`` describes array/TIFF source axes (for example ``HWDC``),
    not MedicAI's canonical internal layout. For NIfTI, the configured reader
    derives axis geometry from the file header. Spacing follows the source
    layout's spatial-axis order and is permuted with the array at ingestion.
    """

    id: str
    images: dict[str, str]
    label: str | dict[str, str] | list[str] | None = None
    split: str | None = None
    input_layout: str | None = None
    label_layout: str | None = None
    spacing: tuple[float, ...] | list[float] | None = None
    affine: list[list[float]] | None = None
    origin: list[float] | None = None
    direction: list[list[float]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.images = {str(name): str(path) for name, path in self.images.items()}
        if isinstance(self.label, dict):
            self.label = {str(name): str(path) for name, path in self.label.items()}
        elif isinstance(self.label, list):
            self.label = [str(path) for path in self.label]
        elif self.label is not None:
            self.label = str(self.label)
        if self.spacing is not None:
            self.spacing = tuple(float(value) for value in self.spacing)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ManifestItem:
    """Compatibility representation consumed by the existing preprocessing stages."""

    case_id: str
    images: list[str]
    labels: str | list[str] | None = None
    spacing: list[float] | None = None
    task_type: str | None = None
    image_layout: str | None = None
    label_layout: str | None = None
    label_output: str | None = None
    regions: list[list[int]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    split: str | None = None
    label_names: list[str] | None = None


def _legacy_task(global_meta: dict[str, Any]) -> TaskSpec:
    names = global_meta.get("class_names", global_meta.get("labels", []))
    if isinstance(names, dict):
        labels = names
    else:
        labels = {str(name): idx for idx, name in enumerate(names)}
    task_type = _canonical_task_type(global_meta.get("task_type", "multi_class"))
    region_values = global_meta.get("regions", {})
    if isinstance(region_values, list):
        regions = {
            f"region_{i + 1}": [int(v) for v in values] for i, values in enumerate(region_values)
        }
    else:
        regions = region_values or {}
    order = global_meta.get("regions_class_order")
    if task_type == "region_based" and order is None:
        order = list(range(1, len(regions) + 1))
    return TaskSpec(
        task_type=task_type,
        modalities=global_meta.get("modalities", []),
        labels=labels,
        regions=regions,
        regions_class_order=order,
    )


def _tiff_sidecar(image_path: str) -> Path:
    path = Path(image_path)
    case_stem = path.stem
    case_stem = re.sub(r"_\d{4}$", "", case_stem)
    return path.with_name(f"{case_stem}.json")


def _load_sidecar_spacing(case: CaseRecord) -> tuple[float, ...] | None:
    for path in case.images.values():
        if Path(path).suffix.lower() not in {".tif", ".tiff"}:
            continue
        sidecar = _tiff_sidecar(path)
        if sidecar.is_file():
            with sidecar.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            if "spacing" not in data:
                raise ValueError(f"TIFF spacing sidecar {sidecar} must define 'spacing'.")
            try:
                return tuple(float(value) for value in data["spacing"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"TIFF spacing sidecar {sidecar} has invalid spacing.") from exc
    return None


class DatasetManifest:
    """Validated dataset manifest with a typed API and legacy pipeline adapter."""

    def __init__(
        self,
        items: list[ManifestItem] | None = None,
        global_meta: dict[str, Any] | None = None,
        *,
        task: TaskSpec | dict[str, Any] | None = None,
        cases: list[CaseRecord | dict[str, Any]] | None = None,
    ) -> None:
        self.global_meta = dict(global_meta or {})
        if task is None:
            self.task = _legacy_task(self.global_meta)
        elif isinstance(task, TaskSpec):
            self.task = task
        else:
            self.task = TaskSpec(**task)

        if cases is None:
            self.cases = self._cases_from_legacy_items(items or [])
        else:
            self.cases = [
                case if isinstance(case, CaseRecord) else CaseRecord(**case) for case in cases
            ]
        self.items: list[ManifestItem] = []
        self._validate_and_adapt()

    @property
    def modalities(self) -> list[str]:
        return self.task.modalities

    @property
    def class_names(self) -> list[str]:
        return list(self.task.labels)

    @property
    def task_type(self) -> str:
        return {
            "multi_class": "multi-class",
            "multi_label": "multi-label",
            "region_based": "multi-label",
        }.get(self.task.task_type, self.task.task_type)

    @property
    def ignore_class_ids(self) -> list[int]:
        return list(self.global_meta.get("ignore_class_ids", []))

    @property
    def target_class_ids(self) -> list[int]:
        value = self.global_meta.get("target_class_ids", [])
        return list(value) if value is not None else []

    @property
    def dataset_name(self) -> str:
        return self.global_meta.get("name", "CustomDataset")

    @property
    def input_layout(self) -> str | None:
        return self.global_meta.get("input_layout", self.global_meta.get("image_layout"))

    @property
    def image_layout(self) -> str | None:
        """Backward-compatible alias for the source ``input_layout``."""
        return self.input_layout

    @property
    def spatial_dims(self) -> int:
        configured = self.global_meta.get("spatial_dims")
        if configured is not None:
            configured = int(configured)
            if configured not in (2, 3):
                raise ValueError("manifest spatial_dims must be 2 or 3.")
            return configured
        layouts = [case.input_layout or self.input_layout for case in self.cases]
        layouts = [layout for layout in layouts if layout]
        ranks = {len(layout.replace("C", "")) for layout in layouts}
        if len(ranks) > 1:
            raise ValueError("All cases in one dataset must use the same spatial rank.")
        return next(iter(ranks)) if ranks else 3

    @property
    def label_layout(self) -> str | None:
        return self.global_meta.get("label_layout")

    @property
    def label_output(self) -> str:
        explicit = self.global_meta.get("label_output")
        if explicit is not None:
            if explicit not in {"auto", "sparse", "regions", "channel_masks"}:
                raise ValueError(f"Unsupported label_output {explicit!r}.")
            return explicit
        if self.task.task_type in {"region_based", "multi_label"}:
            return "regions" if self.task.task_type == "region_based" else "channel_masks"
        return "auto"

    @property
    def regions(self) -> list[list[int]]:
        return [list(values) for values in self.task.regions.values()]

    def _cases_from_legacy_items(self, items: list[ManifestItem]) -> list[CaseRecord]:
        cases = []
        for item in items:
            image_layout = item.image_layout or self.input_layout
            images = {
                modality: path for modality, path in zip(self.modalities, item.images, strict=False)
            }
            if isinstance(item.labels, dict):
                labels = item.labels
            else:
                labels = item.labels
            cases.append(
                CaseRecord(
                    id=item.case_id,
                    images=images,
                    label=labels,
                    split=item.split,
                    input_layout=image_layout,
                    label_layout=item.label_layout or self.label_layout,
                    spacing=item.spacing,
                    meta=item.meta,
                )
            )
        return cases

    def _validate_and_adapt(self) -> None:
        self.task.validate()
        if not self.cases:
            raise ValueError("DatasetManifest.cases must contain at least one case.")
        case_ids = [case.id for case in self.cases]
        if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
            raise ValueError("Case IDs must be non-empty and unique.")

        for case in self.cases:
            if not case.images:
                raise ValueError(f"Case {case.id!r} must provide at least one image modality.")
            if set(case.images) != set(self.modalities):
                missing = sorted(set(self.modalities) - set(case.images))
                extra = sorted(set(case.images) - set(self.modalities))
                raise ValueError(
                    f"Case {case.id!r} image modalities mismatch; missing={missing}, extra={extra}."
                )
            if any(not path for path in case.images.values()):
                raise ValueError(f"Case {case.id!r} contains an empty image path.")
            if case.split is not None and case.split not in _VALID_SPLITS:
                raise ValueError(
                    f"Case {case.id!r} has invalid split {case.split!r}; choose from {sorted(_VALID_SPLITS)}."
                )
            if case.label is None and case.split not in {"test", "inference"}:
                raise ValueError(f"Labeled case {case.id!r} has no label/mask path.")
            if isinstance(case.label, dict) and self.task.task_type != "multi_label":
                raise ValueError(
                    f"Case {case.id!r} uses named mask paths, but its task is not multi_label."
                )
            if isinstance(case.label, dict):
                expected_masks = set(self.task.labels) - {"background"}
                if set(case.label) != expected_masks:
                    raise ValueError(
                        f"Case {case.id!r} mask names must match declared multi-label targets; "
                        f"expected={sorted(expected_masks)}, got={sorted(case.label)}."
                    )

            image_layout = case.input_layout or self.input_layout
            image_dims = _layout_spatial_dims(
                image_layout, field_name=f"case {case.id} input_layout"
            )
            label_layout = case.label_layout or self.label_layout
            if label_layout is None and image_layout is not None:
                label_layout = image_layout.replace("C", "")
            _layout_spatial_dims(label_layout, field_name=f"case {case.id} label_layout")

            suffixes = {Path(path).suffix.lower() for path in case.images.values()}
            if (
                any(
                    str(path).lower().endswith((".nii", ".nii.gz")) for path in case.images.values()
                )
                and image_layout
            ):
                raise ValueError(
                    f"Case {case.id!r} uses NIfTI; omit input_layout so the selected reader "
                    "derives axes from the header."
                )
            needs_layout = any(suffix in _ARRAY_SUFFIXES for suffix in suffixes)
            if needs_layout and image_layout is None:
                raise ValueError(
                    f"Case {case.id!r} uses array/TIFF data; declare input_layout (for example 'DHWC' or 'HWDC')."
                )

            spacing = case.spacing if case.spacing is not None else _load_sidecar_spacing(case)
            has_tiff = any(suffix in {".tif", ".tiff"} for suffix in suffixes)
            if has_tiff and spacing is None:
                raise ValueError(
                    f"TIFF case {case.id!r} requires positive per-case spacing or an official <case>.json sidecar."
                )
            if spacing is not None:
                if any(not math.isfinite(value) or value <= 0 for value in spacing):
                    raise ValueError(
                        f"Case {case.id!r} spacing must contain finite positive values."
                    )
                if image_dims is not None and len(spacing) != image_dims:
                    raise ValueError(
                        f"Case {case.id!r} spacing has {len(spacing)} values, but input_layout "
                        f"{image_layout!r} declares {image_dims} spatial axes."
                    )
                case.spacing = tuple(spacing)

            spatial_dims = image_dims or self.spatial_dims
            if case.affine is not None:
                if len(case.affine) != 4 or any(len(row) != 4 for row in case.affine):
                    raise ValueError(f"Case {case.id!r} affine must have shape (4, 4).")
                if any(not math.isfinite(float(value)) for row in case.affine for value in row):
                    raise ValueError(f"Case {case.id!r} affine must contain finite values.")
            if case.origin is not None:
                if len(case.origin) != spatial_dims or any(
                    not math.isfinite(float(value)) for value in case.origin
                ):
                    raise ValueError(
                        f"Case {case.id!r} origin must contain {spatial_dims} finite values."
                    )
            if case.direction is not None:
                if (
                    len(case.direction) != spatial_dims
                    or any(len(row) != spatial_dims for row in case.direction)
                    or any(
                        not math.isfinite(float(value)) for row in case.direction for value in row
                    )
                ):
                    raise ValueError(
                        f"Case {case.id!r} direction must have shape "
                        f"({spatial_dims}, {spatial_dims}) and finite values."
                    )

            self.items.append(self._adapt_case(case, image_layout, label_layout))

    def _adapt_case(
        self, case: CaseRecord, image_layout: str | None, label_layout: str | None
    ) -> ManifestItem:
        label_names: list[str] | None = None
        if isinstance(case.label, dict):
            label_names = list(case.label)
            label_paths: str | list[str] = list(case.label.values())
        elif isinstance(case.label, list):
            label_paths = case.label
        else:
            label_paths = case.label
        task_type = self.task.task_type
        if task_type == "region_based":
            internal_task = "multi-label"
            label_output = "regions"
        elif task_type == "multi_label":
            internal_task = "multi-label"
            label_output = "channel_masks"
        else:
            internal_task = self.task_type
            label_output = "auto"
        return ManifestItem(
            case_id=case.id,
            images=[case.images[modality] for modality in self.modalities],
            labels=label_paths,
            spacing=list(case.spacing) if case.spacing is not None else None,
            task_type=internal_task,
            image_layout=image_layout,
            label_layout=label_layout,
            label_output=label_output,
            regions=self.regions if task_type == "region_based" else None,
            meta={
                **case.meta,
                **{
                    key: value
                    for key, value in {
                        "affine": case.affine,
                        "origin": case.origin,
                        "direction": case.direction,
                    }.items()
                    if value is not None
                },
                **({"label_names": label_names} if label_names else {}),
            },
            split=case.split,
            label_names=label_names,
        )

    @classmethod
    def from_json(cls, manifest_path: str | Path) -> DatasetManifest:
        """Load either the typed ``task``/``cases`` format or the legacy ``meta``/``items`` format."""
        with Path(manifest_path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if "task" in data and "cases" in data:
            return cls(task=data["task"], cases=data["cases"], global_meta=data.get("meta"))

        global_meta = data.get("meta", {})
        task = _legacy_task(global_meta)
        cases = []
        for index, item in enumerate(data.get("items", [])):
            raw_images = item.get("images", item.get("image"))
            if isinstance(raw_images, dict):
                images = {str(key): str(value) for key, value in raw_images.items()}
            else:
                paths = (
                    [str(v) for v in raw_images]
                    if isinstance(raw_images, list)
                    else [str(raw_images)]
                )
                modalities = task.modalities
                if len(paths) != len(modalities):
                    raise ValueError(
                        f"Legacy case {item.get('id', index)!r} has {len(paths)} images for "
                        f"{len(modalities)} declared modalities."
                    )
                images = dict(zip(modalities, paths, strict=True))
            labels = item.get("labels", item.get("label"))
            if isinstance(labels, list):
                labels = [str(value) for value in labels]
            elif labels is not None:
                labels = str(labels)
            cases.append(
                CaseRecord(
                    id=str(item.get("id", f"case_{index:04d}")),
                    images=images,
                    label=labels,
                    split=item.get("split"),
                    input_layout=item.get("input_layout", item.get("image_layout")),
                    label_layout=item.get("label_layout"),
                    spacing=item.get("spacing"),
                    meta=item.get("meta", {}),
                )
            )
        return cls(task=task, cases=cases, global_meta=global_meta)

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task.to_dict(), "cases": [case.to_dict() for case in self.cases]}

    def to_json(self, path: str | Path) -> None:
        """Serialize the canonical typed manifest as readable JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)
