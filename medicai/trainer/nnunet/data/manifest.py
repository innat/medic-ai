"""Typed dataset and geometry contracts for the nnU-Net trainer workflow."""

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
    "region_based": "region_based",
}
_SPATIAL_AXES = {"HW", "DHW"}
_LAYOUT_KEYS = {"image", "label"}
_ARRAY_SUFFIXES = (".npy", ".tif", ".tiff")


def _canonical_task_type(value: str) -> str:
    try:
        return _TASK_ALIASES[value]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unsupported task_type {value!r}; choose from "
            "'binary', 'multi_class', or 'region_based'."
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


def _resolve_layout_pair(
    input_layout: str | dict[str, str] | None,
    *,
    fallback_label_layout: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve one shared or key-specific layout declaration to image/label layouts."""
    if isinstance(input_layout, dict):
        unknown = set(input_layout) - _LAYOUT_KEYS
        if unknown:
            raise ValueError(
                f"input_layout only supports 'image' and 'label' keys; got {sorted(unknown)}."
            )
        image_layout = input_layout.get("image")
        label_layout = input_layout.get("label")
    else:
        image_layout = input_layout
        label_layout = None

    if image_layout is not None and not isinstance(image_layout, str):
        raise TypeError("input_layout image layout must be a string.")
    if label_layout is not None and not isinstance(label_layout, str):
        raise TypeError("input_layout label layout must be a string.")
    image_layout = image_layout.upper() if image_layout is not None else None
    label_layout = label_layout.upper() if label_layout is not None else None
    if label_layout is None:
        label_layout = fallback_label_layout
        label_layout = label_layout.upper() if label_layout is not None else None
    if label_layout is None and image_layout is not None:
        label_layout = image_layout.replace("C", "")
    return image_layout, label_layout


def _resolve_case_layouts(
    case_layout: str | dict[str, str] | None,
    default_layout: str | dict[str, str] | None,
) -> tuple[str | None, str | None]:
    default_image, default_label = _resolve_layout_pair(default_layout)
    case_image, case_label = _resolve_layout_pair(
        case_layout,
        fallback_label_layout=default_label,
    )
    image_layout = case_image or default_image
    label_layout = case_label or default_label
    if label_layout is None and image_layout is not None:
        label_layout = image_layout.replace("C", "")
    return image_layout, label_layout


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
        task_type: ``binary``, ``multi_class``, or ``region_based``.
        modalities: Ordered modality/channel names, matching every case's image map.
        labels: Ordered label-name to integer-ID mapping. Categorical labels
            must include background ID 0. Multi-class IDs must be consecutive;
            region-based tasks may use sparse IDs such as BraTS' 1, 2, and 4.
        regions: Optional named region to categorical label-ID mapping.
        regions_class_order: Integer labels used to collapse region probabilities
            to one map, in region order. Keep probabilities for overlapping output.
        ignore_class_ids: Optional label sentinels excluded from loss and metrics.
        target_class_ids: Optional declared foreground labels included in targets.
    """

    task_type: str
    modalities: list[str]
    labels: dict[str, int] | list[str]
    regions: dict[str, list[int]] = field(default_factory=dict)
    regions_class_order: list[int] | None = None
    ignore_class_ids: list[int] = field(default_factory=list)
    target_class_ids: list[int] = field(default_factory=list)

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
        self.ignore_class_ids = [int(value) for value in self.ignore_class_ids]
        self.target_class_ids = [int(value) for value in self.target_class_ids]
        self.validate()

    def validate(self) -> None:
        if not self.modalities or any(not item for item in self.modalities):
            raise ValueError("TaskSpec.modalities must contain at least one non-empty name.")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("TaskSpec.modalities must be unique and ordered.")
        if not self.labels or len(set(self.labels.values())) != len(self.labels):
            raise ValueError("TaskSpec.labels must contain unique integer IDs.")

        ids = sorted(self.labels.values())
        if self.labels.get("background") != 0:
            raise ValueError("Categorical/region labels must define 'background' as ID 0.")
        if any(label_id < 0 for label_id in ids):
            raise ValueError("TaskSpec label IDs must be non-negative integers.")
        if self.task_type == "multi_class" and ids != list(range(len(ids))):
            raise ValueError("Multi-class label IDs must be consecutive integers beginning at 0.")

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
        elif self.regions:
            raise ValueError("Declare regions only for region_based tasks.")
        if len(set(self.ignore_class_ids)) != len(self.ignore_class_ids):
            raise ValueError("TaskSpec.ignore_class_ids must be unique.")
        if len(set(self.target_class_ids)) != len(self.target_class_ids):
            raise ValueError("TaskSpec.target_class_ids must be unique.")
        if not set(self.target_class_ids) <= (set(ids) - {0}):
            raise ValueError("TaskSpec.target_class_ids must be declared foreground label IDs.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(init=False)
class CaseRecord:
    """Describe one case with ordered image paths, a label, and source geometry.

    ``input_layout`` describes array/TIFF source axes (for example ``HWDC`` or
    ``{"image": "HWDC", "label": "HWD"}``), not MedicAI's canonical internal
    layout. A shared string applies to both arrays; when it contains ``C``, the
    label layout is inferred by removing that axis. Channel-free images gain a
    channel axis when modalities are stacked. For NIfTI, the configured reader
    derives axis geometry from the file header. Spacing follows the image's
    source spatial-axis order and is permuted with the array at ingestion.
    A single ``image`` path is used for one modality; multiple paths follow the
    modality order declared by :class:`TaskSpec`.

    Args:
        id: Stable, unique case identifier.
        image: One path, or ordered paths matching ``TaskSpec.modalities``.
        label: Path to one categorical label map. Every manifest case is a labeled training case.
        input_layout: Shared source axes such as ``"DHW"``/``"HWDC"``, or a mapping
            with ``"image"`` and optionally ``"label"`` axis strings.
        spacing: Positive voxel spacing ordered like the image's source spatial axes.
        meta: Additional case metadata retained with the record.

    Examples:
        A channel-free 3D array declares only its spatial axes; preprocessing
        adds the trailing image-channel axis while stacking modalities::

            case = CaseRecord(
                id="case_001",
                image="image.npy",
                label="mask.npy",
                input_layout="DHW",
                spacing=(1.0, 1.0, 1.0),
            )

        Different image and label orders use the keyed form::

            input_layout={"image": "HWDC", "label": "HWD"}
    """

    id: str
    image: str | list[str] | dict[str, str]
    label: str
    input_layout: str | dict[str, str] | None = None
    spacing: tuple[float, ...] | list[float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        id: str,
        image: str | list[str] | dict[str, str],
        label: str,
        input_layout: str | dict[str, str] | None = None,
        spacing: tuple[float, ...] | list[float] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.image = image
        self.label = label
        self.input_layout = input_layout
        self.spacing = spacing
        self.meta = {} if meta is None else meta
        self.__post_init__()

    def __post_init__(self) -> None:
        self.id = str(self.id)
        if self.image is None or self.label is None:
            raise ValueError("CaseRecord requires both image and label paths.")
        if isinstance(self.image, dict):
            self.image = {str(name): str(path) for name, path in self.image.items()}
        elif isinstance(self.image, (list, tuple)):
            self.image = [str(path) for path in self.image]
        else:
            self.image = str(self.image)
        if isinstance(self.label, (dict, list)):
            raise ValueError("CaseRecord.label must be one categorical label-map path.")
        self.label = str(self.label)
        if self.spacing is not None:
            self.spacing = tuple(float(value) for value in self.spacing)

    def image_map(self, modalities: list[str]) -> dict[str, str]:
        """Pair this case's paths with the modality order declared by its task."""
        if isinstance(self.image, dict):
            image_map = self.image
            if set(image_map) != set(modalities):
                missing = sorted(set(modalities) - set(image_map))
                extra = sorted(set(image_map) - set(modalities))
                raise ValueError(f"image modalities mismatch; missing={missing}, extra={extra}.")
            return {modality: image_map[modality] for modality in modalities}

        paths = [self.image] if isinstance(self.image, str) else self.image
        if not paths:
            raise ValueError("CaseRecord.image must contain at least one image path.")
        if len(paths) != len(modalities):
            raise ValueError(
                f"Case has {len(paths)} image path(s) for {len(modalities)} declared modalities."
            )
        return dict(zip(modalities, paths, strict=True))

    @property
    def image_paths(self) -> list[str]:
        """Return image paths without modality names, preserving their input order."""
        if isinstance(self.image, dict):
            return list(self.image.values())
        return [self.image] if isinstance(self.image, str) else list(self.image or [])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ManifestItem:
    """Internal normalized case view consumed by preprocessing stages."""

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
    label_names: list[str] | None = None


def _tiff_sidecar(image_path: str) -> Path:
    path = Path(image_path)
    case_stem = path.stem
    case_stem = re.sub(r"_\d{4}$", "", case_stem)
    return path.with_name(f"{case_stem}.json")


def _load_sidecar_spacing(case: CaseRecord) -> tuple[float, ...] | None:
    for path in case.image_paths:
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
    """Validated typed manifest of a segmentation task and labeled cases."""

    def __init__(
        self,
        task: TaskSpec | dict[str, Any],
        cases: list[CaseRecord | dict[str, Any]],
        name: str = "CustomDataset",
        input_layout: str | dict[str, str] | None = None,
        spatial_dims: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.task = task if isinstance(task, TaskSpec) else TaskSpec(**task)
        self.name = str(name)
        self._input_layout = input_layout
        self._spatial_dims = spatial_dims
        self.metadata = dict(metadata or {})
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
        }.get(self.task.task_type, self.task.task_type)

    @property
    def ignore_class_ids(self) -> list[int]:
        return list(self.task.ignore_class_ids)

    @property
    def target_class_ids(self) -> list[int]:
        return list(self.task.target_class_ids)

    @property
    def dataset_name(self) -> str:
        return self.name

    @property
    def input_layout(self) -> str | dict[str, str] | None:
        return self._input_layout

    @property
    def image_layout(self) -> str | None:
        """Resolved source-axis layout for image arrays."""
        return _resolve_case_layouts(
            None,
            self.input_layout,
        )[0]

    @property
    def label_layout(self) -> str | None:
        """Resolved source-axis layout for labels using the manifest default."""
        return _resolve_case_layouts(None, self.input_layout)[1]

    @property
    def spatial_dims(self) -> int:
        configured = self._spatial_dims
        if configured is not None:
            configured = int(configured)
            if configured not in (2, 3):
                raise ValueError("manifest spatial_dims must be 2 or 3.")
            return configured
        layouts = [
            _resolve_case_layouts(
                case.input_layout,
                self.input_layout,
            )[0]
            for case in self.cases
        ]
        layouts = [layout for layout in layouts if layout]
        ranks = {len(layout.replace("C", "")) for layout in layouts}
        if len(ranks) > 1:
            raise ValueError("All cases in one dataset must use the same spatial rank.")
        return next(iter(ranks)) if ranks else 3

    @property
    def label_output(self) -> str:
        if self.task.task_type == "region_based":
            return "regions"
        return "auto"

    @property
    def regions(self) -> list[list[int]]:
        return [list(values) for values in self.task.regions.values()]

    def _validate_and_adapt(self) -> None:
        self.task.validate()
        if not self.cases:
            raise ValueError("DatasetManifest.cases must contain at least one case.")
        dataset_spatial_dims = self.spatial_dims
        case_ids = [case.id for case in self.cases]
        if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
            raise ValueError("Case IDs must be non-empty and unique.")

        for case in self.cases:
            try:
                image_map = case.image_map(self.modalities)
            except ValueError as exc:
                raise ValueError(f"Case {case.id!r}: {exc}") from exc
            if not image_map:
                raise ValueError(f"Case {case.id!r} must provide at least one image modality.")
            if any(not path for path in image_map.values()):
                raise ValueError(f"Case {case.id!r} contains an empty image path.")
            if not case.label:
                raise ValueError(f"Case {case.id!r} must provide a non-empty label path.")
            image_layout, label_layout = _resolve_case_layouts(
                case.input_layout,
                self.input_layout,
            )
            image_dims = _layout_spatial_dims(
                image_layout, field_name=f"case {case.id} input_layout"
            )
            if image_dims is not None and image_dims != dataset_spatial_dims:
                raise ValueError(
                    f"Case {case.id!r} declares {image_dims}D input_layout, but this dataset "
                    f"uses {dataset_spatial_dims}D spatial data."
                )
            label_dims = _layout_spatial_dims(
                label_layout, field_name=f"case {case.id} input_layout['label']"
            )
            expected_dims = image_dims or dataset_spatial_dims
            if label_dims is not None and label_dims != expected_dims:
                raise ValueError(
                    f"Case {case.id!r} label layout declares {label_dims}D data, but its "
                    f"image layout declares {expected_dims}D data."
                )

            suffixes = {Path(path).suffix.lower() for path in image_map.values()}
            if (
                any(
                    str(path).lower().endswith((".nii", ".nii.gz"))
                    for path in image_map.values()
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

            ordered_paths = list(image_map.values())
            case.image = ordered_paths[0] if len(ordered_paths) == 1 else ordered_paths
            self.items.append(self._adapt_case(case, image_layout, label_layout))

    def _adapt_case(
        self, case: CaseRecord, image_layout: str | None, label_layout: str | None
    ) -> ManifestItem:
        task_type = self.task.task_type
        if task_type == "region_based":
            internal_task = "multi-label"
            label_output = "regions"
        else:
            internal_task = self.task_type
            label_output = "auto"
        return ManifestItem(
            case_id=case.id,
            images=list(case.image_map(self.modalities).values()),
            labels=case.label,
            spacing=list(case.spacing) if case.spacing is not None else None,
            task_type=internal_task,
            image_layout=image_layout,
            label_layout=label_layout,
            label_output=label_output,
            regions=self.regions if task_type == "region_based" else None,
            meta=case.meta,
        )

    @classmethod
    def from_json(cls, manifest_path: str | Path) -> DatasetManifest:
        """Load the typed ``task``/``cases`` manifest format."""
        with Path(manifest_path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        required = {"task", "cases"}
        missing = required - set(data)
        if missing:
            raise ValueError(
                "Manifest must use the typed 'task' and 'cases' fields; "
                f"missing {sorted(missing)}."
            )
        return cls(
            task=data["task"],
            cases=data["cases"],
            name=data.get("name", "CustomDataset"),
            input_layout=data.get("input_layout"),
            spatial_dims=data.get("spatial_dims"),
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        cases = []
        for case in self.cases:
            record = case.to_dict()
            ordered_paths = list(case.image_map(self.modalities).values())
            record["image"] = ordered_paths[0] if len(ordered_paths) == 1 else ordered_paths
            cases.append(record)
        return {
            "task": self.task.to_dict(),
            "cases": cases,
            "name": self.name,
            "input_layout": self.input_layout,
            "spatial_dims": self._spatial_dims,
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path) -> None:
        """Serialize the canonical typed manifest as readable JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)
