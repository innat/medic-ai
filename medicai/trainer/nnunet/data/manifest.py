import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_VALID_TASK_TYPES = {"binary", "multi-class", "multi-label"}
_VALID_LABEL_OUTPUTS = {"auto", "sparse", "regions", "channel_masks"}


def _as_path_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


@dataclass
class ManifestItem:
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


class DatasetManifest:
    """
    Parses a top-level manifest JSON structure mapping flexible metadata.
    """

    def __init__(
        self,
        items: list[ManifestItem],
        global_meta: dict[str, Any] | None = None,
    ):
        self.items = items
        self.global_meta = global_meta or {}

    @property
    def modalities(self):
        return self.global_meta.get("modalities", [])

    @property
    def class_names(self):
        return self.global_meta.get("class_names", self.global_meta.get("labels", []))

    @property
    def task_type(self):
        task_type = self.global_meta.get("task_type", "multi-class")
        if task_type not in _VALID_TASK_TYPES:
            raise ValueError(
                f"Unsupported task_type '{task_type}'. " f"Choose from {sorted(_VALID_TASK_TYPES)}."
            )
        return task_type

    @property
    def ignore_class_ids(self):
        return list(self.global_meta.get("ignore_class_ids", []))

    @property
    def target_class_ids(self):
        value = self.global_meta.get("target_class_ids", [])
        return list(value) if value is not None else []

    @property
    def dataset_name(self):
        return self.global_meta.get("name", "CustomDataset")

    @property
    def image_layout(self):
        return self.global_meta.get("image_layout")

    @property
    def label_layout(self):
        return self.global_meta.get("label_layout")

    @property
    def label_output(self):
        value = self.global_meta.get("label_output", "auto")
        if value not in _VALID_LABEL_OUTPUTS:
            raise ValueError(
                f"Unsupported label_output '{value}'. "
                f"Choose from {sorted(_VALID_LABEL_OUTPUTS)}."
            )
        return value

    @property
    def regions(self):
        return [[int(v) for v in region] for region in self.global_meta.get("regions", [])]

    @classmethod
    def from_json(cls, manifest_path: str | Path):
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        global_meta = data.get("meta", {})
        items = []
        global_task_type = global_meta.get("task_type", "multi-class")

        for i, item_dict in enumerate(data.get("items", [])):
            case_id = item_dict.get("id", f"case_{i:04d}")
            images = _as_path_list(item_dict.get("images", item_dict.get("image")))
            raw_labels = item_dict.get("labels", item_dict.get("label"))
            if isinstance(raw_labels, list):
                labels = [str(v) for v in raw_labels]
            elif raw_labels is None:
                labels = None
            else:
                labels = str(raw_labels)

            spacing = item_dict.get("spacing")
            if spacing is not None:
                spacing = [float(v) for v in spacing]

            item = ManifestItem(
                case_id=case_id,
                images=images,
                labels=labels,
                spacing=spacing,
                task_type=item_dict.get("task_type", item_dict.get("type", global_task_type)),
                image_layout=item_dict.get("image_layout"),
                label_layout=item_dict.get("label_layout"),
                label_output=item_dict.get("label_output"),
                regions=[[int(v) for v in region] for region in item_dict.get("regions", [])]
                or None,
                meta=item_dict.get("meta", {}),
            )
            items.append(item)

        manifest = cls(items=items, global_meta=global_meta)
        manifest._validate()
        return manifest

    def _validate(self):
        if not self.modalities:
            raise ValueError("manifest.meta.modalities is required.")
        if not self.class_names:
            raise ValueError("manifest.meta.class_names is required.")
        if self.task_type not in _VALID_TASK_TYPES:
            raise ValueError(
                f"Unsupported manifest.meta.task_type '{self.task_type}'. "
                f"Choose from {sorted(_VALID_TASK_TYPES)}."
            )
        _ = self.label_output
        if not self.items:
            raise ValueError("manifest.items must contain at least one case.")

        for item in self.items:
            if not item.images:
                raise ValueError(f"Manifest item '{item.case_id}' has no images.")
            task_type = item.task_type or self.task_type
            if task_type not in _VALID_TASK_TYPES:
                raise ValueError(f"Unsupported task_type '{task_type}' for case '{item.case_id}'.")
            item_label_output = item.label_output or self.label_output
            if item_label_output not in _VALID_LABEL_OUTPUTS:
                raise ValueError(
                    f"Unsupported label_output '{item_label_output}' for case '{item.case_id}'."
                )
            if task_type == "multi-label" and item.labels is None:
                raise ValueError(
                    f"Manifest item '{item.case_id}' has no labels for multi-label task."
                )
            if task_type == "multi-label" and item_label_output == "regions":
                regions = item.regions or self.regions
                if not regions:
                    raise ValueError(
                        f"Manifest item '{item.case_id}' uses label_output='regions' but no regions are defined."
                    )
