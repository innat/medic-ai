# nnU-Net API

`medicai.trainer.nnunet` is an in-progress Keras 3 implementation of the nnU-Net workflow for MedicAI.
The goal is parity with official nnU-Net v2 where it matters, while exposing a more flexible data contract through a single manifest file instead of the original folder naming convention.

## Quick Start

```python
from medicai.trainer.nnunet import nnUNetPipeline

pipeline = nnUNetPipeline(
    dataset_dir="/path/to/dataset",
    manifest_file="/path/to/dataset/manifest.json",
    output_dir="/path/to/outputs",
)

pipeline.setup(gpu_memory_gb="auto")
pipeline.train(fold=0, epochs=1000)
```

For 2D training, set:

```python
pipeline.configuration = "2d"
```

before preprocessing, training, or prediction.

## Manifest

Both benchmark datasets and custom datasets are supported as long as they are described in `manifest.json`.
The source files can live anywhere and do not need to follow the official nnU-Net `imagesTr` / `labelsTr` layout.

### Common Manifest

```json
{
  "meta": {
    "name": "Task001_Custom",
    "modalities": ["CT"],
    "class_names": ["background", "tumor", "organ"],
    "task_type": "multi-class",
    "ignore_class_ids": [255],
    "target_class_ids": [1, 2],
    "image_layout": "CHW",
    "label_layout": "HW"
  },
  "items": [
    {
      "id": "case_001",
      "images": "/data/case_001_ct.tif",
      "labels": "/data/case_001_seg.tif",
      "spacing": [1.0, 1.0]
    }
  ]
}
```

### Phase 0 Typed Manifest

New manifests can describe the task and cases explicitly. Array and TIFF
inputs must declare their source layout; spacing is expressed in that same
source spatial-axis order and is reordered together with the array into
channel-last `HWC` or `DHWC` form.

```json
{
  "task": {
    "task_type": "multi_class",
    "modalities": ["CT"],
    "labels": {"background": 0, "tumor": 1}
  },
  "cases": [
    {
      "id": "case_001",
      "images": {"CT": "/data/case_001.npy"},
      "label": "/data/case_001_mask.npy",
      "split": "train",
      "input_layout": "HWDC",
      "spacing": [0.7, 0.7, 2.5]
    }
  ]
}
```

The legacy `meta`/`items` manifest remains readable during migration. Its
`image_layout` field is treated as the source layout.

### Layout Fields

- `image_layout` describes raw input tensor layout before MedicAI normalizes it.
- `label_layout` describes raw label tensor layout when labels carry channel dimensions.
- Supported examples:
  - 2D: `HW`, `HWC`, `CHW`
  - 3D: `DHW`, `DHWC`, `CDHW`, `HWD`, `HWDC`, `CHWD`
- `input_layout` is required for array and TIFF image inputs. NIfTI geometry is
  read from its header and does not use `input_layout`.

### Spacing Rules

- If `spacing` is present on an item, it is used.
- NIfTI spacing is read from the file header unless explicitly provided.
- TIFF/NumPy cases require positive `spacing` or an nnU-Net-style per-case JSON
  sidecar containing `spacing`; unit-spacing fallback is intentionally rejected.

## Task Types

### Binary

Use a single integer-encoded segmentation map.

```json
{
  "meta": {
    "modalities": ["CT"],
    "class_names": ["background", "tumor"],
    "task_type": "binary",
    "target_class_ids": [1]
  },
  "items": [
    {
      "id": "case_001",
      "images": "/data/case_001_ct.tif",
      "labels": "/data/case_001_seg.tif"
    }
  ]
}
```

### Multi-Class

Use a single sparse label file with integer class IDs.

```json
{
  "meta": {
    "modalities": ["CT"],
    "class_names": ["background", "tumor", "organ"],
    "task_type": "multi-class"
  },
  "items": [
    {
      "id": "case_001",
      "images": ["/data/case_001_ct.tif"],
      "labels": "/data/case_001_seg.tif"
    }
  ]
}
```

### Multi-Label

Multi-label can be expressed in two ways.

`label_output: "channel_masks"`

- Use one binary mask file per output channel.

```json
{
  "meta": {
    "modalities": ["CT"],
    "class_names": ["background", "tumor", "organ"],
    "task_type": "multi-label",
    "label_output": "channel_masks"
  },
  "items": [
    {
      "id": "case_001",
      "images": "/data/case_001_ct.tif",
      "labels": ["/data/case_001_tumor.tif", "/data/case_001_organ.tif"]
    }
  ]
}
```

`label_output: "regions"`

- Use one sparse label file and expand it into overlapping region channels.
- This is the right shape for BraTS-style training where several MRI modalities map to one sparse segmentation file and the training targets are derived regions.

```json
{
  "meta": {
    "modalities": ["FLAIR", "T1", "T1CE", "T2"],
    "class_names": ["background", "whole_tumor", "tumor_core", "enhancing_tumor"],
    "task_type": "multi-label",
    "label_output": "regions",
    "regions": [[1, 2, 4], [1, 4], [4]]
  },
  "items": [
    {
      "id": "brats_001",
      "images": [
        "/data/brats_001_flair.nii.gz",
        "/data/brats_001_t1.nii.gz",
        "/data/brats_001_t1ce.nii.gz",
        "/data/brats_001_t2.nii.gz"
      ],
      "labels": "/data/brats_001_seg.nii.gz"
    }
  ]
}
```

This region-based pattern matches the BraTS-style target construction shown in the Keras brain tumor segmentation example: https://keras.io/examples/vision/brain_tumor_segmentation/

## Current Behavior

- Preprocessing writes `dataset_fingerprint.json`, `nnunet_plans.json`, preprocessed `.npz` files, and per-case property JSON files.
- Planner defaults remain heuristic-driven, similar to nnU-Net’s AutoML spirit.
- Users can still tweak settings manually by changing pipeline configuration or passing custom losses, metrics, and training arguments.
- 3D prediction uses sliding-window inference.
- 2D prediction uses direct `model.predict`.

## Reused MedicAI Components

- `medicai.losses` for Dice and Dice+CE losses
- `medicai.metrics` for Dice-based metrics
- `medicai.utils.inference` for sliding-window inference
- `medicai.layers.ResizingND` for deep supervision resizing

## Known Gaps

- Planner heuristics are still simplified relative to official nnU-Net v2.
- Cascade training and inference are still incomplete.
- Fold ensembling and learned postprocessing are not fully wired end to end.
- Full nnU-Net v2 augmentation parity is still pending.
- Region-based training is supported at the manifest/preprocessing target-construction level, but the broader official nnU-Net region workflow is not fully replicated yet.

## Roadmap

See:

- `medicai/trainer/nnunet/ROADMAP.md`
