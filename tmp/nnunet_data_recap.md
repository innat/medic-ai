# nnU-Net Data Pipeline Recap

This recap outlines the mandatory steps to prepare your data and the sequence of modules that transform raw medical images into training patches for the model.

## 1. Dataset Preparation (The Manifest)

Modern `medic-ai` nnU-Net is purely **manifest-driven**. You no longer need a specific folder structure; instead, you provide a JSON manifest describing your data.

### Dummy Custom Dataset Example (`manifest.json`)
```json
{
  "meta": {
    "name": "DummyBrainDataset",
    "modalities": ["T1", "T2"],
    "class_names": ["background", "edema", "tumor"],
    "task_type": "multi-class"
  },
  "items": [
    {
      "id": "case_001",
      "images": ["path/to/c001_t1.nii.gz", "path/to/c001_t2.nii.gz"],
      "labels": ["path/to/c001_seg.nii.gz"],
      "spacing": [1.0, 1.0, 1.5]
    }
  ]
}
```

---

## 2. The Data Transformation Flow

The following modules process your data from raw files to model input:

### Phase A: Fingerprinting (Offline)
**Responsible Module**: [`dataset_fingerprint.py`](file:///c:/Users/ASUS/Desktop/medic-ai/medicai/dataloader/nnunet/dataset_fingerprint.py)
1. **Intensity Stats**: Samples foreground pixels from each case to calculate global `mean`, `std`, and `percentiles` (critical for CT).
2. **Spatial Stats**: Analyzes `spacing` and `shapes` (after cropping to nonzero regions) to find the dataset's median properties.
3. **Anisotropy Detection**: Checks if spacing is anisotropic (e.g., ratio > 3.0) to configure special resampling later.

### Phase B: Preprocessing (Offline)
**Responsible Module**: [`preprocessing.py`](file:///c:/Users/ASUS/Desktop/medic-ai/medicai/dataloader/nnunet/preprocessing.py)
This converts raw images into `.npz` files ready for fast loading.
1. **Crop to Non-Zero**: Removes empty background around the subject.
2. **Normalize**: (via [`normalization.py`](file:///c:/Users/ASUS/Desktop/medic-ai/medicai/dataloader/nnunet/normalization.py))
   - **CT**: Global clipping and Z-scoring using fingerprint stats.
   - **MRI**: Individual case Z-scoring (mask-aware if cropping removed > 75% of volume).
3. **Resample**: (via [`resampling.py`](file:///c:/Users/ASUS/Desktop/medic-ai/medicai/dataloader/nnunet/resampling.py))
   - Resamples images to the target spacing using cubic interpolation (or separate-z for anisotropic data).
   - Resamples labels using nearest-neighbor.

### Phase C: Training Dataloader (On-the-fly)
**Responsible Module**: [`dataset.py`](file:///c:/Users/ASUS/Desktop/medic-ai/medicai/dataloader/nnunet/dataset.py)
1. **Deterministic Patch Extraction**: Instead of loading whole volumes, `nnUNetDataset` extracts patches of a fixed `patch_size`.
2. **Foreground Oversampling**: Uses `class_locations` (pre-calculated during preprocessing) to ensure 33% of patches are forced to contain a foreground object.
3. **Constant Padding**: Automatically pads patches that extend beyond image boundaries.

### Phase D: Augmentations (GPU/CPU on-the-fly)
**Responsible Module**: [`augmentations.py`](file:///c:/Users/ASUS/Desktop/medic-ai/medicai/dataloader/nnunet/augmentations.py)
Applies stochastic transforms to the extracted patches:
- Random flipping/mirroring.
- Random rotations.
- (Optional) Intensity/Color jitter.

---

## 3. Responsible Modules Summary

| Module | Purpose | Timing |
| :--- | :--- | :--- |
| `manifest.py` | Data structure definition | Logic |
| `dataset_fingerprint.py` | Global stats & planning | Setup |
| `preprocessing.py` | Crop, Norm, Resample | Pre-train |
| `resampling.py` | Spatial interpolation | Pre-train |
| `normalization.py` | Intensity scaling | Pre-train |
| `dataset.py` | Patch extraction & oversampling | Train |
| `augmentations.py` | Spatial/Pixel variability | Train |

---

## Dummy Workflow Code
To execute this pipeline for a new model:
```python
from medicai.dataloader.nnunet import fingerprint_dataset, preprocess_dataset
from medicai.trainer.nnunet import nnUNetPipeline

# 1. Analyze
fp = fingerprint_dataset("manifest.json")

# 2. Plan & Preprocess
plan = MyPlan(fp) # Inherited from BasePlan
preprocess_dataset(fp, plan, "output_dir", manifest_file="manifest.json")

# 3. Train
pipeline = nnUNetPipeline(plan)
pipeline.train(preprocessed_dir="output_dir")
```
