# MedicAI nnU-Net Workflow

This guide walks through the current `medicai.trainer.nnunet` API from raw
NIfTI files to a trained model and prediction. It is written for a first-time
user: create a `manifest.json`, validate the data, plan preprocessing, save a
reusable cache, train, and then predict. The API follows the nnU-Net workflow
but intentionally does not copy the official command-line/API surface.

The public entry point is `nnUNetPipeline`. Analysis, planning, preprocessing,
training, prediction, and nnU-Net-specific data support live under
`medicai.trainer.nnunet`; reusable generic data-loading code remains in
`medicai.dataloader`.

Install MedicAI and the optional PyGrain dependency used by the training input
pipeline:

```bash
pip install 'medicai[nnunet]'
```

## 1. Understand The Inputs

Suppose the raw 3D data looks like this:

```text
my_dataset/
├── dataset.json                 # Optional; not required or read automatically
└── raw/
    ├── imagesTr/
    │   ├── case_001_0000.nii.gz # Image modality/channel 0
    │   └── case_002_0000.nii.gz
    ├── labelsTr/
    │   ├── case_001.nii.gz      # One categorical label map per case
    │   └── case_002.nii.gz
```

The folders may be named differently; the manifest records the paths. A
manifest contains labeled training cases only; validation partitions are
generated from those cases for cross-validation. Keep independent inference
images outside the manifest and pass them to prediction separately. For
NIfTI, MedicAI reads spacing and source affine from the files. Image modalities
and their label must share the same geometry. Do not set `input_layout` or
`spacing` for NIfTI unless you intentionally need to override header geometry.

An existing official nnU-Net `dataset.json` is optional. It can help you
transcribe modality names and label IDs, but MedicAI does not automatically
convert it or discover cases from the official directory structure. Create a
MedicAI `manifest.json` either by writing JSON or using `TaskSpec`, `CaseRecord`,
and `DatasetManifest` below.

## 2. Create `manifest.json`

### Choose The Task

`TaskSpec` describes the target semantics:

| `task_type` | Label contract |
| --- | --- |
| `"binary"` | One categorical map with background ID `0` and foreground ID `1`. |
| `"multi_class"` | One categorical map with consecutive IDs starting at background `0`. |
| `"region_based"` | One fine-grained categorical map plus named regions, which may overlap; declared label IDs may be sparse. |

### `TaskSpec` Arguments

| Argument | Required? | Meaning |
| --- | --- | --- |
| `task_type` | Yes | `"binary"`, `"multi_class"`, or `"region_based"`. |
| `modalities` | Yes | Ordered names for image channels/modalities; path order in each case must match. |
| `labels` | Yes | Mapping from label names to integer IDs, including `{"background": 0}`. An ordered list of label names is also accepted; a mapping is clearer. |
| `regions` | Optional; required for region-based tasks | Mapping from region name to the categorical label IDs in that region. |
| `regions_class_order` | Optional; required for region-based tasks | One foreground class ID per region, in the same order as `regions`, for hard-map decoding. |
| `ignore_class_ids` | Optional | Label IDs excluded from loss/metrics. |
| `target_class_ids` | Optional | Foreground label IDs included in the target/loss/metric contract. |

Binary example:

```python
task = TaskSpec(
    task_type="binary",
    modalities=["CT"],
    labels={"background": 0, "lesion": 1},
)
```

For a region-based task, overlapping memberships are allowed:

```python
task = TaskSpec(
    task_type="region_based",
    modalities=["T1", "T1ce", "T2", "FLAIR"],
    labels={"background": 0, "edema": 1, "core": 2, "enhancing": 3},
    regions={"whole_tumor": [1, 2, 3], "tumor_core": [2, 3]},
    regions_class_order=[1, 2],
)
```

### `CaseRecord` Arguments

Create one record per case. NIfTI image paths go in `image`; the task modality
names and their order are declared once in `TaskSpec`.

| Argument | Required? | Meaning |
| --- | --- | --- |
| `id` | Yes | Unique stable case identifier; used for cache files and cross-validation folds. |
| `image` | Yes | One NIfTI path for one modality, or a list of paths ordered like `TaskSpec.modalities`. A modality-to-path mapping may also be used. |
| `label` | Yes | Path to one categorical label-map NIfTI. Region-based tasks still use one map. Every manifest case is a labeled training case used for cross-validation. |
| `input_layout` | Optional for NIfTI | Shared axis string, or `{"image": ..., "label": ...}` for arrays with different source axis orders. NIfTI normally derives axes from file geometry. |
| `spacing` | Optional for NIfTI | Source voxel spacing. NIfTI normally derives it from the header. For TIFF, provide this per case or through the supported spacing sidecar. |
| `meta` | Optional | Additional case metadata. |

For example, if each NIfTI file is one image channel, list channels in the same
order as `modalities`:

```python
case = CaseRecord(
    id="case_001",
    image=[
        "/data/my_dataset/raw/imagesTr/case_001_0000.nii.gz",  # T1
        "/data/my_dataset/raw/imagesTr/case_001_0001.nii.gz",  # T2
    ],
    label="/data/my_dataset/raw/labelsTr/case_001.nii.gz",
)
```

For non-NIfTI arrays, `input_layout` describes the raw array axes. A shared
string such as `"DHW"` or `"DHWC"` applies to both image and label. Use a keyed
mapping when their source axes differ, for example
`{"image": "HWDC", "label": "HWD"}`. The `C` axis is optional: if the raw
image has no channel axis, declare only its spatial axes (for example `"DHW"`);
preprocessing adds the trailing channel axis while stacking modalities. Spacing
is ordered like the image's source spatial axes.

### Build And Save The Manifest

The manifest contains labeled training cases only; MedicAI creates
cross-validation folds from all of them. Keep external validation or inference
images outside this manifest and pass them to a separate evaluation/prediction
workflow. Use absolute paths to avoid ambiguity: manifest paths are currently
used as provided, not automatically resolved relative to the manifest file.

```python
from pathlib import Path

from medicai.trainer.nnunet import CaseRecord, DatasetManifest, TaskSpec

root = Path("/data/my_dataset").resolve()

task = TaskSpec(
    task_type="multi_class",
    modalities=["CT"],
    labels={"background": 0, "organ": 1, "lesion": 2},
)

cases = [
    CaseRecord(
        id="case_001",
        image=str(root / "raw/imagesTr/case_001_0000.nii.gz"),
        label=str(root / "raw/labelsTr/case_001.nii.gz"),
    ),
    CaseRecord(
        id="case_002",
        image=str(root / "raw/imagesTr/case_002_0000.nii.gz"),
        label=str(root / "raw/labelsTr/case_002.nii.gz"),
    ),
]

manifest = DatasetManifest(
    task=task,
    cases=cases,
    name="my_dataset",
)
manifest.to_json(root / "manifest.json")
```

`DatasetManifest` takes the typed task and cases, with optional dataset-level
`name`, `input_layout`, `spatial_dims`, and arbitrary `metadata`.

`DatasetManifest` constructor arguments:

| Argument | Required? | Meaning |
| --- | --- | --- |
| `task` | Yes | A `TaskSpec` instance or mapping of `TaskSpec` arguments. |
| `cases` | Yes | List of `CaseRecord` instances or mappings of `CaseRecord` arguments. |
| `name` | Optional | Descriptive dataset name. |
| `input_layout` | Optional | Dataset-wide default source layout, overridable per case. |
| `spatial_dims` | Optional | Explicit spatial rank; otherwise inferred from layouts, defaulting to 3D. |
| `metadata` | Optional | Additional dataset metadata; task and geometry fields should use typed arguments. |

## 3. Validate And Analyze

Create a pipeline. `input_path` is the persistent dataset/workflow directory
for the manifest (by default), fingerprint, plan, and preprocessed cache.
`manifest_file` can point to a manifest elsewhere. `output_path` holds model
and training artifacts.

```python
from medicai.trainer.nnunet import nnUNetPipeline

work_dir = "/experiments/my_dataset"
pipeline = nnUNetPipeline(
    input_path=work_dir,
    manifest_file="/data/my_dataset/manifest.json",
    output_path=f"{work_dir}/models",
)

report = pipeline.analyze()
print(report.errors)
print(report.warnings)
print(report.class_prevalence)
print(report.region_prevalence)
print(report.recommendations)
report.raise_if_errors()
```

`analyze()` reads the source files, validates geometry and label values, and
returns a fingerprint without saving pipeline artifacts. Fix every reported
error before proceeding. Warnings (for example an absent class) should be
reviewed for your dataset.

`nnUNetPipeline` constructor arguments:

| Argument | Required? | Meaning |
| --- | --- | --- |
| `input_path` | Yes | Persistent fingerprint, plan, and preprocessing-cache root; also the default manifest location. |
| `output_path` | Optional | Model/checkpoint root; defaults to `<input_path>/outputs`. |
| `manifest_file` | Optional | Manifest path; defaults to `<input_path>/manifest.json`. |

## 4. Plan Configurations

Planning derives target spacing, patch/batch settings, and network configuration
from the fingerprint and the planner's built-in resource heuristic. Inspect the
result before preprocessing:

```python
plan = pipeline.plan(
    planner=None,  # Optional advanced override; default planner is automatic.
)
print(plan.configurations.keys())
print(plan.selected_configuration)
```

`plan()` arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `planner` | `None` | Optional advanced planner override by registered planner name. When omitted, MedicAI uses its default planner. |

Official nnU-Net defaults to its standard planner; dataset fingerprint
heuristics derive the configurations and their settings. The planner
implementation can be overridden (for example, with a residual-encoder
preset), but users normally leave it at the default. Planning retains all
suitable configurations proposed by the planner; not every dataset gets every
configuration. MedicAI records a recommended configuration for `train()` to
use when `configuration="auto"`. The planner currently uses its built-in
resource assumptions rather than detecting the runtime accelerator. TODO: add
reliable device-aware resource estimation across supported accelerators.

The plan is saved to `<input_path>/nnunet_plans.json`.

## 5. Preprocess And Cache Full Cases

Preprocessing writes channel-last full-case arrays before patch sampling. By
default, it caches every planned configuration so training can choose one later:

```python
preprocess_report = pipeline.preprocess(incremental=True)
print(preprocess_report)
```

`preprocess()` arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `configurations` | `None` | Preprocess every planned configuration. Optionally pass one name or a list to limit cache generation. |
| `incremental` | `True` | Reuse cache entries when sources and preprocessing settings still match. |
| `num_workers` | `None` | Optional preprocessing worker count. |

The cache is stored below `<input_path>/preprocessed/<configuration>/`; each
case has an `.npz` array file and a properties JSON sidecar. Properties retain
source geometry and preprocessing provenance. Patch size is consumed later by
the model and sampler; changing only patch size may permit reuse of the
full-volume cache, while changing spacing, normalization, crop, or resampling
requires preprocessing again.

## Workflow A: One Notebook, Start To Prediction

Run Steps 1-5 above in order, then train:

```python
history = pipeline.train(
    fold=0,
    epochs=1000,
    callbacks=[],
)
```

Then predict:

```python
pipeline.predict(
    input_path="/data/new_case.nii.gz",
    output_path="/experiments/my_dataset/predictions/new_case.nii.gz",
    fold=0,
    configuration="auto",
)
```

Current prediction writes a hard segmentation map. Full training-equivalent
preprocessing/inverse geometry, probability export, and fold ensembling are
still incomplete. Also, the current prediction entry point accepts one image
path; do not assume it handles a multi-modality list like `CaseRecord.image`.

`predict()` arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `input_path` | Required | One input image path. Current API limitation: one path only. |
| `output_path` | Required | Destination path for the predicted hard segmentation map. |
| `fold` | `0` | Trained fold whose default checkpoint is loaded. |
| `configuration` | `"auto"` | Configuration to load; auto uses the compiled configuration when available, otherwise the plan recommendation. |
| `model_weights_path` | `None` | Optional explicit weights path; otherwise use the pipeline checkpoint path. |
| `overlap` | `0.5` | Sliding-window overlap for 3D inference. |
| `mode` | `"gaussian"` | Sliding-window blending mode. |
| `padding_mode` | `"constant"` | Sliding-window padding mode. |
| `cval` | `0.0` | Constant value used when `padding_mode="constant"`. |

## Workflow B: Preprocess Now, Train Later

In the first notebook/session, create the manifest, analyze, plan, and
preprocess. Keep the source dataset and manifest available, and preserve
`input_path` between sessions:

```python
pipeline = nnUNetPipeline(
    input_path="/experiments/my_dataset",
    manifest_file="/data/my_dataset/manifest.json",
    output_path="/experiments/my_dataset/models",
)
report = pipeline.analyze()
report.raise_if_errors()
plan = pipeline.plan()
pipeline.preprocess()
```

In a later session, make a new pipeline with the same paths. If the plan and
cache exist, skip analyze/plan/preprocess and train from the saved state:

```python
pipeline = nnUNetPipeline(
    input_path="/experiments/my_dataset",
    manifest_file="/data/my_dataset/manifest.json",
    output_path="/experiments/my_dataset/models",
)
history = pipeline.train(
    fold=0,
    epochs=1000,
)
```

The source manifest remains relevant for folds/task metadata, and the
preprocessed cache remains under `input_path`. To inspect the cached arrays:

```python
import numpy as np

cache_file = (
    "/experiments/my_dataset/preprocessed/3d_fullres/case_001.npz"
)
with np.load(cache_file, allow_pickle=False) as case:
    image = case["image"]      # DHWC
    label = case["label"]      # categorical or region target, when present
    spacing = case["spacing"]
```

`dataset()` can also expose the PyGrain patch stream for inspection or custom
training integration. The plan and requested cache must already exist:

```python
train_data = pipeline.dataset(
    split="train",
    fold=0,
    configuration="3d_fullres",
)
images, labels = next(iter(train_data))
```

`dataset()` arguments: `split` selects the fold partition (`"train"` or
`"validation"`), `fold` (default `0`), `n_folds` (five by default for generated
KFold), optional `splits` or `splitter`/`groups`, `configuration` (default
saved selection), `num_threads` (default `4`), and `seed` (default `12345`).
Pass the same split arguments as `train()` to inspect its exact fold. Batch
size and epoch iteration count come from the plan/sampler, not user arguments.

## Workflow C: Resume Interrupted Training

Set `resume=True` on the initial training call so the recovery callback is
active from the beginning. If training is interrupted, reconstruct the same
pipeline and repeat the call with the same schedule-critical values:

```python
fit_args = dict(
    fold=0,
    n_folds=5,
    epochs=1000,
    resume=True,
)

# Run this from the start when you want interruption recovery enabled.
history = pipeline.train(**fit_args)

# After interruption, create the same pipeline again and call the same settings.
history = pipeline.train(**fit_args)
```

Keep the same fold, plan/configuration, fold assignment, epochs, compile
settings, callbacks, and train/validation input choice. The run signature rejects some changed settings rather than silently
changing the learning-rate horizon. Recovery restores Keras model/optimizer
progress, but PyGrain/Python sampling RNG and arbitrary callback state are not
guaranteed to resume exactly. For a fresh run instead, use a new output
directory if an interrupted backup exists.

## Compile And Fit Controls

Compilation is optional; `train()` compiles with nnU-Net defaults when needed.
Power users can customize the planned configuration and forward Keras
`Model.compile()` arguments:

```python
pipeline.compile(configuration="auto")
pipeline.compile(
    configuration="3d_fullres",
    optimizer=my_optimizer,
    loss=my_loss,
    metrics=[my_metric],
    jit_compile=True,
)
```

`configuration` selects the MedicAI model; other keyword arguments are passed
to Keras compilation, including `optimizer`, `loss`, `metrics`, `run_eagerly`,
`steps_per_execution`, and `jit_compile`. Omitted objectives use the defaults
for the task's categorical or region-based label contract. The default
optimizer and polynomial learning-rate schedule follow the nnU-Net training
recipe; a custom optimizer disables that default schedule.

`train()` follows the Keras fit pattern with `epochs`, `callbacks`, `x`,
`validation_data`, and additional Keras `Model.fit()` keyword arguments. The
pipeline owns patch batch size and steps per epoch; neither can be overridden.
Internally generated random-patch streams use the nnU-Net iteration-based
epoch length, not `number_of_cases // batch_size`. Use callbacks to customize
fit-time behavior such as learning-rate schedules.

### Cross-Validation Strategies

By default, MedicAI creates deterministic five-fold KFold splits using the
same shuffle seed and fold-generation algorithm as official nnU-Net. Set
`n_folds` to choose another fold count. You can instead provide explicit
folds, or pass a scikit-learn-style splitter. For group-aware splitting, map
every case ID to its patient/site/group ID; MedicAI checks that a group never
crosses from a fold's training partition into its validation partition.

```python
from sklearn.model_selection import GroupKFold

group_by_case = {
    case.id: case.meta["patient_id"]
    for case in manifest.cases
}

history = pipeline.train(
    fold=0,
    splitter=GroupKFold(n_splits=5),
    groups=group_by_case,
    epochs=1000,
)
```

Alternatively, supply folds directly as case-ID lists. This works with any
split-generation library and does not require scikit-learn:

```python
custom_folds = [
    {"train": ["case_002", "case_003"], "val": ["case_001"]},
    {"train": ["case_001", "case_003"], "val": ["case_002"]},
    {"train": ["case_001", "case_002"], "val": ["case_003"]},
]

history = pipeline.train(fold=0, splits=custom_folds, epochs=1000)
```

Explicit folds must partition the same manifest cases in every fold and place
each case in validation exactly once. To check patient/group separation for
explicit folds too, pass the same `groups` mapping. Pass the same `splits` or
`splitter`/`groups` to `pipeline.dataset()` when inspecting the corresponding
fold's patches. Generated default splits are saved and reused when their case
set and fold count still match; changing `n_folds` regenerates them. Custom
splits are used as supplied, and the selected fold assignment is recorded in
training provenance for resume checks.

## Current Limitations

- Planner heuristics and online augmentation are not yet fully at official
  nnU-Net v2 feature parity.
- Validation evaluation, configuration selection, fold ensembling, and
  learned postprocessing are not yet one complete public workflow.
- Prediction does not yet apply the full cached preprocessing and inverse
  geometry path and currently exports hard labels rather than region
  probability maps.
- Cascade training/inference and exact augmentation RNG replay remain
  incomplete.
