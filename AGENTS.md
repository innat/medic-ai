# Medic-AI - Agent Instructions

This file provides implementation guidance for AI coding agents working on
`medicai`. It supplements the human-facing docs with repo-specific rules for
multi-backend medical imaging work.

**Canonical sources**
- Human documentation: `README.md`
- Contribution guide: `CONTRIBUTION.md`
- Public package exports: `medicai/models/__init__.py`,
  `medicai/utils/__init__.py`, `medicai/transforms/__init__.py`,
  `medicai/losses/__init__.py`, `medicai/metrics/__init__.py`
- Package and tooling configuration: `pyproject.toml`, `pytest.ini`,
  `Makefile`, `docs/conf.py`, `.readthedocs.yaml`

## 1. Agent Responsibilities

Contribute like a careful medical ML engineer:

- Prefer precise, minimal changes over broad rewrites.
- Preserve Keras 3 multi-backend compatibility in core library code.
- Keep 2D and 3D behavior aligned unless the feature is explicitly rank-specific.
- Follow TDD for bug fixes: reproduce with a failing test first, then fix.
- Add or update docs when public behavior, APIs, or architectural patterns change.

## 2. Development Environment

- Package manager: `uv`
- Target Python: `>=3.12`
- Core libraries: `keras`, optional backend runtimes `tensorflow`, `torch`, `jax`
- Medical I/O dependency: `nibabel`
- On Windows, prefer running all repository commands from WSL rather than
  PowerShell or Command Prompt. The project tooling, shell commands, and agent
  workflows assume a Unix-like environment.
- Keep local commands aligned with repo tooling: formatting uses Black and
  isort, tests use pytest markers from `pytest.ini`, and docs build through
  Sphinx on Read the Docs.

Setup:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .[dev]
```

Common test commands:

```bash
uv run pytest test/
python -m pytest test/
make test-unit
make test-integration
make test-gpu
```

Backend matrix checks:

```bash
python -m pytest test/backends/test_backend_matrix_losses.py
python -m pytest test/backends/test_backend_matrix_metrics_transforms_models.py
```

Optional backend selection before import:

```bash
KERAS_BACKEND=tensorflow python -m pytest -m "unit"
KERAS_BACKEND=torch python -m pytest -m "unit"
KERAS_BACKEND=jax python -m pytest -m "unit"
```

## 3. Repository Architecture

The repository is organized around reusable building blocks and thin public model
wrappers.

- `medicai/models/`
  Classification and segmentation model families.
- `medicai/layers/`
  Reusable low-level layers such as convolution, pooling, attention, resizing,
  MLP, and regularization helpers.
- `medicai/blocks/`
  Mid-level architectural blocks reused by larger models.
- `medicai/metrics/`, `medicai/losses/`
  Backend-agnostic medical objective functions and metrics.
- `medicai/transforms/`
  TensorFlow-oriented preprocessing and augmentation utilities for imaging data.
- `medicai/utils/`
  Shared helpers for registration, activation validation, encoder resolution,
  inference, visualization, and descriptive mixins.
- `test/`
  Pytest suite, including backend matrix smoke tests and model-level coverage.
- `docs/`
  Sphinx content for guides, examples, and API documentation.

### Public API surfaces

When changing exported symbols, treat package `__init__.py` files as API
contracts, not bookkeeping:

- `medicai/__init__.py` defines top-level lazy-loaded subpackages
- `medicai/models/__init__.py` defines the public model catalog and registry
  helpers
- `medicai/transforms/__init__.py`, `medicai/losses/__init__.py`,
  `medicai/metrics/__init__.py`, and `medicai/utils/__init__.py` define the
  public import paths users are expected to rely on

Avoid adding new public objects in deep modules without exporting them from the
appropriate package surface when they are intended for users.

### Model package layout

Use the existing package patterns instead of concentrating everything into one
file.

- Classification families generally split into:
  - `<family>_layers.py`
  - `<family>_backbone.py`
  - `<family>.py`
- Segmentation families commonly split into:
  - decoder/helper files such as `decoder.py` or `<family>_layers.py`
  - the public model file `<family>.py`

Keep classes single-purpose:

- Layers files: reusable primitive operations
- Backbone files: feature extractor only
- Decoder files: task-specific reconstruction head
- Public model files: wiring, validation, serialization, registration

Avoid duplicating layer logic across model families. If logic is reusable, move
it into `medicai/layers/`, `medicai/blocks/`, or `medicai/utils/`.

## 4. Critical Technical Rules

### Multi-backend enforcement

- Never import backend-specific ops inside core `models/`, `metrics/`, or
  `losses/` logic unless a workaround is unavoidable.
- Prefer `keras.ops` for tensor math and shape-safe operations.
- If a backend-specific branch is required, isolate it, gate it with
  `keras.config.backend()`, and explain why in an inline comment.
- Optional backend runtimes are not guaranteed to be installed in every dev,
  test, or docs environment. Keep backend assumptions explicit and avoid making
  module imports fail unnecessarily.

### 2D and 3D model behavior

- Support both `(B, H, W, C)` and `(B, D, H, W, C)` when the family is intended
  to be rank-agnostic.
- Derive spatial rank from `input_shape` rather than hardcoding 2D assumptions.
- Reuse rank-aware helpers from `medicai.utils`, such as:
  - `get_conv_layer`
  - `get_norm_layer`
  - `get_pooling_layer`
  - `get_reshaping_layer`
  - `get_act_layer`
- For 3D workflows, preserve compatibility with sliding-window inference helpers
  in `medicai.utils.inference` and `medicai.utils.swi_utils`.

### TensorFlow-only transform zone

- `medicai/transforms/` may use TensorFlow ops directly.
- Keep transform code compatible with `tf.data` pipelines, caching, prefetching,
  and thread-safe dataset execution.
- Do not copy TensorFlow-specific transform patterns into backend-agnostic
  model, metric, or loss modules.

### Typing and docstrings

- All public functions and methods must have type hints.
- Use Google-style docstrings.
- Document expected tensor shapes explicitly.
- State expected value ranges when medically relevant rather than assuming
  normalization conventions.
- Treat docstrings as first-class Read the Docs content, not as minimal inline
  notes. In this repository, model and API docstrings are expected to carry
  real user-facing documentation value.
- Follow the level of detail already used in public model classes such as
  `UNet` and `TransUNet`: explain the architecture at a high level, describe
  important argument contracts, and document output behavior clearly.
- For public models, losses, metrics, transforms, and reusable layers, prefer
  docstrings that include:
  - a short conceptual overview
  - precise `Args` entries
  - `Returns` when useful
  - shape expectations for inputs and outputs
  - notes about 2D versus 3D behavior when relevant
  - important constraints or invariants such as required `pyramid_outputs`,
    valid `encoder_depth` values, supported activations, or expected label
    formats
- When a public API has multiple common usage patterns, include small,
  executable-style examples in the docstring. Existing model docstrings in this
  repo use examples, rubric sections, and scenario-based explanations; new
  public APIs should aim for the same standard when they are non-trivial.
- If behavior is subtle enough that a user would need to open the source to
  understand it, the docstring is not complete enough yet.

### Import-time safety for docs

- Read the Docs builds import public modules through Sphinx with mocked Keras
  objects configured in `docs/conf.py`.
- Keep module import side effects minimal: avoid doing real runtime work, heavy
  tensor creation, backend probing, dataset loading, or environment-dependent
  logic at import time.
- Public modules should define classes, functions, constants, and registration
  metadata cleanly enough that autodoc can import them without a live training
  runtime.
- If a feature requires runtime-only behavior, defer it to function or method
  execution rather than module import.

## 5. Project-Specific Model Conventions

Follow the current model APIs already used across the repo.

### Registration and serialization

Public model classes should generally:

- be decorated with `@keras.saving.register_keras_serializable(...)`
- be registered in the Medic-AI registry with `@registration.register(...)`
- inherit `DescribeMixin` when they are end-user model entry points

Examples from the current codebase:

```python
@keras.saving.register_keras_serializable(package="densenet")
@registration.register(family="densenet")
class DenseNet121(DenseNetBase, DescribeMixin):
    ...
```

```python
@keras.saving.register_keras_serializable(package="unet")
@registration.register(name="unet", type="segmentation")
class UNet(keras.Model, DescribeMixin):
    ...
```

### Classification model structure

Classification families usually expose:

- a reusable backbone class returning feature tensors plus `pyramid_outputs`
- a base classification wrapper that adds pooling and the prediction head
- thin named variants that only inject configuration

### Segmentation model structure

Segmentation models usually:

- accept either `encoder_name=...` or a prebuilt `encoder=...`
- resolve built-in encoders through `resolve_encoder(...)`
- require the encoder to expose `pyramid_outputs`
- validate `encoder_depth`, decoder configuration, and class count early
- keep decoder implementation in focused helper files

### Public exports

If you add a public model family or variant:

- update the local package `__init__.py`
- update `medicai/models/__init__.py`
- ensure registry-based creation still works through `create_model()` and
  `list_models()`
- keep registration names stable and lowercase-friendly, since lookup is
  case-insensitive and user-facing

## 6. Common Workflows

### Implementing a New Model

Build new model families with small, focused modules and clear interfaces.

Recommended file layout:

```text
medicai/models/<family>/
├── __init__.py
├── <family>_layers.py      # reusable family-specific layers
├── <family>_backbone.py    # feature extractor only
├── decoder.py              # only for segmentation-style families
├── <family>.py             # public model classes and registration
└── README.md               # family-specific notes if the repo already uses one
```

Implementation checklist:

1. Define the smallest reusable units first.
   Put repeated blocks in `<family>_layers.py` or shared packages like
   `medicai/layers/` or `medicai/blocks/`.
2. Keep the backbone isolated.
   The backbone should focus on feature extraction and expose structured outputs,
   especially `pyramid_outputs` when downstream decoders need multi-scale
   features.
3. Keep task heads separate from feature extraction.
   Classification heads belong in the public wrapper or a small head helper.
   Segmentation decoders belong in a dedicated decoder/helper file.
4. Use repo utilities instead of hardcoding layer choices.
   Prefer `get_conv_layer`, `get_norm_layer`, `get_pooling_layer`,
   `get_act_layer`, `resolve_encoder`, and `validate_activation`.
5. Validate arguments early.
   Fail fast on impossible `num_classes`, unsupported `encoder_depth`,
   incompatible `pooling`, or malformed `input_shape`.
6. Preserve serialization.
   Implement `get_config()` and `from_config()` for configurable classes that
   need round-trip serialization.
7. Export and register the public entry points.
   Add decorators, package exports, and registry entries.

#### Classification models

Recommended pattern:

1. Implement reusable blocks in `<family>_layers.py`.
2. Build a backbone in `<family>_backbone.py`.
3. Build a base classifier in `<family>.py`.
4. Add thin named variants that only pass config values.
5. Export variants from package `__init__.py` and `medicai/models/__init__.py`.

Example skeleton:

```python
@keras.saving.register_keras_serializable(package="densenet")
class DenseNetBase(keras.Model):
    ...


@keras.saving.register_keras_serializable(package="densenet")
@registration.register(family="densenet")
class DenseNet121(DenseNetBase, DescribeMixin):
    ...
```

Classification requirements:

- support 2D and 3D input shapes when the family is rank-agnostic
- use backend-agnostic ops in feature and head logic
- support `include_top`, `pooling`, and validated classifier activation when
  appropriate
- expose stable config for named variants

Tests:

- add forward-shape tests in `test/models/test_models.py` or a dedicated family
  test module
- verify output shape for representative 2D inputs, and 3D if the family claims
  3D support
- add backend smoke coverage if new behavior affects backend portability

#### Segmentation models

Recommended pattern:

1. Put reusable decoder or attention components in `decoder.py` or
   `<family>_layers.py`.
2. Accept either `encoder_name` or a prebuilt `encoder`.
3. Use `resolve_encoder(...)` for built-in backbones.
4. Require `pyramid_outputs` from encoders.
5. Keep the public model responsible for orchestration, validation, and final
   segmentation head creation.

Example skeleton:

```python
@keras.saving.register_keras_serializable(package="transunet")
@registration.register(name="trans_unet", type="segmentation")
class TransUNet(keras.Model, DescribeMixin):
    ...
```

Segmentation requirements:

- document the expected encoder contract clearly
- validate `encoder_depth` and decoder filter lengths
- preserve spatial rank correctness for both 2D and 3D
- use `ResizingND` or head upsampling only when the encoder/decoder stride
  pattern requires it
- keep final classifier activation explicit and validated

Tests:

- add shape tests for end-to-end segmentation outputs
- cover both built-in encoder resolution and, when relevant, custom encoder
  usage
- add edge-case tests for incorrect encoder contracts or invalid depth values

### Implementing a New Metric or Loss

1. Implement the math with `keras.ops`.
2. Keep hot paths vectorized.
3. Validate shapes, class counts, reduction modes, and activation assumptions.
4. Add strict type hints and Google-style docstrings.
5. Add tests in `test/metrics/` or `test/losses/`.
6. Compare against a NumPy reference implementation when feasible.

Also cover:

- empty masks
- one-hot and single-channel cases
- numerically delicate cases such as all-zero denominators
- backend matrix behavior when relevant
- the existing base-class conventions for `from_logits`, class selection,
  ignored classes, smoothing, and reduction behavior

### Implementing a New Transformation

1. Use TensorFlow ops to preserve `tf.data` compatibility.
2. Handle standard channel-last medical tensor layouts cleanly.
3. Be explicit about rank assumptions and boundary behavior.
4. Add tests in `test/transforms/` with realistic 3D mock arrays.
5. Verify deterministic behavior when randomness is involved.

### Bug Fixes and Refactors

- Reproduce first with a failing test.
- Make the smallest fix that resolves the issue.
- Do not silently widen scope into architecture rewrites.
- If deduplication is needed, extract helpers without changing external
  behavior.
- Re-run the narrowest meaningful tests first, then broader coverage as needed.

## 7. Testing Principles

- Framework: `pytest`
- Prefer grouped test classes for related behavior.
- Use `@pytest.mark.parametrize` with readable `id=` labels.
- Seed RNGs with `keras.utils.set_random_seed(...)` when determinism matters.
- Use realistic medical tensor shapes and label layouts.
- Respect the marker split from `pytest.ini`: `unit`, `integration`, `slow`,
  and `gpu`.
- Prefer the narrowest meaningful test target first, then run broader suites as
  confidence increases.
- The suite already applies deterministic seeding in `test/conftest.py`; new
  tests should stay compatible with that expectation.

Important edge cases:

- completely empty masks
- out-of-range intensities
- binary vs multi-class segmentation targets
- 2D and 3D input variants
- backends missing from the environment, which should skip rather than fail

### Formatting and style

- Follow the repository formatter configuration in `pyproject.toml`.
- Target Black formatting with a 100-character line length.
- Keep imports isort-compatible using the existing Black profile.
- Avoid introducing style churn unrelated to the task.

## 8. Before Finalizing

Use this checklist before you stop:

- [ ] No direct `tf.*`, `torch.*`, or `jax.*` calls were introduced in core
      `models/`, `metrics/`, or `losses/` code unless explicitly justified.
- [ ] Public APIs include type hints and Google-style docstrings.
- [ ] Tensor shapes and expected value ranges are documented where relevant.
- [ ] New model code is split into focused files rather than a monolith.
- [ ] Public models are exported, registered, and serializable.
- [ ] Tests cover the changed behavior with realistic medical tensor shapes.
- [ ] README/docs were updated if the public API or supported model list changed.
