# Contributing

Please refer to the current [roadmap](https://github.com/innat/medic-ai/wiki/Roadmap) first. If you have suggestions or ideas, please open a [GitHub issue](https://github.com/innat/medic-ai/issues/new/choose).

## 1. Setup

`medicai` now requires Python `3.10+`. Create a fresh environment before
installing local dependencies.

### Option A: `uv` (recommended)

```bash
git clone https://github.com/innat/medic-ai
cd medic-ai
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .[dev]
```

### Option B: `venv` + `pip`

```bash
git clone https://github.com/innat/medic-ai
cd medic-ai
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .[dev]
```

### Extras

- `.[test]`: install the package plus `pytest`
- `.[docs]`: install the documentation toolchain
- `.[dev]`: install docs, tests, and formatting tools

If you want to run backend-specific tests or examples, install the corresponding
runtime(s) separately:
- TensorFlow backend: `pip install tensorflow`
- Torch backend: `pip install torch`
- JAX backend: `pip install jax`

The package includes `keras`, but it intentionally does not force a specific
version of TensorFlow, PyTorch, or JAX.

## 2. Run tests

Run all tests:

```bash
python -m pytest test/
```

Or, if you prefer the helper target:

```bash
make test-unit
make test-integration
make test-gpu
```

Run by markers:

```bash
# Fast unit tests
python -m pytest -m "unit"

# Integration tests
python -m pytest -m "integration"

# GPU-required tests only (auto-skips if GPU is not available)
python -m pytest -m "gpu"
```

Run a single file or test:

```bash
python -m pytest test/losses/test_dice_loss.py
python -m pytest -k dice_loss
```

## 3. Backend matrix checks (Keras 3)

These tests validate backend compatibility for `tensorflow`, `torch`, and `jax`:
- `test/backends/test_backend_matrix_losses.py`
- `test/backends/test_backend_matrix_metrics_transforms_models.py`

Run them:

```bash
python -m pytest test/backends/test_backend_matrix_losses.py
python -m pytest test/backends/test_backend_matrix_metrics_transforms_models.py
```

Note: these tests may skip a backend if that runtime is not installed in your environment.

## 4. Set backend manually (optional)

Set `KERAS_BACKEND` before importing `keras` or `medicai`.

PowerShell:

```powershell
$env:KERAS_BACKEND="tensorflow"; python -m pytest -m "unit"
$env:KERAS_BACKEND="torch"; python -m pytest -m "unit"
$env:KERAS_BACKEND="jax"; python -m pytest -m "unit"
```

Bash:

```bash
KERAS_BACKEND=tensorflow python -m pytest -m "unit"
KERAS_BACKEND=torch python -m pytest -m "unit"
KERAS_BACKEND=jax python -m pytest -m "unit"
```
