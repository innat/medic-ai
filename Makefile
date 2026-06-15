SHELL := /bin/bash

.PHONY: help venv uv-venv style clean test-unit test-integration test-gpu

help:
	@echo "Commands:"
	@echo "  venv              : create virtual environment and install package with dev extras."
	@echo "  uv-venv           : create uv virtual environment and install package with dev extras."
	@echo "  style             : run formatting tools (black + isort)."
	@echo "  clean             : remove temporary/cache artifacts."
	@echo "  test-unit         : run unit tests."
	@echo "  test-integration  : run integration tests."
	@echo "  test-gpu          : run GPU-marked tests (auto-skip if no GPU)."

venv:
	python3.12 -m venv .venv
	source .venv/bin/activate && \
	python -m pip install -U pip setuptools wheel && \
	python -m pip install -e .[dev]

uv-venv:
	uv venv --python 3.12
	uv pip install --python .venv/bin/python -e .[dev]

style:
	python -m black .
	python -m isort .

clean:
	find . -type f -name "*.DS_Store" -delete
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +

test-unit:
	python -m pytest -m "unit"

test-integration:
	python -m pytest -m "integration"

test-gpu:
	python -m pytest -m "gpu"
