#!/bin/bash
SHELL = /bin/bash

# Styling
.PHONY: style
style:
	black .
	python -m isort .
	pyupgrade

venv:
	python -m venv venv
	source venv/bin/activate && \
	python -m pip install pip setuptools wheel && \
	python -m pip install -e .

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: help
help:
    @echo "Commands:"
    @echo "venv    : creates a virtual environment."
    @echo "style   : executes style formatting."
    @echo "clean   : cleans all unnecessary files."