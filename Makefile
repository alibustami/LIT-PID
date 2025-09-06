SHELL := /bin/bash
PY := 3.11
VENV_DIR ?= lit-pid-env

export UV_PROJECT_ENVIRONMENT := $(CURDIR)/$(VENV_DIR)

.PHONY: create-venv remove-venv precommit fmt lint test

create-venv:
	@echo "START: Creating .venv with uv (Python $(PY))" ; \
	export UV_PROJECT_ENVIRONMENT="lit-pid-env" ; \
	uv venv lit-pid-env --python $(PY) --system-site-packages ; \
	uv sync ; \
	uv run pre-commit install


remove:
	@echo "START: removing .venv and lock" && \
	rm -rf .venv uv.lock dist/ lit-pid-env/

rqts_txt:
	uv export --no-hashes  > requirements.txt

lock:
	uv lock

dev-install:
	uv pip install -e .
