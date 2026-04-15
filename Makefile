# MOBIDICpy Development Makefile
# Provides common development commands for the project
# Note: Assumes virtual environment is already activated

# Variables
PYTHON := python
PIP := pip

.DEFAULT_GOAL := help

# Help target
.PHONY: help
help:  ## Show this help message
	@echo "MOBIDICpy Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup and Installation
.PHONY: install-dev
install-dev:  ## Install package in development mode with dev dependencies
	$(PIP) install --no-cache-dir --editable ".[dev,doc,calibration]"
	$(PYTHON) -W ignore::UserWarning -c "from pyemu.utils import get_pestpp; get_pestpp(':pyemu')"

.PHONY: install
install:  ## Install package with core dependencies
	$(PIP) install .

.PHONY: install-calib
install-calib:  ## Install calibration dependencies
	$(PIP) install .[calibration]
	$(PYTHON) -W ignore::UserWarning -c "from pyemu.utils import get_pestpp; get_pestpp(':pyemu')"

.PHONY: install-doc
install-doc:  ## Install documentation dependencies
	$(PIP) install .[doc]

# Testing
.PHONY: test
test:  ## Run tests with pytest
	$(PYTHON) -m pytest -v

.PHONY: coverage
coverage:  ## Generate test coverage report
	$(PYTHON) -m coverage run
	$(PYTHON) -m coverage report

.PHONY: coverage-html
coverage-html: coverage  ## Generate HTML coverage report
	$(PYTHON) -m coverage html

# Code Quality
.PHONY: lint-check
lint-check:  ## Check code quality with ruff
	$(PYTHON) -m ruff check .

.PHONY: lint
lint:  ## Fix code quality issues automatically
	$(PYTHON) -m ruff check . --fix

.PHONY: format
format: ## Format code with ruff
	${PYTHON} -m ruff format .

# Documentation
.PHONY: docs-serve
docs-serve:  ## Serve documentation locally
	$(PYTHON) -m mkdocs serve

# Build and Release
.PHONY: build
build:  ## Build package for distribution
	$(PYTHON) -m build

.PHONY: bump-patch
bump-patch:  ## Bump patch version
	bump-my-version bump patch

.PHONY: bump-minor
bump-minor:  ## Bump minor version
	bump-my-version bump minor

.PHONY: bump-major
bump-major:  ## Bump major version
	bump-my-version bump major

# Utility targets
.PHONY: clean
clean:  ## Clean temporary files and caches
	rm -rf build/
	rm -rf dist/
	rm -rf site/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.PHONY: all
all: install-dev lint test  ## Run common development tasks (install, lint, test)
