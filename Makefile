# Makefile for Mahalanobis Filter Project

.PHONY: clean install dev test lint format run build-package publish help venv venv-activate

# Python interpreter to use
PYTHON = python3

# Virtual environment directory - use local directory
VENV_DIR = ./venv

# Check if we're in a Google Drive directory
IS_GOOGLE_DRIVE := $(shell pwd | grep 'GoogleDrive' > /dev/null && echo true || echo false)

# Default target
.DEFAULT_GOAL := help

help:
	@echo "Mahalanobis Filter Dashboard - Development Commands"
	@echo ""
	@echo "Usage:"
	@echo "  make venv          Create and set up a virtual environment"
	@echo "  make venv-activate Activate the virtual environment"
	@echo "  make install       Install production dependencies"
	@echo "  make dev           Install development dependencies"
	@echo "  make run           Run the Dash application"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black"
	@echo "  make clean         Remove build artifacts and cache files"
	@echo "  make build-package Build Python package"
	@echo "  make publish       Publish package to PyPI"
	@echo "  make help          Show this help message"
	@echo ""
	@echo "Development Workflow:"
	@echo "  1. make venv          # Create virtual environment"
	@echo "  2. make install       # Install dependencies"
	@echo "  3. make run           # Start the application"
	@echo ""

# Create a virtual environment
venv:
	@if [ "$(IS_GOOGLE_DRIVE)" = "true" ]; then \
		echo "Warning: Google Drive detected. Creating virtual environment in ~/.local/venv/mahalanobis-filter"; \
		mkdir -p ~/.local/venv/mahalanobis-filter; \
		$(PYTHON) -m venv ~/.local/venv/mahalanobis-filter; \
		ln -sf ~/.local/venv/mahalanobis-filter $(VENV_DIR); \
	else \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@echo "\nVirtual environment created in $(VENV_DIR)/"
	@echo "Run 'make venv-activate' to activate it\n"

# Activate virtual environment
venv-activate:
	@echo "Run the following command to activate the virtual environment:"
	@echo "  source $(VENV_DIR)/bin/activate"
	@echo "Or use:"
	@echo "  . $(VENV_DIR)/bin/activate"

# Install production dependencies
install:
	@echo "Installing dependencies from requirements.txt..."
	pip install -r requirements.txt

# Install development dependencies
dev: install
	@echo "Installing development dependencies..."
	$(VENV_DIR)/bin/pip install -e ".[dev]"
	$(VENV_DIR)/bin/pip install black flake8 pylint pytest

# Run the application
run:
	$(VENV_DIR)/bin/python -m mahalanobis_filter.mahalanobis_filter_dash

# Run tests
test:
	$(VENV_DIR)/bin/python -m pytest tests/

# Run linting
lint:
	$(VENV_DIR)/bin/python -m flake8 mahalanobis_filter/
	$(VENV_DIR)/bin/python -m pylint mahalanobis_filter/

# Format code
format:
	$(VENV_DIR)/bin/python -m black mahalanobis_filter/

# Clean build artifacts and cache files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name *.egg-info -exec rm -rf {} +
	find . -type f -name *.pyc -delete

# Clean everything including the virtual environment
clean-all: clean
	rm -rf $(VENV_DIR)/

# Build Python package
build-package: clean
	$(VENV_DIR)/bin/pip install --upgrade build
	$(VENV_DIR)/bin/python -m build

# Publish package to PyPI
publish: build-package
	$(VENV_DIR)/bin/pip install --upgrade twine
	$(VENV_DIR)/bin/python -m twine upload dist/*
