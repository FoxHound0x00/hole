.PHONY: help install format lint test test-fast test-slow clean

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
SRC := hole tests

help:
	@echo "Available targets:"
	@echo "  install     - Install package in editable mode with dev extras"
	@echo "  format      - Run black + isort on $(SRC)"
	@echo "  lint        - Run flake8 + mypy on hole/"
	@echo "  test        - Run the full test suite"
	@echo "  test-fast   - Run tests except the slow end-to-end driver tests"
	@echo "  test-slow   - Run only the slow end-to-end driver tests"
	@echo "  clean       - Remove build / cache artefacts"

install:
	$(PYTHON) -m pip install -e ".[dev]" || $(PYTHON) -m pip install -e .

format:
	$(PYTHON) -m black $(SRC)
	$(PYTHON) -m isort $(SRC)

lint:
	$(PYTHON) -m flake8 hole
	$(PYTHON) -m mypy hole

test:
	$(PYTEST) -q

test-fast:
	$(PYTEST) -q --ignore=tests/test_analysis_drivers.py

test-slow:
	$(PYTEST) -q tests/test_analysis_drivers.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
