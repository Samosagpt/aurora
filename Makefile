# Makefile for Aurora Project
# Provides convenient commands for common development tasks

.PHONY: help install test test-unit test-integration test-all coverage lint format clean docs

# Default target
help:
	@echo "Aurora Development Commands"
	@echo "============================"
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install all dependencies"
	@echo "  make install-test    Install testing dependencies"
	@echo "  make install-dev     Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test            Run all tests"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make test-integration Run integration tests"
	@echo "  make test-fast       Run fast tests only"
	@echo "  make coverage        Run tests with coverage report"
	@echo "  make coverage-html   Generate HTML coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            Run all linters"
	@echo "  make format          Format code with black and isort"
	@echo "  make type-check      Run type checking with mypy"
	@echo "  make security        Run security checks with bandit"
	@echo ""
	@echo "Pre-commit:"
	@echo "  make pre-commit      Install pre-commit hooks"
	@echo "  make pre-commit-run  Run pre-commit on all files"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean           Remove build artifacts and cache"
	@echo "  make clean-test      Remove test artifacts"
	@echo "  make clean-all       Remove all generated files"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs            View documentation"
	@echo ""
	@echo "CI/CD:"
	@echo "  make ci              Run CI checks locally"

# Installation targets
install:
	pip install -r requirements.txt

install-test:
	python setup_testing.py

install-dev: install install-test
	pip install black isort flake8 pylint mypy bandit[toml] pre-commit

# Testing targets
test:
	pytest -v

test-unit:
	pytest -v -m "unit or not (integration or slow or requires_ollama)"

test-integration:
	pytest -v -m "integration"

test-fast:
	pytest -v -m "not slow"

test-all:
	pytest -v --maxfail=1

coverage:
	pytest --cov=. --cov-report=term-missing

coverage-html:
	pytest --cov=. --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality targets
lint:
	@echo "Running flake8..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "Running pylint..."
	pylint **/*.py --fail-under=7.0 || true
	@echo "Running bandit..."
	bandit -r . -f screen || true

format:
	@echo "Running black..."
	black .
	@echo "Running isort..."
	isort .

format-check:
	@echo "Checking black..."
	black --check --diff .
	@echo "Checking isort..."
	isort --check-only --diff .

type-check:
	mypy . --ignore-missing-imports

security:
	bandit -r . -f json -o bandit-report.json
	bandit -r . -f screen

# Pre-commit targets
pre-commit:
	pre-commit install
	pre-commit autoupdate

pre-commit-run:
	pre-commit run --all-files

# Cleaning targets
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	rm -rf .mypy_cache 2>/dev/null || true
	rm -rf build 2>/dev/null || true
	rm -rf dist 2>/dev/null || true

clean-test:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -f .coverage
	rm -f coverage.xml
	rm -f bandit-report.json

clean-all: clean clean-test
	find . -type d -name ".venv" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "venv" -exec rm -rf {} + 2>/dev/null || true

# Documentation targets
docs:
	@echo "Opening documentation..."
	@if [ -f README.md ]; then cat README.md; fi
	@echo "\nFor testing docs, see TESTING.md"
	@echo "For contributing, see CONTRIBUTING.md"

# CI/CD targets
ci: format-check lint test coverage
	@echo "All CI checks passed!"
