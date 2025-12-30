# Testing Guide for Aurora

This guide provides comprehensive information about the testing infrastructure and practices for the Aurora project.

## Table of Contents

- [Overview](#overview)
- [Test Framework](#test-framework)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Writing Tests](#writing-tests)
- [Code Coverage](#code-coverage)
- [Continuous Integration](#continuous-integration)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Troubleshooting](#troubleshooting)

## Overview

Aurora uses **pytest** as its primary testing framework, with comprehensive test coverage across all modules. Our testing infrastructure includes:

- ✅ Unit tests for individual components
- ✅ Integration tests for system-level functionality
- ✅ Automated CI/CD pipeline with GitHub Actions
- ✅ Code coverage tracking and reporting
- ✅ Pre-commit hooks for code quality
- ✅ Multiple Python version support (3.9-3.12)
- ✅ Cross-platform testing (Linux, Windows, macOS)

## Test Framework

### Core Dependencies

```bash
pip install pytest pytest-cov pytest-xdist pytest-timeout pytest-mock
```

### Configuration Files

- **`pytest.ini`** - Main pytest configuration
- **`pyproject.toml`** - Tool configurations (coverage, black, isort, etc.)
- **`.coveragerc`** - Coverage.py configuration
- **`conftest.py`** - Shared fixtures and test utilities
- **`.pre-commit-config.yaml`** - Pre-commit hooks configuration

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest test_aurora_system.py

# Run specific test function
pytest test_aurora_system.py::test_aurora_system

# Run tests in a directory
pytest tests/
```

### Running Tests by Category

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests that don't require Ollama
pytest -m "not requires_ollama"

# Run fast tests (exclude slow ones)
pytest -m "not slow"

# Combine markers
pytest -m "unit and not requires_gpu"
```

### Parallel Test Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

### Test Output and Debugging

```bash
# Show local variables in tracebacks
pytest -l

# Stop on first failure
pytest -x

# Stop after N failures
pytest --maxfail=3

# Show full diff for assertions
pytest -vv

# Capture output
pytest -s  # Don't capture stdout/stderr

# Show print statements
pytest -s -v
```

## Test Categories

Aurora tests are organized using pytest markers:

### Available Markers

| Marker | Description | Usage |
|--------|-------------|-------|
| `unit` | Fast unit tests | `@pytest.mark.unit` |
| `integration` | Integration tests | `@pytest.mark.integration` |
| `slow` | Tests taking >1 second | `@pytest.mark.slow` |
| `requires_ollama` | Needs Ollama running | `@pytest.mark.requires_ollama` |
| `requires_gpu` | Needs GPU/CUDA | `@pytest.mark.requires_gpu` |
| `requires_internet` | Needs internet | `@pytest.mark.requires_internet` |
| `desktop_only` | Desktop environment | `@pytest.mark.desktop_only` |
| `windows_only` | Windows-specific | `@pytest.mark.windows_only` |
| `linux_only` | Linux-specific | `@pytest.mark.linux_only` |

### Example Usage

```python
import pytest

@pytest.mark.unit
def test_simple_function():
    """Fast unit test."""
    assert 1 + 1 == 2

@pytest.mark.integration
@pytest.mark.requires_ollama
def test_ollama_integration():
    """Integration test requiring Ollama."""
    # Test code here
    pass

@pytest.mark.slow
@pytest.mark.requires_gpu
def test_image_generation():
    """Slow test requiring GPU."""
    # Test code here
    pass
```

## Writing Tests

### Test File Organization

```
aurora/
├── test_aurora_system.py      # Aurora system tests
├── test_rag.py                 # RAG system tests
├── test_cords.py               # Coordinate testing
└── tests/                      # Additional tests directory
    ├── unit/                   # Unit tests
    ├── integration/            # Integration tests
    └── data/                   # Test data files
```

### Test Naming Conventions

- Test files: `test_*.py` or `*_test.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Using Fixtures

Aurora provides many pre-configured fixtures in `conftest.py`:

```python
def test_with_fixtures(mock_ollama, temp_rag_db, capture_logs):
    """Example test using fixtures."""
    # mock_ollama provides mocked Ollama client
    response = mock_ollama.generate("test prompt")

    # temp_rag_db provides temporary database path
    assert temp_rag_db.exists()

    # capture_logs captures log output
    assert "expected log message" in capture_logs.text
```

### Common Fixtures

- `project_root_path` - Project root directory
- `test_data_dir` - Test data directory
- `temp_output_dir` - Temporary output directory
- `mock_ollama` - Mocked Ollama client
- `mock_aurora_system` - Mocked Aurora system
- `temp_config_file` - Temporary config file
- `temp_rag_db` - Temporary RAG database
- `sample_text_file` - Sample text file
- `skip_if_no_ollama` - Skip if Ollama unavailable
- `skip_if_no_gpu` - Skip if GPU unavailable
- `skip_if_no_internet` - Skip if offline

### Test Example Template

```python
"""
Tests for [module_name].

This module tests [description of what is being tested].
"""

import pytest
from module_name import function_to_test


class TestModuleName:
    """Test suite for ModuleName."""

    def test_basic_functionality(self):
        """Test basic functionality works correctly."""
        result = function_to_test("input")
        assert result == "expected_output"

    @pytest.mark.parametrize("input,expected", [
        ("a", "A"),
        ("b", "B"),
        ("c", "C"),
    ])
    def test_multiple_inputs(self, input, expected):
        """Test function with multiple inputs."""
        assert function_to_test(input) == expected

    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function_to_test(None)

    @pytest.mark.slow
    def test_performance(self):
        """Test performance with large input."""
        large_input = "x" * 10000
        result = function_to_test(large_input)
        assert len(result) > 0
```

## Code Coverage

### Generating Coverage Reports

```bash
# Run tests with coverage
pytest --cov=.

# Generate HTML report
pytest --cov=. --cov-report=html

# Generate XML report (for CI/CD)
pytest --cov=. --cov-report=xml

# Show missing lines
pytest --cov=. --cov-report=term-missing

# Coverage for specific module
pytest --cov=rag_handler tests/
```

### Viewing Coverage Reports

```bash
# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Configuration

Coverage settings are in `pyproject.toml` and `.coveragerc`:

- Minimum coverage target: 80% (configurable)
- Branch coverage enabled
- Excludes test files, logs, build artifacts

## Continuous Integration

### GitHub Actions Workflows

Aurora uses GitHub Actions for automated testing:

#### CI/CD Pipeline (`.github/workflows/ci.yml`)

- **Linting** - Code quality checks (Black, isort, Flake8, Pylint, Bandit)
- **Unit Tests** - Fast tests on multiple Python versions and OS
- **Integration Tests** - Tests with Ollama and external dependencies
- **Build Check** - Package build verification
- **Security Scan** - Dependency vulnerability scanning
- **Documentation Check** - Ensure docs are present

#### Test Workflow (`.github/workflows/tests.yml`)

- Runs tests on push and pull requests
- Tests multiple Python versions (3.9-3.12)
- Tests on Linux, Windows, and macOS
- Uploads coverage to Codecov
- Scheduled daily test runs

### Running CI Locally

To run CI checks locally before pushing:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

## Pre-commit Hooks

Pre-commit hooks automatically check code quality before commits.

### Installation

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
```

### What Gets Checked

- **Black** - Code formatting
- **isort** - Import sorting
- **Flake8** - Style guide enforcement
- **MyPy** - Type checking
- **Bandit** - Security linting
- **Various checks** - Trailing whitespace, file size, merge conflicts, etc.

### Manual Execution

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Update hook versions
pre-commit autoupdate
```

### Bypassing Hooks

```bash
# Skip hooks for a single commit (not recommended)
git commit --no-verify -m "commit message"
```

## Best Practices

### DO ✅

- Write tests for new features and bug fixes
- Use descriptive test names that explain what is being tested
- Test edge cases and error conditions
- Use fixtures to avoid code duplication
- Mark tests appropriately (unit, integration, slow, etc.)
- Keep tests independent and isolated
- Mock external dependencies (Ollama, API calls, file system)
- Update tests when changing functionality
- Aim for high code coverage (>80%)
- Run tests before submitting pull requests

### DON'T ❌

- Don't commit failing tests
- Don't write tests that depend on external state
- Don't use hardcoded paths or credentials
- Don't test implementation details
- Don't skip tests without good reason
- Don't write overly complex test setup
- Don't ignore CI failures
- Don't commit without running pre-commit hooks

## Troubleshooting

### Common Issues

#### Tests Pass Locally But Fail in CI

```bash
# Ensure you're using correct Python version
python --version

# Install exact dependencies
pip install -r requirements.txt

# Clear pytest cache
pytest --cache-clear

# Check for platform-specific issues
pytest -v -m "not (windows_only or linux_only)"
```

#### Ollama Tests Failing

```bash
# Ensure Ollama is running
ollama list

# Pull required models
ollama pull llama3.2

# Skip Ollama tests
pytest -m "not requires_ollama"
```

#### Import Errors

```bash
# Ensure project is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Install in development mode
pip install -e .
```

#### Coverage Not Working

```bash
# Install coverage tools
pip install pytest-cov coverage

# Clear previous coverage data
coverage erase

# Run with coverage
pytest --cov=. --cov-report=term-missing
```

### Getting Help

- Check existing tests for examples
- Review pytest documentation: <https://docs.pytest.org/>
- Open an issue on GitHub
- Ask in GitHub Discussions

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Happy Testing! 🧪**
