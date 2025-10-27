# Aurora Tests Directory

This directory contains the test suite for the Aurora project.

## Structure

```
tests/
├── __init__.py              # Test package initialization
├── README.md                # This file
├── unit/                    # Unit tests
│   ├── __init__.py
│   ├── test_generation.py
│   ├── test_rag.py
│   └── test_utilities.py
├── integration/             # Integration tests
│   ├── __init__.py
│   ├── test_ollama_integration.py
│   └── test_system_integration.py
└── data/                    # Test data and fixtures
    ├── sample_text.txt
    └── test_config.json
```

## Running Tests

See [TESTING.md](../TESTING.md) for comprehensive testing documentation.

### Quick Start

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run with coverage
pytest --cov=. --cov-report=html
```

## Writing New Tests

1. **Choose the appropriate directory**:
   - `unit/` for fast, isolated tests
   - `integration/` for tests requiring external dependencies

2. **Use appropriate markers**:

   ```python
   import pytest

   @pytest.mark.unit
   def test_function():
       pass

   @pytest.mark.integration
   @pytest.mark.requires_ollama
   def test_with_ollama():
       pass
   ```

3. **Use fixtures from conftest.py**:

   ```python
   def test_with_mock(mock_ollama):
       response = mock_ollama.generate("test")
       assert response is not None
   ```

## Test Guidelines

- Keep tests focused and independent
- Use descriptive test names
- Add docstrings to test functions
- Mock external dependencies
- Clean up resources after tests
- Aim for high code coverage (>80%)

## Resources

- [TESTING.md](../TESTING.md) - Comprehensive testing guide
- [conftest.py](../conftest.py) - Shared fixtures and utilities
- [pytest.ini](../pytest.ini) - Pytest configuration
