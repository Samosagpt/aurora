"""
Pytest configuration and shared fixtures for Aurora tests.

This file provides common test fixtures and configuration that can be used
across all test files in the Aurora project.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================


def pytest_configure(config):
    """Pytest configuration hook."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_ollama: Tests requiring Ollama")
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "requires_internet: Tests requiring internet")
    config.addinivalue_line("markers", "desktop_only: Tests for desktop environment only")
    config.addinivalue_line("markers", "windows_only: Windows-specific tests")
    config.addinivalue_line("markers", "linux_only: Linux-specific tests")


def pytest_collection_modifyitems(config, items):
    """Modify test items based on markers."""
    # Add default marker to tests without explicit markers
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# ============================================================================
# Session-level Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def test_data_dir(project_root_path):
    """Return the test data directory path."""
    test_dir = project_root_path / "tests" / "data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def temp_output_dir(project_root_path, tmp_path_factory):
    """Create a temporary directory for test outputs."""
    temp_dir = tmp_path_factory.mktemp("test_outputs")
    return temp_dir


# ============================================================================
# Environment and System Fixtures
# ============================================================================


@pytest.fixture
def mock_env(monkeypatch):
    """Provide a clean environment for testing."""
    # Save original env
    original_env = dict(os.environ)

    # Clear certain env variables for testing
    test_vars = ["OLLAMA_HOST", "CUDA_VISIBLE_DEVICES", "HF_TOKEN"]
    for var in test_vars:
        monkeypatch.delenv(var, raising=False)

    yield monkeypatch

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_platform(monkeypatch):
    """Mock platform detection."""
    platform_mock = Mock()
    platform_mock.system.return_value = "Linux"
    monkeypatch.setattr("platform.system", platform_mock.system)
    return platform_mock


# ============================================================================
# Ollama Fixtures
# ============================================================================


@pytest.fixture
def mock_ollama():
    """Mock Ollama client for testing."""
    mock = MagicMock()
    mock.list.return_value = {
        "models": [{"name": "llama3.2", "size": 1000000}, {"name": "llava", "size": 2000000}]
    }
    mock.generate.return_value = {"response": "Test response from mock Ollama", "done": True}
    mock.chat.return_value = {"message": {"content": "Test chat response"}, "done": True}
    return mock


@pytest.fixture
def skip_if_no_ollama():
    """Skip test if Ollama is not available."""
    try:
        import ollama

        ollama.list()
    except Exception:
        pytest.skip("Ollama is not available")


# ============================================================================
# GPU/CUDA Fixtures
# ============================================================================


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU is not available")
    except ImportError:
        pytest.skip("PyTorch is not installed")


@pytest.fixture
def mock_torch_no_cuda(monkeypatch):
    """Mock torch to report no CUDA availability."""
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    return torch_mock


# ============================================================================
# File System Fixtures
# ============================================================================


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_file = tmp_path / "test_config.json"
    config_file.write_text('{"test": "value"}')
    return config_file


@pytest.fixture
def temp_rag_db(tmp_path):
    """Create a temporary RAG database file for testing."""
    db_file = tmp_path / "test_rag.json"
    return db_file


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    text_file = tmp_path / "sample.txt"
    text_file.write_text("This is a sample text for testing.\n" * 10)
    return text_file


# ============================================================================
# Aurora System Fixtures
# ============================================================================


@pytest.fixture
def mock_aurora_system():
    """Mock Aurora system for testing."""
    mock = MagicMock()
    mock.get_identity.return_value = "AURORA Test System"
    mock.get_version.return_value = "2.0.0-test"
    mock.get_capabilities.return_value = {
        "modalities": ["text", "image", "vision"],
        "skills": ["generation", "analysis", "automation"],
    }
    return mock


# ============================================================================
# Network Fixtures
# ============================================================================


@pytest.fixture
def skip_if_no_internet():
    """Skip test if internet is not available."""
    import socket

    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
    except OSError:
        pytest.skip("Internet connection is not available")


@pytest.fixture
def mock_requests(monkeypatch):
    """Mock requests library for testing."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_response.text = "Mock response text"

    mock_get = Mock(return_value=mock_response)
    mock_post = Mock(return_value=mock_response)

    monkeypatch.setattr("requests.get", mock_get)
    monkeypatch.setattr("requests.post", mock_post)

    return {"get": mock_get, "post": mock_post, "response": mock_response}


# ============================================================================
# Logging Fixtures
# ============================================================================


@pytest.fixture
def capture_logs(caplog):
    """Capture logs during test execution."""
    import logging

    caplog.set_level(logging.DEBUG)
    return caplog


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary test files after each test."""
    yield
    # Cleanup code here if needed
    pass


# ============================================================================
# Parametrize Helpers
# ============================================================================


def pytest_generate_tests(metafunc):
    """Generate parameterized tests based on markers."""
    # This can be used for custom parametrization logic
    pass
