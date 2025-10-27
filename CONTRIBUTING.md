# Contributing to AURORA

First off, thank you for considering contributing to AURORA! It's people like you that make AURORA such a great tool.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up your development environment** (see below)
4. **Create a branch** for your feature or bugfix
5. **Make your changes** following our coding standards
6. **Test your changes** thoroughly
7. **Submit a pull request**

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, screenshots)
- **Describe the behavior you observed** and what you expected
- **Include your environment details** (OS, Python version, Ollama version)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **List some examples** of how it would be used

### Your First Code Contribution

Unsure where to begin? You can start by looking through `beginner` and `help-wanted` issues:

- **Beginner issues** - issues that should only require a few lines of code
- **Help wanted issues** - issues that may be more involved

### Pull Requests

- Fill in the required template
- Follow the coding style guide
- Include appropriate test cases
- Update documentation as needed
- End all files with a newline

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Ollama (for local LLM inference)

### Local Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/aurora.git
   cd aurora
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Install Ollama models**

   ```bash
   ollama pull llama3.2
   ollama pull llava  # For vision features
   ```

6. **Run the application**

   ```bash
   # Web interface
   streamlit run streamlit_app.py

   # Console interface
   python main.py
   ```

### Running Tests

Aurora has a comprehensive testing infrastructure. For detailed information, see [TESTING.md](TESTING.md).

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest test_aurora_system.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run only unit tests
pytest -m unit

# Run tests in parallel
pytest -n auto
```

**Quick test checklist before submitting PR:**

```bash
# 1. Run pre-commit hooks
pre-commit run --all-files

# 2. Run unit tests
pytest -m "unit or not (integration or slow)"

# 3. Check coverage
pytest --cov=. --cov-report=term-missing

# 4. Run full test suite (if time permits)
pytest -v
```

For more details on:

- Test categories and markers
- Writing new tests
- Using fixtures
- Code coverage
- CI/CD pipeline

Please refer to **[TESTING.md](TESTING.md)**

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: Maximum 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Group imports (stdlib, third-party, local)
- **Naming conventions**:
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`

### Code Quality

- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings for all public functions/classes
- **Comments**: Write clear, concise comments for complex logic
- **Error handling**: Use specific exceptions, avoid bare `except:`
- **Logging**: Use the logging module, not print statements

### Example

```python
from typing import Optional, List

def process_user_input(text: str, model: Optional[str] = None) -> dict:
    """
    Process user input and generate a response.

    Args:
        text: The user's input text
        model: Optional model name to use for generation

    Returns:
        Dictionary containing the response and metadata

    Raises:
        ValueError: If text is empty
        ConnectionError: If Ollama is not available
    """
    if not text.strip():
        raise ValueError("Input text cannot be empty")

    # Implementation here
    pass
```

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally

**Format:**

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Example:**

```
feat(vision): Add support for new vision models

- Integrated Qwen3-VL for improved screen analysis
- Added configuration options for vision models
- Updated documentation with usage examples

Closes #123
```

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features
3. **Ensure all tests pass** before submitting
4. **Update CHANGELOG.md** with your changes
5. **Request review** from at least one core maintainer
6. **Address review comments** promptly

### PR Template

When creating a PR, please include:

- **Description**: What does this PR do?
- **Related Issue**: Link to the issue this PR addresses
- **Type of Change**: Bug fix, feature, documentation, etc.
- **Testing**: How was this tested?
- **Screenshots**: If applicable
- **Checklist**:
  - [ ] Code follows style guidelines
  - [ ] Self-reviewed code
  - [ ] Commented complex code
  - [ ] Updated documentation
  - [ ] Added tests
  - [ ] All tests pass

## Project Structure

Understanding the project structure will help you contribute:

```
aurora/
├── Core AI Engine/
│   ├── Generation.py           # Ollama manager
│   ├── rag_handler.py         # RAG system
│   └── prompthandler.py       # Intent routing
├── Agentic Capabilities/
│   ├── desktop_agent.py       # Desktop control
│   ├── agentic_handler.py     # Tool orchestration
│   └── vision_agent.py        # Autonomous agent
├── Generation Modules/
│   ├── image_gen.py           # Image generation
│   └── video_gen.py           # Video generation
├── Voice & Speech/
│   ├── offline_sr_whisper.py  # Speech recognition
│   └── offline_text2speech.py # Text-to-speech
└── User Interfaces/
    ├── streamlit_app.py       # Web UI
    └── main.py                # Console UI
```

## Community

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions

## Recognition

Contributors will be recognized in:

- README.md (Contributors section)
- CHANGELOG.md (for each release)
- GitHub Contributors page

## Questions?

Feel free to open an issue or discussion if you have questions about contributing!

---

**Thank you for contributing to AURORA!** 🎉
