# Contributing to AutoForge

Thank you for your interest in contributing to AutoForge! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- **Bug Reports**: Submit issues for bugs you encounter
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests for bug fixes or features
- **Documentation**: Improve or add documentation
- **Examples**: Add example scripts or tutorials

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- pip or uv package manager

### Development Setup

1. **Fork and Clone**

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/autoforge.git
cd autoforge
```

2. **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Development Dependencies**

```bash
pip install -e ".[dev]"
```

4. **Set Up Pre-commit Hooks**

```bash
pre-commit install
```

5. **Run Tests**

```bash
pytest
```

## 📝 Development Guidelines

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Linting
- **mypy**: Type checking

Run before committing:

```bash
# Format code
black mltune tests
isort mltune tests

# Lint
ruff check mltune tests

# Type check
mypy mltune
```

### Code Conventions

1. **Type Hints**: All public functions should have type hints
2. **Docstrings**: Use Google-style docstrings
3. **Line Length**: Maximum 100 characters
4. **Imports**: Group imports (stdlib, third-party, local)

Example:

```python
"""Module docstring."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from mltune.core.config import Config


def my_function(
    param1: str,
    param2: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Short description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When param1 is empty.
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    return {"result": param1}
```

### Testing

- All new features must have tests
- Bug fixes should include regression tests
- Maintain test coverage above 80%

Run tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mltune --cov-report=term-missing

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::TestConfig::test_from_dict
```

### Documentation

- Update docstrings for modified functions
- Update README.md for user-facing changes
- Add examples for new features

## 🔀 Pull Request Process

1. **Create a Branch**

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

2. **Make Changes**
   - Write code following our conventions
   - Add/update tests
   - Update documentation

3. **Commit Changes**

We use conventional commits:

```
feat: add new optimizer
fix: resolve issue with config loading
docs: update README
test: add tests for optimizer
refactor: simplify experiment tracking
```

4. **Push and Create PR**

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

5. **PR Requirements**
   - All tests pass
   - Code coverage maintained
   - Documentation updated
   - Pre-commit hooks pass

6. **Review Process**
   - Address review comments
   - Keep PR up to date with main branch
   - Wait for approval from maintainers

## 📋 Issue Guidelines

### Bug Reports

Include:
- Python version
- AutoForge version
- Minimal reproducible example
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative solutions considered

## 🏗️ Project Structure

```
mltune/
├── mltune/           # Main package
│   ├── core/         # Core abstractions
│   ├── optim/        # Optimization algorithms
│   ├── tracker/      # Experiment tracking
│   ├── api/          # REST API
│   ├── utils/        # Utilities
│   └── cli.py        # CLI interface
├── tests/            # Test suite
├── docs/             # Documentation
├── examples/         # Example scripts
└── configs/          # Configuration templates
```

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions

Thank you for contributing to AutoForge! 🎉
