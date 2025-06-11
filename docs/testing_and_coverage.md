# Testing and Coverage Guide

This guide explains how to use the testing and coverage system for the prompting project, which ensures that all new or modified functions and classes have at least 80% unit test coverage.

## Overview

The project uses:
- **pytest** for running tests
- **coverage.py** and **pytest-cov** for measuring code coverage
- **diff-cover** for checking coverage only on changed code
- **Poetry** for dependency management
- **Pre-commit hooks** for automated checks
- **GitHub Actions** for CI/CD integration

## Quick Start

### 1. Install Dependencies

```bash
# Install all dependencies including test dependencies
poetry install --all-extras
```

### 2. Run Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run diff coverage check (only on changed code)
make test-diff-cov
```

### 3. View Coverage Reports

```bash
# Generate HTML coverage report
make coverage-html
# Open htmlcov/index.html in your browser

# Generate terminal coverage report
make coverage-report
```

## Detailed Usage

### Running Tests

#### Basic Test Execution
```bash
# Using pytest directly
poetry run pytest tests/

# Using make
make test

# Run with verbose output
poetry run pytest tests/ -v

# Run specific test file
poetry run pytest tests/test_specific_module.py

# Run tests with pattern matching
poetry run pytest tests/ -k "test_function_name"
```

#### Fast Testing Options
```bash
# Stop on first failure
make test-fast

# Run only failed tests from last run
poetry run pytest --lf

# Run failed tests first, then continue
poetry run pytest --ff
```

### Coverage Analysis

#### Full Coverage
```bash
# Run tests with coverage
poetry run pytest --cov=prompting --cov-report=term-missing tests/

# Or using make
make test-cov
```

#### Coverage Reports
```bash
# Terminal report
poetry run coverage report

# HTML report (interactive)
poetry run coverage html

# XML report (for CI)
poetry run coverage xml
```

#### Diff Coverage (Changed Code Only)
```bash
# Check coverage on changes since main branch
poetry run diff-cover coverage.xml --compare-branch=main --fail-under=80

# Using the script
bash scripts/check_diff_coverage.sh

# Or using make
make test-diff-cov
```

### Configuration

#### Coverage Configuration
Coverage is configured in `pyproject.toml` and `.coveragerc`:

- **Source**: Only `prompting` package is measured
- **Omit**: Test files, `__pycache__`, virtual environments
- **Minimum**: 80% coverage required
- **Reports**: Terminal, HTML, and XML formats

#### Pytest Configuration
Pytest is configured in `pyproject.toml`:

- **Test Discovery**: `test_*.py` and `*_test.py` files
- **Test Paths**: `tests/` directory
- **Markers**: `slow`, `integration`, `unit`

## CI/CD Integration

### GitHub Actions
The CI pipeline automatically:

1. Runs tests with coverage
2. Checks diff coverage (80% minimum on changed code)
3. Uploads coverage reports to Codecov (optional)
4. Fails the build if coverage is below threshold

### Pre-commit Hooks
Pre-commit hooks run:

1. pytest checks
2. Diff coverage checks
3. Code formatting (ruff)
4. Linting checks

```bash
# Install pre-commit hooks
make pre-commit-install

# Run pre-commit on all files
make pre-commit-run
```

## Writing Tests

### Test Structure
```python
# tests/test_module.py
import pytest
from prompting.module import MyClass, my_function

class TestMyClass:
    def test_initialization(self):
        obj = MyClass()
        assert obj is not None

    def test_method_behavior(self):
        obj = MyClass()
        result = obj.method()
        assert result == expected_value

def test_my_function():
    result = my_function(input_value)
    assert result == expected_output
```

### Test Markers
```python
import pytest

@pytest.mark.unit
def test_unit_function():
    """Fast unit test"""
    pass

@pytest.mark.integration
def test_integration_feature():
    """Slower integration test"""
    pass

@pytest.mark.slow
def test_expensive_operation():
    """Very slow test"""
    pass
```

### Running Specific Test Types
```bash
# Run only unit tests
poetry run pytest -m unit

# Skip slow tests
poetry run pytest -m "not slow"

# Run integration tests
poetry run pytest -m integration
```

## Coverage Requirements

### Diff Coverage
- **New/Modified Code**: Must have ≥80% coverage
- **Comparison**: Against `main` branch
- **Enforcement**: CI fails if requirement not met

### Full Coverage
- **Global Minimum**: 80% overall coverage
- **Reporting**: HTML reports available for detailed analysis
- **Exclusions**: Test files, migrations, generated code

## Troubleshooting

### Common Issues

#### 1. Coverage Too Low
```bash
# Check which lines are missing coverage
poetry run coverage report --show-missing

# Generate HTML report for visual inspection
make coverage-html
```

#### 2. Diff Coverage Failing
```bash
# Check what changed files need more tests
poetry run diff-cover coverage.xml --compare-branch=main

# See specific line coverage
poetry run diff-cover coverage.xml --compare-branch=main --html-report diff-cover.html
```

#### 3. Tests Not Found
```bash
# Check test discovery
poetry run pytest --collect-only

# Verify test file naming (must be test_*.py or *_test.py)
ls tests/test_*.py
```

### Performance Tips

1. **Use markers** to categorize tests by speed
2. **Run fast tests first** during development
3. **Use diff coverage** to focus on changes
4. **Parallel execution** for large test suites:
   ```bash
   poetry add --group dev pytest-xdist
   poetry run pytest -n auto
   ```

## Best Practices

### Test Writing
- Write tests for all public functions and methods
- Include edge cases and error conditions
- Use descriptive test names
- Keep tests focused and independent
- Mock external dependencies

### Coverage Strategy
- Focus on critical business logic
- Don't aim for 100% coverage everywhere
- Use `# pragma: no cover` sparingly for untestable code
- Prioritize diff coverage for new changes

### Development Workflow
1. Write failing test first (TDD)
2. Implement minimal code to pass
3. Run `make test-diff-cov` to check coverage
4. Refactor with confidence
5. Use pre-commit hooks to catch issues early

## Available Commands

```bash
# Testing
make test              # Run all tests
make test-cov          # Run tests with coverage
make test-diff-cov     # Run diff coverage check
make test-fast         # Run with fail-fast

# Coverage
make coverage-report   # Terminal coverage report
make coverage-html     # HTML coverage report
make coverage-xml      # XML coverage report

# Development
make lint              # Run linter
make format            # Run formatter
make lint-fix          # Run linter with auto-fix
make clean             # Clean coverage files

# Pre-commit
make pre-commit-install # Install hooks
make pre-commit-run     # Run all hooks

# Help
make help              # Show all commands
```

## Integration with IDEs

### VS Code
Add to `.vscode/settings.json`:
```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "python.linting.enabled": true,
    "coverage-gutters.coverageFileNames": ["coverage.xml"]
}
```

### PyCharm
1. Configure pytest as test runner
2. Enable coverage in run configurations
3. Set coverage source to `prompting` package

This setup ensures that your code maintains high quality with comprehensive test coverage while making the testing workflow efficient and automated.
