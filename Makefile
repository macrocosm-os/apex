# Makefile for prompting project
SHELL:=/bin/bash

.PHONY: test test-cov test-diff-cov install lint format clean help promote-changes

# Test commands
test:
	poetry run pytest tests/ -v

test-cov:
	poetry run pytest --cov=prompting --cov-report=term-missing --cov-report=html --cov-report=xml tests/

test-diff-cov:
	bash scripts/check_diff_coverage.sh

test-fast:
	poetry run pytest tests/ -x --ff

# Coverage commands
coverage-report:
	poetry run coverage report

coverage-html:
	poetry run coverage html
	@echo "HTML coverage report generated in htmlcov/index.html"

coverage-xml:
	poetry run coverage xml

# Development commands
install:
	poetry install --all-extras

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

lint-fix:
	poetry run ruff check --fix .

# Pre-commit
pre-commit-install:
	poetry run pre-commit install

pre-commit-run:
	poetry run pre-commit run --all-files

# Original commands
promote-changes:
	./scripts/promote_changes.sh

# Clean commands
clean:
	rm -rf .coverage htmlcov/ coverage.xml .pytest_cache/ __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Help
help:
	@echo "Available commands:"
	@echo "  test              - Run tests"
	@echo "  test-cov          - Run tests with coverage"
	@echo "  test-diff-cov     - Run diff coverage check"
	@echo "  test-fast         - Run tests with fail-fast"
	@echo "  coverage-report   - Generate coverage report"
	@echo "  coverage-html     - Generate HTML coverage report"
	@echo "  coverage-xml      - Generate XML coverage report"
	@echo "  install           - Install dependencies"
	@echo "  lint              - Run linter"
	@echo "  format            - Run formatter"
	@echo "  lint-fix          - Run linter with auto-fix"
	@echo "  pre-commit-install- Install pre-commit hooks"
	@echo "  pre-commit-run    - Run pre-commit on all files"
	@echo "  promote-changes   - Run promote changes script"
	@echo "  clean             - Clean up coverage and cache files"
	@echo "  help              - Show this help message"
