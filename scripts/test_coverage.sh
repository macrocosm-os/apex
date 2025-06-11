#!/bin/bash

# Test coverage script for prompting project
# This script runs tests with coverage and generates reports

set -e

echo "🧪 Running tests with coverage..."

# Run tests with coverage
poetry run pytest --cov=prompting --cov-report=term-missing --cov-report=html --cov-report=xml tests/

echo "📊 Coverage reports generated:"
echo "  - HTML report: htmlcov/index.html"
echo "  - XML report: coverage.xml"
echo "  - Terminal report: displayed above"

# Check if coverage meets minimum threshold
echo "✅ Checking coverage threshold..."
poetry run coverage report --fail-under=80
