#!/bin/bash
# Quick start script for MLTune

set -e

echo "=================================================="
echo "MLTune - Machine Learning Training & Tuning Platform"
echo "=================================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "Quick Start:"
echo "  1. Run an optimization: mltune optimize config.yaml"
echo "  2. Start the API server: mltune server"
echo "  3. View experiments: mltune experiments"
echo ""
echo "For more information, see README.md or run: mltune --help"
