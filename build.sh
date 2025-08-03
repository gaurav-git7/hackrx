#!/bin/bash

# Build script for Render deployment
# This script ensures proper installation order and handles common build issues

echo "🚀 Starting build process..."

# Set environment variables for better compatibility
export PYTHONPATH="${PYTHONPATH}:${PWD}"
export TOKENIZERS_PARALLELISM=false

# Upgrade pip and install essential build tools first
echo "📦 Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install system dependencies
echo "🔧 Installing system dependencies..."
pip install packaging>=21.0

# Install PyTorch CPU version for better compatibility
echo "🧠 Installing PyTorch CPU version..."
pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install requirements with verbose output for debugging
echo "📋 Installing project dependencies..."
pip install -r requirements.txt --verbose

# Verify critical packages
echo "🔍 Verifying critical packages..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import sentence_transformers; print('Sentence transformers imported successfully')"
python -c "import fastapi; print('FastAPI imported successfully')"

echo "✅ Build process completed successfully!" 