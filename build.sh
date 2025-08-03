#!/bin/bash

# Build script for Render deployment
# This script ensures proper installation order and handles common build issues

echo "🚀 Starting build process..."

# Upgrade pip and install essential build tools first
echo "📦 Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install system dependencies
echo "🔧 Installing system dependencies..."
pip install packaging>=21.0

# Install requirements with verbose output for debugging
echo "📋 Installing project dependencies..."
pip install -r requirements.txt --verbose

echo "✅ Build process completed successfully!" 