#!/bin/bash
# CellSight Cancer Prediction App - Startup Script
# ===================================================
# Requirements: Python 3.8+, Flask, scikit-learn, Pillow, joblib

echo "🔬 Starting CellSight Cancer Detection App..."

# Install dependencies if needed
pip install flask pillow scikit-learn joblib --break-system-packages -q 2>/dev/null || true

# Set models directory (default: same folder as this script)
export MODELS_DIR="${MODELS_DIR:-$(dirname "$0")}"

echo "📦 Models directory: $MODELS_DIR"
echo "🌐 Starting server at http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""

python3 "$(dirname "$0")/app.py"
