#!/bin/bash
# Start Prediction Service on Linux/Mac

echo "======================================"
echo "🔮 Starting Prediction Service"
echo "======================================"

cd "$(dirname "$0")"

echo ""
echo "📦 Checking dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "⚠️  FastAPI not installed"
    echo "Installing dependencies..."
    pip install -r requirements-prediction-service.txt
fi

echo ""
echo "🚀 Starting service on http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""

py prediction_service.py --port 5000
