#!/bin/bash
# Better Impuls Viewer Launcher Script

echo "🌟 Better Impuls Viewer - Desktop Application"
echo "============================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed or not in PATH"
    echo "Please install Node.js 16+ and try again"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Check Python dependencies
echo "🐍 Checking Python dependencies..."
if ! python -c "import fastapi, uvicorn, pandas, numpy" &> /dev/null; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
fi

echo "✅ All dependencies are ready"

# Determine mode
if [ "$1" == "dev" ] || [ "$1" == "development" ]; then
    echo "🚀 Starting in development mode..."
    npm run dev
elif [ "$1" == "test" ]; then
    echo "🧪 Running backend integration test..."
    npm run test:backend
else
    echo "🚀 Starting Better Impuls Viewer..."
    npm start
fi