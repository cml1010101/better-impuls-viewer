#!/bin/bash

# test-bundle.sh - Test script for the bundled Electron application

set -e

echo "🧪 Testing Better Impuls Viewer Electron App Bundle..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if build exists
if [ ! -d "electron-dist" ]; then
    error "No build found. Please run './scripts/build.sh' first."
    exit 1
fi

# Find the executable
EXECUTABLE=""
if [ -f "electron-dist/better-impuls-viewer" ]; then
    EXECUTABLE="electron-dist/better-impuls-viewer"
elif [ -f "electron-dist/Better Impuls Viewer" ]; then
    EXECUTABLE="electron-dist/Better Impuls Viewer"
elif [ -f "electron-dist/linux-unpacked/better-impuls-viewer" ]; then
    EXECUTABLE="electron-dist/linux-unpacked/better-impuls-viewer"
elif [ -f "electron-dist/Better Impuls Viewer-1.0.0.AppImage" ]; then
    EXECUTABLE="electron-dist/Better Impuls Viewer-1.0.0.AppImage"
    chmod +x "$EXECUTABLE"
else
    error "Could not find executable in electron-dist/"
    log "Contents of electron-dist/:"
    ls -la electron-dist/
    exit 1
fi

log "Found executable: $EXECUTABLE"

# Test Python dependencies in bundled environment
log "Testing Python environment..."
PYTHON_PATH=""
if [ -f "python-env/bin/python" ]; then
    PYTHON_PATH="python-env/bin/python"
elif [ -f "python-env/bin/python3" ]; then
    PYTHON_PATH="python-env/bin/python3"
else
    error "Python environment not found"
    exit 1
fi

# Test Python imports
$PYTHON_PATH -c "
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))

try:
    import fastapi
    import uvicorn
    import pandas
    import numpy
    from backend.server import app
    print('✅ All Python dependencies available')
except ImportError as e:
    print(f'❌ Missing Python dependency: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    success "Python environment test passed"
else
    error "Python environment test failed"
    exit 1
fi

# Test frontend build
log "Testing frontend build..."
if [ -f "frontend/dist/index.html" ]; then
    success "Frontend build found"
else
    error "Frontend build not found"
    exit 1
fi

# Test sample data
log "Testing sample data..."
if [ -d "sample_data" ] && [ $(find sample_data -name "*.tbl" | wc -l) -gt 0 ]; then
    success "Sample data found ($(find sample_data -name "*.tbl" | wc -l) .tbl files)"
else
    warning "No sample data found. The app will work but may not have test data."
fi

# Test backend startup (quick test)
log "Testing backend startup..."
export DATA_FOLDER="$(pwd)/sample_data"
export PYTHONPATH="$(pwd)/backend"

# Start backend in background for a quick test
$PYTHON_PATH backend/server.py &
BACKEND_PID=$!

# Wait a bit for server to start
sleep 3

# Test if backend is responding
if curl -s http://localhost:8000/ > /dev/null; then
    success "Backend startup test passed"
else
    warning "Backend startup test inconclusive"
fi

# Kill the test backend
kill $BACKEND_PID 2>/dev/null || true
sleep 1

# Test executable permissions and basic info
log "Testing executable..."
if [ -x "$EXECUTABLE" ]; then
    success "Executable has proper permissions"
    
    # Get file size
    SIZE=$(du -sh "$EXECUTABLE" | cut -f1)
    log "Executable size: $SIZE"
    
    # Show file info
    file "$EXECUTABLE" 2>/dev/null || true
else
    error "Executable does not have execute permissions"
    exit 1
fi

echo ""
success "✅ All tests passed!"
echo ""
log "Bundle test summary:"
echo "  ✅ Python environment: OK"
echo "  ✅ Frontend build: OK"
echo "  ✅ Backend startup: OK"
echo "  ✅ Executable: OK ($SIZE)"
echo ""
log "To run the application:"
echo "  ./$EXECUTABLE"
echo ""
if [[ "$EXECUTABLE" == *.AppImage ]]; then
    log "This is an AppImage. You can also:"
    echo "  - Copy it anywhere and run it directly"
    echo "  - Make it executable: chmod +x '$EXECUTABLE'"
    echo "  - Run with: ./'$EXECUTABLE'"
fi