#!/bin/bash

# build.sh - Complete build script for Better Impuls Viewer Electron app

set -e  # Exit on any error

echo "🚀 Building Better Impuls Viewer Electron App..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function for colored output
log() {
    echo -e "${BLUE}[BUILD]${NC} $1"
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

# Clean up previous builds
log "Cleaning up previous builds..."
rm -rf dist electron-dist build python-env bundled-backend node_modules frontend/node_modules

# Install Electron dependencies
log "Installing Electron dependencies..."
npm install

# Build frontend
log "Building React frontend..."
cd frontend
npm install
npm run build
cd ..

success "Frontend built successfully"

# Create Python virtual environment and install backend dependencies
log "Setting up Python backend environment..."
python3 -m venv python-env

# Activate virtual environment and install dependencies
source python-env/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn

# Install Python dependencies from requirements.txt
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    error "requirements.txt not found"
    exit 1
fi

# Add additional required packages for bundled environment
pip install python-dotenv

deactivate

success "Python environment set up successfully"

# Validate the build
log "Validating build components..."

# Check if frontend dist exists
if [ ! -d "frontend/dist" ]; then
    error "Frontend build failed - dist directory not found"
    exit 1
fi

# Check if Python environment exists
if [ ! -d "python-env" ]; then
    error "Python environment setup failed"
    exit 1
fi

# Check if main electron file exists
if [ ! -f "electron/main.js" ]; then
    error "Electron main file not found"
    exit 1
fi

success "Build validation passed"

# Create the Electron application package
log "Creating Electron application package..."
npm run package

if [ $? -eq 0 ]; then
    success "Electron application built successfully!"
    
    # Check for generated files
    if [ -d "electron-dist" ]; then
        log "Build artifacts created in electron-dist/"
        ls -la electron-dist/
        
        # Calculate total size
        total_size=$(du -sh electron-dist/ | cut -f1)
        log "Total package size: $total_size"
    fi
else
    error "Electron packaging failed"
    exit 1
fi

echo ""
success "✅ Build completed successfully!"
echo ""
log "Next steps:"
echo "  1. Test the application: ./scripts/test-bundle.sh"
echo "  2. The packaged application is in the electron-dist/ directory"
echo ""