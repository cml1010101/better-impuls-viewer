#!/bin/bash

# dev-setup.sh - Quick development environment setup for Better Impuls Viewer Electron app

set -e

echo "🚀 Setting up Better Impuls Viewer for Electron Development..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[SETUP]${NC} $1"
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

# Check prerequisites
log "Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    error "Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    error "Node.js version $NODE_VERSION is too old. Please install Node.js 18+ first."
    exit 1
fi

success "Node.js $(node --version) found"

# Check Python
if ! command -v python3 &> /dev/null; then
    error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
success "Python $(python3 --version) found"

# Install Electron dependencies
log "Installing Electron dependencies..."
npm install

# Install frontend dependencies
log "Installing frontend dependencies..."
cd frontend
npm install
cd ..

success "Dependencies installed successfully"

# Check Python dependencies
log "Checking Python backend dependencies..."
MISSING_DEPS=""

python3 -c "
import sys
required_packages = ['fastapi', 'uvicorn', 'pandas', 'numpy', 'astropy', 'torch', 'scipy', 'sklearn', 'dotenv', 'requests', 'pydantic']
missing = []

for package in required_packages:
    try:
        if package == 'sklearn':
            __import__('sklearn')
        elif package == 'dotenv':
            __import__('dotenv')
        else:
            __import__(package)
    except ImportError:
        missing.append(package)

if missing:
    print('MISSING:' + ','.join(missing))
    sys.exit(1)
else:
    print('ALL_OK')
" 2>/dev/null

if [ $? -ne 0 ]; then
    error "Some Python dependencies are missing. Install them with:"
    echo "pip install fastapi uvicorn pandas numpy astropy torch scipy scikit-learn python-dotenv requests pydantic"
    exit 1
fi

success "All Python dependencies are available"

echo ""
success "✅ Development environment setup completed!"
echo ""
log "Next steps:"
echo ""
echo "🔧 Development Mode (choose one):"
echo ""
echo "  Option 1 - Full development with hot reload:"
echo "    Terminal 1: cd backend && python3 server.py"
echo "    Terminal 2: cd frontend && npm run dev"  
echo "    Terminal 3: npm run electron-dev"
echo ""
echo "  Option 2 - Quick start:"
echo "    npm run electron-dev"
echo ""
echo "📦 Building and Packaging:"
echo "    ./scripts/build.sh          # Complete build"
echo "    ./scripts/test-bundle.sh    # Test the build"
echo ""
echo "📚 Documentation:"
echo "    See ELECTRON_BUNDLE.md for complete documentation"
echo ""