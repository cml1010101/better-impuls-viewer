#!/bin/bash

# Better Impuls Viewer - Production Build Script
# This script creates production builds for different platforms

echo "🏗️  Building Better Impuls Viewer for production..."

# Function to show usage
show_usage() {
    echo "Usage: $0 [platform]"
    echo "Platforms:"
    echo "  linux    - Build for Linux (AppImage)"
    echo "  windows  - Build for Windows (NSIS installer)"
    echo "  mac      - Build for macOS (DMG)"
    echo "  all      - Build for all platforms"
    echo "  pack     - Build unpacked version for testing"
    echo ""
    echo "Example: $0 linux"
}

# Check if platform is provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

PLATFORM=$1

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed or not in PATH"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing root dependencies..."
    npm install
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

# Build the application
echo "🔨 Building frontend..."
npm run build-frontend

echo "🔨 Building backend..."
npm run build-backend

# Create production builds based on platform
case $PLATFORM in
    linux)
        echo "🐧 Building for Linux..."
        npx electron-builder --linux
        ;;
    windows)
        echo "🪟 Building for Windows..."
        npx electron-builder --win
        ;;
    mac)
        echo "🍎 Building for macOS..."
        npx electron-builder --mac
        ;;
    all)
        echo "🌍 Building for all platforms..."
        npm run dist-all
        ;;
    pack)
        echo "📦 Creating unpacked build for testing..."
        npm run pack
        ;;
    *)
        echo "❌ Unknown platform: $PLATFORM"
        show_usage
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo "✅ Build completed successfully!"
    echo "📁 Output directory: dist-electron/"
    ls -la dist-electron/
else
    echo "❌ Build failed!"
    exit 1
fi