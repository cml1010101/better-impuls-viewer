#!/usr/bin/env bash

# Setup script for Better Impuls Viewer Python dependencies
# This script should be run after installing the application

set -e

echo "Better Impuls Viewer - Python Dependencies Setup"
echo "================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH."
    echo ""
    echo "Please install Python 3.8 or newer from:"
    echo "https://www.python.org/downloads/"
    echo ""
    echo "On macOS, you can also install Python using Homebrew:"
    echo "  brew install python"
    echo ""
    echo "On Ubuntu/Debian:"
    echo "  sudo apt update && sudo apt install python3 python3-pip"
    exit 1
fi

# Determine Python command
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Found Python: $($PYTHON_CMD --version)"

# Check if pip is available
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "Error: pip is not available."
    echo ""
    echo "Please install pip:"
    echo "  python3 -m ensurepip --upgrade"
    echo ""
    echo "Or install it manually from:"
    echo "https://pip.pypa.io/en/stable/installation/"
    exit 1
fi

echo "Found pip: $($PYTHON_CMD -m pip --version)"

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APP_DIR="$SCRIPT_DIR"

# Look for requirements.txt in common locations
REQUIREMENTS_FILE=""
if [ -f "$APP_DIR/requirements.txt" ]; then
    REQUIREMENTS_FILE="$APP_DIR/requirements.txt"
elif [ -f "$APP_DIR/resources/app.asar" ]; then
    # Try to extract requirements.txt from asar if available
    if command -v npx &> /dev/null; then
        echo "Extracting requirements.txt from application bundle..."
        npx asar extract "$APP_DIR/resources/app.asar" "$APP_DIR/temp_extract" 2>/dev/null || true
        if [ -f "$APP_DIR/temp_extract/requirements.txt" ]; then
            cp "$APP_DIR/temp_extract/requirements.txt" "$APP_DIR/requirements.txt"
            REQUIREMENTS_FILE="$APP_DIR/requirements.txt"
            rm -rf "$APP_DIR/temp_extract"
        fi
    fi
fi

if [ -z "$REQUIREMENTS_FILE" ]; then
    echo "Installing common dependencies manually..."
    PACKAGES="fastapi uvicorn pandas numpy matplotlib astropy torch scipy python-dotenv scikit-learn"
    for package in $PACKAGES; do
        echo "Installing $package..."
        $PYTHON_CMD -m pip install --user "$package" || {
            echo "Warning: Failed to install $package, continuing..."
        }
    done
else
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    $PYTHON_CMD -m pip install --user -r "$REQUIREMENTS_FILE" || {
        echo "Warning: Some packages failed to install."
        echo "You may need to install them individually or check your internet connection."
    }
fi

# Verify installation
echo ""
echo "Verifying installation..."
if $PYTHON_CMD -c "import fastapi, uvicorn, pandas, numpy; print('Core dependencies verified successfully!')" 2>/dev/null; then
    echo "✓ Installation completed successfully!"
    echo ""
    echo "You can now run Better Impuls Viewer."
else
    echo "✗ Verification failed. Some dependencies may not be installed correctly."
    echo ""
    echo "You can try installing dependencies manually:"
    echo "  $PYTHON_CMD -m pip install --user fastapi uvicorn pandas numpy matplotlib astropy torch scipy python-dotenv scikit-learn"
fi

echo ""
echo "Setup complete!"