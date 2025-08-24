#!/bin/bash
set -e

# Deploy Frontend Script for Better Impuls Viewer
# Builds the frontend and deploys to /var/www/impuls-viewer

echo "=== Better Impuls Viewer Frontend Deployment ==="

# Check if we're in the correct directory
if [ ! -f "frontend/package.json" ]; then
    echo "Error: Must run from the root of the better-impuls-viewer repository"
    exit 1
fi

# Check if Node.js dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

# Build the frontend
echo "Building frontend..."
cd frontend && npm run build && cd ..

# Check if build was successful
if [ ! -d "frontend/dist" ]; then
    echo "Error: Frontend build failed - dist directory not found"
    exit 1
fi

# Deployment target directory
TARGET_DIR="/var/www/impuls-viewer"

# Create target directory if it doesn't exist
echo "Preparing target directory: $TARGET_DIR"
sudo mkdir -p "$TARGET_DIR"

# Backup existing deployment if it exists
if [ -d "$TARGET_DIR/index.html" ] || [ -n "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]; then
    BACKUP_DIR="$TARGET_DIR.backup.$(date +%Y%m%d_%H%M%S)"
    echo "Backing up existing deployment to: $BACKUP_DIR"
    sudo cp -r "$TARGET_DIR" "$BACKUP_DIR"
fi

# Deploy new build
echo "Deploying frontend to: $TARGET_DIR"
sudo cp -r frontend/dist/* "$TARGET_DIR/"

# Set proper permissions
echo "Setting permissions..."
# sudo chown -R www-data:www-data "$TARGET_DIR"
sudo chmod -R 755 "$TARGET_DIR"

# Verify deployment
if [ -f "$TARGET_DIR/index.html" ]; then
    echo "✅ Deployment successful!"
    echo "Frontend deployed to: $TARGET_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Configure your web server (nginx/apache) to serve from $TARGET_DIR"
    echo "2. Ensure backend API is running on port 8000"
    echo "3. Update CORS settings in backend if needed"
else
    echo "❌ Deployment failed - index.html not found in target directory"
    exit 1
fi