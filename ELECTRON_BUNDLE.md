# Electron Bundle for Better Impuls Viewer

This document describes the Electron application bundle that packages both the React frontend and FastAPI backend into a single desktop application.

## Overview

The Better Impuls Viewer Electron app provides:
- Single-file desktop application (.AppImage for Linux)
- Bundled Python backend with all dependencies
- React frontend with data visualization capabilities
- Sample astronomical data included
- No need for manual backend/frontend setup

## Architecture

```
better-impuls-viewer/
├── electron/
│   └── main.js                 # Electron main process
├── frontend/                   # React frontend
│   ├── src/
│   ├── dist/                   # Built frontend (created during build)
│   └── package.json
├── backend/                    # FastAPI backend
│   ├── server.py               # Main backend server
│   ├── process.py              # Data processing functions
│   └── display.py              # Display utilities
├── python-env/                 # Bundled Python environment (created during build)
├── sample_data/                # Sample astronomical data
├── scripts/
│   ├── build.sh                # Build script
│   └── test-bundle.sh          # Test script
├── package.json                # Electron configuration
├── requirements.txt            # Python dependencies
└── electron-dist/              # Final packaged application (created during build)
```

## Build Process

### Prerequisites

1. **Node.js** (v18+ recommended)
2. **Python 3.8+**
3. **npm** or **yarn**

### Building the Application

1. **Clone and navigate to the repository:**
   ```bash
   git clone <repository>
   cd better-impuls-viewer
   ```

2. **Run the build script:**
   ```bash
   ./scripts/build.sh
   ```

   This script will:
   - Clean previous builds
   - Install Electron dependencies
   - Build the React frontend
   - Create a Python virtual environment
   - Install Python dependencies
   - Package the Electron application

3. **Test the bundle:**
   ```bash
   ./scripts/test-bundle.sh
   ```

### Manual Build Steps

If you prefer to build manually:

```bash
# 1. Install Electron dependencies
npm install

# 2. Build frontend
cd frontend
npm install
npm run build
cd ..

# 3. Create Python environment
python3 -m venv python-env
source python-env/bin/activate
pip install -r requirements.txt
deactivate

# 4. Package with Electron
npm run package
```

## Application Components

### Electron Main Process (`electron/main.js`)

The main process:
- Starts the Python backend server
- Creates the application window
- Manages the lifecycle of both frontend and backend
- Handles graceful shutdown

### Backend Integration

The backend (`backend/server.py`) is automatically started by Electron and provides:
- REST API for data access
- Data processing capabilities
- File management for astronomical data
- Caching for performance

### Frontend Integration

The React frontend (`frontend/src/`) communicates with the backend via HTTP and includes:
- Interactive data visualization
- Campaign selection interface
- Periodogram analysis
- Phase-folding capabilities

## Configuration

### Environment Variables

The application respects these environment variables:

- `ELECTRON_IS_DEV`: Set to "1" for development mode
- `DATA_FOLDER`: Override default data folder path
- `PYTHONPATH`: Additional Python paths

### Data Sources

By default, the application looks for data in:
1. `~/Documents/impuls-data` (user's documents)
2. `./sample_data` (bundled sample data)
3. Environment variable `DATA_FOLDER`

## Development Workflow

### Setting Up Development Environment

1. **Install Prerequisites:**
   ```bash
   # Node.js 18+ and npm
   node --version  # Should be 18+
   npm --version
   
   # Python 3.8+
   python3 --version  # Should be 3.8+
   
   # Install Python dependencies for backend development
   pip install fastapi uvicorn pandas numpy astropy torch scipy scikit-learn python-dotenv requests pydantic
   ```

2. **Clone and Install Dependencies:**
   ```bash
   git clone <repository>
   cd better-impuls-viewer
   
   # Install Electron dependencies
   npm install
   
   # Install frontend dependencies
   cd frontend
   npm install
   cd ..
   ```

### Running in Development Mode

#### Option 1: One-Command Development (Recommended)

Start everything automatically with the integrated development script:

```bash
# Start all services in development mode with one command
npm run dev
```

This automatically:
- ✅ **Checks if frontend dev server is running**, starts it if needed
- ✅ **Waits for frontend to be ready** before launching Electron
- ✅ **Starts Electron** which automatically starts the Python backend
- ✅ **Provides hot reloading** for React frontend changes
- ✅ **Opens Chrome DevTools** for debugging
- ✅ **Manages all processes** and cleanup on exit

#### Option 2: Manual Multi-Terminal Setup

For advanced debugging, start components separately:

```bash
# Terminal 1: Start the backend server
cd backend
python3 server.py

# Terminal 2: Start the frontend dev server
cd frontend
npm run dev

# Terminal 3: Start Electron in development mode
npm run dev-simple
```

This setup provides:
- **Separate process control** for each component
- **Direct backend debugging** with Python debugger access
- **Independent process management**

#### Option 3: Simple Electron Only

If frontend dev server is already running:

```bash
# Start Electron only (expects frontend on port 5173)
npm run dev-simple
```

### Development Features

**In development mode, the Electron app:**
- ✅ Uses system Python (no virtual environment needed)
- ✅ Loads frontend from Vite dev server with hot reload
- ✅ Automatically opens Chrome DevTools for debugging
- ✅ Shows detailed console logs for both frontend and backend
- ✅ Allows real-time code changes without restart
- ✅ Supports breakpoints and debugging in both React and Python code

**Environment Detection:**
```javascript
// The app detects development mode via:
const isDev = process.env.ELECTRON_IS_DEV === '1';
```

### Development Scripts

```bash
# Available npm scripts for development:
npm run dev                    # 🚀 One-command development (starts everything)
npm run dev-simple             # Start Electron only (expects frontend running)
npm run electron-dev           # Raw Electron dev mode (same as dev-simple)
npm run start-frontend         # Start frontend dev server only
npm run start-backend          # Start backend server only
npm run build-frontend         # Build frontend only
npm run build-backend          # Set up Python virtual environment
npm run build                  # Build both frontend and backend
npm run clean                  # Clean all build artifacts
```

**Recommended workflow:**
1. **First time setup**: `./scripts/dev-setup.sh`
2. **Daily development**: `npm run dev`
3. **Building for production**: `./scripts/build.sh`

### Debugging

#### Frontend Debugging
- **DevTools**: Automatically opened in development mode
- **React DevTools**: Install browser extension for component debugging
- **Console Logs**: Check browser console for frontend errors
- **Network Tab**: Monitor API calls to backend

#### Backend Debugging
- **Python Console**: Direct access to Python debugger (pdb, ipdb)
- **API Testing**: Use curl or Postman to test endpoints directly
- **Logs**: Backend logs appear in terminal where you started server.py

```bash
# Example: Test backend API directly
curl http://localhost:8000/stars
curl http://localhost:8000/telescopes/1
curl http://localhost:8000/data/1/Hubble/1
```

#### Electron Main Process Debugging
- **Console Logs**: Check terminal where electron was started
- **Process Monitoring**: Backend process logs appear in Electron console

### Development vs Production

| Feature | Development Mode | Production Mode |
|---------|------------------|-----------------|
| Frontend Source | Vite dev server (http://localhost:5173) | Built files in `frontend/dist/` |
| Backend Python | System Python | Bundled virtual environment |
| Hot Reloading | ✅ Enabled | ❌ Disabled |
| DevTools | ✅ Auto-opened | ❌ Disabled |
| File Watching | ✅ Enabled | ❌ Disabled |
| Optimization | ❌ Development builds | ✅ Production optimized |
| Startup Time | ⚡ Fast (no build) | 🐌 Slower (bundle loading) |
| File Size | 📦 Minimal | 📦 ~100-300MB bundle |

## Production Packaging

### Quick Packaging

For a complete build and package:

```bash
# One-command build and package
./scripts/build.sh

# Test the generated package
./scripts/test-bundle.sh
```

### Manual Packaging Steps

#### Step 1: Build Components

```bash
# 1. Clean previous builds
npm run clean

# 2. Install dependencies
npm install
cd frontend && npm install && cd ..

# 3. Build frontend
npm run build-frontend

# 4. Set up Python environment
npm run build-backend

# 5. Create Electron package
npm run package
```

#### Step 2: Verify the Build

```bash
# Check build artifacts
ls -la electron-dist/

# Test the application
./scripts/test-bundle.sh
```

### Packaging Options

#### Available Build Targets

```bash
# Build for current platform (auto-detected)
npm run package

# Build distributable package
npm run dist

# Clean and rebuild everything
npm run clean && npm run dist
```

#### Build Configuration

The electron-builder configuration in `package.json`:

```json
{
  "build": {
    "appId": "com.better-impuls-viewer.app",
    "productName": "Better Impuls Viewer",
    "directories": {
      "output": "electron-dist"
    },
    "files": [
      "electron/**/*",
      "frontend/dist/**/*", 
      "backend/**/*",
      "sample_data/**/*",
      "python-env/**/*"
    ],
    "linux": {
      "target": "AppImage",
      "category": "Science"
    }
  }
}
```

#### Cross-Platform Building

```bash
# Build for specific platforms (requires platform-specific dependencies)
npx electron-builder --linux
npx electron-builder --win
npx electron-builder --mac

# Build for multiple platforms
npx electron-builder --linux --win --mac
```

### Generated Package Structure

```
electron-dist/
├── Better Impuls Viewer-1.0.0.AppImage  # Linux AppImage (self-contained)
├── linux-unpacked/                      # Unpacked Linux build
│   ├── better-impuls-viewer             # Executable
│   ├── resources/                       # App resources
│   │   ├── app.asar                    # Packaged application
│   │   ├── backend/                    # Python backend
│   │   ├── frontend/dist/              # Built React app
│   │   ├── python-env/                 # Virtual environment
│   │   └── sample_data/                # Sample data
│   └── ...
└── latest-linux.yml                     # Update metadata
```

### Package Testing and Validation

#### Automated Testing

```bash
# Run comprehensive test suite
./scripts/test-bundle.sh
```

This tests:
- ✅ Python environment integrity
- ✅ Frontend build completeness  
- ✅ Backend startup functionality
- ✅ Sample data availability
- ✅ Executable permissions
- ✅ File sizes and dependencies

#### Manual Testing

```bash
# Run the packaged application
./electron-dist/Better\ Impuls\ Viewer-1.0.0.AppImage

# Or run unpacked version
./electron-dist/linux-unpacked/better-impuls-viewer
```

**Test checklist:**
- [ ] Application starts without errors
- [ ] Backend server starts automatically  
- [ ] Frontend loads and displays correctly
- [ ] All API endpoints respond correctly
- [ ] Data visualization works
- [ ] Settings and authentication flows work
- [ ] Application closes cleanly

### Distribution

#### AppImage Distribution

The generated `.AppImage` file:
- **Self-contained**: No installation required
- **Portable**: Run from anywhere  
- **Cross-distribution**: Works on most Linux distributions
- **Single file**: Easy to distribute and download

```bash
# Make executable and run
chmod +x "Better Impuls Viewer-1.0.0.AppImage"
./"Better Impuls Viewer-1.0.0.AppImage"
```

#### Size Optimization

To reduce package size:

```bash
# Check package contents and sizes
du -sh electron-dist/
du -sh python-env/
du -sh frontend/dist/

# Optimize Python environment
pip install --no-cache-dir <packages>
pip uninstall <unused-packages>

# Optimize frontend bundle
cd frontend && npm run build -- --minify
```

### Troubleshooting Packaging

#### Common Issues

1. **Large Package Size (>500MB)**
   ```bash
   # Check what's taking space
   du -sh python-env/* | sort -h
   
   # Remove unnecessary packages
   pip uninstall <package-name>
   ```

2. **Missing Python Dependencies**
   ```bash
   # Test Python environment
   python-env/bin/python -c "import fastapi, pandas, numpy"
   
   # Reinstall if needed
   rm -rf python-env
   npm run build-backend
   ```

3. **Frontend Build Failures**
   ```bash
   # Clean and rebuild frontend
   cd frontend
   rm -rf dist node_modules
   npm install
   npm run build
   ```

4. **Electron Packaging Errors**
   ```bash
   # Clean electron cache
   npx electron-builder install-app-deps
   npm run clean
   npm run package
   ```

#### Debug Mode for Packaging

```bash
# Enable verbose electron-builder output
DEBUG=electron-builder npm run package

# Test with development environment
ELECTRON_IS_DEV=1 ./electron-dist/better-impuls-viewer
```

### CI/CD and Automated Builds

The repository includes GitHub Actions workflow for automated building:

```yaml
# .github/workflows/release.yml
# Automatically builds and publishes on GitHub releases
```

For local CI-like builds:
```bash
# Simulate CI environment
npm ci  # Clean install
npm run clean
npm run dist
./scripts/test-bundle.sh
```

## Packaging Details

### Electron Builder Configuration

The `package.json` includes electron-builder configuration:

```json
{
  "build": {
    "appId": "com.better-impuls-viewer.app",
    "productName": "Better Impuls Viewer",
    "directories": {
      "output": "electron-dist"
    },
    "files": [
      "electron/**/*",
      "frontend/dist/**/*",
      "backend/**/*",
      "sample_data/**/*",
      "python-env/**/*"
    ],
    "linux": {
      "target": "AppImage",
      "category": "Science"
    }
  }
}
```

### File Inclusion

The packaged application includes:
- Electron main process and configuration
- Built React frontend
- Python backend source code
- Complete Python virtual environment
- Sample astronomical data
- All necessary dependencies

## Troubleshooting

### Common Issues

1. **Python dependencies missing:**
   - Ensure all packages in `requirements.txt` are installable
   - Check Python version compatibility

2. **Backend fails to start:**
   - Verify Python environment is properly created
   - Check for port conflicts (default: 8000)
   - Review backend logs in console

3. **Frontend not loading:**
   - Ensure frontend build completed successfully
   - Check for JavaScript errors in DevTools

4. **DevTools Autofill errors (harmless):**
   - Errors like "Autofill.enable wasn't found" are suppressed automatically
   - These are Chrome DevTools features not available in Electron
   - No impact on application functionality
   - Fixed in latest version to prevent any restart loops

5. **Application restart loop:**
   - If the app keeps restarting, check for JavaScript errors in main process
   - Ensure all Electron event handlers are properly implemented
   - Latest version includes fixes for console message handling

6. **Large file sizes:**
   - Python environment can be 50-100MB
   - Consider excluding unused Python packages
   - Use `.gitignore` to exclude build artifacts

### Debug Mode

Enable debug output:
```bash
ELECTRON_IS_DEV=1 ./electron-dist/better-impuls-viewer
```

### Log Files

Application logs are printed to:
- Console (development mode)
- System console/terminal (production mode)

## Security Considerations

The Electron application:
- Disables Node.js integration in renderer
- Enables context isolation
- Uses secure defaults for web security
- Validates all external URLs

## Performance Optimization

- Backend uses caching for expensive operations
- Frontend implements efficient React patterns
- Python environment includes only necessary packages
- Data processing is optimized for large datasets

## Distribution

### AppImage (Linux)

The generated `.AppImage` file:
- Is self-contained and portable
- Requires no installation
- Can be run from any location
- Includes all dependencies

### Future Platforms

The configuration supports building for:
- Windows (`.exe` with installer)
- macOS (`.dmg` or `.app`)
- Additional Linux formats (`.deb`, `.rpm`)

### Automated Release Distribution

The repository includes a GitHub Actions workflow that automatically builds and distributes the application when a new release is created. See [RELEASE_WORKFLOW.md](RELEASE_WORKFLOW.md) for details on:

- Creating releases that trigger automatic builds
- Downloading pre-built AppImage files
- Verifying download integrity with checksums
- Manual workflow testing

## File Size Management

To keep the distribution size reasonable:

1. **Exclude unnecessary files in `.gitignore`:**
   ```gitignore
   node_modules/
   dist/
   electron-dist/
   python-env/
   *.AppImage
   ```

2. **Optimize Python environment:**
   - Use minimal Python installation
   - Exclude development packages
   - Consider using PyInstaller for Python bundling

3. **Frontend optimization:**
   - Use production builds
   - Enable tree shaking
   - Compress static assets

## API Reference

The bundled backend provides the same API as the standalone version:

- `GET /stars` - List available stars
- `GET /telescopes/{star_number}` - Get telescopes for star
- `GET /campaigns/{star_number}/{telescope}` - Get campaigns
- `GET /data/{star_number}/{telescope}/{campaign_id}` - Get light curve data
- `GET /periodogram/{star_number}/{telescope}/{campaign_id}` - Get periodogram
- `GET /phase_fold/{star_number}/{telescope}/{campaign_id}?period=X` - Get phase-folded data

All endpoints are accessible at `http://localhost:8000` when the application is running.