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

## Development vs Production

### Development Mode

```bash
# Start in development mode
npm run electron-dev
```

In development:
- Frontend runs on Vite dev server (http://localhost:5173)
- Backend runs with system Python
- Hot reloading for frontend changes
- Debug tools enabled

### Production Mode

```bash
# Run the packaged application
./electron-dist/better-impuls-viewer
# or
./electron-dist/Better\ Impuls\ Viewer-1.0.0.AppImage
```

In production:
- Frontend served from built files
- Backend runs with bundled Python environment
- Optimized for performance
- Single executable file

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

4. **Large file sizes:**
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