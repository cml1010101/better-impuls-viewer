# Better Impuls Viewer - Electron Desktop App

This directory contains the Electron desktop application that packages both the frontend and backend into a single executable.

## Architecture

The application consists of three main components:

1. **Frontend**: React + TypeScript application built with Vite (in `frontend/`)
2. **Backend**: Python FastAPI server (in `backend/`)
3. **Electron**: Desktop app wrapper (in `electron/`)

## Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- Git

## Quick Start

### Development Mode

1. Install dependencies:
```bash
npm install
cd frontend && npm install && cd ..
pip3 install -r requirements.txt
```

2. Start development mode:
```bash
./dev-start.sh
```
or manually:
```bash
npm run electron-dev
```

This will start:
- Backend API server on http://localhost:8000
- Frontend dev server on http://localhost:5173
- Electron app window

### Production Build

1. Build for testing (unpacked):
```bash
./build.sh pack
```

2. Build for specific platform:
```bash
./build.sh linux    # Creates AppImage for Linux
./build.sh windows  # Creates NSIS installer for Windows
./build.sh mac      # Creates DMG for macOS
```

3. Build for all platforms:
```bash
./build.sh all
```

## Available Scripts

From the root directory:

- `npm run electron` - Start Electron app (production mode)
- `npm run electron-dev` - Start in development mode with hot reload
- `npm run build` - Build both frontend and backend
- `npm run build-frontend` - Build only frontend
- `npm run build-backend` - Build only backend
- `npm run pack` - Create unpacked Electron build for testing
- `npm run dist` - Create distribution package for current platform
- `npm run dist-all` - Create distribution packages for all platforms

## Project Structure

```
better-impuls-viewer/
├── electron/              # Electron main process
│   ├── main.js            # Main Electron process
│   └── preload.js         # Preload script for security
├── frontend/              # React frontend
│   ├── src/               # Source code
│   ├── dist/              # Built frontend (after build)
│   └── package.json       # Frontend dependencies
├── backend/               # Python backend
│   ├── server.py          # FastAPI server
│   ├── models.py          # Data models
│   └── ...                # Other backend modules
├── package.json           # Root Electron package
├── requirements.txt       # Python dependencies
├── dev-start.sh          # Development launcher script
└── build.sh              # Production build script
```

## How It Works

1. **Electron Main Process** (`electron/main.js`):
   - Starts the Python backend as a child process
   - Creates the main application window
   - Loads the frontend (dev server in development, built files in production)
   - Handles app lifecycle and cleanup

2. **Backend Integration**:
   - Python FastAPI server runs as a subprocess
   - Communicates via HTTP API calls from frontend
   - Automatically starts when Electron app launches
   - Properly terminated when app closes

3. **Frontend Integration**:
   - React app builds to static files for production
   - Communicates with backend via fetch/axios calls
   - Served from Electron in production mode

## Development Tips

1. **Hot Reload**: In development mode, the frontend supports hot reload while the backend runs separately
2. **Debugging**: Use Chrome DevTools for frontend debugging (automatically opens in dev mode)
3. **Backend Logs**: Check console output for backend logs and errors
4. **Port Conflicts**: Ensure ports 8000 (backend) and 5173 (frontend dev) are available

## Building for Distribution

The build process:

1. Builds the frontend to static files (`frontend/dist/`)
2. Packages Python backend files as extraResources
3. Creates platform-specific installers/packages:
   - **Linux**: AppImage (portable executable)
   - **Windows**: NSIS installer (.exe)
   - **macOS**: DMG disk image

## Configuration

Key configuration files:

- `package.json` - Electron build configuration and scripts
- `electron/main.js` - Main process configuration
- `frontend/vite.config.ts` - Frontend build configuration
- `backend/config.py` - Backend configuration

## Troubleshooting

1. **Backend won't start**: Check Python installation and dependencies
2. **Frontend build errors**: Check Node.js version and dependencies
3. **Electron crashes**: Check console output and ensure all dependencies are installed
4. **Port conflicts**: Change ports in configuration files if needed

## Dependencies

The application uses:

- **Electron**: Desktop app framework
- **React + TypeScript**: Frontend framework
- **Vite**: Frontend build tool
- **FastAPI**: Backend API framework
- **Plotly.js**: Data visualization

For a complete list, see `package.json` and `requirements.txt`.