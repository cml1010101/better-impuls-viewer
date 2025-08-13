# Better Impuls Viewer Development Instructions

**Always follow these instructions first and only fall back to additional search and context gathering if the information here is incomplete or found to be in error.**

## Overview

Better Impuls Viewer is an Electron-based desktop application for astronomical data analysis with interactive visualization. It combines a React/TypeScript frontend, FastAPI/Python backend with PyTorch ML capabilities, and Electron for desktop distribution.

## Bootstrap Commands - Run These First

Always run these commands in this exact order before any development work:

```bash
# Install Python dependencies (takes ~2-3 minutes)
pip install -r requirements.txt

# Install root Node.js dependencies (takes ~1 minute)
npm install

# Install frontend dependencies (takes ~1 minute) 
cd frontend && npm install && cd ..
```

## Development Workflow

### Backend Development
```bash
# Start backend server (CRITICAL: must run from repository root)
python backend/server.py

# Backend will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

**IMPORTANT**: Always run the backend server from the repository root directory, NOT from the `backend/` directory. The server expects `sample_data/` to be in the current working directory.

### Frontend Development
```bash
# Start frontend dev server (from frontend directory)
cd frontend && npm run dev

# Frontend will be available at http://localhost:5173
# Hot reloading enabled automatically
```

### Electron Development Mode
```bash
# Start both frontend and Electron together (recommended for development)
npm run dev

# This automatically:
# 1. Starts frontend dev server on port 5173
# 2. Waits for frontend to be ready
# 3. Launches Electron pointing to dev server
```

**Note**: Electron GUI will not display in headless environments but the process will start successfully for testing purposes.

## Building and Packaging

### Frontend Build
```bash
# Build frontend for production (takes ~50 seconds)
cd frontend && npm run build

# Output: frontend/dist/ directory
```

### Backend Build
```bash
# Build backend executable with PyInstaller
# NEVER CANCEL: Takes 12-15 minutes to complete. Set timeout to 20+ minutes.
cd backend && pyinstaller --onefile server.py --name impuls_backend

# Output: backend/dist/impuls_backend (4GB+ executable including PyTorch)
```

### Complete Build
```bash
# Build both frontend and backend
# NEVER CANCEL: Takes 15-18 minutes total. Set timeout to 25+ minutes.
npm run build
```

### Electron Packaging
```bash
# Create unpacked Electron app for testing
# NEVER CANCEL: Takes 15-20 minutes. Set timeout to 30+ minutes.
npm run pack

# Create distributable packages (.AppImage, .exe, .dmg)
# NEVER CANCEL: Takes 20-25 minutes. Set timeout to 35+ minutes.
npm run dist
```

## Testing

### Backend Integration Test
```bash
# Test backend functionality without GUI (takes ~15 seconds)
npm run test:backend
```

### Python Test Suite
```bash
# Test CSV functionality (takes ~5 seconds)
python test_csv_functionality.py

# Test star range parsing (takes <1 second)
python test_star_ranges.py

# Note: Some ML training tests may fail due to API changes - this is expected
```

### Frontend Linting
```bash
# Lint frontend code (takes ~10 seconds) - always run before committing
cd frontend && npm run lint
```

## API Testing

The backend provides these key endpoints (server must be running from repository root):

```bash
# List available stars (returns [1,2,3,...,14])
curl http://localhost:8000/stars

# List telescopes for star 1 (returns ["hubble","kepler","tess"])
curl http://localhost:8000/telescopes/1

# Check ML model status
curl http://localhost:8000/model_status

# View API documentation
# Open http://localhost:8000/docs in browser
```

## Validation Workflows

### Complete Development Validation
After making any changes, always run this complete validation sequence:

```bash
# 1. Lint frontend code
cd frontend && npm run lint && cd ..

# 2. Test backend integration
npm run test:backend

# 3. Build frontend
cd frontend && npm run build && cd ..

# 4. Start backend and test API
python backend/server.py &
sleep 5
curl http://localhost:8000/stars
curl http://localhost:8000/telescopes/1
pkill -f "python backend/server.py"
```

### Manual UI Testing Scenarios
When testing the application UI, always verify these core workflows:

1. **Star Selection**: Select different stars (1-14) and verify data loads
2. **Telescope Selection**: Switch between Hubble, Kepler, and TESS data
3. **Light Curve Display**: Verify time-series data visualization appears
4. **Periodogram Analysis**: Test automatic period detection functionality
5. **Phase Folding**: Enter a period value and verify phase-folded curve generation

## Project Structure

```
better-impuls-viewer/
├── main.js                  # Electron main process
├── dev.js                   # Development launcher script
├── package.json            # Root dependencies and scripts
├── requirements.txt        # Python dependencies
├── backend/
│   ├── server.py            # FastAPI application (start from root!)
│   ├── config.py           # Configuration settings
│   ├── models.py           # Pydantic data models
│   ├── data_processing.py  # Data analysis functions
│   ├── period_detection.py # ML period detection
│   └── model_training.py   # PyTorch CNN training
├── frontend/
│   ├── src/                # React/TypeScript source
│   ├── dist/               # Built frontend (generated)
│   └── package.json       # Frontend dependencies
├── sample_data/            # Astronomical datasets (stars 1-14)
└── .github/
    └── copilot-instructions.md  # This file
```

## Common Issues and Solutions

### "Data file not found" errors
- **Cause**: Backend server started from wrong directory
- **Solution**: Always run `python backend/server.py` from repository root, not from `backend/` directory

### Empty API responses
- **Cause**: Backend looking for data in wrong location
- **Solution**: Ensure `sample_data/` directory exists in current working directory when starting backend

### Frontend build warnings about chunk size
- **Expected behavior**: Plotly.js creates large bundles, warnings are normal

### PyInstaller build warnings
- **Expected behavior**: CUDA/GPU library warnings are normal in CPU-only environments

### Electron display issues in containers
- **Expected behavior**: Electron GUI cannot display in headless environments but the process starts successfully

## CI/CD Integration

The GitHub Actions workflow (`.github/workflows/release.yml`) builds the application for Linux, Windows, and macOS. Build times per platform:

- **Linux**: 20-25 minutes
- **Windows**: 25-30 minutes  
- **macOS**: 25-30 minutes

Always ensure your changes pass frontend linting before pushing to avoid CI failures.

## Performance Notes

- **Frontend build**: ~50 seconds
- **Backend build**: 12-15 minutes (PyTorch compilation)
- **Full Electron packaging**: 15-25 minutes
- **Python dependency installation**: 2-3 minutes
- **Node.js dependency installation**: 1-2 minutes

## Development Environment Requirements

- **Python**: 3.8+ (tested with 3.12)
- **Node.js**: 16+ (tested with 20.19)
- **Memory**: 8GB+ recommended for backend builds
- **Disk Space**: 10GB+ for complete build artifacts

## Machine Learning Features

The application includes a PyTorch CNN for automatic period detection and astronomical object classification. ML model training requires CSV data and can take 10-30 minutes depending on dataset size.

## Key Dependencies

- **Backend**: FastAPI, PyTorch, pandas, astropy, scikit-learn, uvicorn
- **Frontend**: React 19, TypeScript, Vite, Plotly.js
- **Desktop**: Electron 28, electron-builder
- **Development**: ESLint, PyInstaller

Always run the complete validation workflow after making changes to ensure all components work together correctly.