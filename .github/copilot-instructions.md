# Better Impuls Viewer Development Instructions

**Always follow these instructions first and only fall back to additional search and context gathering if the information here is incomplete or found to be in error.**

## Overview

Better Impuls Viewer is a modern web application for astronomical data analysis with interactive visualization. It combines a React/TypeScript frontend, FastAPI/Python backend with PyTorch ML capabilities, and can be deployed to any web hosting service.

## Bootstrap Commands - Run These First

Always run these commands in this exact order before any development work:

```bash
# Install Python dependencies (takes ~2-3 minutes)
pip install -r backend/requirements.txt

# Install frontend dependencies (takes ~1 minute) 
cd frontend && npm install && cd ..
```

## Development Workflow

### Backend Development
```bash
# Start backend server (CRITICAL: must run from repository root)
python backend/app.py

# Backend will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

**IMPORTANT**: Always run the backend server from the repository root directory, NOT from the `backend/` directory. The server expects `impuls-data/` to be in the current working directory.

### Frontend Development
```bash
# Start frontend dev server (from frontend directory)
cd frontend && npm run dev

# Frontend will be available at http://localhost:5173
# Hot reloading enabled automatically
```

### Development Mode (Frontend Only)
The project currently only has the frontend and backend as separate components. There are no root-level npm scripts or development utilities like concurrent runners.

To develop the application:
1. Start the backend: `python backend/app.py`
2. In a separate terminal, start the frontend: `cd frontend && npm run dev`

## Building and Deployment

### Frontend Build
```bash
# Build frontend for production (takes ~50 seconds)
cd frontend && npm run build

# Output: frontend/dist/ directory
```

### Frontend Preview
```bash
# Serve the production build locally
cd frontend && npm run preview
```

**Note**: The project currently does not have root-level build scripts or deployment automation.

## Testing

### Frontend Linting
```bash
# Lint frontend code (takes ~10 seconds) - always run before committing
cd frontend && npm run lint
```

**Note**: The project currently does not have automated test suites or test scripts. All testing must be done manually by running the application and testing the functionality.

## API Testing

The backend provides these key endpoints (server must be running from repository root):

```bash
# Test basic API connectivity
curl http://localhost:8000/api/

# List available stars (returns star information)
curl http://localhost:8000/api/stars

# Get star information
curl http://localhost:8000/api/star/1

# List surveys for star 1 (returns available survey names)
curl http://localhost:8000/api/star/1/surveys

# View API documentation
# Open http://localhost:8000/docs in browser
```

**Note**: All API endpoints are prefixed with `/api/`. The actual data returned will depend on the ML model training status and available data files.

## Validation Workflows

### Complete Development Validation
After making any changes, always run this validation sequence:

```bash
# 1. Lint frontend code
cd frontend && npm run lint && cd ..

# 2. Start backend and test API (in background)
python backend/app.py &
sleep 5

# 3. Test basic API endpoints
curl http://localhost:8000/api/
curl http://localhost:8000/api/stars

# 4. Stop the backend server
pkill -f "python backend/app.py"

# 5. Build frontend
cd frontend && npm run build && cd ..
```

**Note**: Some endpoints may fail if ML models are not trained or if data parsing issues exist. Check the console output for specific errors.

### Manual UI Testing Scenarios
When testing the application UI, always verify these core workflows:

1. **Star Selection**: Select different stars (1-14) and verify data loads
2. **Survey Selection**: Switch between available survey types (hubble, kepler, tess)
3. **Light Curve Display**: Verify time-series data visualization appears
4. **Periodogram Analysis**: Test automatic period detection functionality
5. **Phase Folding**: Enter a period value and verify phase-folded curve generation

**Note**: The exact number of available stars and surveys depends on the data files present in the `impuls-data/` directory.

## Project Structure

```
better-impuls-viewer/
├── .github/
│   ├── copilot-instructions.md  # This file
│   └── workflows/              # GitHub Actions workflows
├── .gitignore                  # Git ignore rules
├── README.md                  # Project documentation
├── backend/                   # Python FastAPI backend
│   ├── app.py                # Main FastAPI application (start from root!)
│   ├── config.py             # Configuration settings
│   ├── models.py             # Pydantic data models
│   ├── database.py           # Star database management
│   ├── data_processing.py    # Data analysis functions
│   ├── periodizer.py         # Period detection algorithms
│   ├── train.py              # ML model training
│   ├── eval.py               # Model evaluation
│   └── requirements.txt      # Python dependencies
├── frontend/                 # React/TypeScript frontend
│   ├── src/                  # React/TypeScript source
│   ├── dist/                 # Built frontend (generated)
│   ├── package.json          # Frontend dependencies and scripts
│   ├── vite.config.ts        # Vite configuration
│   └── eslint.config.js      # ESLint configuration
├── impuls-data/              # Astronomical datasets (stars 1-14)
│   ├── impuls_stars.csv      # Star catalog metadata
│   └── *.tbl                 # Time-series data files
└── sample_data/              # Additional sample datasets
```

## Common Issues and Solutions

### "Data file not found" errors
- **Cause**: Backend server started from wrong directory
- **Solution**: Always run `python backend/app.py` from repository root, not from `backend/` directory

### Empty API responses
- **Cause**: Backend looking for data in wrong location
- **Solution**: Ensure `impuls-data/` directory exists in current working directory when starting backend

### Coordinate parsing errors
- **Cause**: Mismatch between coordinate format in CSV and parsing logic
- **Solution**: Check the format of coordinates in `impuls-data/impuls_stars.csv` matches the expected format in `database.py`

### Frontend build warnings about chunk size
- **Expected behavior**: Plotly.js creates large bundles, warnings are normal

### Missing ML model errors
- **Cause**: PyTorch models haven't been trained yet
- **Solution**: Train models using the training scripts or accept that ML features may not work initially

### Node.js/Python version issues
- **Solution**: Ensure Python 3.8+ and Node.js 16+ are installed

## CI/CD Integration

The application can be deployed to various web hosting platforms including Netlify, Vercel, GitHub Pages, and traditional web servers. The frontend build output in `frontend/dist/` contains the static files that can be deployed to any web hosting service.

**Note**: Currently no automated deployment scripts exist. Deployment must be done manually by building the frontend and uploading the files.

## Performance Notes

- **Frontend build**: ~50 seconds
- **Backend startup**: ~5-10 seconds (may fail if data format issues exist)
- **Python dependency installation**: 2-3 minutes
- **Node.js dependency installation**: 1-2 minutes

## Development Environment Requirements

- **Python**: 3.8+ (tested with 3.12)
- **Node.js**: 16+ (tested with 20.19)
- **Memory**: 4GB+ recommended for development
- **Disk Space**: 2GB+ for dependencies and builds

## Machine Learning Features

The application includes a PyTorch CNN for automatic period detection and astronomical object classification. ML model training requires CSV data and can take 10-30 minutes depending on dataset size.

**Note**: ML features may not work initially until models are properly trained and data format issues are resolved.

## Key Dependencies

- **Backend**: FastAPI, PyTorch, pandas, astropy, scikit-learn, uvicorn, lightkurve
- **Frontend**: React 19, TypeScript, Vite, Plotly.js
- **Development**: ESLint, Node.js tooling

Always run the complete validation workflow after making changes to ensure all components work together correctly.

## Getting Started Checklist

For new developers working on this project:

1. **Clone the repository**
2. **Install Python dependencies**: `pip install -r backend/requirements.txt`
3. **Install Node dependencies**: `cd frontend && npm install && cd ..`
4. **Check data files**: Ensure `impuls-data/impuls_stars.csv` exists and has correct format
5. **Test backend startup**: `python backend/app.py` (expect coordinate parsing issues initially)
6. **Test frontend**: `cd frontend && npm run dev`
7. **Lint frontend**: `cd frontend && npm run lint`
8. **Build frontend**: `cd frontend && npm run build`

The application may require debugging of data parsing and ML model training before full functionality is available.