# Better Impuls Viewer - Dependency Setup Guide

If you see an error about "Missing Dependencies" when running Better Impuls Viewer, this means that Python and/or the required Python packages are not installed on your system.

## Quick Setup

### Windows
1. Navigate to the Better Impuls Viewer installation folder
2. Right-click on `setup-dependencies.ps1` 
3. Select "Run with PowerShell"
4. Follow the on-screen instructions
5. Restart Better Impuls Viewer

### macOS and Linux
1. Open Terminal
2. Navigate to the Better Impuls Viewer installation folder
3. Run: `./setup-dependencies.sh`
4. Follow the on-screen instructions
5. Restart Better Impuls Viewer

## Manual Setup

If the automatic setup scripts don't work, you can install the dependencies manually:

### 1. Install Python
- Download and install Python 3.8 or newer from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation (Windows)

### 2. Install Required Packages
Open a terminal/command prompt and run:

```bash
pip install fastapi uvicorn pandas numpy matplotlib astropy torch scipy python-dotenv scikit-learn
```

Or if you have access to the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Verify Installation
You can verify the installation by running:

```bash
python -c "import fastapi, uvicorn, pandas, numpy; print('Dependencies installed successfully!')"
```

## Troubleshooting

### Python Not Found
If you get a "Python not found" error:
- On Windows: Try using `py` instead of `python`
- On macOS/Linux: Try using `python3` instead of `python`
- Make sure Python is added to your system PATH

### Package Installation Fails
If package installation fails:
- Try using `pip install --user` to install for the current user only
- Check your internet connection
- On Windows, try running Command Prompt as Administrator
- On macOS/Linux, you might need to install packages system-wide with `sudo`

### Still Having Issues?
1. Restart your computer after installing Python
2. Try uninstalling and reinstalling Python
3. Contact support with the specific error message you're seeing

## Development Note
This app requires Python because it includes a FastAPI backend for astronomical data processing. The Python dependencies handle complex mathematical operations, data visualization, and machine learning tasks that are essential for the application's functionality.