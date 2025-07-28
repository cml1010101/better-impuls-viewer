# Setup script for Better Impuls Viewer Python dependencies
# This script should be run after installing the application

Write-Host "Better Impuls Viewer - Python Dependencies Setup" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Check if Python is available
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
}

if (-not $pythonCmd) {
    Write-Host "Error: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.8 or newer from:"
    Write-Host "https://www.python.org/downloads/"
    Write-Host ""
    Write-Host "Make sure to check 'Add Python to PATH' during installation."
    exit 1
}

$pythonVersion = & $pythonCmd --version
Write-Host "Found Python: $pythonVersion" -ForegroundColor Green

# Check if pip is available
try {
    $pipVersion = & $pythonCmd -m pip --version
    Write-Host "Found pip: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: pip is not available." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install pip:"
    Write-Host "  $pythonCmd -m ensurepip --upgrade"
    Write-Host ""
    Write-Host "Or download and install from:"
    Write-Host "https://pip.pypa.io/en/stable/installation/"
    exit 1
}

# Get the directory containing this script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appDir = $scriptDir

# Look for requirements.txt
$requirementsFile = $null
$requirementsPath = Join-Path $appDir "requirements.txt"
if (Test-Path $requirementsPath) {
    $requirementsFile = $requirementsPath
    Write-Host "Found requirements file: $requirementsFile" -ForegroundColor Green
} else {
    Write-Host "requirements.txt not found, will install common dependencies manually" -ForegroundColor Yellow
}

if ($requirementsFile) {
    Write-Host "Installing dependencies from $requirementsFile..." -ForegroundColor Green
    try {
        & $pythonCmd -m pip install --user -r $requirementsFile
    } catch {
        Write-Host "Warning: Some packages failed to install." -ForegroundColor Yellow
        Write-Host "You may need to install them individually or check your internet connection."
    }
} else {
    Write-Host "Installing common dependencies manually..." -ForegroundColor Green
    $packages = @("fastapi", "uvicorn", "pandas", "numpy", "matplotlib", "astropy", "torch", "scipy", "python-dotenv", "scikit-learn")
    foreach ($package in $packages) {
        Write-Host "Installing $package..." -ForegroundColor Cyan
        try {
            & $pythonCmd -m pip install --user $package
        } catch {
            Write-Host "Warning: Failed to install $package, continuing..." -ForegroundColor Yellow
        }
    }
}

# Verify installation
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Green
try {
    & $pythonCmd -c "import fastapi, uvicorn, pandas, numpy; print('Core dependencies verified successfully!')"
    Write-Host "✓ Installation completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run Better Impuls Viewer."
} catch {
    Write-Host "✗ Verification failed. Some dependencies may not be installed correctly." -ForegroundColor Red
    Write-Host ""
    Write-Host "You can try installing dependencies manually:"
    Write-Host "  $pythonCmd -m pip install --user fastapi uvicorn pandas numpy matplotlib astropy torch scipy python-dotenv scikit-learn"
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")