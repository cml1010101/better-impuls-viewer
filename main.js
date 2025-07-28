const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const isDev = require('electron-is-dev');

// Keep a global reference of the window object
let mainWindow;
let backendProcess;

// Backend configuration
const BACKEND_PORT = 8000;
const FRONTEND_PORT = 5173;

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      sandbox: false  // Disable sandbox for container compatibility
    },
    icon: path.join(__dirname, 'assets/icon.png'), // Add icon if available
    title: 'Better Impuls Viewer',
    show: false  // Don't show window until ready
  });

  // Load the app
  if (isDev) {
    // Development mode: load from dev server
    console.log('Running in development mode');
    mainWindow.loadURL(`http://localhost:${FRONTEND_PORT}`);
    // Open DevTools in development
    mainWindow.webContents.openDevTools();
  } else {
    // Production mode: load from built files
    console.log('Running in production mode');
    const frontendPath = path.join(__dirname, 'frontend/dist/index.html');
    mainWindow.loadFile(frontendPath);
  }

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    require('electron').shell.openExternal(url);
    return { action: 'deny' };
  });
}

function startBackend() {
  return new Promise((resolve, reject) => {
    console.log('Starting backend server...');
    
    const backendPath = path.join(__dirname, 'backend');
    const serverScript = path.join(backendPath, 'server.py');
    
    // Use the same Python executable finding logic
    const pythonExecutable = findPythonExecutable();
    
    if (!pythonExecutable) {
      reject(new Error('Python executable not found'));
      return;
    }
    
    console.log('Using Python command:', pythonExecutable);
    
    // Start the Python backend
    backendProcess = spawn(pythonExecutable, [serverScript], {
      cwd: backendPath,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let startupTimeout;

    backendProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log(`Backend stdout: ${output}`);
      
      // Check if server is ready
      if (output.includes('Uvicorn running') || output.includes('Application startup complete')) {
        if (startupTimeout) {
          clearTimeout(startupTimeout);
          startupTimeout = null;
        }
        console.log('Backend server is ready!');
        resolve();
      }
    });

    backendProcess.stderr.on('data', (data) => {
      const output = data.toString();
      console.error(`Backend stderr: ${output}`);
      
      // Also check stderr for server ready messages (uvicorn logs to stderr)
      if (output.includes('Uvicorn running') || output.includes('Application startup complete')) {
        if (startupTimeout) {
          clearTimeout(startupTimeout);
          startupTimeout = null;
        }
        console.log('Backend server is ready!');
        resolve();
      }
    });

    backendProcess.on('error', (error) => {
      console.error('Failed to start backend:', error);
      reject(error);
    });

    backendProcess.on('close', (code) => {
      console.log(`Backend process exited with code ${code}`);
      if (code !== 0) {
        reject(new Error(`Backend exited with code ${code}`));
      }
    });

    // Set a timeout for backend startup
    startupTimeout = setTimeout(() => {
      console.log('Backend startup timeout - assuming ready');
      resolve();
    }, 10000); // 10 second timeout
  });
}

function stopBackend() {
  if (backendProcess) {
    console.log('Stopping backend server...');
    backendProcess.kill();
    backendProcess = null;
  }
}

// Find available Python executable
function findPythonExecutable() {
  const pythonCommands = ['python3', 'python', 'py'];
  
  for (const cmd of pythonCommands) {
    try {
      require('child_process').execSync(`${cmd} --version`, { stdio: 'ignore' });
      return cmd;
    } catch (error) {
      // Continue to next command
    }
  }
  return null;
}

// Check if Python and required packages are available
async function checkDependencies() {
  return new Promise((resolve) => {
    const pythonExecutable = findPythonExecutable();
    
    if (!pythonExecutable) {
      resolve({ success: false, reason: 'python_not_found' });
      return;
    }
    
    const checkProcess = spawn(pythonExecutable, ['-c', 
      'import fastapi, uvicorn, pandas, numpy, matplotlib, astropy, torch, scipy, dotenv, sklearn; print("Dependencies OK")'
    ]);
    
    checkProcess.on('close', (code) => {
      resolve({ success: code === 0, reason: code === 0 ? null : 'packages_missing' });
    });

    checkProcess.on('error', () => {
      resolve({ success: false, reason: 'python_error' });
    });
  });
}

// Automatically install Python dependencies
async function installDependencies() {
  return new Promise((resolve) => {
    const pythonExecutable = findPythonExecutable();
    
    if (!pythonExecutable) {
      resolve({ success: false, error: 'Python not found on system' });
      return;
    }

    console.log('Installing Python dependencies automatically...');
    
    // Try to install from requirements.txt first, then fallback to individual packages
    const requirementsPath = path.join(__dirname, 'requirements.txt');
    let installCommand = ['-m', 'pip', 'install', '--user', '-r', requirementsPath];
    
    // Check if requirements.txt exists
    if (!require('fs').existsSync(requirementsPath)) {
      // Fallback to installing individual packages
      const packages = ['fastapi', 'uvicorn', 'pandas', 'numpy', 'matplotlib', 'astropy', 'torch', 'scipy', 'python-dotenv', 'scikit-learn'];
      installCommand = ['-m', 'pip', 'install', '--user'].concat(packages);
    }
    
    const installProcess = spawn(pythonExecutable, installCommand, {
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let output = '';
    let errorOutput = '';
    
    installProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    installProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });
    
    installProcess.on('close', (code) => {
      if (code === 0) {
        console.log('Dependencies installed successfully');
        resolve({ success: true });
      } else {
        console.error('Failed to install dependencies:', errorOutput);
        resolve({ success: false, error: errorOutput });
      }
    });

    installProcess.on('error', (error) => {
      console.error('Error running pip install:', error);
      resolve({ success: false, error: error.message });
    });
  });
}

// Show progress dialog during dependency installation
function showInstallationProgress() {
  const progressWindow = new BrowserWindow({
    width: 400,
    height: 200,
    show: false,
    resizable: false,
    minimizable: false,
    maximizable: false,
    closable: false,
    alwaysOnTop: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  progressWindow.loadURL(`data:text/html;charset=utf-8,
    <html>
      <head>
        <style>
          body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; 
            padding: 20px; 
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: calc(100vh - 40px);
            text-align: center;
          }
          .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin-bottom: 20px;
          }
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          h2 { margin: 0 0 10px 0; color: #333; }
          p { margin: 5px 0; color: #666; }
        </style>
      </head>
      <body>
        <div class="spinner"></div>
        <h2>Setting up Better Impuls Viewer</h2>
        <p>Installing required dependencies...</p>
        <p>This may take a few minutes.</p>
      </body>
    </html>
  `);

  progressWindow.once('ready-to-show', () => {
    progressWindow.show();
  });

  return progressWindow;
}

// App event handlers
app.whenReady().then(async () => {
  console.log('Electron app is ready');

  // Check dependencies first
  const depsResult = await checkDependencies();
  
  if (!depsResult.success) {
    if (depsResult.reason === 'python_not_found') {
      dialog.showErrorBox(
        'Python Required', 
        'Python 3.8 or newer is required but not found on your system.\n\n' +
        'Please install Python from https://python.org/downloads/\n' +
        'Make sure to check "Add Python to PATH" during installation.\n\n' +
        'After installing Python, restart Better Impuls Viewer.'
      );
      app.quit();
      return;
    }
    
    // Dependencies are missing, install them automatically
    console.log('Dependencies missing, installing automatically...');
    
    const progressWindow = showInstallationProgress();
    
    try {
      const installResult = await installDependencies();
      progressWindow.close();
      
      if (!installResult.success) {
        dialog.showErrorBox(
          'Installation Failed', 
          'Failed to install required Python packages automatically.\n\n' +
          'Error: ' + (installResult.error || 'Unknown error') + '\n\n' +
          'Please try installing manually:\n' +
          'pip install fastapi uvicorn pandas numpy matplotlib astropy torch scipy python-dotenv scikit-learn'
        );
        app.quit();
        return;
      }
      
      // Verify installation worked
      const verifyResult = await checkDependencies();
      if (!verifyResult.success) {
        dialog.showErrorBox(
          'Installation Verification Failed', 
          'Dependencies were installed but verification failed.\n\n' +
          'Please restart Better Impuls Viewer and try again.'
        );
        app.quit();
        return;
      }
      
      console.log('Dependencies installed and verified successfully');
      
    } catch (error) {
      progressWindow.close();
      console.error('Error during dependency installation:', error);
      dialog.showErrorBox(
        'Installation Error', 
        'An error occurred while installing dependencies:\n\n' +
        error.message + '\n\n' +
        'Please try installing manually:\n' +
        'pip install fastapi uvicorn pandas numpy matplotlib astropy torch scipy python-dotenv scikit-learn'
      );
      app.quit();
      return;
    }
  }

  try {
    // Start backend server
    await startBackend();
    
    // Create main window
    createWindow();

  } catch (error) {
    console.error('Failed to start application:', error);
    dialog.showErrorBox(
      'Startup Error', 
      `Failed to start the backend server:\n${error.message}\n\nPlease check the console for more details.`
    );
    app.quit();
  }
});

app.on('window-all-closed', () => {
  // Stop backend when all windows are closed
  stopBackend();
  
  // On macOS, keep app running even when all windows are closed
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // On macOS, re-create window when dock icon is clicked
  if (mainWindow === null) {
    createWindow();
  }
});

app.on('before-quit', () => {
  // Ensure backend is stopped before quitting
  stopBackend();
});

// Handle app termination
process.on('SIGINT', () => {
  stopBackend();
  app.quit();
});

process.on('SIGTERM', () => {
  stopBackend();
  app.quit();
});