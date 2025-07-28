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
    
    // Start the Python backend
    backendProcess = spawn('python', [serverScript], {
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

// Check if Python and required packages are available
async function checkDependencies() {
  return new Promise((resolve) => {
    const checkProcess = spawn('python', ['-c', 
      'import fastapi, uvicorn, pandas, numpy; print("Dependencies OK")'
    ]);
    
    checkProcess.on('close', (code) => {
      resolve(code === 0);
    });

    checkProcess.on('error', () => {
      resolve(false);
    });
  });
}

// Show error dialog for missing dependencies
function showDependencyError() {
  dialog.showErrorBox(
    'Missing Dependencies',
    'Python and required packages are not installed or not in PATH.\n\n' +
    'Please install:\n' +
    '1. Python 3.8+\n' +
    '2. Run: pip install -r requirements.txt\n\n' +
    'Then restart the application.'
  );
}

// App event handlers
app.whenReady().then(async () => {
  console.log('Electron app is ready');

  // Check dependencies first
  const depsOk = await checkDependencies();
  if (!depsOk) {
    showDependencyError();
    app.quit();
    return;
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