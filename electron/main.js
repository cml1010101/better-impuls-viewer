const { app, BrowserWindow, shell, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const { execSync } = require('child_process');
const fs = require('fs');

// Keep a global reference of the window object
let mainWindow;
let backendProcess;

const isDev = process.env.ELECTRON_IS_DEV === '1';

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      webSecurity: true,
      enableWebSQL: false,
      spellcheck: false,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'assets', 'icon.png') // Optional: add an icon
  });

  // Filter out DevTools autofill errors to reduce console noise
  mainWindow.webContents.on('console-message', (event, level, message) => {
    // Suppress known DevTools autofill errors that don't affect functionality
    if (message.includes('Autofill.enable') || 
        message.includes('Autofill.setAddresses') ||
        message.includes("wasn't found")) {
      // Just return without logging these messages - they're harmless DevTools warnings
      return;
    }
    // Log other console messages normally
    if (level === 1) {
      console.log(`Frontend console: ${message}`);
    } else if (level === 2) {
      console.warn(`Frontend console warning: ${message}`);
    } else if (level === 3) {
      console.error(`Frontend console error: ${message}`);
    }
  });

  // In development, load from vite dev server
  // In production, load the built frontend
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    const frontendPath = path.join(__dirname, '..', 'frontend', 'dist', 'index.html');
    mainWindow.loadFile(frontendPath);
  }

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function getBackendExecutablePath() {
  if (isDev) {
    // In development, use system Python
    return 'python3';
  } else {
    // In production, use bundled Python environment
    const bundledPython = path.join(__dirname, '..', 'python-env', 'bin', 'python');
    if (fs.existsSync(bundledPython)) {
      return bundledPython;
    }
    // Fallback to system Python if bundled doesn't exist
    return 'python3';
  }
}

function startBackendServer() {
  const pythonPath = getBackendExecutablePath();
  const backendPath = isDev 
    ? path.join(__dirname, '..', 'backend', 'server.py')
    : path.join(__dirname, '..', 'backend', 'server.py');

  console.log('Starting backend server...');
  console.log('Python path:', pythonPath);
  console.log('Backend script path:', backendPath);

  // Set environment variables for the backend
  const env = { 
    ...process.env,
    PYTHONPATH: path.join(__dirname, '..', 'backend'),
    DATA_FOLDER: isDev 
      ? path.join(__dirname, '..', 'sample_data')
      : path.join(__dirname, '..', 'sample_data')
  };

  try {
    backendProcess = spawn(pythonPath, [backendPath], {
      env,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    backendProcess.stdout.on('data', (data) => {
      console.log(`Backend stdout: ${data}`);
    });

    backendProcess.stderr.on('data', (data) => {
      console.error(`Backend stderr: ${data}`);
    });

    backendProcess.on('close', (code) => {
      console.log(`Backend process exited with code ${code}`);
      if (code !== 0 && mainWindow) {
        // Backend crashed, show error message
        console.error('Backend process crashed');
      }
    });

    backendProcess.on('error', (error) => {
      console.error('Failed to start backend process:', error);
    });

    console.log('Backend server started with PID:', backendProcess.pid);
  } catch (error) {
    console.error('Error starting backend:', error);
  }
}

function stopBackendServer() {
  if (backendProcess) {
    console.log('Stopping backend server...');
    backendProcess.kill('SIGTERM');
    backendProcess = null;
  }
}

// Check if backend dependencies are available
function checkBackendDependencies() {
  const pythonPath = getBackendExecutablePath();
  
  try {
    // Try to import required packages
    const testScript = `
import sys
try:
    import fastapi
    import uvicorn
    import pandas
    import numpy
    print("Dependencies OK")
    sys.exit(0)
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)
`;
    
    execSync(`${pythonPath} -c "${testScript}"`, { 
      stdio: 'pipe',
      timeout: 10000 
    });
    return true;
  } catch (error) {
    console.error('Backend dependencies check failed:', error.message);
    return false;
  }
}

// Wait for backend to be ready
function waitForBackend(callback, retries = 30) {
  const http = require('http');
  
  const options = {
    hostname: 'localhost',
    port: 8000,
    path: '/',
    method: 'GET',
    timeout: 1000
  };

  const req = http.request(options, (res) => {
    console.log('Backend is ready!');
    callback();
  });

  req.on('error', (error) => {
    if (retries > 0) {
      console.log(`Waiting for backend... (${31 - retries}/30)`);
      setTimeout(() => waitForBackend(callback, retries - 1), 1000);
    } else {
      console.error('Backend failed to start within timeout');
      callback();
    }
  });

  req.on('timeout', () => {
    req.destroy();
    if (retries > 0) {
      setTimeout(() => waitForBackend(callback, retries - 1), 1000);
    } else {
      console.error('Backend failed to start within timeout');
      callback();
    }
  });

  req.end();
}

// IPC handlers for Electron-specific functionality
ipcMain.handle('select-data-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: 'Select Data Folder',
    message: 'Choose the folder containing your star data files (.tbl format)'
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  
  return null;
});

// App event handlers
app.whenReady().then(() => {
  console.log('Electron app ready');
  
  // Check backend dependencies first
  if (!isDev && !checkBackendDependencies()) {
    console.error('Backend dependencies not available');
    // Could show an error dialog here
  }
  
  // Start backend server
  startBackendServer();
  
  // Wait for backend to be ready, then create window
  waitForBackend(() => {
    createWindow();
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopBackendServer();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopBackendServer();
});

// Handle app termination
process.on('SIGINT', () => {
  stopBackendServer();
  app.quit();
});

process.on('SIGTERM', () => {
  stopBackendServer();
  app.quit();
});