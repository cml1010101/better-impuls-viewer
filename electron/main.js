const { app, BrowserWindow, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

// Keep a global reference of the window object
let mainWindow;
let backendProcess = null;
const isDev = process.env.NODE_ENV === 'development';

// Backend configuration
const BACKEND_PORT = 8000;
const FRONTEND_DEV_PORT = 5173;

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      webSecurity: true,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'assets', 'icon.png'), // Add an icon if you have one
    show: false // Don't show until ready
  });

  // Determine the URL to load
  let indexUrl;
  if (isDev) {
    // In development, use the Vite dev server
    indexUrl = `http://localhost:${FRONTEND_DEV_PORT}`;
  } else {
    // In production, load the built frontend files
    const frontendPath = path.join(__dirname, '..', 'frontend', 'dist', 'index.html');
    indexUrl = `file://${frontendPath}`;
  }

  // Load the frontend
  mainWindow.loadURL(indexUrl);

  // Show window when ready to prevent visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    // Open DevTools in development
    if (isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

function startBackend() {
  return new Promise((resolve, reject) => {
    const backendPath = isDev 
      ? path.join(__dirname, '..', 'backend')
      : path.join(process.resourcesPath, 'backend');
    
    const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    const serverScript = path.join(backendPath, 'server.py');

    console.log('Starting backend from:', backendPath);
    console.log('Python executable:', pythonExecutable);
    console.log('Server script:', serverScript);

    // Check if server script exists
    if (!fs.existsSync(serverScript)) {
      console.error('Backend server script not found:', serverScript);
      reject(new Error(`Backend server script not found: ${serverScript}`));
      return;
    }

    // Start the Python backend
    backendProcess = spawn(pythonExecutable, [serverScript], {
      cwd: backendPath,
      env: {
        ...process.env,
        PYTHONPATH: backendPath,
        PORT: BACKEND_PORT.toString()
      }
    });

    backendProcess.stdout.on('data', (data) => {
      console.log('Backend stdout:', data.toString());
      // Check if server is ready
      if (data.toString().includes('Uvicorn running on')) {
        console.log('Backend server is ready');
        resolve();
      }
    });

    backendProcess.stderr.on('data', (data) => {
      console.error('Backend stderr:', data.toString());
    });

    backendProcess.on('error', (error) => {
      console.error('Failed to start backend:', error);
      reject(error);
    });

    backendProcess.on('close', (code) => {
      console.log(`Backend process exited with code ${code}`);
      if (code !== 0 && code !== null) {
        reject(new Error(`Backend process exited with code ${code}`));
      }
    });

    // Timeout if backend doesn't start within 30 seconds
    setTimeout(() => {
      if (backendProcess && !backendProcess.killed) {
        reject(new Error('Backend startup timeout'));
      }
    }, 30000);
  });
}

function stopBackend() {
  if (backendProcess && !backendProcess.killed) {
    console.log('Stopping backend process...');
    backendProcess.kill('SIGTERM');
    
    // Force kill after 5 seconds if still running
    setTimeout(() => {
      if (backendProcess && !backendProcess.killed) {
        console.log('Force killing backend process...');
        backendProcess.kill('SIGKILL');
      }
    }, 5000);
  }
}

// App event handlers
app.whenReady().then(async () => {
  try {
    console.log('App is ready, starting backend...');
    await startBackend();
    console.log('Backend started successfully, creating window...');
    createWindow();
  } catch (error) {
    console.error('Failed to start application:', error);
    dialog.showErrorBox(
      'Startup Error',
      `Failed to start the application:\n\n${error.message}\n\nPlease check that Python 3 is installed and all dependencies are available.`
    );
    app.quit();
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', (event) => {
  console.log('App is quitting, stopping backend...');
  stopBackend();
});

// Handle app crashes and unhandled errors
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  stopBackend();
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});