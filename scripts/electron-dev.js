#!/usr/bin/env node

/**
 * Electron Development Script
 * 
 * This script provides a streamlined development experience by:
 * 1. Starting the frontend dev server (if not already running)
 * 2. Waiting for it to be ready
 * 3. Starting Electron in development mode
 * 
 * The Electron app will automatically start the backend server.
 */

const { spawn, exec } = require('child_process');
const http = require('http');
const path = require('path');
const fs = require('fs');

// Configuration
const FRONTEND_PORT = 5173;
const FRONTEND_HOST = 'localhost';
const FRONTEND_URL = `http://${FRONTEND_HOST}:${FRONTEND_PORT}`;

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

function log(message, color = colors.blue) {
  console.log(`${color}[ELECTRON-DEV]${colors.reset} ${message}`);
}

function success(message) {
  console.log(`${colors.green}[SUCCESS]${colors.reset} ${message}`);
}

function error(message) {
  console.log(`${colors.red}[ERROR]${colors.reset} ${message}`);
}

function warning(message) {
  console.log(`${colors.yellow}[WARNING]${colors.reset} ${message}`);
}

// Check if a port is in use
function checkPort(port) {
  return new Promise((resolve) => {
    const options = {
      hostname: 'localhost',
      port: port,
      path: '/',
      method: 'GET',
      timeout: 1000
    };

    const req = http.request(options, (res) => {
      resolve(true); // Port is in use
    });

    req.on('error', () => {
      resolve(false); // Port is not in use
    });

    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });

    req.end();
  });
}

// Wait for frontend dev server to be ready
function waitForFrontend(maxRetries = 30) {
  return new Promise((resolve, reject) => {
    let retries = 0;
    
    const check = async () => {
      try {
        const isReady = await checkPort(FRONTEND_PORT);
        if (isReady) {
          success(`Frontend dev server is ready at ${FRONTEND_URL}`);
          resolve();
        } else if (retries < maxRetries) {
          retries++;
          log(`Waiting for frontend dev server... (${retries}/${maxRetries})`);
          setTimeout(check, 1000);
        } else {
          reject(new Error('Frontend dev server failed to start within timeout'));
        }
      } catch (err) {
        reject(err);
      }
    };
    
    check();
  });
}

// Start frontend dev server
function startFrontendServer() {
  return new Promise((resolve, reject) => {
    log('Starting frontend dev server...');
    
    const frontendPath = path.join(__dirname, '..', 'frontend');
    
    // Check if frontend directory exists
    if (!fs.existsSync(frontendPath)) {
      reject(new Error('Frontend directory not found'));
      return;
    }
    
    // Check if package.json exists
    const packageJsonPath = path.join(frontendPath, 'package.json');
    if (!fs.existsSync(packageJsonPath)) {
      reject(new Error('Frontend package.json not found. Run "cd frontend && npm install" first.'));
      return;
    }
    
    // Start the dev server
    const frontendProcess = spawn('npm', ['run', 'dev'], {
      cwd: frontendPath,
      stdio: ['pipe', 'pipe', 'pipe'],
      shell: true
    });
    
    let hasStarted = false;
    
    frontendProcess.stdout.on('data', (data) => {
      const output = data.toString();
      // Look for Vite's ready message
      if (output.includes('Local:') || output.includes('localhost:5173')) {
        if (!hasStarted) {
          hasStarted = true;
          resolve(frontendProcess);
        }
      }
      // Forward frontend logs with prefix
      process.stdout.write(`${colors.cyan}[FRONTEND]${colors.reset} ${output}`);
    });
    
    frontendProcess.stderr.on('data', (data) => {
      process.stderr.write(`${colors.cyan}[FRONTEND]${colors.reset} ${data}`);
    });
    
    frontendProcess.on('error', (err) => {
      if (!hasStarted) {
        reject(err);
      }
    });
    
    frontendProcess.on('close', (code) => {
      if (code !== 0 && !hasStarted) {
        reject(new Error(`Frontend dev server failed to start (exit code ${code})`));
      }
    });
    
    // Timeout after 30 seconds
    setTimeout(() => {
      if (!hasStarted) {
        frontendProcess.kill();
        reject(new Error('Frontend dev server failed to start within 30 seconds'));
      }
    }, 30000);
  });
}

// Start Electron
function startElectron() {
  return new Promise((resolve, reject) => {
    log('Starting Electron...');
    
    // Use --no-sandbox in CI environments or when DISPLAY is not available
    const isCI = process.env.CI || !process.env.DISPLAY;
    const electronScript = isCI ? 'electron-dev-ci' : 'electron-dev';
    
    if (isCI) {
      log('Detected CI environment, using --no-sandbox flag');
    }
    
    const electronProcess = spawn('npm', ['run', electronScript], {
      stdio: 'inherit',
      shell: true,
      env: {
        ...process.env,
        ELECTRON_IS_DEV: '1'
      }
    });
    
    electronProcess.on('error', (err) => {
      reject(err);
    });
    
    electronProcess.on('close', (code) => {
      log(`Electron exited with code ${code}`);
      resolve(code);
    });
    
    // Return the process so we can manage it
    resolve(electronProcess);
  });
}

// Main function
async function main() {
  try {
    log('🚀 Starting Electron development environment...');
    console.log();
    
    // Check if frontend is already running
    const frontendRunning = await checkPort(FRONTEND_PORT);
    
    let frontendProcess = null;
    
    if (frontendRunning) {
      success(`Frontend dev server already running at ${FRONTEND_URL}`);
    } else {
      // Start frontend dev server
      frontendProcess = await startFrontendServer();
      
      // Wait for it to be ready
      await waitForFrontend();
    }
    
    console.log();
    log('🎯 All services ready! Starting Electron...');
    console.log();
    
    // Start Electron
    const electronProcess = await startElectron();
    
    // Handle cleanup on exit
    const cleanup = () => {
      log('Cleaning up processes...');
      if (frontendProcess) {
        frontendProcess.kill('SIGTERM');
      }
      if (electronProcess && electronProcess.kill) {
        electronProcess.kill('SIGTERM');
      }
      process.exit(0);
    };
    
    process.on('SIGINT', cleanup);
    process.on('SIGTERM', cleanup);
    
  } catch (err) {
    error(`Failed to start development environment: ${err.message}`);
    console.log();
    console.log('💡 Troubleshooting tips:');
    console.log('  1. Make sure you have run: npm install');
    console.log('  2. Make sure you have run: cd frontend && npm install');
    console.log('  3. Check that ports 5173 and 8000 are available');
    console.log('  4. Verify Python dependencies: pip install -r requirements.txt');
    console.log();
    process.exit(1);
  }
}

// Run if this script is executed directly
if (require.main === module) {
  // Check for help flag
  if (process.argv.includes('--help') || process.argv.includes('-h')) {
    console.log(`
${colors.bright}Electron Development Script${colors.reset}

${colors.green}Usage:${colors.reset}
  npm run dev                 # Run this script via npm
  node scripts/electron-dev.js  # Run this script directly

${colors.green}What this script does:${colors.reset}
  1. Checks if frontend dev server is running on port 5173
  2. Starts frontend dev server if needed (npm run dev in frontend/)
  3. Waits for frontend to be ready
  4. Starts Electron in development mode
  5. Electron automatically starts the Python backend

${colors.green}Environment:${colors.reset}
  • Frontend: Vite dev server with hot reload
  • Backend: Python FastAPI server (auto-started by Electron)
  • DevTools: Automatically opened for debugging

${colors.green}Requirements:${colors.reset}
  • Node.js and npm installed
  • Frontend dependencies: cd frontend && npm install
  • Python dependencies: pip install -r requirements.txt

${colors.green}Troubleshooting:${colors.reset}
  • Run './scripts/dev-setup.sh' for automatic setup
  • Make sure ports 5173 and 8000 are available
  • Check that all dependencies are installed
`);
    process.exit(0);
  }
  
  main();
}

module.exports = { main };