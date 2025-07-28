#!/usr/bin/env node
/**
 * Test script to verify Electron backend integration without GUI
 */

const { spawn } = require('child_process');
const path = require('path');

// Backend configuration
const BACKEND_PORT = 8000;

function startBackend() {
  return new Promise((resolve, reject) => {
    console.log('Starting backend server...');
    
    const backendPath = path.join(__dirname, 'backend');
    const serverScript = path.join(backendPath, 'server.py');
    
    // Start the Python backend
    const backendProcess = spawn('python', [serverScript], {
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
        console.log('✅ Backend server is ready!');
        
        // Test API endpoint
        testAPI().then(() => {
          console.log('✅ API test successful!');
          backendProcess.kill();
          process.exit(0);
        }).catch((err) => {
          console.error('❌ API test failed:', err);
          backendProcess.kill();
          process.exit(1);
        });
      }
    });

    backendProcess.stderr.on('data', (data) => {
      const output = data.toString();
      console.error(`Backend stderr: ${output}`);
      
      // Also check stderr for server ready messages
      if (output.includes('Uvicorn running') || output.includes('Application startup complete')) {
        if (startupTimeout) {
          clearTimeout(startupTimeout);
          startupTimeout = null;
        }
        console.log('✅ Backend server is ready!');
        
        // Test API endpoint
        testAPI().then(() => {
          console.log('✅ API test successful!');
          backendProcess.kill();
          process.exit(0);
        }).catch((err) => {
          console.error('❌ API test failed:', err);
          backendProcess.kill();
          process.exit(1);
        });
      }
    });

    backendProcess.on('error', (error) => {
      console.error('❌ Failed to start backend:', error);
      reject(error);
    });

    // Set a timeout for backend startup
    startupTimeout = setTimeout(() => {
      console.log('⏰ Backend startup timeout');
      reject(new Error('Backend startup timeout'));
    }, 15000); // 15 second timeout
  });
}

async function testAPI() {
  // Give the server a moment to be fully ready
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  try {
    const response = await fetch(`http://localhost:${BACKEND_PORT}/stars`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const data = await response.json();
    console.log('API Response:', data);
    return data;
  } catch (error) {
    throw new Error(`API test failed: ${error.message}`);
  }
}

// Check if Node.js has fetch (Node 18+)
if (typeof fetch === 'undefined') {
  console.log('Installing node-fetch for older Node.js versions...');
  require('child_process').execSync('npm install node-fetch@2', { stdio: 'inherit' });
  global.fetch = require('node-fetch');
}

console.log('🧪 Testing Electron backend integration...');
startBackend().catch((error) => {
  console.error('❌ Test failed:', error);
  process.exit(1);
});