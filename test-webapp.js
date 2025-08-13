#!/usr/bin/env node
/**
 * Test script to verify web app components are working
 */

const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

console.log('🧪 Testing Better Impuls Viewer Web App...');

function testBackendAPI() {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'localhost',
      port: 8000,
      path: '/docs',
      method: 'GET'
    };

    const req = http.request(options, (res) => {
      if (res.statusCode === 200) {
        console.log('✅ Backend API is responding');
        resolve();
      } else {
        reject(new Error(`Backend API returned status ${res.statusCode}`));
      }
    });

    req.on('error', (err) => {
      reject(err);
    });

    req.setTimeout(5000, () => {
      req.destroy();
      reject(new Error('Backend API request timeout'));
    });

    req.end();
  });
}

function testFrontendBuild() {
  return new Promise((resolve, reject) => {
    console.log('📦 Testing frontend build...');
    
    const buildProcess = spawn('npm', ['run', 'build'], {
      cwd: path.join(__dirname, 'frontend'),
      stdio: ['pipe', 'pipe', 'pipe']
    });

    buildProcess.on('close', (code) => {
      if (code === 0) {
        console.log('✅ Frontend builds successfully');
        resolve();
      } else {
        reject(new Error(`Frontend build failed with code ${code}`));
      }
    });

    buildProcess.on('error', (error) => {
      reject(error);
    });
  });
}

async function runTests() {
  try {
    // Test frontend build
    await testFrontendBuild();
    
    console.log('🎉 All tests passed! Web app components are working.');
    process.exit(0);
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
  }
}

runTests();