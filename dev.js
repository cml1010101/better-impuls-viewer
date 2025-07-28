#!/usr/bin/env node
/**
 * Development script for Better Impuls Viewer Electron app
 * Starts frontend dev server and then launches Electron in dev mode
 */

const { spawn } = require('child_process');
const path = require('path');

const FRONTEND_PORT = 5173;

console.log('🚀 Starting Better Impuls Viewer in development mode...');

// Start frontend dev server
console.log('📦 Starting frontend dev server...');
const frontendProcess = spawn('npm', ['run', 'dev'], {
  cwd: path.join(__dirname, 'frontend'),
  stdio: ['pipe', 'pipe', 'pipe']
});

let frontendReady = false;

frontendProcess.stdout.on('data', (data) => {
  const output = data.toString();
  console.log(`Frontend: ${output}`);
  
  if (output.includes('Local:') && !frontendReady) {
    frontendReady = true;
    console.log('✅ Frontend dev server is ready!');
    
    // Wait a bit then start Electron
    setTimeout(() => {
      console.log('🖥️  Starting Electron...');
      const electronProcess = spawn('npm', ['run', 'electron:dev'], {
        stdio: 'inherit',
        env: { ...process.env, DISPLAY: process.env.DISPLAY || ':0' }
      });
      
      electronProcess.on('close', (code) => {
        console.log('🖥️  Electron closed, stopping frontend dev server...');
        frontendProcess.kill();
        process.exit(code);
      });
    }, 2000);
  }
});

frontendProcess.stderr.on('data', (data) => {
  console.error(`Frontend error: ${data}`);
});

frontendProcess.on('error', (error) => {
  console.error('❌ Failed to start frontend dev server:', error);
  process.exit(1);
});

// Handle process termination
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down development environment...');
  frontendProcess.kill();
  process.exit(0);
});

process.on('SIGTERM', () => {
  frontendProcess.kill();
  process.exit(0);
});