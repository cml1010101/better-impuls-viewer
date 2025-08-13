#!/usr/bin/env node
/**
 * Development script for Better Impuls Viewer Web App
 * Starts frontend dev server and backend API server concurrently
 */

const { spawn } = require('child_process');
const path = require('path');

console.log('🚀 Starting Better Impuls Viewer Web App in development mode...');

// Start frontend dev server
console.log('📦 Starting frontend dev server...');
const frontendProcess = spawn('npm', ['run', 'dev'], {
  cwd: path.join(__dirname, 'frontend'),
  stdio: ['pipe', 'pipe', 'pipe']
});

// Start backend server
console.log('🐍 Starting backend API server...');
const backendProcess = spawn('python', ['server.py'], {
  cwd: path.join(__dirname, 'backend'),
  stdio: ['pipe', 'pipe', 'pipe']
});

// Handle frontend output
frontendProcess.stdout.on('data', (data) => {
  const output = data.toString();
  console.log(`Frontend: ${output}`);
});

frontendProcess.stderr.on('data', (data) => {
  console.error(`Frontend error: ${data}`);
});

// Handle backend output
backendProcess.stdout.on('data', (data) => {
  const output = data.toString();
  console.log(`Backend: ${output}`);
});

backendProcess.stderr.on('data', (data) => {
  console.error(`Backend error: ${data}`);
});

// Handle process errors
frontendProcess.on('error', (error) => {
  console.error('❌ Failed to start frontend dev server:', error);
  process.exit(1);
});

backendProcess.on('error', (error) => {
  console.error('❌ Failed to start backend server:', error);
  process.exit(1);
});

// Handle process termination
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down development environment...');
  frontendProcess.kill();
  backendProcess.kill();
  process.exit(0);
});

process.on('SIGTERM', () => {
  frontendProcess.kill();
  backendProcess.kill();
  process.exit(0);
});

// Handle individual process exits
frontendProcess.on('close', (code) => {
  console.log(`📦 Frontend dev server exited with code ${code}`);
  backendProcess.kill();
  process.exit(code);
});

backendProcess.on('close', (code) => {
  console.log(`🐍 Backend server exited with code ${code}`);
  frontendProcess.kill();
  process.exit(code);
});