#!/usr/bin/env node
/**
 * Production deployment script for Better Impuls Viewer Web App
 * Builds frontend and provides instructions for backend deployment
 */

const { spawn } = require('child_process');
const path = require('path');

console.log('🚀 Building Better Impuls Viewer for production deployment...');

function buildFrontend() {
  return new Promise((resolve, reject) => {
    console.log('📦 Building frontend for production...');
    
    const buildProcess = spawn('npm', ['run', 'build'], {
      cwd: path.join(__dirname, 'frontend'),
      stdio: 'inherit'
    });

    buildProcess.on('close', (code) => {
      if (code === 0) {
        console.log('✅ Frontend build completed successfully');
        console.log('📁 Built files are in: frontend/dist/');
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

async function deploy() {
  try {
    await buildFrontend();
    
    console.log('\n🎉 Production build completed successfully!\n');
    console.log('📋 Deployment Instructions:');
    console.log('');
    console.log('1. Frontend (Static Files):');
    console.log('   - Serve the contents of frontend/dist/ using any web server');
    console.log('   - Examples: nginx, Apache, Netlify, Vercel, GitHub Pages');
    console.log('');
    console.log('2. Backend (API Server):');
    console.log('   - Deploy backend/ folder to a Python hosting service');
    console.log('   - Install requirements: pip install -r requirements.txt');
    console.log('   - Run: python server.py (or use uvicorn for production)');
    console.log('   - Examples: Heroku, Railway, DigitalOcean, AWS Lambda');
    console.log('');
    console.log('3. Environment Configuration:');
    console.log('   - Update CORS_ORIGINS in backend/config.py with your production domain');
    console.log('   - Set API_BASE_URL in frontend if backend is on different domain');
    console.log('');
    console.log('🌐 Your web app is ready for deployment!');
    
  } catch (error) {
    console.error('❌ Deployment preparation failed:', error.message);
    process.exit(1);
  }
}

deploy();