const { contextBridge, ipcRenderer } = require('electron');

// Expose a limited API to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  // Example: Add any API methods you need to expose to the frontend
  // For now, this is just a placeholder for future communication
  getVersion: () => process.versions.electron,
  platform: process.platform
});