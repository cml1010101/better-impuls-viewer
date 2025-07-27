/// <reference types="vite/client" />

// Electron API types
interface ElectronAPI {
  selectDataFolder: () => Promise<string | null>;
  isElectron: boolean;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}

export {};
