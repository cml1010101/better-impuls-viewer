// Utility to determine the API base URL based on the environment
export const getApiBaseUrl = (): string => {
  // Check if we're running in Electron
  if (window && (window as any).process && (window as any).process.type) {
    // We're in Electron, use localhost
    return 'http://localhost:8000';
  }
  
  // Check if we're in development mode (Vite dev server)
  if (import.meta.env.DEV) {
    return 'http://localhost:8000';
  }
  
  // In production (whether Electron or web), use localhost for now
  // This could be made configurable via environment variables
  return 'http://localhost:8000';
};

export const API_BASE = getApiBaseUrl();