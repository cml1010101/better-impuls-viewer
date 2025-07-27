"""
Configuration management for Better Impuls Viewer backend.
Handles environment variables and application settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (optional, for development)
load_dotenv()


class Config:
    """Application configuration settings."""
    
    # Google Sheets configuration - now managed through credentials system
    # Legacy environment variable support for development
    GOOGLE_SHEET_URL = os.getenv("GOOGLE_SHEET_URL")
    GOOGLE_SERVICE_ACCOUNT_KEY_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY_PATH", "google_sheets_service_account.json")
    
    # API configuration
    CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "file://"]
    
    # Data processing configuration
    DEFAULT_THRESHOLD = 1.0
    DYNAMIC_THRESHOLD_MULTIPLIER = 1.5
    
    # Period detection configuration
    MIN_PERIOD = 0.1
    MAX_PERIOD = 100.0
    
    # Model configuration
    MODEL_SAVE_PATH = "trained_cnn_model.pth"
    DEVICE = "cpu"  # Can be changed to "cuda" if GPU is available

    @classmethod
    def get_data_dir(cls):
        """Get the data directory path, supporting Electron environment and user configuration."""
        # First check if user has configured a custom data folder
        configured_path = cls._get_configured_data_folder()
        if configured_path and os.path.exists(configured_path):
            return configured_path
        
        # Check if we're running in a bundled Electron app
        data_folder_env = os.environ.get('DATA_FOLDER')
        if data_folder_env:
            full_path = os.path.abspath(data_folder_env)
            if os.path.exists(full_path):
                return full_path
        
        # Check common locations for data folder
        possible_paths = [
            os.path.expanduser('~/Documents/impuls-data'),
            os.path.join(os.path.dirname(__file__), '..', 'sample_data'),
            '../sample_data',
            './sample_data',
            'sample_data'
        ]
        
        for path in possible_paths:
            full_path = os.path.abspath(path)
            if os.path.exists(full_path):
                return full_path
        
        # Create sample_data directory as fallback if it doesn't exist
        fallback_path = os.path.abspath('sample_data')
        try:
            os.makedirs(fallback_path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create fallback data directory {fallback_path}: {e}")
        
        return fallback_path
    
    @classmethod
    def _get_configured_data_folder(cls):
        """Get configured data folder from credentials manager, avoiding circular imports."""
        try:
            # Dynamically import to avoid circular dependency
            import importlib
            credentials_module = importlib.import_module('credentials_manager')
            get_credentials_manager = getattr(credentials_module, 'get_credentials_manager')
            credentials_manager = get_credentials_manager()
            return credentials_manager.get_data_folder_path()
        except Exception:
            # If credentials manager is not available, return None
            return None
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        # Note: Validation is now handled by credentials manager
        # This method is kept for backward compatibility
        from credentials_manager import get_credentials_manager
        try:
            credentials_manager = get_credentials_manager()
            if not credentials_manager.get_google_sheets_url():
                print("Warning: Google Sheets URL not configured in credentials manager")
                return False
            return True
        except Exception:
            print("Warning: Could not validate credentials")
            return False

# Initialize DATA_DIR as a module-level variable but allow dynamic updates
try:
    DATA_DIR = Config.get_data_dir()
    if not DATA_DIR:
        DATA_DIR = os.path.abspath('sample_data')
        os.makedirs(DATA_DIR, exist_ok=True)
except Exception as e:
    print(f"Warning: Error initializing DATA_DIR: {e}")
    DATA_DIR = os.path.abspath('sample_data')
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception as create_error:
        print(f"Error creating fallback DATA_DIR: {create_error}")

def get_data_folder():
    """Get the current data folder path, checking for updates from credentials manager."""
    global DATA_DIR
    try:
        # Re-evaluate the data directory in case configuration changed
        current_dir = Config.get_data_dir()
        if current_dir:
            DATA_DIR = current_dir
        return DATA_DIR
    except Exception as e:
        print(f"Warning: Error getting data folder: {e}")
        return DATA_DIR

CLASS_NAMES = [
    "sinusoidal",
    "double dip",
    "shape changer",
    "beater",
    "beater/complex peak",
    "resolved close peaks",
    "resolved distant peaks",
    "eclipsing binaries",
    "pulsator",
    "burster",
    "dipper",
    "co-rotating optically thin material",
    "long term trend",
    "stochastic"
]