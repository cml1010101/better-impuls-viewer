"""
Configuration management for Better Impuls Viewer backend.
Handles environment variables and application settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration settings."""
    
    # Google Sheets configuration
    GOOGLE_SHEET_URL = os.getenv("GOOGLE_SHEET_URL")
    GOOGLE_SERVICE_ACCOUNT_KEY_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY_PATH", "google_sheets_service_account.json")
    
    # API configuration
    CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]
    
    # Data processing configuration
    DEFAULT_THRESHOLD = 1.0
    DYNAMIC_THRESHOLD_MULTIPLIER = 1.5
    
    # Period detection configuration
    MIN_PERIOD = 0.1
    MAX_PERIOD = 100.0
    
    # Model configuration
    MODEL_SAVE_PATH = "trained_cnn_model.pth"
    DEVICE = "cpu"  # Can be changed to "cuda" if GPU is available

    DATA_DIR = os.path.expanduser('~/Documents/impuls-data') if os.path.exists(os.path.expanduser('~/Documents/impuls-data')) else '../sample_data'
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if not cls.GOOGLE_SHEET_URL:
            print("Warning: GOOGLE_SHEET_URL not set in environment variables")
            return False
        return True

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