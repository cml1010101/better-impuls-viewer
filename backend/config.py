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
    
    # API configuration
    CORS_ORIGINS = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
        "http://localhost:4173",  # Vite preview server
        "http://localhost:8000",  # Backend self-requests
        "http://127.0.0.1:5173",  # Alternative localhost
        "http://127.0.0.1:3000",
        "http://127.0.0.1:4173",
        "http://127.0.0.1:8000"
    ]
    
    # Data processing configuration
    DEFAULT_THRESHOLD = 1.0
    DYNAMIC_THRESHOLD_MULTIPLIER = 1.5
    
    # Period detection configuration
    MIN_PERIOD = 0.1
    MAX_PERIOD = 100.0
    
    # Model configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'model.pth')
    DEVICE = "cpu"  # Can be changed to "cuda" if GPU is available

    DEFAULT_DATA_DIR = 'sample_data'
    DATA_DIR = os.getenv('DATA_DIR', DEFAULT_DATA_DIR)

    IMPULS_STARS_PATH = os.path.join(DATA_DIR, 'impuls_stars.csv')

    SED_USERNAME = os.getenv('SED_USERNAME', None)
    SED_PASSWORD = os.getenv('SED_PASSWORD', None)
    SED_API_URL = os.getenv('SED_API_URL', 'k2clusters.ipac.caltech.edu/impuls/seds')
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if not os.path.exists(cls.CSV_TRAINING_DATA_PATH):
            print(f"Warning: CSV training data file not found: {cls.CSV_TRAINING_DATA_PATH}")
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