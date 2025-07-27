"""
Google Sheets integration for loading training data.
Handles authentication and data extraction from Google Sheets.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import requests
from urllib.parse import urlparse
import re
from config import Config, CLASS_NAMES
from models import TrainingDataPoint

# --- New Imports for Google Sheets API Authentication ---
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
# --- End New Imports ---


class GoogleSheetsLoader:
    """Load and process training data from Google Sheets."""
    
    def __init__(self, sheet_url: str = None):
        """Initialize with Google Sheets URL and authenticate."""
        self.sheet_url = sheet_url or Config.GOOGLE_SHEET_URL
        if not self.sheet_url:
            raise ValueError("Google Sheets URL not provided")
        
        # Extract spreadsheet ID from the URL
        pattern = r'/spreadsheets/d/([a-zA-Z0-9-_]+)'
        match = re.search(pattern, self.sheet_url)
        if not match:
            raise ValueError("Invalid Google Sheets URL format. Could not extract Spreadsheet ID.")
        self.spreadsheet_id = match.group(1)
        
        # --- Authentication Setup ---
        self.client = self._authenticate_google_sheets()
        # --- End Authentication Setup ---

    def _authenticate_google_sheets(self):
        """Authenticates with Google Sheets API using a Service Account."""
        scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly'] # For read-only access
        
        service_account_path = Config.GOOGLE_SERVICE_ACCOUNT_KEY_PATH
        
        if not os.path.exists(service_account_path):
            raise FileNotFoundError(
                f"Service account key file not found at: {service_account_path}. "
                "Please ensure it's in the correct location and named 'google_sheets_service_account.json' "
                "or update Config.GOOGLE_SERVICE_ACCOUNT_KEY_PATH."
            )

        try:
            creds = Credentials.from_service_account_file(service_account_path, scopes=scopes)
            client = gspread.authorize(creds)
            print("Successfully authenticated with Google Sheets API.")
            return client
        except Exception as e:
            raise Exception(f"Failed to authenticate with Google Sheets API: {e}")

    # The _convert_to_csv_url method is no longer strictly needed if using gspread
    # but can be kept for consistency or if you ever needed direct CSV export for other reasons.
    # We will prioritize gspread's more robust data fetching.
    def _convert_to_csv_url(self, sheets_url: str) -> str:
        """Convert Google Sheets URL to CSV export URL (for direct download fallback/comparison)."""
        # Extract the spreadsheet ID from the URL
        pattern = r'/spreadsheets/d/([a-zA-Z0-9-_]+)'
        match = re.search(pattern, sheets_url)
        if not match:
            raise ValueError("Invalid Google Sheets URL format")
        
        spreadsheet_id = match.group(1)
        # Convert to CSV export URL
        csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv"
        return csv_url
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from Google Sheets using gspread."""
        try:
            # Open the spreadsheet by its ID
            spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            
            # Assuming you want the first worksheet. If not, specify by name:
            # worksheet = spreadsheet.worksheet("Your Sheet Name")
            worksheet = spreadsheet.get_worksheet(0) # Get the first worksheet
            
            # Get all values as a list of lists, then convert to DataFrame
            # get_all_records() returns a list of dictionaries, which is often easier
            # get_all_values() returns a list of lists, where the first list is the header
            data_list = worksheet.get_all_values() 
            
            if not data_list:
                print("No data found in the Google Sheet.")
                return pd.DataFrame()
                
            # Assume first row is header
            df = pd.DataFrame(data_list[1:], columns=data_list[0])
            
            print(f"Loaded {len(df)} rows from Google Sheets via API")
            return df
            
        except gspread.exceptions.SpreadsheetNotFound:
            print(f"Error: Spreadsheet with ID '{self.spreadsheet_id}' not found or you don't have access.")
            raise
        except Exception as e:
            print(f"Error loading Google Sheets data via API: {e}")
            raise
    
    def extract_training_data(self, stars_to_extract: list[int]) -> List[TrainingDataPoint]: # type: ignore
        """
        Extract training data from Google Sheets.
        
        Expected columns:
        - A: Star number
        - F: CDIPS period 1
        - G: CDIPS period 2
        - H: ELEANOR period 1
        - I: ELEANOR period 2
        - J: QLP period 1
        - K: QLP period 2
        - L: SPOC period 1
        - M: SPOC period 2
        - N: TESS 16 period 1
        - O: TESS 16 period 2
        - P: TASOC period 1
        - Q: TASOC period 2
        - R: TGLC period 1
        - S: TGLC period 2
        - T: EVEREST period 1
        - U: EVEREST period 2
        - V: K2SC period 1
        - W: K2SC period 2
        - X: K2SFF period 1
        - Y: K2SFF period 2
        - Z: K2VARCAT period 1
        - AA: K2VARCAT period 2
        - AB: ZTF_R period 1
        - AC: ZTF_R period 2
        - AD: ZTF_G period 1
        - AE: ZTF_G period 2
        - AF: W1 period 1
        - AG: W1 period 2
        - AH: W2 period 1
        - AI: W2 period 2
        - AN: LC category
        
        Returns:
            List of TrainingDataPoint objects with valid periods
        """
        df = self.load_raw_data()
        training_data = []
        sensors = {
            'cdips': (df.columns[5], df.columns[6]),
            'eleanor': (df.columns[7], df.columns[8]),
            'qlp': (df.columns[9], df.columns[10]),
            'spoc': (df.columns[11], df.columns[12]),
            't16': (df.columns[13], df.columns[14]),
            'tasoc': (df.columns[15], df.columns[16]),
            'tglc': (df.columns[17], df.columns[18]),
            'everest': (df.columns[19], df.columns[20]),
            'k2sc': (df.columns[21], df.columns[22]),
            'k2sff': (df.columns[23], df.columns[24]),
            'k2varcat': (df.columns[25], df.columns[26]),
            'ztf_r': (df.columns[27], df.columns[28]),
            'ztf_g': (df.columns[29], df.columns[30]),
            'w1': (df.columns[31], df.columns[32]),
            'w2': (df.columns[33], df.columns[34]),
        }

        CATEGORY_COL_NAME = df.columns[35]  # Assuming LC category is in column AL (index 35)
        
        print(f"Processing {len(df)} rows from Google Sheets...")
        print(f"Expected columns: A (star), AK (period1), AL (period2), AN (category)")
        
        for index, row in df.iterrows():
            if index == 0:
                # Skip header row
                continue
            star_number = index - 1
            if stars_to_extract is not None and star_number not in stars_to_extract:
                continue
            for sensor, (period1_col, period2_col) in sensors.items():
                try:
                    period1 = row[period1_col]
                    period2 = row[period2_col]
                    
                    # Skip if both periods are NaN or empty
                    if pd.isna(period1) and pd.isna(period2):
                        continue
                    # If period1 is "no", skip this sensor
                    if isinstance(period1, str) and period1.lower() == "no":
                        continue

                    # Load period values, converting to float if necessary
                    period1 = float(period1) if not pd.isna(period1) else None
                    period2 = float(period2) if not pd.isna(period2) else None
                    
                    # Normalize LC category
                    lc_category = self._normalize_lc_category(row[CATEGORY_COL_NAME])
                    
                    # Load time series data for this star
                    time_series, flux_series = self._load_star_data(star_number)
                    
                    # Skip if no valid time series data
                    if len(time_series) == 0 or len(flux_series) == 0:
                        continue
                    
                    # Create TrainingDataPoint object
                    training_data.append(TrainingDataPoint(
                        star_number=star_number,
                        period_1=period1,
                        period_2=period2,
                        lc_category=lc_category,
                        time_series=time_series,
                        flux_series=flux_series
                    ))
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                    continue
        
        print(f"Extracted {len(training_data)} valid training examples")
        return training_data
    
    def _normalize_lc_category(self, category: str) -> str:
        """
        Normalize LC category strings to standard classifications.
        
        Maps various category strings from Google Sheets to the 14 main types:
        - sinusoidal, sinusoidal? -> sinusoidal
        - double dip, double_dip -> double dip
        - shape changer, shape_changer -> shape changer
        - beater -> beater
        - beater/complex peak -> beater/complex peak
        - resolved close peaks -> resolved close peaks
        - resolved distant peaks -> resolved distant peaks
        - eclipsing binaries -> eclipsing binaries
        - pulsator -> pulsator
        - burster -> burster
        - dipper, dipper? -> dipper
        - co-rotating optically thin material -> co-rotating optically thin material
        - long term trend -> long term trend
        - stochastic -> stochastic
        """
        category = category.lower().strip()
        
        # Remove question marks and normalize
        category = category.replace('?', '').strip()
        
        # Map common variations to the exact CLASS_NAMES
        if 'sinusoidal' in category:
            return 'sinusoidal'
        elif 'double' in category and 'dip' in category:
            return 'double dip'
        elif 'shape' in category and 'chang' in category:
            return 'shape changer'
        elif 'beater' in category and ('complex' in category or 'peak' in category):
            return 'beater/complex peak'
        elif 'beater' in category:
            return 'beater'
        elif 'resolved' in category and 'close' in category:
            return 'resolved close peaks'
        elif 'resolved' in category and 'distant' in category:
            return 'resolved distant peaks'
        elif ('distant' in category and 'peak' in category) and 'resolved' not in category:
            return 'resolved distant peaks'  # Legacy mapping
        elif ('close' in category and 'peak' in category) and 'resolved' not in category:
            return 'resolved close peaks'  # Legacy mapping
        elif 'eclipsing' in category or 'binary' in category:
            return 'eclipsing binaries'
        elif 'pulsator' in category or 'pulsating' in category:
            return 'pulsator'
        elif 'burster' in category or 'burst' in category:
            return 'burster'
        elif 'dipper' in category:
            return 'dipper'
        elif 'co-rotating' in category or ('rotating' in category and 'material' in category):
            return 'co-rotating optically thin material'
        elif 'long' in category and 'trend' in category:
            return 'long term trend'
        elif 'stochastic' in category or 'irregular' in category:
            return 'stochastic'
        else:
            # Try to match against CLASS_NAMES directly
            for class_name in CLASS_NAMES:
                if class_name.lower().replace(' ', '') in category.replace(' ', ''):
                    return class_name
            
            # Default fallback - pick first category for unknown
            print(f"Warning: Unknown category '{category}', defaulting to 'stochastic'")
            return 'stochastic'
    
    def _load_star_data(self, star_number: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load time series data for a specific star from sample_data directory.
        
        Args:
            star_number: Star identifier
            
        Returns:
            Tuple of (time_series, flux_series) arrays
        """
        import os
        # Assuming data_processing.py is in the same directory or accessible
        # If data_processing is not imported or accessible here, you'll need to adjust
        from .data_processing import find_longest_x_campaign, sort_data, remove_y_outliers
        
        # Look for data files in sample_data directory
        sample_data_dir = Config.DATA_DIR
        
        if not os.path.exists(sample_data_dir):
            print(f"Sample data directory not found: {sample_data_dir}")
            return np.array([]), np.array([])
        
        # Find files for this star
        star_files = []
        for file in os.listdir(sample_data_dir):
            if file.startswith(f"{star_number}-") and file.endswith(".tbl"):
                star_files.append(file)
        
        if not star_files:
            # print(f"No data files found for star {star_number}") # Suppress this for cleaner output if many stars have no local data
            return np.array([]), np.array([])
        
        # Use the first available file (could be enhanced to combine multiple telescopes)
        file_path = os.path.join(sample_data_dir, star_files[0])
        
        try:
            # Load data using pandas
            # Ensure correct delimiter and skip rows based on your .tbl file format
            data = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None)
            # Assuming first two columns are time and flux, adjust if different
            # You might need to inspect your .tbl files to confirm the exact column indices
            time_series = data.iloc[:, 0].values
            flux_series = data.iloc[:, 1].values
            
            data_array = np.column_stack((time_series, flux_series))
            
            # Process data using existing functions
            data_array = find_longest_x_campaign(data_array, 1.0)
            data_array = remove_y_outliers(data_array)
            data_array = sort_data(data_array)
            
            if len(data_array) > 0:
                return data_array[:, 0], data_array[:, 1]  # time, flux
            else:
                return np.array([]), np.array([])
                
        except Exception as e:
            print(f"Error loading data for star {star_number} from {file_path}: {e}")
            return np.array([]), np.array([])
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of LC categories in the training data."""
        training_data = self.extract_training_data()
        categories = {}
        
        for point in training_data:
            category = point.lc_category
            categories[category] = categories.get(category, 0) + 1
        
        return categories


# Example usage and testing
if __name__ == "__main__":
    # Ensure you have a config.py with GOOGLE_SHEET_URL and optionally GOOGLE_SERVICE_ACCOUNT_KEY_PATH
    # And your google_sheets_service_account.json file in the specified path

    # Create dummy config and models for testing purposes if they don't exist
    class MockConfig:
        GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1BTMH782_sIq03m4x-v_mP2zT4o6kQ6JtE-Vp3Yg0pP0/edit#gid=0" # Replace with your actual test sheet URL
        GOOGLE_SERVICE_ACCOUNT_KEY_PATH = "google_sheets_service_account.json"
        DATA_DIR = "sample_data" # Ensure this directory exists with some .tbl files

    class MockTrainingDataPoint:
        def __init__(self, star_number, period_1, period_2, lc_category, time_series, flux_series):
            self.star_number = star_number
            self.period_1 = period_1
            self.period_2 = period_2
            self.lc_category = lc_category
            self.time_series = time_series
            self.flux_series = flux_series

    class MockDataProcessing:
        # Simple mock functions to avoid circular dependencies for testing
        @staticmethod
        def find_longest_x_campaign(data_array, threshold):
            return data_array
        @staticmethod
        def sort_data(data_array):
            return data_array
        @staticmethod
        def remove_y_outliers(data_array):
            return data_array

    # Monkey patch Config and TrainingDataPoint if running this file directly for testing
    # In a real application, you'd ensure these are properly imported and available.
    try:
        from config import Config
        from models import TrainingDataPoint
        from data_processing import find_longest_x_campaign, sort_data, remove_y_outliers
    except ImportError:
        print("Running with mock Config, TrainingDataPoint, and data_processing for testing. Ensure actual files exist for production.")
        Config = MockConfig
        TrainingDataPoint = MockTrainingDataPoint
        import sys
        # Temporarily add mock data_processing to a module for _load_star_data to find it
        sys.modules['data_processing'] = MockDataProcessing
        
    # Create a dummy sample_data directory and a dummy .tbl file for testing _load_star_data
    if not os.path.exists(Config.DATA_DIR):
        os.makedirs(Config.DATA_DIR)
    dummy_star_number = 12345
    dummy_tbl_file = os.path.join(Config.DATA_DIR, f"{dummy_star_number}-somefile.tbl")
    if not os.path.exists(dummy_tbl_file):
        with open(dummy_tbl_file, 'w') as f:
            f.write("# Header line 1\n")
            f.write("# Header line 2\n")
            f.write("# Header line 3\n")
            for i in range(10):
                f.write(f"{i*0.1}\t{np.sin(i*0.1)}\t{i}\n")
        print(f"Created dummy .tbl file: {dummy_tbl_file}")


    if Config.GOOGLE_SHEET_URL:
        try:
            loader = GoogleSheetsLoader()
            training_data = loader.extract_training_data()
            print(f"Successfully loaded {len(training_data)} training examples")
            
            # Show category distribution
            categories = loader.get_category_distribution()
            print("Category distribution:", categories)
            
            # Print a sample training point
            if training_data:
                print("\nSample TrainingDataPoint:")
                sample_point = training_data[0]
                print(f"  Star Number: {sample_point.star_number}")
                print(f"  Period 1: {sample_point.period_1}")
                print(f"  Period 2: {sample_point.period_2}")
                print(f"  LC Category: {sample_point.lc_category}")
                print(f"  Time Series Length: {len(sample_point.time_series)}")
                print(f"  Flux Series Length: {len(sample_point.flux_series)}")
                
        except Exception as e:
            print(f"Error testing Google Sheets loader: {e}")
    else:
        print("GOOGLE_SHEET_URL not set in Config.")