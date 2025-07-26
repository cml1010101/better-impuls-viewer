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
from config import Config
from models import TrainingDataPoint


class GoogleSheetsLoader:
    """Load and process training data from Google Sheets."""
    
    def __init__(self, sheet_url: str = None):
        """Initialize with Google Sheets URL."""
        self.sheet_url = sheet_url or Config.GOOGLE_SHEET_URL
        if not self.sheet_url:
            raise ValueError("Google Sheets URL not provided")
    
    def _convert_to_csv_url(self, sheets_url: str) -> str:
        """Convert Google Sheets URL to CSV export URL."""
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
        """Load raw data from Google Sheets."""
        try:
            csv_url = self._convert_to_csv_url(self.sheet_url)
            response = requests.get(csv_url)
            response.raise_for_status()
            
            # Read CSV data into DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            print(f"Loaded {len(df)} rows from Google Sheets")
            return df
            
        except Exception as e:
            print(f"Error loading Google Sheets data: {e}")
            raise
    
    def extract_training_data(self) -> List[TrainingDataPoint]:
        """
        Extract training data from Google Sheets.
        
        Expected columns:
        - A: Star number
        - AK: Period 1 
        - AL: Period 2
        - AN: LC category
        
        Returns:
            List of TrainingDataPoint objects with valid periods
        """
        df = self.load_raw_data()
        training_data = []
        
        # Column mapping (convert to 0-based indexing)
        star_col = 0  # Column A
        period1_col = 36  # Column AK (A=0, K=10, AK=36) 
        period2_col = 37  # Column AL
        category_col = 39  # Column AN
        
        for index, row in df.iterrows():
            try:
                # Extract basic data
                star_number = int(row.iloc[star_col]) if pd.notna(row.iloc[star_col]) else None
                if star_number is None:
                    continue
                
                # Extract periods
                period_1 = row.iloc[period1_col] if pd.notna(row.iloc[period1_col]) else None
                period_2 = row.iloc[period2_col] if pd.notna(row.iloc[period2_col]) else None
                
                # Filter out invalid periods (-9 means no valid period)
                if period_1 == -9:
                    period_1 = None
                if period_2 == -9:
                    period_2 = None
                
                # Skip if no valid periods
                if period_1 is None and period_2 is None:
                    continue
                
                # Extract category
                lc_category = str(row.iloc[category_col]) if pd.notna(row.iloc[category_col]) else "unknown"
                
                # Load actual time series data for this star
                # This will be implemented to load from the sample_data directory
                time_series, flux_series = self._load_star_data(star_number)
                
                if len(time_series) > 0:
                    training_point = TrainingDataPoint(
                        star_number=star_number,
                        period_1=period_1,
                        period_2=period_2,
                        lc_category=lc_category,
                        time_series=time_series.tolist(),
                        flux_series=flux_series.tolist()
                    )
                    training_data.append(training_point)
                    
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue
        
        print(f"Extracted {len(training_data)} valid training examples")
        return training_data
    
    def _load_star_data(self, star_number: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load time series data for a specific star from sample_data directory.
        
        Args:
            star_number: Star identifier
            
        Returns:
            Tuple of (time_series, flux_series) arrays
        """
        import os
        from data_processing import find_longest_x_campaign, sort_data, remove_y_outliers
        
        # Look for data files in sample_data directory
        sample_data_dir = "/home/runner/work/better-impuls-viewer/better-impuls-viewer/sample_data"
        
        if not os.path.exists(sample_data_dir):
            print(f"Sample data directory not found: {sample_data_dir}")
            return np.array([]), np.array([])
        
        # Find files for this star
        star_files = []
        for file in os.listdir(sample_data_dir):
            if file.startswith(f"{star_number}-") and file.endswith(".tbl"):
                star_files.append(file)
        
        if not star_files:
            print(f"No data files found for star {star_number}")
            return np.array([]), np.array([])
        
        # Use the first available file (could be enhanced to combine multiple telescopes)
        file_path = os.path.join(sample_data_dir, star_files[0])
        
        try:
            # Load data using pandas
            data = pd.read_table(file_path, header=None, sep=r'\s+', skiprows=[0, 1, 2])
            data_array = data.to_numpy()
            
            # Process data using existing functions
            data_array = find_longest_x_campaign(data_array, 1.0)
            data_array = remove_y_outliers(data_array)
            data_array = sort_data(data_array)
            
            if len(data_array) > 0:
                return data_array[:, 0], data_array[:, 1]  # time, flux
            else:
                return np.array([]), np.array([])
                
        except Exception as e:
            print(f"Error loading data for star {star_number}: {e}")
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
    # Test the Google Sheets loader
    if Config.GOOGLE_SHEET_URL:
        loader = GoogleSheetsLoader()
        try:
            training_data = loader.extract_training_data()
            print(f"Successfully loaded {len(training_data)} training examples")
            
            # Show category distribution
            categories = loader.get_category_distribution()
            print("Category distribution:", categories)
            
        except Exception as e:
            print(f"Error testing Google Sheets loader: {e}")
    else:
        print("GOOGLE_SHEET_URL not set in environment variables")