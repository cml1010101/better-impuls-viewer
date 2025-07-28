"""
CSV data loading for training data.
Handles CSV file loading and data extraction for model training.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import csv
import os
import argparse
import sys
from config import Config, CLASS_NAMES
from models import TrainingDataPoint


def parse_star_range(star_range: Union[str, List[int], None]) -> Optional[List[int]]:
    """
    Parse star range specification into a list of star numbers.
    
    Args:
        star_range: Can be:
            - None: Return None (extract all stars)
            - List[int]: Return as-is
            - String: Parse range like "30:50" to [30, 31, 32, ..., 50]
            
    Returns:
        List of star numbers or None
        
    Examples:
        parse_star_range("30:50") -> [30, 31, 32, ..., 50]
        parse_star_range("5:8") -> [5, 6, 7, 8]
        parse_star_range("42") -> [42]
        parse_star_range([1, 5, 10]) -> [1, 5, 10]
        parse_star_range(None) -> None
    """
    if star_range is None:
        return None
    
    if isinstance(star_range, list):
        return star_range
    
    if isinstance(star_range, str):
        star_range = star_range.strip()
        
        # Check if it's a range (contains colon)
        if ':' in star_range:
            try:
                start_str, end_str = star_range.split(':', 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                
                if start > end:
                    raise ValueError(f"Invalid range: start ({start}) must be <= end ({end})")
                
                return list(range(start, end + 1))
            except ValueError as e:
                raise ValueError(f"Invalid star range format '{star_range}': {e}")
        else:
            # Single star number
            try:
                return [int(star_range)]
            except ValueError:
                raise ValueError(f"Invalid star number '{star_range}'")
    
    raise ValueError(f"Invalid star range type: {type(star_range)}. Expected str, list, or None")


class CSVDataLoader:
    """Load and process training data from CSV files."""
    
    def __init__(self, csv_file_path: str):
        """Initialize with CSV file path."""
        self.csv_file_path = csv_file_path
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        try:
            df = pd.read_csv(self.csv_file_path)
            print(f"Loaded {len(df)} rows from CSV file: {self.csv_file_path}")
            return df
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            raise
    
    def extract_training_data(self, stars_to_extract: Union[str, List[int], None] = None) -> List[TrainingDataPoint]:
        """
        Extract training data from CSV file.
        
        Expected CSV format:
        - star_number: Integer star identifier
        - period_1: Primary period (float, -9 or NaN for no valid period)
        - period_2: Secondary period (float, -9 or NaN for no valid period, optional)
        - lc_category: Light curve category (string)
        - sensor: Sensor name (string, optional, defaults to 'csv')
        
        Args:
            stars_to_extract: Can be:
                - None: Extract all available stars
                - List[int]: Specific star numbers to extract
                - String: Range like "30:50" or single number like "42"
        
        Returns:
            List of TrainingDataPoint objects
        """
        df = self.load_raw_data()
        training_data = []
        
        # Parse star range specification
        parsed_stars = parse_star_range(stars_to_extract)
        if parsed_stars is not None:
            print(f"Extracting data for stars: {parsed_stars[:10]}{'...' if len(parsed_stars) > 10 else ''} ({len(parsed_stars)} total)")
        else:
            print("Extracting data for all available stars")
        
        # Validate required columns
        required_columns = ['star_number', 'period_1', 'lc_category']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV. Available columns: {list(df.columns)}")
        
        print(f"Processing {len(df)} rows from CSV...")
        
        for index, row in df.iterrows():
            try:
                star_number = int(row['star_number'])
                
                # Filter by star range if specified
                if parsed_stars is not None and star_number not in parsed_stars:
                    continue
                
                # Extract periods
                period_1 = self._extract_period(row['period_1'])
                period_2 = self._extract_period(row.get('period_2', -9))
                
                # Skip if no valid periods
                if period_1 is None:
                    continue
                
                # Get category and sensor
                lc_category = self._normalize_lc_category(str(row['lc_category']))
                sensor = str(row.get('sensor', 'csv'))
                
                # Load time series data for this star
                time_series, flux_series = self._load_star_data(star_number)
                
                # Skip if no valid time series data
                if len(time_series) == 0 or len(flux_series) == 0:
                    continue
                
                # Create training data point
                training_data.append(TrainingDataPoint(
                    star_number=star_number,
                    period_1=period_1,
                    period_2=period_2,
                    lc_category=lc_category,
                    time_series=time_series.tolist(),
                    flux_series=flux_series.tolist(),
                    sensor=sensor,
                    period_type='csv_provided',
                    period_confidence=0.8  # Default confidence for CSV-provided periods
                ))
                
            except Exception as e:
                print(f"Error processing row {index} (star {row.get('star_number', 'unknown')}): {e}")
                continue
        
        print(f"Extracted {len(training_data)} training examples from CSV")
        return training_data
    
    def _extract_period(self, period_value) -> Optional[float]:
        """Extract valid period from CSV value."""
        try:
            if pd.isna(period_value) or period_value == -9 or period_value == "-9" or period_value == "no":
                return None
            period_float = float(period_value)
            return period_float if period_float > 0 else None
        except (ValueError, TypeError):
            return None
    
    def _normalize_lc_category(self, category: str) -> str:
        """
        Normalize LC category strings to standard classifications.
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
            
            # Default fallback
            print(f"Warning: Unknown category '{category}', defaulting to 'stochastic'")
            return 'stochastic'
    
    def _load_star_data(self, star_number: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load time series data for a specific star from sample_data directory.
        """
        from data_processing import find_longest_x_campaign, sort_data, remove_y_outliers
        
        # Look for data files in sample_data directory
        sample_data_dir = Config.DATA_DIR
        
        if not os.path.exists(sample_data_dir):
            print(f"Sample data directory not found: {sample_data_dir}")
            return np.array([]), np.array([])
        
        # Find files for this star - try both formats
        star_files = []
        for file in os.listdir(sample_data_dir):
            # Try both 3-digit padded and unpadded star numbers
            if ((file.startswith(f"{str(star_number).zfill(3)}-") or 
                 file.startswith(f"{star_number}-")) and 
                file.endswith(".tbl")):
                star_files.append(file)
        
        if not star_files:
            return np.array([]), np.array([])
        
        # Use the first available file
        file_path = os.path.join(sample_data_dir, star_files[0])
        
        try:
            # Load data using pandas
            data = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, skiprows=[0, 1, 2])
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


# Example usage and testing
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Extract training data from CSV")
    parser.add_argument('--stars', type=str, help='Star range to extract (e.g., "30:50", "42", or comma-separated list "1,5,10")')
    parser.add_argument('--csv-input', type=str, required=True, help='CSV file containing training data')
    parser.add_argument('--test', action='store_true', help='Run test with mock data')
    
    args = parser.parse_args()
    
    # Parse stars argument
    stars_to_extract = None
    if args.stars:
        # Handle comma-separated lists
        if ',' in args.stars:
            try:
                stars_to_extract = [int(s.strip()) for s in args.stars.split(',')]
            except ValueError as e:
                print(f"Error parsing star list '{args.stars}': {e}")
                sys.exit(1)
        else:
            # Handle range or single star
            stars_to_extract = args.stars
    
    if args.test:
        print("Running test mode...")
        # Test star range parsing
        print("\nTesting star range parsing:")
        test_ranges = ["30:35", "42", "1,5,10", None]
        for test_range in test_ranges:
            try:
                parsed = parse_star_range(test_range)
                print(f"  '{test_range}' -> {parsed}")
            except ValueError as e:
                print(f"  '{test_range}' -> ERROR: {e}")
    
    try:
        print(f"Loading training data from CSV: {args.csv_input}")
        loader = CSVDataLoader(args.csv_input)
        
        print(f"Extracting training data...")
        if stars_to_extract:
            print(f"Stars to extract: {stars_to_extract}")
        training_data = loader.extract_training_data(stars_to_extract)
        print(f"Successfully loaded {len(training_data)} training examples from CSV")
        
        # Show category distribution
        categories = {}
        for point in training_data:
            cat = point.lc_category
            categories[cat] = categories.get(cat, 0) + 1
        
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
            print(f"  Sensor: {getattr(sample_point, 'sensor', 'N/A')}")
            print(f"  Period Type: {getattr(sample_point, 'period_type', 'N/A')}")
            
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        sys.exit(1)