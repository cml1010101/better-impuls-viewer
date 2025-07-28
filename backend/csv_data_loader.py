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
        """Load raw data from CSV file, ignoring headers and using column indices."""
        try:
            # Load CSV without headers, treating all rows as data
            df = pd.read_csv(self.csv_file_path, header=None)
            print(f"Loaded {len(df)} rows from CSV file: {self.csv_file_path}")
            return df
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            raise
    
    def extract_training_data(self, stars_to_extract: Union[str, List[int], None] = None) -> List[TrainingDataPoint]:
        """
        Extract training data from CSV file using hardcoded column indices like Google Sheets.
        
        Expected CSV format (no header, hardcoded column positions):
        - Column 0 (A): Star number
        - Columns 5-6 (F-G): CDIPS period 1 & 2
        - Columns 7-8 (H-I): ELEANOR period 1 & 2  
        - Columns 9-10 (J-K): QLP period 1 & 2
        - Columns 11-12 (L-M): SPOC period 1 & 2
        - Columns 13-14 (N-O): TESS 16 period 1 & 2
        - Columns 15-16 (P-Q): TASOC period 1 & 2
        - Columns 17-18 (R-S): TGLC period 1 & 2
        - Columns 19-20 (T-U): EVEREST period 1 & 2
        - Columns 21-22 (V-W): K2SC period 1 & 2
        - Columns 23-24 (X-Y): K2SFF period 1 & 2
        - Columns 25-26 (Z-AA): K2VARCAT period 1 & 2
        - Columns 27-28 (AB-AC): ZTF_R period 1 & 2
        - Columns 29-30 (AD-AE): ZTF_G period 1 & 2
        - Columns 31-32 (AF-AG): W1 period 1 & 2
        - Columns 33-34 (AH-AI): W2 period 1 & 2
        - Column 39 (AN): LC category
        
        Args:
            stars_to_extract: Can be:
                - None: Extract all available stars
                - List[int]: Specific star numbers to extract
                - String: Range like "30:50" or single number like "42"
        
        Returns:
            List of TrainingDataPoint objects with multiple entries per star (one per sensor)
        """
        df = self.load_raw_data()
        training_data = []
        
        # Parse star range specification
        parsed_stars = parse_star_range(stars_to_extract)
        if parsed_stars is not None:
            print(f"Extracting data for stars: {parsed_stars[:10]}{'...' if len(parsed_stars) > 10 else ''} ({len(parsed_stars)} total)")
        else:
            print("Extracting data for all available stars")
        
        # Define sensor mappings using hardcoded column indices (same as Google Sheets)
        sensors = {
            'cdips': (5, 6),      # Columns F-G
            'eleanor': (7, 8),    # Columns H-I
            'qlp': (9, 10),       # Columns J-K
            'spoc': (11, 12),     # Columns L-M
            't16': (13, 14),      # Columns N-O
            'tasoc': (15, 16),    # Columns P-Q
            'tglc': (17, 18),     # Columns R-S
            'everest': (19, 20),  # Columns T-U
            'k2sc': (21, 22),     # Columns V-W
            'k2sff': (23, 24),    # Columns X-Y
            'k2varcat': (25, 26), # Columns Z-AA
            'ztf_r': (27, 28),    # Columns AB-AC
            'ztf_g': (29, 30),    # Columns AD-AE
            'w1': (31, 32),       # Columns AF-AG
            'w2': (33, 34),       # Columns AH-AI
        }

        CATEGORY_COL_INDEX = 40  # Column for LC category (was column AN in Google Sheets, but we have it at index 40)
        
        print(f"Processing {len(df)} rows from CSV...")
        print(f"Processing multiple telescopes/sensors per star")
        
        for index, row in df.iterrows():
            try:
                # Skip if we don't have enough columns for star number
                if len(row) <= 0:
                    continue
                    
                star_number = int(row.iloc[0])  # Column A (index 0)
                
                # Filter by star range if specified
                if parsed_stars is not None and star_number not in parsed_stars:
                    continue
                
                # Get LC category (with safety check for column existence)
                lc_category = 'stochastic'  # Default fallback
                if len(row) > CATEGORY_COL_INDEX:
                    try:
                        lc_category = self._normalize_lc_category(str(row.iloc[CATEGORY_COL_INDEX]))
                    except:
                        print(f"Warning: Could not normalize category for star {star_number}, defaulting to 'stochastic'")
                        lc_category = 'stochastic'
                
                # Load time series data for this star
                time_series, flux_series = self._load_star_data(star_number)
                
                # Skip if no valid time series data
                if len(time_series) == 0 or len(flux_series) == 0:
                    continue
                
                # Process each sensor's data to extract training samples
                for sensor, (period1_col_idx, period2_col_idx) in sensors.items():
                    try:
                        # Skip if we don't have enough columns for this sensor
                        if len(row) <= max(period1_col_idx, period2_col_idx):
                            continue
                            
                        # Get correct periods from CSV columns
                        correct_periods = self._extract_correct_periods_by_index(row, period1_col_idx, period2_col_idx)
                        
                        if len(correct_periods) == 0:
                            continue  # Skip this sensor if no valid periods
                        
                        # Generate 5 periods for training per light curve (same as Google Sheets approach)
                        training_periods = self._generate_training_periods(
                            time_series, flux_series, correct_periods
                        )
                        
                        # Create training data points for each period
                        for period_info in training_periods:
                            period_value = period_info['period']
                            period_type = period_info['type']  # 'correct', 'periodogram_peak', or 'random'
                            confidence = period_info['confidence']
                            
                            training_data.append(TrainingDataPoint(
                                star_number=star_number,
                                period_1=period_value,
                                period_2=None,  # Store as single period for training
                                lc_category=lc_category,
                                time_series=time_series.tolist(),
                                flux_series=flux_series.tolist(),
                                # Add metadata for training
                                sensor=sensor,
                                period_type=period_type,
                                period_confidence=confidence
                            ))
                            
                    except Exception as e:
                        print(f"Error processing sensor {sensor} for star {star_number}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue
        
        print(f"Extracted {len(training_data)} training examples from CSV with multiple telescopes/sensors")
        return training_data
    
    def _extract_correct_periods_by_index(self, row: pd.Series, period1_col_idx: int, period2_col_idx: int) -> List[float]:
        """Extract valid correct periods from a CSV row using column indices."""
        correct_periods = []
        
        # Extract period 1
        try:
            period1 = row.iloc[period1_col_idx]
            if not pd.isna(period1) and period1 != -9 and period1 != "-9" and period1 != "no":
                period1_float = float(period1)
                if period1_float > 0:
                    correct_periods.append(period1_float)
        except (ValueError, TypeError, IndexError):
            pass
        
        # Extract period 2
        try:
            period2 = row.iloc[period2_col_idx]
            if not pd.isna(period2) and period2 != -9 and period2 != "-9" and period2 != "no":
                period2_float = float(period2)
                if period2_float > 0:
                    correct_periods.append(period2_float)
        except (ValueError, TypeError, IndexError):
            pass
        
        return correct_periods
    
    def _generate_training_periods(self, time_series: np.ndarray, flux_series: np.ndarray, correct_periods: List[float]) -> List[Dict]:
        """
        Generate 5 training periods for each light curve (same strategy as Google Sheets):
        - 1-2 correct periods (from CSV data)
        - 2 peaks from periodogram that are not correct
        - 2 random periods
        """
        from period_detection import calculate_lomb_scargle
        
        training_periods = []
        
        # Add correct periods
        for period in correct_periods[:2]:  # Maximum 2 correct periods
            training_periods.append({
                'period': period,
                'type': 'correct',
                'confidence': 0.9
            })
        
        try:
            # Generate periodogram to find other peaks
            data_array = np.column_stack((time_series, flux_series))
            frequencies, power = calculate_lomb_scargle(data_array)
            periods = 1.0 / frequencies
            
            # Find peaks in periodogram (excluding correct periods)
            periodogram_peaks = []
            for i in range(1, len(power) - 1):
                if power[i] > power[i-1] and power[i] > power[i+1]:
                    period = periods[i]
                    # Exclude periods too close to correct ones
                    if not any(abs(period - cp) / cp < 0.1 for cp in correct_periods):
                        periodogram_peaks.append((period, power[i]))
            
            # Sort by power and take top peaks
            periodogram_peaks.sort(key=lambda x: x[1], reverse=True)
            
            # Add up to 2 periodogram peaks
            for period, power_val in periodogram_peaks[:2]:
                if len(training_periods) < 4:  # Leave room for random periods
                    training_periods.append({
                        'period': period,
                        'type': 'periodogram_peak',
                        'confidence': 0.3
                    })
        
        except Exception as e:
            print(f"Error generating periodogram: {e}")
        
        # Fill remaining slots with random periods
        min_period = 0.5
        max_period = min(50.0, (time_series[-1] - time_series[0]) / 4.0)
        
        while len(training_periods) < 5:
            random_period = np.random.uniform(min_period, max_period)
            # Ensure it's not too close to existing periods
            if not any(abs(random_period - tp['period']) / tp['period'] < 0.1 for tp in training_periods):
                training_periods.append({
                    'period': random_period,
                    'type': 'random',
                    'confidence': 0.1
                })
        
        return training_periods[:5]  # Ensure exactly 5 periods
    
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