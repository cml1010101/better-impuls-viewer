"""
CSV-based training data loader for Better Impuls Viewer.
Replaces Google Sheets integration with CSV file upload functionality.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import csv
import os
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


class CSVTrainingDataLoader:
    """Load and process training data from uploaded CSV files."""
    
    def __init__(self, csv_file_path: str):
        """Initialize with CSV file path."""
        self.csv_file_path = csv_file_path
        if not os.path.exists(csv_file_path):
            raise ValueError(f"CSV file not found: {csv_file_path}")
        
        # Validate CSV format
        self._validate_csv_format()
    
    def _validate_csv_format(self) -> None:
        """Validate that the CSV has the expected format."""
        try:
            df = pd.read_csv(self.csv_file_path)
            
            # Check if we have the minimum required columns
            required_columns = ['Star', 'LC_Category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"CSV missing required columns: {missing_columns}")
            
            # Check for period columns (at least one sensor should have period data)
            period_columns = [col for col in df.columns if 'period' in col.lower()]
            if len(period_columns) < 2:  # Should have at least period_1 and period_2 for one sensor
                raise ValueError("CSV should contain period columns (e.g., 'CDIPS_period_1', 'CDIPS_period_2', etc.)")
            
            print(f"CSV validation successful. Found {len(df)} rows and {len(df.columns)} columns.")
            print(f"Available period columns: {period_columns}")
            
        except Exception as e:
            raise ValueError(f"Invalid CSV format: {e}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        try:
            df = pd.read_csv(self.csv_file_path)
            print(f"Loaded {len(df)} rows from CSV file")
            return df
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            raise
    
    def extract_training_data(self, stars_to_extract: Union[str, List[int], None] = None) -> List[TrainingDataPoint]:
        """
        Extract training data from CSV file.
        
        For each light curve, generates 5 training samples:
        - 1-2 correct periods (from CSV data)
        - 2 peaks from periodogram that are not correct
        - 2 random periods
        
        Args:
            stars_to_extract: Can be:
                - None: Extract all available stars
                - List[int]: Specific star numbers to extract
                - String: Range like "30:50" or single number like "42"
        
        Expected CSV columns:
        - Star: Star number/identifier
        - LC_Category: Light curve category
        - {Sensor}_period_1: First period for each sensor (e.g., CDIPS_period_1)
        - {Sensor}_period_2: Second period for each sensor (e.g., CDIPS_period_2)
        
        Supported sensors: CDIPS, ELEANOR, QLP, SPOC, TESS16, TASOC, TGLC, EVEREST,
                          K2SC, K2SFF, K2VARCAT, ZTF_R, ZTF_G, W1, W2
        
        Returns:
            List of TrainingDataPoint objects with 5 periods per light curve
        """
        df = self.load_raw_data()
        training_data = []
        
        # Parse star range specification
        parsed_stars = parse_star_range(stars_to_extract)
        if parsed_stars is not None:
            print(f"Extracting data for stars: {parsed_stars[:10]}{'...' if len(parsed_stars) > 10 else ''} ({len(parsed_stars)} total)")
        else:
            print("Extracting data for all available stars")
        
        # Detect sensor columns dynamically
        sensors = self._detect_sensor_columns(df)
        print(f"Detected sensors: {list(sensors.keys())}")
        
        print(f"Processing {len(df)} rows from CSV...")
        print(f"Generating 5 periods per light curve (correct + periodogram peaks + random)")
        
        for index, row in df.iterrows():
            try:
                # Get star number
                star_number = row['Star']
                if isinstance(star_number, str):
                    try:
                        star_number = int(star_number)
                    except ValueError:
                        print(f"Warning: Invalid star number '{star_number}' at row {index}, skipping")
                        continue
                
                if parsed_stars is not None and star_number not in parsed_stars:
                    continue
                
                # Normalize LC category
                try:
                    lc_category = self._normalize_lc_category(row['LC_Category'])
                except:
                    print(f"Warning: Could not normalize category for star {star_number}, defaulting to 'stochastic'")
                    lc_category = 'stochastic'
                
                # Load time series data for this star
                time_series, flux_series = self._load_star_data(star_number)
                
                # Skip if no valid time series data
                if len(time_series) == 0 or len(flux_series) == 0:
                    continue
                
                # Process each sensor's data to extract training samples
                for sensor, (period1_col, period2_col) in sensors.items():
                    try:
                        # Get correct periods from CSV
                        correct_periods = self._extract_correct_periods(row, period1_col, period2_col)
                        
                        if len(correct_periods) == 0:
                            continue  # Skip this sensor if no valid periods
                        
                        # Generate 5 periods for training per light curve
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
        
        print(f"Extracted {len(training_data)} training examples with 5-period strategy")
        return training_data
    
    def _detect_sensor_columns(self, df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
        """Detect sensor columns in the CSV based on column names."""
        sensors = {}
        
        # List of known sensors
        known_sensors = [
            'CDIPS', 'ELEANOR', 'QLP', 'SPOC', 'TESS16', 'TASOC', 'TGLC', 
            'EVEREST', 'K2SC', 'K2SFF', 'K2VARCAT', 'ZTF_R', 'ZTF_G', 'W1', 'W2'
        ]
        
        for sensor in known_sensors:
            # Look for columns like {SENSOR}_period_1 and {SENSOR}_period_2
            period1_col = f"{sensor}_period_1"
            period2_col = f"{sensor}_period_2"
            
            # Also check for alternative naming conventions
            alt_period1_col = f"{sensor.lower()}_period_1"
            alt_period2_col = f"{sensor.lower()}_period_2"
            
            if period1_col in df.columns and period2_col in df.columns:
                sensors[sensor.lower()] = (period1_col, period2_col)
            elif alt_period1_col in df.columns and alt_period2_col in df.columns:
                sensors[sensor.lower()] = (alt_period1_col, alt_period2_col)
        
        return sensors
    
    def _extract_correct_periods(self, row: pd.Series, period1_col: str, period2_col: str) -> List[float]:
        """Extract valid correct periods from a CSV row."""
        correct_periods = []
        
        # Extract period 1
        try:
            period1 = row[period1_col]
            if not pd.isna(period1) and period1 != -9 and period1 != "-9" and period1 != "no" and period1 != "":
                period1_float = float(period1)
                if period1_float > 0:
                    correct_periods.append(period1_float)
        except (ValueError, TypeError, KeyError):
            pass
        
        # Extract period 2
        try:
            period2 = row[period2_col]
            if not pd.isna(period2) and period2 != -9 and period2 != "-9" and period2 != "no" and period2 != "":
                period2_float = float(period2)
                if period2_float > 0:
                    correct_periods.append(period2_float)
        except (ValueError, TypeError, KeyError):
            pass
        
        return correct_periods
    
    def _generate_training_periods(self, time_series: np.ndarray, flux_series: np.ndarray, 
                                 correct_periods: List[float]) -> List[Dict]:
        """
        Generate 5 training periods per light curve:
        - 1-2 correct periods (high confidence)
        - 2 periodogram peaks that are not correct (medium confidence)
        - 2 random periods (low confidence)
        """
        training_periods = []
        
        # 1. Add correct periods (1-2 samples)
        for i, period in enumerate(correct_periods[:2]):  # Max 2 correct periods
            training_periods.append({
                'period': period,
                'type': 'correct',
                'confidence': np.random.uniform(0.85, 0.95)  # High confidence for correct periods
            })
        
        # 2. Generate periodogram peaks that are not correct
        periodogram_peaks = self._find_incorrect_periodogram_peaks(
            time_series, flux_series, correct_periods, n_peaks=2
        )
        
        for period in periodogram_peaks:
            training_periods.append({
                'period': period,
                'type': 'periodogram_peak',
                'confidence': np.random.uniform(0.3, 0.6)  # Medium confidence for periodogram peaks
            })
        
        # 3. Generate random periods
        random_periods = self._generate_random_periods(time_series, n_periods=2)
        
        for period in random_periods:
            training_periods.append({
                'period': period,
                'type': 'random',
                'confidence': np.random.uniform(0.05, 0.25)  # Low confidence for random periods
            })
        
        # Ensure we have exactly 5 periods by padding if necessary
        while len(training_periods) < 5:
            # Add more random periods if we don't have enough
            extra_random = self._generate_random_periods(time_series, n_periods=1)[0]
            training_periods.append({
                'period': extra_random,
                'type': 'random',
                'confidence': np.random.uniform(0.05, 0.25)
            })
        
        # Limit to exactly 5 periods
        return training_periods[:5]
    
    def _find_incorrect_periodogram_peaks(self, time_series: np.ndarray, flux_series: np.ndarray, 
                                        correct_periods: List[float], n_peaks: int = 2) -> List[float]:
        """Find periodogram peaks that are not the correct periods."""
        try:
            # Import period detection functions
            from period_detection import find_periodogram_periods
            
            # Create data array for periodogram analysis
            data_array = np.column_stack([time_series, flux_series])
            
            # Find top periodogram peaks
            period_power_pairs = find_periodogram_periods(data_array, top_n=20)
            
            if len(period_power_pairs) == 0:
                # Fallback to random periods if periodogram fails
                return self._generate_random_periods(time_series, n_peaks)
            
            incorrect_peaks = []
            tolerance = 0.1  # 10% tolerance for considering periods "close"
            
            for period, power in period_power_pairs:
                # Check if this period is too close to any correct period
                is_close_to_correct = False
                for correct_period in correct_periods:
                    if abs(period - correct_period) / correct_period < tolerance:
                        is_close_to_correct = True
                        break
                
                # Add to incorrect peaks if not close to correct periods
                if not is_close_to_correct:
                    incorrect_peaks.append(period)
                    if len(incorrect_peaks) >= n_peaks:
                        break
            
            # If we don't have enough incorrect peaks, fill with random
            while len(incorrect_peaks) < n_peaks:
                random_period = self._generate_random_periods(time_series, n_periods=1)[0]
                incorrect_peaks.append(random_period)
            
            return incorrect_peaks[:n_peaks]
            
        except Exception as e:
            print(f"Error finding periodogram peaks: {e}, using random periods instead")
            return self._generate_random_periods(time_series, n_peaks)
    
    def _generate_random_periods(self, time_series: np.ndarray, n_periods: int = 2) -> List[float]:
        """Generate random periods within reasonable astronomical ranges."""
        time_span = np.max(time_series) - np.min(time_series)
        
        # Generate random periods between 0.1 days and min(time_span/3, 50 days)
        min_period = 0.1
        max_period = min(time_span / 3.0, 50.0)
        
        if max_period <= min_period:
            max_period = min_period + 1.0
        
        random_periods = []
        for _ in range(n_periods):
            # Use log-uniform distribution to better sample period space
            log_min = np.log10(min_period)
            log_max = np.log10(max_period)
            log_period = np.random.uniform(log_min, log_max)
            period = 10 ** log_period
            random_periods.append(period)
        
        return random_periods
    
    def _normalize_lc_category(self, category: str) -> str:
        """
        Normalize LC category strings to standard classifications.
        
        Maps various category strings from CSV to the 14 main types:
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
        if pd.isna(category):
            return 'stochastic'
        
        category = str(category).lower().strip()
        
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
        Load time series data for a specific star from data directory.
        
        Args:
            star_number: Star identifier
            
        Returns:
            Tuple of (time_series, flux_series) arrays
        """
        from data_processing import find_longest_x_campaign, sort_data, remove_y_outliers
        
        # Get data folder from config
        sample_data_dir = Config.get_data_folder()
        
        if not os.path.exists(sample_data_dir):
            print(f"Data directory not found: {sample_data_dir}")
            return np.array([]), np.array([])
        
        # Find files for this star
        star_files = []
        for file in os.listdir(sample_data_dir):
            if file.startswith(f"{str(star_number).zfill(3)}-") and file.endswith(".tbl"):
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
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of LC categories in the training data."""
        training_data = self.extract_training_data()
        categories = {}
        
        for point in training_data:
            category = point.lc_category
            categories[category] = categories.get(category, 0) + 1
        
        return categories
    
    def export_training_data_to_csv(self, output_dir: str = "ml-dataset", 
                                   stars_to_extract: Union[str, List[int], None] = None) -> str:
        """
        Export training data to CSV format for external analysis and training.
        
        Creates CSV files containing phase-folded light curves with corresponding
        category and confidence information suitable for machine learning training.
        
        Args:
            output_dir: Directory to save CSV files
            stars_to_extract: Can be:
                - None: Export all available stars
                - List[int]: Specific star numbers to export
                - String: Range like "30:50" or single number like "42"
            
        Returns:
            Path to the exported CSV file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract training data using existing method
        print("Extracting training data from CSV...")
        training_data = self.extract_training_data(stars_to_extract)
        
        if len(training_data) == 0:
            raise ValueError("No training data to export")
        
        # Generate phase-folded data for each training example
        csv_rows = []
        processed_count = 0
        
        print(f"Processing {len(training_data)} training examples for CSV export...")
        
        for data_point in training_data:
            try:
                # Create data array for phase folding
                time_series = np.array(data_point.time_series)
                flux_series = np.array(data_point.flux_series)
                data_array = np.column_stack([time_series, flux_series])
                
                # Skip if period is invalid
                if data_point.period_1 is None or data_point.period_1 <= 0:
                    continue
                
                # Phase-fold the data
                from period_detection import phase_fold_data
                folded_curve = phase_fold_data(data_array, data_point.period_1, n_bins=100)
                
                # Skip if folding failed
                if np.all(folded_curve == 0) or np.all(np.isnan(folded_curve)):
                    continue
                
                # Create phase array (0 to 1)
                phases = np.linspace(0, 1, len(folded_curve))
                
                # Create row data with metadata
                base_row = {
                    'star_number': data_point.star_number,
                    'period': data_point.period_1,
                    'lc_category': data_point.lc_category,
                    'sensor': data_point.sensor or 'unknown',
                    'period_type': data_point.period_type or 'unknown',
                    'period_confidence': data_point.period_confidence or 0.5
                }
                
                # Add phase-folded data points
                for phase, flux in zip(phases, folded_curve):
                    row = base_row.copy()
                    row.update({
                        'phase': phase,
                        'flux': flux
                    })
                    csv_rows.append(row)
                
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count}/{len(training_data)} training examples...")
                    
            except Exception as e:
                print(f"Error processing training example for star {data_point.star_number}: {e}")
                continue
        
        if len(csv_rows) == 0:
            raise ValueError("No valid phase-folded data generated for CSV export")
        
        # Write to CSV file
        csv_filename = f"training_data_{len(training_data)}_examples.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        fieldnames = ['star_number', 'period', 'lc_category', 'sensor', 'period_type', 
                     'period_confidence', 'phase', 'flux']
        
        print(f"Writing {len(csv_rows)} rows to {csv_path}...")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"Successfully exported training data to: {csv_path}")
        print(f"Total rows: {len(csv_rows)}")
        print(f"Unique training examples: {processed_count}")
        
        return csv_path