from dataclasses import dataclass
import numpy as np

def calculate_dynamic_campaign_threshold(data: np.ndarray, multiplier: float = 5.0) -> float:
    """
    Calculate a dynamic threshold for campaign detection based on data characteristics.
    
    The threshold is determined by analyzing the distribution of time gaps between
    consecutive observations. Campaigns are separated when gaps are significantly
    larger than the typical observational cadence.
    
    Args:
        data (np.ndarray): A NumPy array of shape (n, 2) where each row
                           is a (x, y) coordinate pair.
        multiplier (float): Factor to multiply the median gap to determine the threshold.
                           Higher values create fewer, longer campaigns.
    
    Returns:
        float: The calculated threshold for campaign detection.
    """
    if data.shape[0] < 2:
        return 1.0  # Default threshold for minimal data
    
    # Calculate gaps between consecutive time points
    time_gaps = np.diff(data[:, 0])
    
    # Remove negative gaps (shouldn't happen with sorted data)
    time_gaps = time_gaps[time_gaps > 0]
    
    if len(time_gaps) == 0:
        return 1.0
    
    # Use the median gap as the base measure
    median_gap = np.median(time_gaps)
    
    # Use multiplier to determine campaign separation threshold
    # Typical astronomical observations have regular cadence within campaigns
    # but large gaps between campaigns
    threshold = median_gap * multiplier
    
    # Ensure threshold is reasonable (not too small or too large)
    threshold = max(threshold, median_gap * 2)  # At least 2x median
    threshold = min(threshold, np.max(time_gaps) * 0.5)  # At most half the largest gap
    
    return threshold

def find_all_campaigns(data: np.ndarray, x_threshold: float = None) -> list[np.ndarray]:
    """
    Finds all campaigns (series of consecutive data points) where the x-values
    are "close" to each other within a specified or dynamically calculated threshold.

    A "campaign" is defined as a sequence of points where the absolute
    difference between the x-value of the current point and the x-value
    of the previous point is less than or equal to the x_threshold.

    Args:
        data (np.ndarray): A NumPy array of shape (n, 2) where each row
                           is a (x, y) coordinate pair.
        x_threshold (float, optional): The maximum allowed absolute difference between
                             consecutive x-values for them to be considered
                             "close" and part of the same campaign. If None,
                             a dynamic threshold will be calculated.

    Returns:
        List[np.ndarray]: A list of NumPy arrays, each representing a campaign.
                         Returns an empty list if no data or no campaigns are found.
    """
    if data.shape[0] == 0:
        print("Input data is empty.")
        return []

    if data.shape[0] == 1:
        print("Input data has only one point. Returning the point as the only campaign.")
        return [data]

    # Calculate dynamic threshold if not provided
    if x_threshold is None:
        x_threshold = calculate_dynamic_campaign_threshold(data)
        print(f"Using dynamic campaign threshold: {x_threshold:.3f}")

    campaigns = []
    current_campaign = np.array([data[0]]) # Start with the first point

    # Iterate through the data starting from the second point
    for i in range(1, data.shape[0]):
        # Check if the current point's x-value is close to the previous point's x-value
        if np.abs(data[i, 0] - data[i-1, 0]) <= x_threshold:
            # If close, add the point to the current campaign
            current_campaign = np.vstack((current_campaign, data[i]))
        else:
            # If not close, the current campaign ends.
            # Add the current campaign to the list if it has data
            if len(current_campaign) > 0:
                campaigns.append(current_campaign)

            # Start a new campaign with the current point
            current_campaign = np.array([data[i]])

    # After the loop, add the last campaign if it has data
    if len(current_campaign) > 0:
        campaigns.append(current_campaign)

    return campaigns

def remove_y_outliers(data: np.ndarray, iqr_multiplier: float = 3.0) -> np.ndarray:
    """
    Removes outliers from the y-values of the input data using the
    Interquartile Range (IQR) method.

    Points are considered outliers if their y-value is below
    Q1 - iqr_multiplier * IQR or above Q3 + iqr_multiplier * IQR,
    where Q1 and Q3 are the first and third quartiles, and IQR is the
    Interquartile Range (Q3 - Q1).

    Args:
        data (np.ndarray): A NumPy array of shape (n, 2) where each row
                           is a (x, y) coordinate pair.
        iqr_multiplier (float): The multiplier for the IQR to define the
                               outlier bounds. A common value for "mild"
                               outliers is 1.5, while 3.0 is often used
                               for "extreme" outliers.

    Returns:
        np.ndarray: A new NumPy array with outliers removed.
                    Returns an empty array if the input data is empty
                    or if all points are removed as outliers.
    """
    if data.shape[0] == 0:
        print("Input data is empty. No outliers to remove.")
        return np.array([])
    
    if data.shape[1] != 2:
        raise ValueError("Input data must be a 2D array with two columns (x and y).")

    y_data = data[:, 1]

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(y_data, 25)
    Q3 = np.percentile(y_data, 75)

    # Calculate Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    # Filter data: keep only points where y-value is within the bounds
    filtered_indices = np.where((y_data >= lower_bound) & (y_data <= upper_bound))
    filtered_data = data[filtered_indices]

    if filtered_data.shape[0] == 0:
        print("All data points were identified as outliers and removed.")

    return filtered_data

def distribute_uniformly(data: np.ndarray) -> np.ndarray:
    x = data[:, 0]
    y = data[:, 1]
    uniform_x = np.linspace(np.min(x), np.max(x), num=len(x))
    uniform_y = np.interp(uniform_x, x, y)
    return uniform_y


from astropy.timeseries import LombScargle

def calculate_lomb_scargle(
    data: np.ndarray, samples_per_peak: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Lomb-Scargle periodogram of the given data.

    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where the first column is the x-axis data (time) and the second column is the y-axis data (flux).
    samples_per_peak : int, optional
        The number of frequency samples to use across each significant peak.
        Increasing this value can help resolve plateaus into sharper peaks
        by providing a denser frequency grid. Default is 10.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - frequency (np.ndarray): The frequencies at which the periodogram was calculated.
        - power (np.ndarray): The Lomb-Scargle power at each frequency.
    """
    if data.shape[1] != 2:
        raise ValueError("Input data must be a 2D array with two columns (x and y).")

    time = data[:, 0]
    flux = data[:, 1]

    # Calculate the time span to determine appropriate frequency range
    time_span = np.max(time) - np.min(time)

    # Set frequency range to cover periods from 0.1 days to 2 * time_span
    # but limit the maximum period to avoid very low frequency noise
    max_period = min(2 * time_span, 50.0)  # Cap at 50 days
    min_period = 0.1  # Minimum period of 0.1 days

    min_frequency = 1.0 / max_period
    max_frequency = 1.0 / min_period

    # Calculate Lomb-Scargle periodogram with explicit frequency range
    ls = LombScargle(time, flux)
    frequency, power = ls.autopower(
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        samples_per_peak=samples_per_peak,
    )

    return frequency, power

from scipy.signal import find_peaks

def generate_candidate_periods(frequencies: np.ndarray, powers: np.ndarray, num_candidates: int = 4) -> list[float]:
    """Generate multiple period candidates based on the Lomb-Scargle power spectrum."""
    # Find peaks in the power spectrum
    peaks, _ = find_peaks(powers, height=0.1 * np.max(powers))
    
    if len(peaks) == 0:
        print("No significant peaks found in the power spectrum.")
        return []

    # Sort peaks by power
    sorted_peak_indices = np.argsort(powers[peaks])[::-1]
    top_peaks = sorted_peak_indices[:num_candidates]
    return [1.0 / frequencies[idx] for idx in top_peaks]

def phase_fold_data(time: np.ndarray, flux: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Phase fold the light curve data."""
    phase = (time % period) / period
    
    # Sort by phase
    sort_indices = np.argsort(phase)
    phase_sorted = phase[sort_indices]
    flux_sorted = flux[sort_indices]
    
    return phase_sorted, flux_sorted

@dataclass
class Campaign:
    data: np.ndarray
    length: float
    periodogram: tuple[np.ndarray, np.ndarray]
    best_folded_data: list[tuple[float, np.ndarray]]

@dataclass
class LightCurve:
    raw_data: np.ndarray
    campaigns: list[Campaign]

@dataclass
class Star:
    star_number: int
    coordinates: tuple[float, float]
    name: str | None = None
    surveys: dict[str, LightCurve] = None

def create_star_from_raw_light_curve(star_number: int, coordinates: tuple[float, float], raw_light_curves: dict[str, np.ndarray], name: str | None = None, *, min_length: int) -> Star:
    """
    Creates a Star object from raw light curve data.

    Args:
        star_number (int): The identifier for the star.
        coordinates (tuple): The (RA, Dec) coordinates of the star.
        raw_light_curve (np.ndarray): A NumPy array of shape (n, 2) where each row is a (x, y) coordinate pair.
        name (str, optional): The name of the star. Defaults to None.

    Returns:
        Star: A Star object containing the provided information.
    """
    survey_data = {}
    for survey_name, raw_light_curve in raw_light_curves.items():
        campaign_data = find_all_campaigns(raw_light_curve)
        campaigns = []
        for campaign in campaign_data:
            cleaned_data = remove_y_outliers(campaign)
            if cleaned_data.shape[0] < min_length:
                continue
            frequency, power = calculate_lomb_scargle(cleaned_data)
            best_periods = generate_candidate_periods(frequency, power)
            folded_data = []
            for period in best_periods:
                phase, flux = phase_fold_data(cleaned_data[:, 0], cleaned_data[:, 1], period)
                folded_data.append((period, distribute_uniformly(np.column_stack((phase, flux)))))
            campaigns.append(Campaign(
                data=cleaned_data,
                length=cleaned_data[-1, 0] - cleaned_data[0, 0],
                periodogram=(frequency, power),
                best_folded_data=folded_data    
            ))
        survey_data[survey_name] = LightCurve(
            raw_data=raw_light_curve,
            campaigns=campaigns
        )
    return Star(
        star_number=star_number,
        coordinates=coordinates,
        name=name,
        surveys=survey_data
    )


from torch.utils.data import Dataset
import torch

@dataclass
class StarTrainingSample:
    inputs: list[tuple[float, torch.Tensor]]
    correct_period: float | None = None
    correct_category: int | None = None

import os
import pandas as pd
from astropy.coordinates import SkyCoord, Angle

class StarDataset(Dataset):
    def __init__(self, stars: list[Star] = []):
        self.stars = stars

    def save_to_file(self, filepath: str):
        torch.save(self.stars, filepath, pickle_protocol=4)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "StarDataset":
        stars = torch.load(filepath, weights_only=False)
        return cls(stars)

    @classmethod
    def str_to_coords(cls, coords_str: str) -> SkyCoord:
        """Convert a string representation of coordinates to SkyCoord."""
        ra_str = coords_str[:9]
        dec_str = coords_str[9:]
        ra_hr = ra_str[:2]
        ra_min = ra_str[2:4]
        ra_sec = ra_str[4:]
        dec_deg = dec_str[:3]
        dec_min = dec_str[3:5]
        dec_sec = dec_str[5:]
        ra = Angle(f"{ra_hr}h{ra_min}m{ra_sec}s")
        dec = Angle(f"{dec_deg}d{dec_min}m{dec_sec}s")
        return SkyCoord(ra=ra, dec=dec, unit='deg')
    
    @classmethod
    def get_path_to_lc(cls, path_to_pinput: str, star_number: int, survey: str):
        filename = f"{star_number:03d}-{survey}.tbl"
        return f"{path_to_pinput}/{filename}"

    @classmethod
    def load_tbl_file(cls, filepath: str) -> np.ndarray:
        """Load a .tbl file and return its contents.
        This is a placeholder implementation. Replace with actual file reading logic.
        """
        data = pd.read_table(filepath, header=None, sep=r'\s+', skiprows=[0, 1, 2])
        data_array = data.to_numpy()
        
        # If data has more than 2 columns, use only the first 2 (time, flux)
        if data_array.shape[1] > 2:
            result = data_array[:, :2]
        else:
            result = data_array
        
        return result

    @classmethod
    def load_from_folder(cls, path: str):
        """Load the star list from a file."""
        stars = []
        # path should be the directory containing impuls_stars.csv
        csv_path = os.path.join(path, 'impuls_stars.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"impuls_stars.csv not found in {path}")
            
        data = pd.read_csv(csv_path, header=None, names=['star_number', 'name', 'coordinates'])
        for _, row in data.iterrows():
            # Handle different star number formats (e.g., "1.1" -> 1)
            star_number_str = str(row['star_number'])
            if '.' in star_number_str:
                star_number = int(float(star_number_str))
            else:
                star_number = int(star_number_str)
            name = row['name'] if pd.notna(row['name']) else None
            coordinates = cls.str_to_coords(row['coordinates'])
            surveys = {}
            for survey in [
                'cdips', 'eleanor', 'qlp', 'spoc', 't16', 'tasoc', 'tglc',
                'everest', 'k2sc', 'k2sff', 'k2varcat', 'ztf-r', 'ztf-g',
                'neowise-1', 'neowise-2']:
                lc_path = cls.get_path_to_lc(path, star_number, survey)
                if os.path.exists(lc_path):
                    raw_light_curve = cls.load_tbl_file(lc_path)
                    surveys[survey] = raw_light_curve
            if len(surveys) == 0:
                continue
            star = create_star_from_raw_light_curve(
                star_number=star_number,
                coordinates=(coordinates.ra.deg, coordinates.dec.deg),
                raw_light_curves=surveys,
                name=name,
                min_length=50
            )
            stars.append(star)
        return cls(stars)

    def __len__(self):
        return len(self.stars)

    def __getitem__(self, idx) -> StarTrainingSample:
        assert 0 <= idx < len(self.stars)
        star: Star = self.stars[idx]
        inputs = []
        for survey_name, light_curve in star.surveys.items():
            for campaign in light_curve.campaigns:
                for period, folded_data in campaign.best_folded_data:
                    inputs.append((period, torch.tensor(folded_data, dtype=torch.float32)))
        return StarTrainingSample(inputs=inputs)
    
    def get_star(self, star_number: int) -> Star | None:
        for star in self.stars:
            if star.star_number == star_number:
                return star
        return None

if __name__ == "__main__":
    import argparse
    import sys
    
    def main():
        parser = argparse.ArgumentParser(description="Dataset utilities for astronomical data processing")
        parser.add_argument('command', choices=['convert', 'info'], 
                          help='Command to execute')
        parser.add_argument('--input', '-i', required=True,
                          help='Input directory path containing data files')
        parser.add_argument('--output', '-o', 
                          help='Output dataset file path (default: input_dir/stars_dataset.pkl)')
        parser.add_argument('--min-length', type=int, default=50,
                          help='Minimum campaign length for processing (default: 50)')
        
        args = parser.parse_args()
        
        if args.command == 'convert':
            # Convert folder to dataset file
            print(f"Loading data from folder: {args.input}")
            try:
                dataset = StarDataset.load_from_folder(args.input)
                print(f"Successfully loaded {len(dataset)} stars")
                
                if args.output is None:
                    output_path = os.path.join(args.input, 'stars_dataset.pkl')
                else:
                    output_path = args.output
                
                print(f"Saving dataset to: {output_path}")
                dataset.save_to_file(output_path)
                print("Dataset conversion completed successfully!")
                
                # Print summary
                total_campaigns = sum(len(lc.campaigns) for star in dataset.stars for lc in star.surveys.values())
                surveys = set()
                for star in dataset.stars:
                    if star.surveys:
                        surveys.update(star.surveys.keys())
                
                print(f"\nSummary:")
                print(f"  Stars: {len(dataset)}")
                print(f"  Total campaigns: {total_campaigns}")
                print(f"  Surveys found: {sorted(surveys)}")
                
            except Exception as e:
                print(f"Error converting folder to dataset: {e}")
                sys.exit(1)
                
        elif args.command == 'info':
            # Show dataset information
            dataset_path = os.path.join(args.input, 'stars_dataset.pkl')
            if not os.path.exists(dataset_path):
                print(f"No dataset file found at {dataset_path}")
                sys.exit(1)
            
            try:
                dataset = StarDataset.load_from_file(dataset_path)
                print(f"Dataset file: {dataset_path}")
                print(f"Stars: {len(dataset)}")
                
                total_campaigns = 0
                surveys = set()
                star_numbers = []
                
                for star in dataset.stars:
                    star_numbers.append(star.star_number)
                    if star.surveys:
                        surveys.update(star.surveys.keys())
                        total_campaigns += sum(len(lc.campaigns) for lc in star.surveys.values())
                
                print(f"Total campaigns: {total_campaigns}")
                print(f"Surveys: {sorted(surveys)}")
                print(f"Star numbers: {sorted(star_numbers)}")
                
            except Exception as e:
                print(f"Error reading dataset: {e}")
                sys.exit(1)
    
    main()
