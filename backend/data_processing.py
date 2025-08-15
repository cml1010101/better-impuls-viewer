"""
Data processing functions for Better Impuls Viewer.
Handles campaign detection, data sorting, outlier removal and basic preprocessing.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import os


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


def find_all_campaigns(data: np.ndarray, x_threshold: float = None) -> List[np.ndarray]:
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

from astropy.timeseries import LombScargle

def calculate_lomb_scargle(
    data: np.ndarray, samples_per_peak: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
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


def phase_fold_data(time: np.ndarray, flux: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    """Phase fold the light curve data."""
    phase = (time % period) / period
    
    # Sort by phase
    sort_indices = np.argsort(phase)
    phase_sorted = phase[sort_indices]
    flux_sorted = flux[sort_indices]
    
    return phase_sorted, flux_sorted


def detect_period_lomb_scargle(lc_data: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Detect period using Lomb-Scargle periodogram and return periodogram data."""
    try:
        frequency, power = calculate_lomb_scargle(lc_data)
        
        # Find the peak frequency
        peak_idx = np.argmax(power)
        peak_frequency = frequency[peak_idx]
        
        # Convert to period
        period = 1.0 / peak_frequency
        
        # Ensure period is in reasonable range (0.1 to 100 days)
        period = np.clip(period, 0.1, 100.0)
        
        return period, frequency, power
    except Exception as e:
        print(f"Error in period detection: {e}")
        # Return default values
        frequency = np.linspace(0.1, 1.0, 900)
        power = np.random.normal(0.5, 0.1, 900)
        return 2.0, frequency, power


def generate_candidate_periods(true_period: float, num_candidates: int = 4) -> List[float]:
    """Generate multiple period candidates around the true period."""
    import random
    
    candidates = [true_period]  # Include true period
    
    # Add some variations
    for i in range(num_candidates - 1):
        # Add some noise and harmonics
        variation = np.random.uniform(0.8, 1.2)
        harmonic = np.random.choice([0.5, 2.0]) if i % 2 == 0 else 1.0
        candidate = true_period * variation * harmonic
        candidate = np.clip(candidate, 0.1, 100.0)  # Keep in reasonable range
        candidates.append(candidate)
    
    return candidates