"""
Data processing functions for Better Impuls Viewer.
Handles campaign detection, data sorting, outlier removal and basic preprocessing.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import os


def find_longest_x_campaign(data: np.ndarray, x_threshold: float) -> np.ndarray:
    """
    Finds the longest series of consecutive data points where the x-values
    are "close" to each other within a specified threshold.

    A "campaign" is defined as a sequence of points where the absolute
    difference between the x-value of the current point and the x-value
    of the previous point is less than or equal to the x_threshold.

    Args:
        data (np.ndarray): A NumPy array of shape (n, 2) where each row
                           is a (x, y) coordinate pair.
        x_threshold (float): The maximum allowed absolute difference between
                             consecutive x-values for them to be considered
                             "close" and part of the same campaign.

    Returns:
        np.ndarray: A NumPy array representing the longest campaign found.
                    Returns an empty array if no data or no campaign is found.
    """
    if data.shape[0] == 0:
        print("Input data is empty.")
        return np.array([])

    if data.shape[0] == 1:
        print("Input data has only one point. Returning the point as the campaign.")
        return data

    longest_campaign = np.array([])
    current_campaign = np.array([data[0]]) # Start with the first point

    # Iterate through the data starting from the second point
    for i in range(1, data.shape[0]):
        # Check if the current point's x-value is close to the previous point's x-value
        if np.abs(data[i, 0] - data[i-1, 0]) <= x_threshold:
            # If close, add the point to the current campaign
            current_campaign = np.vstack((current_campaign, data[i]))
        else:
            # If not close, the current campaign ends.
            # Check if this campaign is longer than the longest one found so far.
            if len(current_campaign) > len(longest_campaign):
                longest_campaign = current_campaign

            # Start a new campaign with the current point
            current_campaign = np.array([data[i]])

    # After the loop, check the last campaign as it might be the longest
    if len(current_campaign) > len(longest_campaign):
        longest_campaign = current_campaign

    return longest_campaign


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


def sort_data(data: np.ndarray) -> np.ndarray:
    """
    Sort the data based on the first column (x-axis data).
    
    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where the first column is the x-axis data.
    
    Returns
    -------
    np.ndarray
        The sorted data.
    """
    if data.shape[1] != 2:
        raise ValueError("Input data must be a 2D array with two columns (x and y).")
    
    # Sort by the first column (x-axis)
    sorted_data = data[np.argsort(data[:, 0])]
    
    return sorted_data


def calculate_fourier_transform(data: np.ndarray) -> np.ndarray:
    """
    Calculate the Fourier Transform of the given data.
    
    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where the first column is the x-axis data and the second column is the y-axis data.
    
    Returns
    -------
    np.ndarray
        The Fourier Transform of the y-axis data.
    """
    if data.shape[1] != 2:
        raise ValueError("Input data must be a 2D array with two columns (x and y).")
    
    # The data might not be evenly spaced, so we will need to normalize it before applying the Fourier Transform.
    x_data = data[:, 0]
    dt = np.mean(np.diff(x_data))
    if dt <= 0:
        raise ValueError("The x-axis data must be strictly increasing.")
    
    linear_x_data = np.arange(x_data[0], x_data[-1], dt)
    # Interpolate y_data to match the linear x_data
    y_data = np.interp(linear_x_data, x_data, data[:, 1])

    fourier_transform = np.fft.fft(y_data)
    
    return fourier_transform


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


def load_star_data_file(file_path: str) -> np.ndarray:
    """
    Load and preprocess a single star data file.
    
    Parameters
    ----------
    file_path : str
        Path to the .tbl data file
    
    Returns
    -------
    np.ndarray
        Processed data array with time and flux columns
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        # Load data using pandas
        data = pd.read_table(file_path, header=None, sep=r'\s+', skiprows=[0, 1, 2])
        data_array = data.to_numpy()
        
        # Process data using existing functions
        data_array = find_longest_x_campaign(data_array, 1.0)
        data_array = remove_y_outliers(data_array)
        data_array = sort_data(data_array)
        
        return data_array
        
    except Exception as e:
        raise ValueError(f"Error loading data file {file_path}: {e}")


def get_star_data_files(star_number: int, data_dir: str) -> List[str]:
    """
    Find all data files for a specific star number.
    
    Parameters
    ----------
    star_number : int
        Star identifier
    data_dir : str
        Directory containing data files
    
    Returns
    -------
    List[str]
        List of file paths for the star
    """
    if not os.path.exists(data_dir):
        return []
    
    star_files = []
    for file in os.listdir(data_dir):
        if file.startswith(f"{str(star_number).zfill(3)}-") and file.endswith(".tbl"):
            star_files.append(os.path.join(data_dir, file))
    
    return star_files


def calculate_data_statistics(data: np.ndarray) -> dict:
    """
    Calculate basic statistics for a data array.
    
    Parameters
    ----------
    data : np.ndarray
        2D array with time and flux columns
    
    Returns
    -------
    dict
        Dictionary containing basic statistics
    """
    if data.shape[0] == 0:
        return {
            "data_points": 0,
            "duration": 0.0,
            "time_range": (0.0, 0.0),
            "flux_range": (0.0, 0.0),
            "flux_std": 0.0
        }
    
    time_col = data[:, 0]
    flux_col = data[:, 1]
    
    return {
        "data_points": len(data),
        "duration": float(np.max(time_col) - np.min(time_col)),
        "time_range": (float(np.min(time_col)), float(np.max(time_col))),
        "flux_range": (float(np.min(flux_col)), float(np.max(flux_col))),
        "flux_std": float(np.std(flux_col))
    }