import numpy as np
from astropy.timeseries import LombScargle
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import find_peaks
from typing import Tuple, List, Dict, Optional

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
        list[np.ndarray]: A list of NumPy arrays, each representing a campaign.
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

def calculate_lomb_scargle(data: np.ndarray, samples_per_peak: int = 10) -> tuple[np.ndarray, np.ndarray]:
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
    tuple[np.ndarray, np.ndarray]
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
        samples_per_peak=samples_per_peak
    )
    
    return frequency, power

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


def find_periodogram_periods(data: np.ndarray, top_n: int = 5) -> List[Tuple[float, float]]:
    """
    Find the most significant periods from Lomb-Scargle periodogram analysis.
    
    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where the first column is time and the second column is flux.
    top_n : int, optional
        Number of top periods to return. Default is 5.
    
    Returns
    -------
    List[Tuple[float, float]]
        List of (period, power) tuples sorted by power (highest first).
    """
    if data.shape[1] != 2 or data.shape[0] < 10:
        return []
    
    # Calculate periodogram with extended frequency range and higher resolution
    frequencies, powers = calculate_lomb_scargle(data, samples_per_peak=50)
    
    # Convert frequencies to periods, avoiding division by zero
    valid_freq_mask = frequencies > 0
    frequencies = frequencies[valid_freq_mask]
    powers = powers[valid_freq_mask]
    periods = 1.0 / frequencies
    
    # Filter to astronomically reasonable period range (0.5 to 50 days)
    # This excludes very short periods that are likely noise artifacts
    valid_period_mask = (periods >= 0.5) & (periods <= 50) & np.isfinite(periods)
    periods = periods[valid_period_mask]
    powers = powers[valid_period_mask]
    
    if len(periods) == 0:
        return []
    
    # Apply a period-dependent weight to favor typical variable star periods
    # Give higher weight to periods in the 1-20 day range
    period_weights = np.ones(len(periods))
    typical_range = (periods >= 1.0) & (periods <= 20.0)
    period_weights[typical_range] *= 2.0  # Double the weight for typical periods
    
    # Apply weights to powers
    weighted_powers = powers * period_weights
    
    # Use a more sophisticated peak detection
    # First, try to find significant peaks above background
    power_median = np.median(weighted_powers)
    power_mad = np.median(np.abs(weighted_powers - power_median))  # Median absolute deviation
    height_threshold = power_median + 3 * power_mad  # More robust threshold
    
    # Find peaks with minimum separation to avoid close duplicates
    min_separation = max(1, len(weighted_powers) // 50)
    peak_indices, _ = find_peaks(weighted_powers, height=height_threshold, distance=min_separation)
    
    if len(peak_indices) == 0:
        # If no significant peaks found, use a lower threshold
        height_threshold = power_median + 1.5 * power_mad
        peak_indices, _ = find_peaks(weighted_powers, height=height_threshold, distance=min_separation)
    
    if len(peak_indices) == 0:
        # If still no peaks, take the highest weighted powers
        peak_indices = np.argsort(weighted_powers)[-top_n:][::-1]
    
    # Get period-power pairs for peaks (use original powers, not weighted)
    peak_periods = periods[peak_indices]
    peak_powers = powers[peak_indices]  # Original powers for reporting
    peak_weighted_powers = weighted_powers[peak_indices]  # For sorting
    
    # Sort by weighted power (highest first) and return top_n
    sorted_indices = np.argsort(peak_weighted_powers)[::-1]
    results = [(peak_periods[i], peak_powers[i]) for i in sorted_indices[:top_n]]
    
    return results


class SinusoidalModel(nn.Module):
    """PyTorch model for fitting sinusoidal functions to time series data."""
    
    def __init__(self, n_periods: int = 2):
        super(SinusoidalModel, self).__init__()
        self.n_periods = n_periods
        
        # Parameters for each sinusoid: [amplitude, period, phase, offset]
        self.amplitudes = nn.Parameter(torch.randn(n_periods) * 0.01)
        self.periods = nn.Parameter(torch.randn(n_periods) * 5 + 5)  # Initialize around 5 days
        self.phases = nn.Parameter(torch.randn(n_periods) * 2 * np.pi)
        self.offset = nn.Parameter(torch.randn(1))
    
    def forward(self, t):
        """Forward pass through the model."""
        result = self.offset
        
        for i in range(self.n_periods):
            # Ensure period is positive
            period = torch.abs(self.periods[i]) + 0.1
            amplitude = self.amplitudes[i]
            phase = self.phases[i]
            
            # Add sinusoid: amplitude * sin(2π * t / period + phase)
            result = result + amplitude * torch.sin(2 * np.pi * t / period + phase)
        
        return result


def fit_sinusoidal_periods(data: np.ndarray, n_periods: int = 2, max_epochs: int = 1000) -> List[Tuple[float, float]]:
    """
    Use PyTorch to fit sinusoidal curves and extract periods.
    
    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where the first column is time and the second column is flux.
    n_periods : int, optional
        Number of sinusoidal components to fit. Default is 2.
    max_epochs : int, optional
        Maximum number of training epochs. Default is 1000.
    
    Returns
    -------
    List[Tuple[float, float]]
        List of (period, confidence) tuples sorted by confidence.
    """
    if data.shape[1] != 2 or data.shape[0] < 20:
        return []
    
    # Prepare data
    time = torch.tensor(data[:, 0], dtype=torch.float32)
    flux = torch.tensor(data[:, 1], dtype=torch.float32)
    
    # Normalize time to start from 0
    time = time - time.min()
    
    # Normalize flux to have zero mean and unit variance
    flux_mean = flux.mean()
    flux_std = flux.std()
    if flux_std > 0:
        flux = (flux - flux_mean) / flux_std
    
    # Create model
    model = SinusoidalModel(n_periods=n_periods)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience = 100
    patience_counter = 0
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        predicted = model(time)
        loss = criterion(predicted, flux)
        loss.backward()
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Extract periods and calculate confidence based on amplitude
    periods_with_confidence = []
    
    with torch.no_grad():
        for i in range(n_periods):
            period = torch.abs(model.periods[i]).item() + 0.1
            amplitude = torch.abs(model.amplitudes[i]).item()
            
            # Filter reasonable periods (0.1 to 100 days)
            if 0.1 <= period <= 100:
                # Use amplitude as a proxy for confidence
                confidence = amplitude / (flux_std.item() if flux_std > 0 else 1.0)
                periods_with_confidence.append((period, confidence))
    
    # Sort by confidence (highest first)
    periods_with_confidence.sort(key=lambda x: x[1], reverse=True)
    
    return periods_with_confidence


def classify_periodicity(periods_data: List[Tuple[float, float]], 
                        torch_periods: List[Tuple[float, float]]) -> Dict[str, any]:
    """
    Classify the type of variability based on detected periods.
    
    Parameters
    ----------
    periods_data : List[Tuple[float, float]]
        Periods from periodogram analysis as (period, power) pairs.
    torch_periods : List[Tuple[float, float]]
        Periods from torch fitting as (period, confidence) pairs.
    
    Returns
    -------
    Dict[str, any]
        Classification results including type and confidence.
    """
    # Combine periods from both methods and their significance scores
    all_periods_with_scores = []
    
    # Add periodogram periods (use power as significance)
    for period, power in periods_data:
        all_periods_with_scores.append((period, power, 'periodogram'))
    
    # Add torch periods (normalize confidence to be comparable with power)
    if torch_periods:
        max_torch_conf = max(conf for _, conf in torch_periods)
        for period, confidence in torch_periods:
            # Normalize confidence to 0-1 range similar to periodogram power
            normalized_conf = confidence / max_torch_conf if max_torch_conf > 0 else 0
            all_periods_with_scores.append((period, normalized_conf, 'torch'))
    
    if len(all_periods_with_scores) == 0:
        return {
            "type": "other",
            "confidence": 0.0,
            "description": "No significant periods detected"
        }
    
    # Sort by significance score (highest first)
    all_periods_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get the most significant periods
    primary_period = all_periods_with_scores[0][0]
    primary_score = all_periods_with_scores[0][1]
    
    # Look for a strong secondary period
    secondary_period = None
    secondary_score = 0
    
    if len(all_periods_with_scores) > 1:
        # Look for a secondary period that's not too close to the primary
        for period, score, method in all_periods_with_scores[1:]:
            # Avoid harmonics (periods within 20% of 2x, 3x, 0.5x, 0.33x of primary)
            ratio1 = period / primary_period
            ratio2 = primary_period / period
            
            is_harmonic = (
                abs(ratio1 - 2.0) < 0.2 or  # 2x harmonic
                abs(ratio1 - 3.0) < 0.2 or  # 3x harmonic
                abs(ratio2 - 2.0) < 0.2 or  # 0.5x harmonic
                abs(ratio2 - 3.0) < 0.2 or  # 0.33x harmonic
                abs(ratio1 - 1.0) < 0.1     # Too close to primary
            )
            
            if not is_harmonic and score > 0.1:  # Minimum significance threshold
                secondary_period = period
                secondary_score = score
                break
    
    # Classification logic
    confidence_boost = min(primary_score * 2, 0.9)  # Convert score to confidence
    
    # Check for binary system indicators
    if secondary_period is not None:
        ratio = max(primary_period, secondary_period) / min(primary_period, secondary_period)
        
        # Strong binary indicators
        if ratio > 3.0:  # Very different time scales
            return {
                "type": "binary",
                "confidence": min(0.9, 0.7 + confidence_boost * 0.2),
                "description": f"Binary system with periods {primary_period:.3f} and {secondary_period:.3f} days (ratio {ratio:.1f}:1)"
            }
        elif 1.8 <= ratio <= 2.2:  # Close to 2:1 ratio (ellipsoidal variation)
            return {
                "type": "binary",
                "confidence": min(0.95, 0.8 + confidence_boost * 0.15),
                "description": f"Ellipsoidal binary with periods {primary_period:.3f} and {secondary_period:.3f} days (ratio ~2:1)"
            }
        elif ratio > 1.5:  # Moderate difference
            return {
                "type": "binary",
                "confidence": min(0.8, 0.6 + confidence_boost * 0.2),
                "description": f"Possible binary with periods {primary_period:.3f} and {secondary_period:.3f} days"
            }
    
    # Single dominant period - regular variable
    if primary_score > 0.1:  # Significant detection
        # Check if it's in typical variable star period range
        if 0.1 <= primary_period <= 100:
            if 0.5 <= primary_period <= 50:  # Most common range
                confidence = min(0.9, 0.7 + confidence_boost * 0.2)
                period_type = "Regular variable star"
            else:
                confidence = min(0.8, 0.6 + confidence_boost * 0.2)
                period_type = "Variable star"
            
            return {
                "type": "regular",
                "confidence": confidence,
                "description": f"{period_type} with period {primary_period:.3f} days"
            }
    
    # Default case - unclear or complex variability
    return {
        "type": "other",
        "confidence": max(0.3, confidence_boost),
        "description": f"Complex variability pattern (primary period: {primary_period:.3f} days)"
    }


def determine_automatic_periods(data: np.ndarray) -> Dict[str, any]:
    """
    Automatically determine periods using both periodogram and torch fitting methods.
    
    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where the first column is time and the second column is flux.
    
    Returns
    -------
    Dict[str, any]
        Complete analysis results including periods, classification, and metadata.
    """
    if data.shape[1] != 2 or data.shape[0] < 10:
        return {
            "primary_period": None,
            "secondary_period": None,
            "classification": {
                "type": "other",
                "confidence": 0.0,
                "description": "Insufficient data for analysis"
            },
            "methods": {
                "periodogram": {"success": False, "periods": []},
                "torch_fitting": {"success": False, "periods": []}
            },
            "error": "Insufficient data points for period analysis"
        }
    
    # Method 1: Enhanced periodogram analysis
    periodogram_success = False
    periodogram_periods = []
    periodogram_error = None
    
    try:
        periodogram_periods = find_periodogram_periods(data, top_n=3)
        periodogram_success = len(periodogram_periods) > 0
    except Exception as e:
        periodogram_error = str(e)
    
    # Method 2: PyTorch sinusoidal fitting
    torch_success = False
    torch_periods = []
    torch_error = None
    
    try:
        torch_periods = fit_sinusoidal_periods(data, n_periods=2, max_epochs=500)
        torch_success = len(torch_periods) > 0
    except Exception as e:
        torch_error = str(e)
    
    # Combine results to determine best periods
    primary_period = None
    secondary_period = None
    
    # Prioritize periods that appear in both methods or have high confidence
    if periodogram_success and torch_success:
        # Try to find matching periods between methods
        pg_periods = [p[0] for p in periodogram_periods]
        torch_period_values = [p[0] for p in torch_periods]
        
        # Look for similar periods (within 10% tolerance)
        matches = []
        for pg_p in pg_periods:
            for torch_p in torch_period_values:
                if abs(pg_p - torch_p) / max(pg_p, torch_p) < 0.1:
                    matches.append((pg_p + torch_p) / 2)  # Average the matching periods
        
        if matches:
            primary_period = matches[0]
            if len(matches) > 1:
                secondary_period = matches[1]
        else:
            # No matches, use highest confidence from each method
            primary_period = periodogram_periods[0][0]
            if len(torch_periods) > 0:
                secondary_period = torch_periods[0][0]
    
    elif periodogram_success:
        primary_period = periodogram_periods[0][0]
        if len(periodogram_periods) > 1:
            secondary_period = periodogram_periods[1][0]
    
    elif torch_success:
        primary_period = torch_periods[0][0]
        if len(torch_periods) > 1:
            secondary_period = torch_periods[1][0]
    
    # Classify the periodicity
    classification = classify_periodicity(periodogram_periods, torch_periods)
    
    # Prepare final results
    result = {
        "primary_period": primary_period,
        "secondary_period": secondary_period,
        "classification": classification,
        "methods": {
            "periodogram": {
                "success": periodogram_success,
                "periods": [{"period": p, "power": pow} for p, pow in periodogram_periods],
                "error": periodogram_error
            },
            "torch_fitting": {
                "success": torch_success,
                "periods": [{"period": p, "confidence": c} for p, c in torch_periods],
                "error": torch_error
            }
        }
    }
    
    return result


