import numpy as np
from astropy.timeseries import LombScargle

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
    
    # Calculate Lomb-Scargle periodogram with configurable samples_per_peak
    # Increasing samples_per_peak generates a denser frequency grid,
    # which helps in resolving sharp peaks that might appear as plateaus
    # if the sampling is too coarse.
    ls = LombScargle(time, flux)
    frequency, power = ls.autopower(samples_per_peak=samples_per_peak)
    
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

def detect_wise_folding_period(data: np.ndarray, max_period: float = 10.0) -> float:
    """
    Detect the folding period in WISE data by looking for artificial periodicity.
    
    WISE data is often already phase-folded on itself, creating artificial periods.
    This function attempts to detect this folding period by analyzing the 
    time structure of the data.
    
    Args:
        data (np.ndarray): A NumPy array of shape (n, 2) where each row
                           is a (time, flux) coordinate pair.
        max_period (float): Maximum period to consider for folding detection.
    
    Returns:
        float: Detected folding period, or 0 if no clear folding is detected.
    """
    if data.shape[0] < 10:
        return 0.0
    
    times = data[:, 0]
    time_diffs = np.diff(times)
    
    # Check for backwards jumps (negative time differences) which indicate folding
    backwards_jumps = time_diffs[time_diffs < -0.1]
    
    if len(backwards_jumps) > 0:
        # Look for the pattern: if we have backwards jumps, the period is likely 
        # the range before the jump plus the absolute value of the jump
        time_span = np.max(times) - np.min(times)
        
        # Find the largest backwards jump
        largest_backwards = np.min(backwards_jumps)  # Most negative
        
        # The folding period might be related to the time span or jump size
        detected_period = min(time_span, abs(largest_backwards))
        
        if detected_period > 0.1 and detected_period < max_period:
            print(f"Detected WISE folding period from backwards jumps: {detected_period:.3f}")
            return detected_period
    
    # Alternative detection: if the time span is suspiciously small for astronomical data
    time_span = np.max(times) - np.min(times)
    if time_span < 2.0:  # Less than 2 days total time span suggests folding
        print(f"Detected WISE folding period from small time span: {time_span:.3f}")
        return time_span
    
    # Look for repeated patterns in time spacing that might indicate folding
    # If we see regular large gaps, it might be folding artifacts
    large_gaps = time_diffs[time_diffs > 10]  # Gaps larger than 10 days
    if len(large_gaps) > 1:
        # Check if these gaps are similar (indicating folding structure)
        gap_std = np.std(large_gaps)
        gap_mean = np.mean(large_gaps)
        
        if gap_std < gap_mean * 0.1:  # Gaps are very similar
            # This suggests artificial folding - the period might be the time before the gap
            print(f"Detected WISE folding from regular gaps: period ~{gap_mean:.3f}")
            return min(gap_mean, max_period)
    
    return 0.0

def unfold_wise_data(data: np.ndarray, folding_period: float) -> np.ndarray:
    """
    Unfold WISE data that has been artificially phase-folded.
    
    Args:
        data (np.ndarray): A NumPy array of shape (n, 2) where each row
                           is a (time, flux) coordinate pair.
        folding_period (float): The period over which the data was folded.
    
    Returns:
        np.ndarray: Unfolded data with corrected time values.
    """
    if folding_period <= 0:
        return data
    
    unfolded_data = data.copy()
    times = unfolded_data[:, 0]
    
    # Track cycles to unfold the data
    current_cycle = 0
    last_time = times[0]
    
    for i in range(1, len(times)):
        # If time decreases significantly, we've started a new cycle
        if times[i] < last_time - folding_period * 0.5:
            current_cycle += 1
        
        # Adjust time by adding the cycle offset
        unfolded_data[i, 0] = times[i] + current_cycle * folding_period
        last_time = times[i]
    
    print(f"Unfolded WISE data with {current_cycle + 1} cycles")
    return unfolded_data

def is_wise_telescope(telescope: str) -> bool:
    """Check if the telescope is a WISE telescope (w1 or w2)."""
    return telescope.lower() in ['w1', 'w2', 'wise']

def process_wise_data(data: np.ndarray) -> np.ndarray:
    """
    Process WISE data by detecting and unfolding any artificial phase folding.
    
    Args:
        data (np.ndarray): Raw WISE data
    
    Returns:
        np.ndarray: Processed and potentially unfolded WISE data
    """
    # Detect if data is folded
    folding_period = detect_wise_folding_period(data)
    
    if folding_period > 0:
        # Unfold the data
        return unfold_wise_data(data, folding_period)
    else:
        print("No WISE folding detected, returning original data")
        return data
