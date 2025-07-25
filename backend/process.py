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

def find_all_campaigns(data: np.ndarray, x_threshold: float) -> list[np.ndarray]:
    """
    Finds all campaigns (series of consecutive data points) where the x-values
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
        list[np.ndarray]: A list of NumPy arrays, each representing a campaign.
                         Returns an empty list if no data or no campaigns are found.
    """
    if data.shape[0] == 0:
        print("Input data is empty.")
        return []

    if data.shape[0] == 1:
        print("Input data has only one point. Returning the point as the only campaign.")
        return [data]

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
