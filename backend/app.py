import argparse
import os
from display import plot_data
from process import find_longest_x_campaign, sort_data, calculate_fourier_transform, calculate_lomb_scargle, remove_y_outliers
import pandas as pd
import numpy as np

def main(args=None):
    default_folder = '~/Documents/impuls-data'
    parser = argparse.ArgumentParser(description="Process and display data from a file.")
    parser.add_argument('-f', '--folder', type=str, default=default_folder, help='Folder containing the data files.')
    parser.add_argument('star_number', type=int, help='Star number to process.')
    args = parser.parse_args(args)

    folder = os.path.expanduser(args.folder)
    star_number = args.star_number
    all_files = os.listdir(folder)
    star_dfs: list[str] = [f for f in all_files if f.startswith(f'{star_number}') and f.endswith('.tbl')]
    if not star_dfs:
        print(f"No files found for star number {star_number} in folder {folder}.")
        return
    for star_df in star_dfs:
        # Ignore wise data files
        if star_df.endswith('w1.tbl') or star_df.endswith('w2.tbl'):
            continue
        # Retrieve telescope name from the filename
        telescope_name = star_df.split('-')[1].split('.')[0]
        file_path = os.path.join(folder, star_df)
        print(f"Processing file: {file_path}")
        data = pd.read_table(file_path, header=None, sep=r'\s+', skiprows=[0, 1, 2])

        # Convert DataFrame to numpy array
        data = data.to_numpy()
        print(data)
        print(data.shape)
        # Crop the largest consecutive segment of data
        data = find_longest_x_campaign(data, 0.1) # Use a threshold of 0.1 for x-axis values
        data = remove_y_outliers(data)
        # Sort the data based on the first column (x-axis)
        data = sort_data(data)

        frequencies, powers = calculate_lomb_scargle(data)

        print(data)

        # Calculate periods from frequencies. Handle zero frequency to avoid division by zero.
        periods = np.zeros_like(frequencies)
        non_zero_freq_indices = frequencies != 0
        periods[non_zero_freq_indices] = 1.0 / frequencies[non_zero_freq_indices]

        # Remove periods that are too small or too large
        valid_period_indices = (periods > 0.1) & (periods < 20)
        periods = periods[valid_period_indices]

        # Display the data and Fourier Transform results, passing periods instead of frequencies
        plot_data(data, powers, periods, star_number, telescope_name) # Changed fourier_results to powers

if __name__ == "__main__":
    main()