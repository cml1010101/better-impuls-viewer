import numpy as np
import matplotlib.pyplot as plt

def plot_data(data: np.ndarray, fourier_results: np.ndarray, periods: np.ndarray, star_number: int, telescope_name: str):
    """
    Plot the original data and the Fourier Transform results (displayed as periods), 
    with an interactive overlay plot for phase-folded data.
    
    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where the first column is the x-axis data (e.g., time) and the second column is the y-axis data.
    fourier_results : np.ndarray
        The Fourier Transform results of the y-axis data.
    periods : np.ndarray
        The periods corresponding to the fourier_results. These should be calculated as 1/freqs,
        where freqs are typically obtained from np.fft.fftfreq.
    star_number : int
        The star number to be displayed in the plot title.
    telescope_name : str
        The name of the telescope to be displayed in the plot title.
    """
    # --- Robustness check: Ensure fourier_results and periods have consistent lengths ---
    if len(fourier_results) != len(periods):
        print(f"Warning: Length of fourier_results ({len(fourier_results)}) does not match length of periods ({len(periods)}). This may cause indexing errors in plotting. Truncating to smallest length.")
        min_len = min(len(fourier_results), len(periods))
        fourier_results = fourier_results[:min_len]
        periods = periods[:min_len]
    else:
        min_len = len(fourier_results)

    # Create a figure with 3 subplots.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) 

    # --- Plot Original Data ---
    axes[0].scatter(data[:, 0], data[:, 1], label='Original Data', color='blue', s=10)
    axes[0].set_title(f'Star {star_number} - {telescope_name} - Original Data')
    axes[0].set_xlabel('X-axis')
    axes[0].set_ylabel('Y-axis')
    axes[0].legend()
    axes[0].grid(True)
    
    # --- Plot Fourier Transform Results (as Periods) ---
    # Filter out infinite periods (corresponding to zero frequency) and negative periods.
    # We only care about positive, finite periods for physical interpretation.
    # Use a small epsilon to avoid division by zero or issues with very large numbers.
    valid_period_indices = (periods > 0) & np.isfinite(periods)
    
    # Plot using the arrays that have been ensured to have consistent lengths.
    axes[1].plot(periods[valid_period_indices], np.abs(fourier_results[valid_period_indices]), label='Fourier Transform (Periods)', color='orange')
    axes[1].set_title(f'Star {star_number} - {telescope_name} - Fourier Transform (Periods)')
    axes[1].set_xlabel('Period (units of X)')
    axes[1].set_ylabel('Magnitude')
    axes[1].legend()
    
    # Set x-axis limit to show only positive, finite periods.
    if np.any(valid_period_indices):
        # Sort periods for better visualization if desired, though plot handles unsorted x-values.
        # It's often useful to reverse the x-axis for periods if they originated from frequencies,
        # so that smaller periods (higher frequencies) are on the right.
        axes[1].set_xlim(periods[valid_period_indices].min() * 0.9, periods[valid_period_indices].max() * 1.1)
        axes[1].set_xscale('log') # Log scale is often useful for periods due to wide range
    else:
        axes[1].set_xlim(0.1, 10) # Default x-limit if no valid periods are found, and use log scale.
        axes[1].set_xscale('log')
    axes[1].grid(True)

    # --- Placeholder for the Interactive Overlay (Folded) Plot ---
    axes[2].set_title('Click on FT (Periods) plot to see folded data')
    axes[2].set_xlabel('Phase (0-1)')
    axes[2].set_ylabel('Y-axis')
    axes[2].grid(True)

    # Store necessary data and axes references in the figure object for easy access within the callback function.
    fig.data = data
    fig.periods = periods # Store the (potentially truncated) periods for the click handler
    fig.fourier_results = fourier_results
    fig.ax_folded = axes[2] # Reference to the third subplot for folded data
    fig.ax_ft = axes[1]     # Reference to the Fourier Transform subplot

    # Define the callback function to handle mouse clicks on the plot.
    def on_click(event):
        if event.inaxes == fig.ax_ft: 
            clicked_period = event.xdata
            
            if clicked_period is not None and clicked_period > 0 and np.isfinite(clicked_period): 
                # Calculate the phase for each data point.
                phase = (fig.data[:, 0] % clicked_period) / clicked_period

                # Clear the previous folded plot content and draw the new one.
                fig.ax_folded.clear()
                fig.ax_folded.plot(phase, fig.data[:, 1], 'o', markersize=3, alpha=0.5, color='green')
                fig.ax_folded.set_title(f'Folded Data (Period: {clicked_period:.4f} units)')
                fig.ax_folded.set_xlabel('Phase (0-1)')
                fig.ax_folded.set_ylabel('Y-axis')
                fig.ax_folded.grid(True)
                
                fig.canvas.draw_idle() 
            elif clicked_period is not None and (clicked_period <= 0 or not np.isfinite(clicked_period)):
                fig.ax_folded.clear()
                fig.ax_folded.set_title('Cannot fold with invalid period (<=0 or infinite)')
                fig.ax_folded.set_xlabel('Phase (0-1)')
                fig.ax_folded.set_ylabel('Y-axis')
                fig.ax_folded.grid(True)
                fig.canvas.draw_idle()

    # Connect the 'button_press_event' (mouse click) to our on_click callback function.
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Adjust subplot parameters for a tight layout.
    plt.tight_layout()
    plt.show()

# --- Example Usage (for testing plot_data independently) ---
if __name__ == "__main__":
    # 1. Generate some sample data
    np.random.seed(42)
    sampling_rate = 100 
    duration = 10 
    num_samples = int(sampling_rate * duration)
    x_data = np.linspace(0, duration, num_samples, endpoint=False)

    y_data = 2 * np.sin(2 * np.pi * 1 * x_data) + \
             1.5 * np.sin(2 * np.pi * 5 * x_data + np.pi/4) + \
             np.random.normal(0, 0.5, num_samples)
    
    data = np.vstack((x_data, y_data)).T

    # 2. Perform Fourier Transform and calculate periods
    sample_spacing = x_data[1] - x_data[0] 
    fourier_results = np.fft.fft(y_data)
    freqs = np.fft.fftfreq(len(y_data), d=sample_spacing)
    
    # Calculate periods from frequencies. Handle zero frequency to avoid division by zero.
    periods = np.zeros_like(freqs)
    non_zero_freq_indices = freqs != 0
    periods[non_zero_freq_indices] = 1.0 / freqs[non_zero_freq_indices]
    
    # Debug info
    print(f"Debug Info from example: len(y_data)={len(y_data)}, len(fourier_results)={len(fourier_results)}, len(freqs)={len(freqs)}, len(periods)={len(periods)}")

    # 3. Call the plotting function with the generated data, FT results, and periods.
    plot_data(data, fourier_results, periods, 789, "Kepler Space Telescope")