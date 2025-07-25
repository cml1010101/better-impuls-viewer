#!/usr/bin/env python3
"""
Generate sample astronomical data in the correct format for the Better Impuls Viewer.

This script creates data files in the format expected by the application:
- One file per star/telescope combination (e.g., '1-hubble.tbl')
- Each file contains multiple campaigns separated by time gaps
- Campaigns are extracted using the find_all_campaigns function
"""

import numpy as np
import os

def create_sample_astronomical_data():
    """Create realistic sample astronomical data with multiple campaigns per file"""
    
    # Create sample_data directory if it doesn't exist
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Remove old format files if they exist
    old_files = [f for f in os.listdir(sample_dir) if f.endswith('.tbl')]
    for f in old_files:
        os.remove(os.path.join(sample_dir, f))
    
    # Configuration for each star/telescope combination
    configs = [
        # Star 1
        {"star": 1, "telescope": "hubble", "base_period": 2.5, "noise_level": 0.01},
        {"star": 1, "telescope": "kepler", "base_period": 2.4, "noise_level": 0.008},
        {"star": 1, "telescope": "tess", "base_period": 2.6, "noise_level": 0.012},
        
        # Star 2  
        {"star": 2, "telescope": "hubble", "base_period": 7.8, "noise_level": 0.015},
        {"star": 2, "telescope": "kepler", "base_period": 7.7, "noise_level": 0.010},
        {"star": 2, "telescope": "tess", "base_period": 7.9, "noise_level": 0.018},
        
        # Star 3
        {"star": 3, "telescope": "hubble", "base_period": 15.3, "noise_level": 0.020},
        {"star": 3, "telescope": "kepler", "base_period": 15.1, "noise_level": 0.012},
        {"star": 3, "telescope": "tess", "base_period": 15.5, "noise_level": 0.025},
    ]
    
    for config in configs:
        star = config["star"]
        telescope = config["telescope"]
        period = config["base_period"]
        noise = config["noise_level"]
        
        # Create multiple campaigns for this star/telescope combination
        all_time = []
        all_flux = []
        
        # Campaign parameters - separated by large gaps to be distinguishable
        campaign_lengths = [80, 120, 60]  # Different lengths for variety
        start_times = [0, 200, 400]  # Large gaps between campaigns (120+ days)
        
        base_flux = 0.743255  # Normalized flux level
        
        for i, (length, start_time) in enumerate(zip(campaign_lengths, start_times)):
            # Generate time series for this campaign with reasonable cadence
            time_points = np.linspace(start_time, start_time + length, int(length * 2))  # ~2 points per day
            
            # Generate periodic signal with slight variations per campaign
            campaign_period = period * (1 + 0.05 * (i - 1))  # Small period variations between campaigns
            amplitude = 0.01 * (1 + 0.2 * i)  # Varying amplitude
            
            # Create periodic signal with harmonics (realistic for variable stars)
            phase = 2 * np.pi * time_points / campaign_period
            signal = amplitude * np.sin(phase) + 0.3 * amplitude * np.sin(2 * phase)
            
            # Add noise and baseline
            flux_points = base_flux + signal + np.random.normal(0, noise, len(time_points))
            
            # Add some outliers (5% chance per point) to simulate real data
            outlier_mask = np.random.random(len(flux_points)) < 0.05
            flux_points[outlier_mask] += np.random.normal(0, 5 * noise, np.sum(outlier_mask))
            
            all_time.extend(time_points)
            all_flux.extend(flux_points)
        
        # Sort by time
        combined_data = list(zip(all_time, all_flux))
        combined_data.sort(key=lambda x: x[0])
        
        # Write to file in astronomical data format
        filename = f"{star}-{telescope}.tbl"
        filepath = os.path.join(sample_dir, filename)
        
        with open(filepath, 'w') as f:
            # Write header (similar to real astronomical data format)
            f.write("# Time (days) | Flux (normalized)\n")
            f.write("# Generated sample data for astronomical analysis\n")
            f.write("# Multiple campaigns with gaps\n")
            
            for time_val, flux_val in combined_data:
                f.write(f"{time_val:.6f}\t{flux_val:.6f}\n")
        
        print(f"Created {filename} with {len(combined_data)} data points")

if __name__ == "__main__":
    create_sample_astronomical_data()
    print("Sample data generation complete!")
    print("\nData format:")
    print("- Each file contains multiple observing campaigns")
    print("- Campaigns are separated by large time gaps (>100 days)")
    print("- Backend extracts campaigns using find_all_campaigns() function")
    print("- Top 3 campaigns by data count are returned to frontend")