#!/usr/bin/env python3
"""
Generate sample astronomical data in the correct format for the Better Impuls Viewer.

This script creates data files in the format expected by the application:
- One file per star/telescope combination (e.g., '1-hubble.tbl')
- Each file contains multiple campaigns separated by time gaps
- Campaigns are extracted using the find_all_campaigns function
- Updated to include examples of new classification types
"""

import numpy as np
import os

def create_sample_astronomical_data():
    """Create realistic sample astronomical data with multiple campaigns per file and new classification types"""
    
    # Create sample_data directory if it doesn't exist
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Remove old format files if they exist
    old_files = [f for f in os.listdir(sample_dir) if f.endswith('.tbl')]
    for f in old_files:
        os.remove(os.path.join(sample_dir, f))
    
    # Configuration for each star/telescope combination with new classification types
    configs = [
        # Star 1 - Dipper type (eclipsing/transiting objects)
        {"star": 1, "telescope": "hubble", "base_period": 2.5, "noise_level": 0.01, "type": "dipper"},
        {"star": 1, "telescope": "kepler", "base_period": 2.4, "noise_level": 0.008, "type": "dipper"},
        {"star": 1, "telescope": "tess", "base_period": 2.6, "noise_level": 0.012, "type": "dipper"},
        
        # Star 2 - Distant peaks type
        {"star": 2, "telescope": "hubble", "base_period": 7.8, "noise_level": 0.015, "type": "distant_peaks"},
        {"star": 2, "telescope": "kepler", "base_period": 7.7, "noise_level": 0.010, "type": "distant_peaks"},
        {"star": 2, "telescope": "tess", "base_period": 7.9, "noise_level": 0.018, "type": "distant_peaks"},
        
        # Star 3 - Sinusoidal type (regular variable stars)
        {"star": 3, "telescope": "hubble", "base_period": 15.3, "noise_level": 0.020, "type": "sinusoidal"},
        {"star": 3, "telescope": "kepler", "base_period": 15.1, "noise_level": 0.012, "type": "sinusoidal"},
        {"star": 3, "telescope": "tess", "base_period": 15.5, "noise_level": 0.025, "type": "sinusoidal"},
        
        # Star 4 - Close peak type
        {"star": 4, "telescope": "hubble", "base_period": 5.2, "noise_level": 0.018, "type": "close_peak"},
        {"star": 4, "telescope": "kepler", "base_period": 5.1, "noise_level": 0.012, "type": "close_peak"},
        {"star": 4, "telescope": "tess", "base_period": 5.3, "noise_level": 0.022, "type": "close_peak"},
        
        # Star 5 - Other/irregular type
        {"star": 5, "telescope": "hubble", "base_period": 12.7, "noise_level": 0.035, "type": "other"},
        {"star": 5, "telescope": "kepler", "base_period": 12.5, "noise_level": 0.025, "type": "other"},
        {"star": 5, "telescope": "tess", "base_period": 12.9, "noise_level": 0.040, "type": "other"},
    ]
    
    for config in configs:
        star = config["star"]
        telescope = config["telescope"]
        period = config["base_period"]
        noise = config["noise_level"]
        var_type = config["type"]
        
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
            
            # Generate periodic signal based on variable type
            campaign_period = period * (1 + 0.05 * (i - 1))  # Small period variations between campaigns
            amplitude = 0.01 * (1 + 0.2 * i)  # Varying amplitude
            
            # Create type-specific signal
            phase = (time_points - start_time) % campaign_period / campaign_period
            
            if var_type == "dipper":
                # Transit-like dips
                dip_width = 0.1  # 10% of period
                dip_depth = amplitude * 2
                dip_mask = np.abs(phase - 0.5) < dip_width/2
                signal = np.zeros(len(phase))
                signal[dip_mask] = -dip_depth * np.exp(-((phase[dip_mask] - 0.5)/(dip_width/4))**2)
                
            elif var_type == "distant_peaks":
                # Two peaks separated by significant phase
                peak1_phase = 0.2
                peak2_phase = 0.7
                peak_width = 0.08
                signal = (amplitude * np.exp(-((phase - peak1_phase)/peak_width)**2) + 
                         0.7 * amplitude * np.exp(-((phase - peak2_phase)/peak_width)**2))
                
            elif var_type == "close_peak":
                # Two peaks close together
                peak1_phase = 0.4
                peak2_phase = 0.55
                peak_width = 0.06
                signal = (amplitude * np.exp(-((phase - peak1_phase)/peak_width)**2) + 
                         0.8 * amplitude * np.exp(-((phase - peak2_phase)/peak_width)**2))
                
            elif var_type == "sinusoidal":
                # Clean sinusoidal pattern
                signal = amplitude * np.sin(2 * np.pi * phase)
                # Add some harmonics for realism
                signal += 0.3 * amplitude * np.sin(4 * np.pi * phase + np.pi/4)
                
            else:  # "other" - irregular
                # More irregular pattern with multiple components
                signal = (amplitude * np.sin(2 * np.pi * phase) + 
                         0.5 * amplitude * np.sin(3 * np.pi * phase + np.pi/3) +
                         0.3 * amplitude * np.random.normal(0, 1, len(phase)))
            
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
            f.write(f"# Generated sample data for astronomical analysis - Type: {var_type}\n")
            f.write("# Multiple campaigns with gaps\n")
            
            for time_val, flux_val in combined_data:
                f.write(f"{time_val:.6f}\t{flux_val:.6f}\n")
        
        print(f"Created {filename} with {len(combined_data)} data points (type: {var_type})")

if __name__ == "__main__":
    create_sample_astronomical_data()
    print("Sample data generation complete!")
    print("\nData format:")
    print("- Each file contains multiple observing campaigns")
    print("- Campaigns are separated by large time gaps (>100 days)")
    print("- Backend extracts campaigns using find_all_campaigns() function")
    print("- Top 3 campaigns by data count are returned to frontend")
    print("\nNew classification types included:")
    print("- Star 1: Dipper type (eclipsing/transiting objects)")
    print("- Star 2: Distant peaks type (double-peaked variables)")
    print("- Star 3: Sinusoidal type (regular variables)")
    print("- Star 4: Close peak type (close binary/pulsating)")
    print("- Star 5: Other type (irregular/complex)")