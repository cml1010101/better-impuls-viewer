#!/usr/bin/env python3
"""
Test script for auto-normalization (linear detrending) functionality.

This script tests the detrend_linear function by:
1. Loading sample data 
2. Adding artificial linear trends
3. Testing detrending functionality
4. Validating trend removal while preserving periodic signals
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add backend to path for imports
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.append(backend_path)

from data_processing import detrend_linear, remove_y_outliers, calculate_lomb_scargle
import pandas as pd

def create_synthetic_data_with_trend(n_points=200, period=5.0, trend_slope=0.01):
    """Create synthetic light curve data with a linear trend and periodic signal."""
    time = np.linspace(0, 50, n_points)  # 50 days of observations
    
    # Create periodic signal (sinusoidal with period=5 days)
    flux_periodic = 1.0 + 0.1 * np.sin(2 * np.pi * time / period)
    
    # Add linear trend
    linear_trend = trend_slope * time
    flux_with_trend = flux_periodic + linear_trend
    
    # Add small amount of noise
    noise = np.random.normal(0, 0.02, n_points)
    flux_with_trend += noise
    
    return np.column_stack([time, flux_with_trend]), np.column_stack([time, flux_periodic])

def test_detrending_functionality():
    """Test the detrending function with synthetic data."""
    print("Testing auto-normalization (linear detrending) functionality...")
    
    # Test 1: Synthetic data with positive trend
    print("\n1. Testing with synthetic data (positive trend)...")
    
    synthetic_with_trend, synthetic_original = create_synthetic_data_with_trend(
        n_points=200, period=5.0, trend_slope=0.02
    )
    
    # Apply detrending
    detrended_data = detrend_linear(synthetic_with_trend)
    
    # Calculate trend removed
    trend_removed = np.mean(synthetic_with_trend[:, 1]) - np.mean(detrended_data[:, 1])
    trend_slope_original = (synthetic_with_trend[-1, 1] - synthetic_with_trend[0, 1]) / (synthetic_with_trend[-1, 0] - synthetic_with_trend[0, 0])
    trend_slope_detrended = (detrended_data[-1, 1] - detrended_data[0, 1]) / (detrended_data[-1, 0] - detrended_data[0, 0])
    
    print(f"   Original trend slope: {trend_slope_original:.6f}")
    print(f"   Detrended slope: {trend_slope_detrended:.6f}")
    print(f"   Slope reduction: {abs(trend_slope_original - trend_slope_detrended):.6f}")
    
    # Test 2: Real sample data
    print("\n2. Testing with real sample data...")
    
    try:
        # Load a sample data file
        sample_file = os.path.join(os.path.dirname(__file__), 'sample_data', '1-hubble.tbl')
        if os.path.exists(sample_file):
            data = pd.read_csv(sample_file, sep='\t', comment='#', header=0)
            real_data = np.column_stack([data.iloc[:, 0].values, data.iloc[:, 1].values])
            
            # Calculate original trend
            original_slope = (real_data[-1, 1] - real_data[0, 1]) / (real_data[-1, 0] - real_data[0, 0])
            
            # Apply detrending
            detrended_real = detrend_linear(real_data)
            detrended_slope = (detrended_real[-1, 1] - detrended_real[0, 1]) / (detrended_real[-1, 0] - detrended_real[0, 0])
            
            print(f"   Real data original slope: {original_slope:.6f}")
            print(f"   Real data detrended slope: {detrended_slope:.6f}")
            print(f"   Real data slope reduction: {abs(original_slope - detrended_slope):.6f}")
            
        else:
            print("   Sample data file not found, skipping real data test")
            
    except Exception as e:
        print(f"   Error loading real data: {e}")
    
    # Test 3: Periodogram comparison
    print("\n3. Testing periodogram before and after detrending...")
    
    try:
        # Use synthetic data for periodogram test
        freq_before, power_before = calculate_lomb_scargle(synthetic_with_trend)
        freq_after, power_after = calculate_lomb_scargle(detrended_data)
        
        # Find peak power at expected period (5 days)
        periods_before = 1 / freq_before
        periods_after = 1 / freq_after
        
        # Find power near the expected period (5 days)
        expected_period = 5.0
        idx_before = np.argmin(np.abs(periods_before - expected_period))
        idx_after = np.argmin(np.abs(periods_after - expected_period))
        
        power_at_expected_before = power_before[idx_before]
        power_at_expected_after = power_after[idx_after]
        
        print(f"   Power at expected period ({expected_period}d) before detrending: {power_at_expected_before:.4f}")
        print(f"   Power at expected period ({expected_period}d) after detrending: {power_at_expected_after:.4f}")
        print(f"   Signal improvement: {power_at_expected_after/power_at_expected_before:.2f}x")
        
    except Exception as e:
        print(f"   Error in periodogram test: {e}")
    
    # Test 4: Edge cases
    print("\n4. Testing edge cases...")
    
    # Empty data
    empty_data = np.array([]).reshape(0, 2)
    result_empty = detrend_linear(empty_data)
    print(f"   Empty data test: {'PASS' if result_empty.shape[0] == 0 else 'FAIL'}")
    
    # Single point
    single_point = np.array([[1.0, 2.0]])
    result_single = detrend_linear(single_point)
    print(f"   Single point test: {'PASS' if np.array_equal(result_single, single_point) else 'FAIL'}")
    
    # Two points
    two_points = np.array([[1.0, 2.0], [2.0, 3.0]])
    result_two = detrend_linear(two_points)
    expected_detrended = np.array([[1.0, 2.0], [2.0, 2.0]])  # Should remove the slope
    slope_removed = abs((result_two[1, 1] - result_two[0, 1]) / (result_two[1, 0] - result_two[0, 0])) < 1e-10
    print(f"   Two points test: {'PASS' if slope_removed else 'FAIL'}")
    
    print("\nAuto-normalization test completed!")

def test_integration_with_pipeline():
    """Test integration with the full processing pipeline."""
    print("\n5. Testing integration with processing pipeline...")
    
    try:
        # Create data with trend and outliers
        synthetic_with_trend, _ = create_synthetic_data_with_trend(
            n_points=200, period=5.0, trend_slope=0.03
        )
        
        # Add some outliers
        outlier_indices = [50, 100, 150]
        for idx in outlier_indices:
            synthetic_with_trend[idx, 1] += 0.5  # Add large outlier
        
        print("   Testing full pipeline: outlier removal + detrending...")
        
        # Step 1: Remove outliers
        cleaned_data = remove_y_outliers(synthetic_with_trend)
        print(f"   Outliers removed: {len(synthetic_with_trend) - len(cleaned_data)} points")
        
        # Step 2: Apply detrending
        detrended_data = detrend_linear(cleaned_data)
        
        # Check that trend is removed
        original_slope = (cleaned_data[-1, 1] - cleaned_data[0, 1]) / (cleaned_data[-1, 0] - cleaned_data[0, 0])
        detrended_slope = (detrended_data[-1, 1] - detrended_data[0, 1]) / (detrended_data[-1, 0] - detrended_data[0, 0])
        
        print(f"   Pipeline - original slope: {original_slope:.6f}")
        print(f"   Pipeline - detrended slope: {detrended_slope:.6f}")
        print(f"   Pipeline test: {'PASS' if abs(detrended_slope) < abs(original_slope) * 0.1 else 'FAIL'}")
        
    except Exception as e:
        print(f"   Pipeline integration test error: {e}")

if __name__ == "__main__":
    test_detrending_functionality()
    test_integration_with_pipeline()
    print("\nAll tests completed!")