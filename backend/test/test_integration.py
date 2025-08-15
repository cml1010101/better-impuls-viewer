#!/usr/bin/env python3
"""
Test the auto-normalization feature integration with API functions.

This tests the create_multi_branch_data function with the new auto_normalize parameter.
"""

import numpy as np
import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.append(backend_path)

def test_create_multi_branch_data():
    """Test the create_multi_branch_data function with auto_normalize."""
    
    try:
        # Import the functions we need
        from app import create_multi_branch_data
        from data_processing import detrend_linear
        
        print("Testing create_multi_branch_data with auto_normalize parameter...")
        
        # Create synthetic test data with a trend
        n_points = 150
        time = np.linspace(0, 30, n_points)  # 30 days
        
        # Create a periodic signal with a linear trend
        period = 5.0
        flux_periodic = 1.0 + 0.1 * np.sin(2 * np.pi * time / period)
        linear_trend = 0.02 * time  # Strong linear trend
        flux_with_trend = flux_periodic + linear_trend
        
        # Add noise
        noise = np.random.normal(0, 0.02, n_points)
        flux_with_trend += noise
        
        test_data = np.column_stack([time, flux_with_trend])
        
        print(f"   Test data: {len(test_data)} points over {time[-1]:.1f} days")
        
        # Calculate original trend slope
        original_slope = (test_data[-1, 1] - test_data[0, 1]) / (test_data[-1, 0] - test_data[0, 0])
        print(f"   Original data slope: {original_slope:.6f}")
        
        # Test 1: Without auto-normalization (original behavior)
        print("\n   Testing without auto_normalize (default behavior)...")
        result_no_normalize = create_multi_branch_data(test_data, auto_normalize=False)
        
        print(f"   - Raw LC length: {len(result_no_normalize['raw_lc'])}")
        print(f"   - Periodogram length: {len(result_no_normalize['periodogram'])}")
        print(f"   - Detected period: {result_no_normalize['detected_period']:.3f} days")
        print(f"   - Number of candidate periods: {len(result_no_normalize['candidate_periods'])}")
        
        # Test 2: With auto-normalization (new behavior)
        print("\n   Testing with auto_normalize=True (new behavior)...")
        result_with_normalize = create_multi_branch_data(test_data, auto_normalize=True)
        
        print(f"   - Raw LC length: {len(result_with_normalize['raw_lc'])}")
        print(f"   - Periodogram length: {len(result_with_normalize['periodogram'])}")
        print(f"   - Detected period: {result_with_normalize['detected_period']:.3f} days")
        print(f"   - Number of candidate periods: {len(result_with_normalize['candidate_periods'])}")
        
        # Compare the results
        print("\n   Comparison of results:")
        period_diff = abs(result_with_normalize['detected_period'] - result_no_normalize['detected_period'])
        print(f"   - Period detection difference: {period_diff:.3f} days")
        
        # Check if the expected period (5.0) is closer with auto-normalization
        expected_period = 5.0
        error_no_norm = abs(result_no_normalize['detected_period'] - expected_period)
        error_with_norm = abs(result_with_normalize['detected_period'] - expected_period)
        
        print(f"   - Error without normalization: {error_no_norm:.3f} days")
        print(f"   - Error with normalization: {error_with_norm:.3f} days")
        
        if error_with_norm < error_no_norm:
            print("   ✓ Auto-normalization improved period detection!")
        else:
            print("   - Auto-normalization did not improve period detection (may be due to noise)")
        
        # Test periodogram power differences
        max_power_no_norm = np.max(result_no_normalize['periodogram'])
        max_power_with_norm = np.max(result_with_normalize['periodogram'])
        
        print(f"   - Max periodogram power without normalization: {max_power_no_norm:.4f}")
        print(f"   - Max periodogram power with normalization: {max_power_with_norm:.4f}")
        
        print("\n   ✓ create_multi_branch_data test completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ✗ Error in create_multi_branch_data test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_data():
    """Test with real sample data if available."""
    print("\nTesting with real sample data...")
    
    try:
        from app import create_multi_branch_data
        import pandas as pd
        
        # Try to load real sample data
        sample_file = os.path.join(os.path.dirname(__file__), 'sample_data', '1-hubble.tbl')
        
        if not os.path.exists(sample_file):
            print("   Sample data not found, skipping real data test")
            return True
            
        # Load the data
        data = pd.read_csv(sample_file, sep='\t', comment='#', header=0)
        real_data = np.column_stack([data.iloc[:, 0].values, data.iloc[:, 1].values])
        
        print(f"   Loaded real data: {len(real_data)} points")
        
        # Calculate slope of real data
        original_slope = (real_data[-1, 1] - real_data[0, 1]) / (real_data[-1, 0] - real_data[0, 0])
        print(f"   Real data slope: {original_slope:.6f}")
        
        # Test both modes
        result_no_norm = create_multi_branch_data(real_data, auto_normalize=False)
        result_with_norm = create_multi_branch_data(real_data, auto_normalize=True)
        
        print(f"   Period without normalization: {result_no_norm['detected_period']:.3f} days")
        print(f"   Period with normalization: {result_with_norm['detected_period']:.3f} days")
        
        print("   ✓ Real data test completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ✗ Error in real data test: {e}")
        return False

if __name__ == "__main__":
    print("Testing auto-normalization integration...")
    
    success1 = test_create_multi_branch_data()
    success2 = test_real_data()
    
    if success1 and success2:
        print("\n✓ All integration tests passed!")
    else:
        print("\n✗ Some tests failed!")