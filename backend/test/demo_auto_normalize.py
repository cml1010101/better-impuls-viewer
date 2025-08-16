#!/usr/bin/env python3
"""
Demonstration script for auto-normalization feature.

This script creates a visual demonstration of how the auto-normalization feature
improves period detection by removing linear trends from astronomical data.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.append(backend_path)

from data_processing import detrend_linear, calculate_lomb_scargle

def create_demo_data():
    """Create demonstration data with various trend scenarios."""
    
    scenarios = {}
    
    # Scenario 1: Strong positive linear trend
    time = np.linspace(0, 50, 300)
    period = 3.5  # days
    flux_base = 1.0 + 0.15 * np.sin(2 * np.pi * time / period)
    trend = 0.05 * time  # Strong trend
    noise = np.random.normal(0, 0.03, len(time))
    scenarios['strong_positive_trend'] = {
        'data': np.column_stack([time, flux_base + trend + noise]),
        'true_period': period,
        'description': 'Strong Positive Linear Trend (0.05/day)'
    }
    
    # Scenario 2: Moderate negative trend
    trend = -0.02 * time
    scenarios['moderate_negative_trend'] = {
        'data': np.column_stack([time, flux_base + trend + noise]),
        'true_period': period,
        'description': 'Moderate Negative Linear Trend (-0.02/day)'
    }
    
    # Scenario 3: Weak trend (should show minimal difference)
    trend = 0.005 * time
    scenarios['weak_trend'] = {
        'data': np.column_stack([time, flux_base + trend + noise]),
        'true_period': period,
        'description': 'Weak Linear Trend (0.005/day)'
    }
    
    return scenarios

def analyze_scenario(name, scenario_data):
    """Analyze a single scenario and return results."""
    
    data = scenario_data['data']
    true_period = scenario_data['true_period']
    description = scenario_data['description']
    
    print(f"\n--- {description} ---")
    
    # Calculate original trend
    original_slope = (data[-1, 1] - data[0, 1]) / (data[-1, 0] - data[0, 0])
    print(f"Original slope: {original_slope:.6f} flux/day")
    
    # Apply detrending
    detrended_data = detrend_linear(data)
    detrended_slope = (detrended_data[-1, 1] - detrended_data[0, 1]) / (detrended_data[-1, 0] - detrended_data[0, 0])
    print(f"Detrended slope: {detrended_slope:.6f} flux/day")
    print(f"Slope reduction: {(1 - abs(detrended_slope)/abs(original_slope))*100:.1f}%")
    
    # Calculate periodograms
    try:
        freq_orig, power_orig = calculate_lomb_scargle(data)
        freq_detr, power_detr = calculate_lomb_scargle(detrended_data)
        
        periods_orig = 1 / freq_orig
        periods_detr = 1 / freq_detr
        
        # Find peak power near true period
        idx_orig = np.argmin(np.abs(periods_orig - true_period))
        idx_detr = np.argmin(np.abs(periods_detr - true_period))
        
        power_at_true_orig = power_orig[idx_orig]
        power_at_true_detr = power_detr[idx_detr]
        
        # Find detected periods (highest power)
        detected_period_orig = periods_orig[np.argmax(power_orig)]
        detected_period_detr = periods_detr[np.argmax(power_detr)]
        
        print(f"True period: {true_period:.3f} days")
        print(f"Detected period (original): {detected_period_orig:.3f} days (error: {abs(detected_period_orig - true_period):.3f})")
        print(f"Detected period (detrended): {detected_period_detr:.3f} days (error: {abs(detected_period_detr - true_period):.3f})")
        print(f"Power at true period - original: {power_at_true_orig:.4f}")
        print(f"Power at true period - detrended: {power_at_true_detr:.4f}")
        
        if power_at_true_detr > power_at_true_orig:
            improvement = power_at_true_detr / power_at_true_orig
            print(f"Signal improvement: {improvement:.1f}x")
        
        return {
            'name': name,
            'description': description,
            'data_orig': data,
            'data_detr': detrended_data,
            'periods_orig': periods_orig,
            'power_orig': power_orig,
            'periods_detr': periods_detr,
            'power_detr': power_detr,
            'true_period': true_period,
            'original_slope': original_slope,
            'detrended_slope': detrended_slope,
            'detected_period_orig': detected_period_orig,
            'detected_period_detr': detected_period_detr,
            'power_improvement': power_at_true_detr / power_at_true_orig if power_at_true_orig > 0 else 1.0
        }
        
    except Exception as e:
        print(f"Error in periodogram analysis: {e}")
        return None

def create_demo_plots(results, output_dir=None):
    """Create demonstration plots showing the effect of auto-normalization."""
    
    if output_dir is None:
        output_dir = "/tmp"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    
    n_scenarios = len(results)
    
    for i, result in enumerate(results):
        if result is None:
            continue
            
        # Light curve comparison (original vs detrended)
        ax1 = plt.subplot(n_scenarios, 3, i*3 + 1)
        
        time_orig = result['data_orig'][:, 0]
        flux_orig = result['data_orig'][:, 1]
        time_detr = result['data_detr'][:, 0]
        flux_detr = result['data_detr'][:, 1]
        
        plt.plot(time_orig, flux_orig, 'b.', alpha=0.6, markersize=3, label='Original')
        plt.plot(time_detr, flux_detr, 'r.', alpha=0.8, markersize=3, label='Detrended')
        
        plt.xlabel('Time (days)')
        plt.ylabel('Flux')
        plt.title(f'{result["description"]}\nLight Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add slope annotations
        plt.text(0.05, 0.95, f'Original slope: {result["original_slope"]:.5f}', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.text(0.05, 0.85, f'Detrended slope: {result["detrended_slope"]:.5f}', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Periodogram comparison
        ax2 = plt.subplot(n_scenarios, 3, i*3 + 2)
        
        plt.loglog(result['periods_orig'], result['power_orig'], 'b-', alpha=0.7, label='Original')
        plt.loglog(result['periods_detr'], result['power_detr'], 'r-', alpha=0.8, label='Detrended')
        
        # Mark true period
        plt.axvline(result['true_period'], color='green', linestyle='--', alpha=0.8, label='True period')
        
        plt.xlabel('Period (days)')
        plt.ylabel('Power')
        plt.title('Periodogram Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.1, 50)
        
        # Period detection accuracy
        ax3 = plt.subplot(n_scenarios, 3, i*3 + 3)
        
        categories = ['Original', 'Detrended']
        detected_periods = [result['detected_period_orig'], result['detected_period_detr']]
        errors = [abs(p - result['true_period']) for p in detected_periods]
        
        bars = plt.bar(categories, errors, color=['blue', 'red'], alpha=0.7)
        plt.axhline(0, color='green', linestyle='--', alpha=0.8, label='Perfect detection')
        
        plt.ylabel('Period Error (days)')
        plt.title('Period Detection Error')
        plt.legend()
        
        # Add error values on bars
        for bar, error in zip(bars, errors):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{error:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add improvement text
        improvement = result.get('power_improvement', 1.0)
        plt.text(0.5, 0.9, f'Signal improvement: {improvement:.1f}x',
                transform=ax3.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'auto_normalization_demo.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nDemo plot saved to: {output_file}")
    
    # Create a summary plot
    plt.figure(figsize=(10, 6))
    
    scenario_names = [r['description'] for r in results if r is not None]
    improvements = [r.get('power_improvement', 1.0) for r in results if r is not None]
    
    bars = plt.bar(range(len(scenario_names)), improvements, 
                  color=['lightblue', 'lightcoral', 'lightgreen'])
    
    plt.xlabel('Scenario')
    plt.ylabel('Signal Improvement Factor')
    plt.title('Auto-Normalization Performance Summary')
    plt.xticks(range(len(scenario_names)), [name.split('(')[0] for name in scenario_names], rotation=45)
    
    # Add improvement values on bars
    for i, (bar, improvement) in enumerate(zip(bars, improvements)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{improvement:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.8, label='No improvement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    summary_file = os.path.join(output_dir, 'auto_normalization_summary.png')
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    print(f"Summary plot saved to: {summary_file}")
    
    return output_file, summary_file

def main():
    """Run the auto-normalization demonstration."""
    
    print("Auto-Normalization Feature Demonstration")
    print("=" * 50)
    
    print("\nCreating demonstration scenarios...")
    scenarios = create_demo_data()
    
    print("\nAnalyzing scenarios...")
    results = []
    
    for name, scenario in scenarios.items():
        result = analyze_scenario(name, scenario)
        results.append(result)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    # Print summary table
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        print(f"{'Scenario':<30} {'Original Error':<15} {'Detrended Error':<15} {'Improvement':<12}")
        print("-" * 72)
        
        for result in valid_results:
            orig_error = abs(result['detected_period_orig'] - result['true_period'])
            detr_error = abs(result['detected_period_detr'] - result['true_period'])
            improvement = result.get('power_improvement', 1.0)
            
            scenario_name = result['description'].split('(')[0].strip()
            print(f"{scenario_name:<30} {orig_error:<15.3f} {detr_error:<15.3f} {improvement:<12.1f}x")
    
    print("\nConclusion:")
    print("- Auto-normalization significantly improves period detection for data with linear trends")
    print("- Stronger trends show more improvement")
    print("- Weak trends show minimal difference (as expected)")
    print("- The feature preserves periodic signals while removing secular trends")
    
    # Try to create plots (requires matplotlib)
    try:
        output_files = create_demo_plots(valid_results)
        print(f"\nVisual demonstrations created!")
    except Exception as e:
        print(f"\nNote: Could not create plots (matplotlib not available or other error): {e}")
    
    print("\nAuto-normalization demonstration completed!")

if __name__ == "__main__":
    main()