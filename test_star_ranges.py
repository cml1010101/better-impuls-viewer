#!/usr/bin/env python3
"""
Test script to demonstrate star range functionality in csv_training_data.py
"""

import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Test the star range parsing function
from csv_training_data import parse_star_range


def main():
    print("Testing star range parsing functionality:")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        "30:50",      # Range
        "42",         # Single star
        "1:5",        # Small range
        "100:105",    # Large numbers
        [1, 5, 10],   # List input
        None,         # No restriction
    ]
    
    for test_case in test_cases:
        try:
            result = parse_star_range(test_case)
            if result is None:
                print(f"parse_star_range({test_case!r}) -> None (all stars)")
            elif len(result) <= 10:
                print(f"parse_star_range({test_case!r}) -> {result}")
            else:
                print(f"parse_star_range({test_case!r}) -> [{result[0]}, {result[1]}, ..., {result[-2]}, {result[-1]}] ({len(result)} stars)")
        except Exception as e:
            print(f"parse_star_range({test_case!r}) -> ERROR: {e}")
    
    print("\n" + "=" * 50)
    print("Command line usage examples:")
    print("python backend/model_training.py --csv-file training_data.csv --stars '30:50' --force-retrain")
    print("python backend/model_training.py --csv-file training_data.csv --stars '42'")
    print("python backend/model_training.py --csv-file training_data.csv --stars '1,5,10,20' --force-retrain")
    print("python backend/model_training.py --csv-file training_data.csv --stars '30:100' --force-retrain")
    print("python backend/model_training.py --csv-file training_data.csv --stars '42'")
    
    print("\n" + "=" * 50)
    print("Available functionality:")
    print("• Extract training data for specific star ranges")
    print("• Export phase-folded light curves to CSV")
    print("• Train CNN models on subset of stars")
    print("• Support for range notation (30:50)")
    print("• Support for individual stars (42)")
    print("• Support for comma-separated lists (1,5,10)")


if __name__ == "__main__":
    main()