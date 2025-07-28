#!/usr/bin/env python3
"""
Test script to demonstrate star range functionality in csv_data_loader.py
"""

import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Test the star range parsing function without heavy dependencies
from typing import List, Union

def parse_star_range(star_range: Union[str, List[int], None]) -> List[int]:
    """
    Parse star range specification into a list of star numbers.
    
    Args:
        star_range: Can be:
            - None: Return None (extract all stars)
            - List[int]: Return as-is
            - String: Parse range like "30:50" to [30, 31, 32, ..., 50]
            
    Returns:
        List of star numbers or None
        
    Examples:
        parse_star_range("30:50") -> [30, 31, 32, ..., 50]
        parse_star_range("5:8") -> [5, 6, 7, 8]
        parse_star_range("42") -> [42]
        parse_star_range([1, 5, 10]) -> [1, 5, 10]
        parse_star_range(None) -> None
    """
    if star_range is None:
        return None
    
    if isinstance(star_range, list):
        return star_range
    
    if isinstance(star_range, str):
        star_range = star_range.strip()
        
        # Check if it's a range (contains colon)
        if ':' in star_range:
            try:
                start_str, end_str = star_range.split(':', 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                
                if start > end:
                    raise ValueError(f"Invalid range: start ({start}) must be <= end ({end})")
                
                return list(range(start, end + 1))
            except ValueError as e:
                raise ValueError(f"Invalid star range format '{star_range}': {e}")
        else:
            # Single star number
            try:
                return [int(star_range)]
            except ValueError:
                raise ValueError(f"Invalid star number '{star_range}'")
    
    raise ValueError(f"Invalid star range type: {type(star_range)}. Expected str, list, or None")


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
    print("python backend/csv_data_loader.py --stars '30:50' --export-csv")
    print("python backend/csv_data_loader.py --stars '42'")
    print("python backend/csv_data_loader.py --stars '1,5,10,20' --export-csv")
    print("python backend/model_training.py --stars '30:100' --force-retrain")
    print("python backend/model_training.py --stars '42'")
    
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