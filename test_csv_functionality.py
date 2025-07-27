#!/usr/bin/env python3
"""
Test CSV data loading functionality.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, 'backend')

from google_sheets import CSVDataLoader, parse_star_range
from model_training import ModelTrainer
from config import Config

def test_csv_data_loader():
    """Test the CSV data loader with sample data."""
    print("=== Testing CSV Data Loader ===")
    
    csv_file = "sample_training_data.csv"
    if not os.path.exists(csv_file):
        print(f"Sample CSV file not found: {csv_file}")
        return False
    
    try:
        # Test basic loading
        loader = CSVDataLoader(csv_file)
        df = loader.load_raw_data()
        print(f"✓ Successfully loaded CSV with {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Test training data extraction
        training_data = loader.extract_training_data()
        print(f"✓ Extracted {len(training_data)} training examples")
        
        if training_data:
            sample = training_data[0]
            print(f"✓ Sample training point: Star {sample.star_number}, Period {sample.period_1}, Category {sample.lc_category}")
        
        # Test star range filtering
        filtered_data = loader.extract_training_data(stars_to_extract="1:3")
        print(f"✓ Filtered data (stars 1-3): {len(filtered_data)} examples")
        
        return True
        
    except Exception as e:
        print(f"✗ CSV loader test failed: {e}")
        return False

def test_model_trainer_csv():
    """Test model trainer with CSV data source."""
    print("\n=== Testing Model Trainer with CSV ===")
    
    try:
        trainer = ModelTrainer()
        
        # Test CSV loading
        training_data = trainer.load_training_data(
            data_source="csv", 
            csv_file_path="sample_training_data.csv"
        )
        print(f"✓ Model trainer loaded {len(training_data)} examples from CSV")
        
        return True
        
    except Exception as e:
        print(f"✗ Model trainer CSV test failed: {e}")
        return False

def test_star_range_parsing():
    """Test star range parsing functionality."""
    print("\n=== Testing Star Range Parsing ===")
    
    test_cases = [
        ("1:5", [1, 2, 3, 4, 5]),
        ("42", [42]),
        ("10:12", [10, 11, 12]),
        (None, None),
        ([1, 3, 5], [1, 3, 5])
    ]
    
    all_passed = True
    for input_range, expected in test_cases:
        try:
            result = parse_star_range(input_range)
            if result == expected:
                print(f"✓ '{input_range}' -> {result}")
            else:
                print(f"✗ '{input_range}' -> {result} (expected {expected})")
                all_passed = False
        except Exception as e:
            print(f"✗ '{input_range}' -> ERROR: {e}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("Testing CSV Training Data Functionality\n")
    
    # Run tests
    csv_test = test_csv_data_loader()
    trainer_test = test_model_trainer_csv()
    range_test = test_star_range_parsing()
    
    # Summary
    print(f"\n=== Test Results ===")
    print(f"CSV Data Loader: {'PASS' if csv_test else 'FAIL'}")
    print(f"Model Trainer CSV: {'PASS' if trainer_test else 'FAIL'}")
    print(f"Star Range Parsing: {'PASS' if range_test else 'FAIL'}")
    
    if all([csv_test, trainer_test, range_test]):
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)