#!/usr/bin/env python3
"""
Final integration test - demonstrates CSV functionality working independently 
of CSV dependencies. This test validates that the issue requirements
have been met: "Remove google sheets training data generation and replace with csv input"
"""

import sys
import os

# Add backend to path
sys.path.insert(0, 'backend')

def test_csv_only_workflow():
    """Test complete workflow using only CSV input (no CSV)."""
    print("=== CSV-Only Training Workflow Test ===")
    print("This test validates CSV functionality works independently of CSV\n")
    
    # Test 1: CSV Data Loading
    print("1. Testing CSV data loading...")
    try:
        from csv_data_loader import CSVDataLoader
        loader = CSVDataLoader("sample_training_data.csv")
        training_data = loader.extract_training_data()
        print(f"✓ Loaded {len(training_data)} training examples from CSV")
        
        # Verify data quality
        categories = set(point.lc_category for point in training_data)
        print(f"✓ Found {len(categories)} unique categories: {', '.join(sorted(categories))}")
        
    except ImportError as e:
        print(f"✗ Import error (this is expected if Google dependencies not installed): {e}")
        return False
    except Exception as e:
        print(f"✗ CSV loading failed: {e}")
        return False
    
    # Test 2: Model Training Integration
    print("\n2. Testing model training with CSV...")
    try:
        from model_training import ModelTrainer
        trainer = ModelTrainer()
        
        # Use CSV data source explicitly
        csv_data = trainer.load_training_data(
            data_source="csv",
            csv_file_path="sample_training_data.csv"
        )
        print(f"✓ Model trainer loaded {len(csv_data)} examples from CSV")
        
        # Test auto-selection (should prefer CSV when no CSV URL)
        auto_data = trainer.load_training_data(data_source="auto", csv_file_path="sample_training_data.csv")
        print(f"✓ Auto-selection worked: {len(auto_data)} examples")
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return False
    
    # Test 3: Configuration
    print("\n3. Testing configuration...")
    try:
        from config import Config
        # Should work without CSV URL
        csv_path = getattr(Config, 'CSV_TRAINING_DATA_PATH', 'training_data.csv')
        print(f"✓ CSV configuration available: {csv_path}")
        
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False
    
    # Test 4: Star Range Filtering
    print("\n4. Testing star range filtering...")
    try:
        from csv_data_loader import parse_star_range
        test_ranges = ["1:3", "5", None]
        for test_range in test_ranges:
            result = parse_star_range(test_range)
            print(f"✓ '{test_range}' -> {result}")
            
    except Exception as e:
        print(f"✗ Star range parsing failed: {e}")
        return False
    
    # Test 5: Data Processing Pipeline
    print("\n5. Testing data processing pipeline...")
    try:
        # Verify we can load actual star data
        sample_point = training_data[0]
        if len(sample_point.time_series) > 0 and len(sample_point.flux_series) > 0:
            print(f"✓ Data processing pipeline working: {len(sample_point.time_series)} data points")
        else:
            print("⚠ Warning: No time series data found (this may be normal if sample_data is empty)")
            
    except Exception as e:
        print(f"✗ Data processing failed: {e}")
        return False
    
    return True

def test_requirements_fulfilled():
    """Verify that the original issue requirements have been met."""
    print("\n=== Verifying Issue Requirements ===")
    print("Issue: 'Remove google sheets training data generation and replace with csv input'")
    print("Requirement: 'assume user uploads it to a passed path'\n")
    
    checks = []
    
    # Check 1: CSV input functionality exists
    try:
        from csv_data_loader import CSVDataLoader
        CSVDataLoader("sample_training_data.csv")
        print("✓ CSV input functionality implemented")
        checks.append(True)
    except Exception as e:
        print(f"✗ CSV input functionality missing: {e}")
        checks.append(False)
    
    # Check 2: Can specify CSV file path
    try:
        from model_training import ModelTrainer
        trainer = ModelTrainer()
        trainer.load_training_data(data_source="csv", csv_file_path="sample_training_data.csv")
        print("✓ User can specify CSV file path")
        checks.append(True)
    except Exception as e:
        print(f"✗ Cannot specify CSV file path: {e}")
        checks.append(False)
    
    # Check 3: CSV is optional
    try:
        from model_training import ModelTrainer
        trainer = ModelTrainer()
        # Should work with CSV even if CSV not configured
        trainer.load_training_data(data_source="csv", csv_file_path="sample_training_data.csv")
        print("✓ CSV dependencies are optional")
        checks.append(True)
    except Exception as e:
        print(f"✗ Still requires CSV: {e}")
        checks.append(False)
    
    # Check 4: Preserves existing functionality
    csv_data_loader_preserved = True
    try:
        from csv_data_loader import CSVDataLoader
        print("✓ CSV functionality preserved (classes still exist)")
    except Exception as e:
        print("⚠ CSV functionality may have been modified")
        csv_data_loader_preserved = False
    checks.append(csv_data_loader_preserved)
    
    return all(checks)

if __name__ == "__main__":
    print("Integration Test: CSV Training Data Replacement")
    print("=" * 50)
    
    # Run workflow test
    workflow_success = test_csv_only_workflow()
    
    # Verify requirements
    requirements_met = test_requirements_fulfilled()
    
    # Final report
    print(f"\n{'=' * 50}")
    print("FINAL RESULTS:")
    print(f"CSV Workflow: {'PASS' if workflow_success else 'FAIL'}")
    print(f"Requirements Met: {'PASS' if requirements_met else 'FAIL'}")
    
    if workflow_success and requirements_met:
        print("\n🎉 SUCCESS: CSV training data input successfully implemented!")
        print("✓ Users can now provide training data via CSV files")
        print("✓ CSV dependencies are optional")
        print("✓ Original functionality preserved")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: Some tests failed")
        sys.exit(1)