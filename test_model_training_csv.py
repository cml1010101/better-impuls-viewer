#!/usr/bin/env python3
"""
Test model training with CSV data source.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, 'backend')

from model_training import ModelTrainer

def test_model_training_csv():
    """Test model training with CSV data."""
    print("=== Testing Model Training with CSV ===")
    
    try:
        trainer = ModelTrainer()
        
        # Test loading training data from CSV
        training_data = trainer.load_training_data(
            data_source="csv",
            csv_file_path="sample_training_data.csv",
            stars_to_extract="1:3"
        )
        
        print(f"✓ Successfully loaded {len(training_data)} training examples")
        
        # Show data source preference
        print("\n=== Testing Data Source Auto-Selection ===")
        
        # Test auto selection (should prefer CSV since no Google Sheets URL)
        auto_data = trainer.load_training_data(
            data_source="auto",
            csv_file_path="sample_training_data.csv"
        )
        print(f"✓ Auto-selection loaded {len(auto_data)} examples")
        
        return True
        
    except Exception as e:
        print(f"✗ Model training test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_training_csv()
    if success:
        print("\n✓ Model training CSV tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Model training CSV tests failed!")
        sys.exit(1)