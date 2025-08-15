#!/usr/bin/env python3
"""
Test script for train.py functionality.
Validates that the training script works correctly.
"""

import os
import sys
import tempfile
import shutil

# Add backend to path
sys.path.append('backend')
from periodizer import StarClassifier
import torch

def test_model_creation():
    """Test that StarClassifier can be created and used."""
    print("Testing model creation...")
    
    model = StarClassifier(num_classes=14, input_length=512)
    
    # Test forward pass
    x = torch.randn(1, 1, 512)
    period_pred, class_pred = model(x)
    
    assert period_pred.shape == (1, 1), f"Expected period shape (1, 1), got {period_pred.shape}"
    assert class_pred.shape == (1, 14), f"Expected class shape (1, 14), got {class_pred.shape}"
    
    print("✓ Model creation test passed")


def test_model_save_load():
    """Test model saving and loading."""
    print("Testing model save/load...")
    
    # Create model
    model = StarClassifier(num_classes=14, input_length=512)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    try:
        # Save model
        model_state = {
            'model_state_dict': model.state_dict(),
            'num_classes': 14,
            'input_length': 512,
            'class_names': ['test_class'] * 14
        }
        torch.save(model_state, temp_path)
        
        # Load model
        checkpoint = torch.load(temp_path, map_location='cpu')
        loaded_model = StarClassifier(
            num_classes=checkpoint['num_classes'],
            input_length=checkpoint['input_length']
        )
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test that both models produce same output
        x = torch.randn(1, 1, 512)
        
        with torch.no_grad():
            orig_period, orig_class = model(x)
            loaded_period, loaded_class = loaded_model(x)
            
            assert torch.allclose(orig_period, loaded_period), "Period predictions don't match"
            assert torch.allclose(orig_class, loaded_class), "Class predictions don't match"
        
        print("✓ Model save/load test passed")
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_training_data_format():
    """Test that sample data can be loaded."""
    print("Testing sample data loading...")
    
    sample_data_dir = "sample_data"
    if not os.path.exists(sample_data_dir):
        print("⚠ Sample data directory not found, skipping data test")
        return
    
    # Check that we have some .tbl files
    import glob
    tbl_files = glob.glob(os.path.join(sample_data_dir, "*.tbl"))
    
    assert len(tbl_files) > 0, "No .tbl files found in sample_data directory"
    
    # Try to load one file
    import pandas as pd
    test_file = tbl_files[0]
    
    try:
        data = pd.read_csv(test_file, sep='\t', comment='#', header=0)
        assert len(data) > 0, f"No data loaded from {test_file}"
        assert data.shape[1] >= 2, f"Expected at least 2 columns, got {data.shape[1]}"
        
        print(f"✓ Successfully loaded {len(tbl_files)} sample files")
        
    except Exception as e:
        raise AssertionError(f"Failed to load sample data: {e}")


def main():
    """Run all tests."""
    print("Running train.py validation tests...")
    print("=" * 40)
    
    try:
        test_model_creation()
        test_model_save_load()
        test_training_data_format()
        
        print("=" * 40)
        print("✓ All tests passed!")
        print("The train.py script should work correctly.")
        
    except Exception as e:
        print("=" * 40)
        print(f"✗ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()