#!/usr/bin/env python3
"""
Test the complete training workflow with sample data.
This creates sample CSV data and tests the end-to-end training process.
"""

import pandas as pd
import numpy as np
import tempfile
import os
from backend.model_training import ModelTrainer
from backend.models import TrainingDataPoint
from backend.csv_training_data import CSVTrainingDataLoader

def create_mock_csv_data():
    """Create mock CSV training data for testing."""
    
    # Create sample training data in CSV format
    training_data = [
        {"Star": 1, "LC_Category": "dipper", "CDIPS_period_1": 2.5, "CDIPS_period_2": -9},
        {"Star": 2, "LC_Category": "distant peaks", "CDIPS_period_1": 7.8, "CDIPS_period_2": -9},
        {"Star": 3, "LC_Category": "sinusoidal", "CDIPS_period_1": 15.3, "CDIPS_period_2": -9},
        {"Star": 4, "LC_Category": "close peak", "CDIPS_period_1": 5.2, "CDIPS_period_2": -9},
        {"Star": 5, "LC_Category": "other", "CDIPS_period_1": 12.7, "CDIPS_period_2": -9},
    ]
    
    # Create a temporary CSV file
    df = pd.DataFrame(training_data)
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    csv_path = temp_file.name
    temp_file.close()
    
    # Write CSV data
    df.to_csv(csv_path, index=False)
    
    return csv_path

def test_training_workflow():
    """Test the complete training workflow."""
    print("=== Testing Complete Training Workflow ===\n")
    
    try:
        # Create mock CSV training data
        print("1. Creating mock CSV training data...")
        csv_path = create_mock_csv_data()
        print(f"   Created CSV file: {csv_path}\n")
        
        # Test CSV loader
        print("2. Testing CSV data loader...")
        loader = CSVTrainingDataLoader(csv_path)
        training_data = loader.extract_training_data()
        print(f"   Loaded {len(training_data)} training samples\n")
        
        # Initialize trainer
        print("3. Initializing model trainer...")
        trainer = ModelTrainer(model_save_path="test_model.pth")
        
        # Preprocess data
        print("4. Preprocessing training data...")
        folded_curves, confidence_labels, class_labels, class_names = trainer.preprocess_training_data(training_data)
        print(f"   Generated {len(folded_curves)} folded curves")
        print(f"   Classes: {class_names}\n")
        
        # Create data loaders
        print("5. Creating data loaders...")
        train_loader, val_loader = trainer.create_data_loaders(folded_curves, confidence_labels, class_labels)
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}\n")
        
        # Train model (small number of epochs for testing)
        print("6. Training model (quick test)...")
        trainer.n_epochs = 20  # Quick training for testing
        model = trainer.train_model(train_loader, val_loader, len(class_names))
        print("   Model training completed\n")
        
        # Save model
        print("7. Saving model...")
        model_path = trainer.save_model(model, class_names, 0.1, 20, len(folded_curves))
        print(f"   Model saved to: {model_path}\n")
        
        # Cleanup
        print("8. Cleaning up...")
        os.unlink(csv_path)  # Remove temporary CSV file
        if os.path.exists("test_model.pth"):
            os.unlink("test_model.pth")  # Remove test model
        print("   Cleanup completed\n")
        
        print("=== Training Workflow Test SUCCESSFUL ===")
        return True
        
    except Exception as e:
        print(f"=== Training Workflow Test FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup on error
        try:
            if 'csv_path' in locals():
                os.unlink(csv_path)
            if os.path.exists("test_model.pth"):
                os.unlink("test_model.pth")
        except:
            pass
        return False

if __name__ == "__main__":
    test_training_workflow()