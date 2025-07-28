#!/usr/bin/env python3
"""
Test the complete training workflow with sample data.
This simulates the CSV data and tests the end-to-end training process.
"""

import pandas as pd
import numpy as np
from backend.model_training import ModelTrainer
from backend.models import TrainingDataPoint
from backend.csv_data_loader import CSVDataLoader

def create_mock_training_data():
    """Create mock training data that simulates CSV structure."""
    
    # Sample data: star_number, period_1, period_2, lc_category
    training_samples = [
        (1, 2.5, -9, "dipper"),       # Star 1: dipper type
        (2, 7.8, -9, "distant peaks"), # Star 2: distant peaks
        (3, 15.3, -9, "sinusoidal"),  # Star 3: sinusoidal
        (4, 5.2, -9, "close peak"),   # Star 4: close peak  
        (5, 12.7, -9, "other"),       # Star 5: other/irregular
    ]
    
    training_data = []
    loader = CSVDataLoader('https://dummy-url')
    
    for star_number, period_1, period_2, category in training_samples:
        try:
            # Load the actual time series data
            time_series, flux_series = loader._load_star_data(star_number)
            
            if len(time_series) > 0:
                # Convert -9 to None for periods
                p1 = None if period_1 == -9 else period_1
                p2 = None if period_2 == -9 else period_2
                
                training_point = TrainingDataPoint(
                    star_number=star_number,
                    period_1=p1,
                    period_2=p2,
                    lc_category=category,
                    time_series=time_series.tolist(),
                    flux_series=flux_series.tolist()
                )
                training_data.append(training_point)
                print(f"Added training data for star {star_number} ({category})")
            
        except Exception as e:
            print(f"Error loading star {star_number}: {e}")
    
    return training_data

def test_training_workflow():
    """Test the complete training workflow."""
    print("=== Testing Complete Training Workflow ===\n")
    
    try:
        # Create mock training data
        print("1. Creating mock training data...")
        training_data = create_mock_training_data()
        print(f"   Created {len(training_data)} training samples\n")
        
        # Initialize trainer
        print("2. Initializing model trainer...")
        trainer = ModelTrainer(model_save_path="test_model.pth")
        
        # Preprocess data
        print("3. Preprocessing training data...")
        folded_curves, confidence_labels, class_labels, class_names = trainer.preprocess_training_data(training_data)
        print(f"   Generated {len(folded_curves)} folded curves")
        print(f"   Classes: {class_names}\n")
        
        # Create data loaders
        print("4. Creating data loaders...")
        train_loader, val_loader = trainer.create_data_loaders(folded_curves, confidence_labels, class_labels)
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}\n")
        
        # Train model (small number of epochs for testing)
        print("5. Training model (quick test)...")
        trainer.n_epochs = 20  # Quick training for testing
        model = trainer.train_model(train_loader, val_loader, len(class_names))
        print("   Model training completed\n")
        
        # Save model
        print("6. Saving model...")
        model_path = trainer.save_model(model, class_names, 0.1, 20, len(folded_curves))
        print(f"   Model saved to: {model_path}\n")
        
        print("=== Training Workflow Test SUCCESSFUL ===")
        return True
        
    except Exception as e:
        print(f"=== Training Workflow Test FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_training_workflow()