"""
Model training pipeline for CNN period validation using CSV data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any, Union
import os
import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

from config import Config, CLASS_NAMES
from models import TrainingDataPoint, ModelTrainingResult
from csv_data_loader import CSVDataLoader, parse_star_range
from period_detection import PeriodValidationCNN, phase_fold_data


class ModelTrainer:
    """Handles training of the CNN model using CSV data."""
    
    def __init__(self, model_save_path: str = None):
        """Initialize the model trainer."""
        self.model_save_path = model_save_path or Config.MODEL_SAVE_PATH
        self.device = Config.DEVICE
        self.label_encoder = LabelEncoder()
        
        # Training parameters
        self.batch_size = 32
        self.n_epochs = 100
        self.learning_rate = 0.001
        self.validation_split = 0.2
        
    def load_training_data(self, stars_to_extract: Union[str, List[int], None] = None, 
                          csv_file_path: str = None) -> List[TrainingDataPoint]:
        """
        Load training data from CSV file.
        
        Args:
            stars_to_extract: Star range specification
            csv_file_path: Path to CSV file (overrides Config.CSV_TRAINING_DATA_PATH)
            
        Returns:
            List of TrainingDataPoint objects
        """
        csv_path = csv_file_path or Config.CSV_TRAINING_DATA_PATH
        
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV training data file not found: {csv_path}")
            
        return self._load_training_data_from_csv(stars_to_extract, csv_path)
    
    def _load_training_data_from_csv(self, stars_to_extract: Union[str, List[int], None] = None, 
                                    csv_file_path: str = None) -> List[TrainingDataPoint]:
        """Load training data from CSV file."""
        csv_path = csv_file_path or Config.CSV_TRAINING_DATA_PATH
        loader = CSVDataLoader(csv_path)
        return loader.extract_training_data(stars_to_extract)
    
    def prepare_training_samples(self, training_data: List[TrainingDataPoint], 
                               n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training samples by phase-folding the data.
        
        Args:
            training_data: List of TrainingDataPoint objects
            n_bins: Number of phase bins for phase-folded curves
            
        Returns:
            Tuple of (X, y, categories) where:
            - X: Phase-folded light curves (n_samples, n_bins)
            - y: Encoded category labels
            - categories: Original category names
        """
        print(f"Preparing {len(training_data)} training samples with {n_bins} phase bins...")
        
        phase_folded_curves = []
        categories = []
        valid_samples = 0
        
        for i, data_point in enumerate(training_data):
            try:
                # Skip if no valid period
                if data_point.period_1 is None or data_point.period_1 <= 0:
                    continue
                
                # Create data array for phase folding
                time_series = np.array(data_point.time_series)
                flux_series = np.array(data_point.flux_series)
                
                if len(time_series) == 0 or len(flux_series) == 0:
                    continue
                
                data_array = np.column_stack([time_series, flux_series])
                
                # Phase-fold the data
                folded_curve = phase_fold_data(data_array, data_point.period_1, n_bins=n_bins)
                
                # Skip if folding failed or resulted in invalid data
                if np.all(folded_curve == 0) or np.all(np.isnan(folded_curve)) or len(folded_curve) != n_bins:
                    continue
                
                # Normalize the flux values
                folded_curve = (folded_curve - np.mean(folded_curve)) / (np.std(folded_curve) + 1e-8)
                
                phase_folded_curves.append(folded_curve)
                categories.append(data_point.lc_category)
                valid_samples += 1
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(training_data)} samples ({valid_samples} valid)")
                    
            except Exception as e:
                print(f"Error processing sample {i} (star {data_point.star_number}): {e}")
                continue
        
        if len(phase_folded_curves) == 0:
            raise ValueError("No valid training samples generated")
        
        print(f"Successfully prepared {len(phase_folded_curves)} valid training samples")
        
        # Convert to numpy arrays
        X = np.array(phase_folded_curves)
        
        # Ensure X is 2D (samples, features)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        elif len(X.shape) > 2:
            X = X.reshape(len(X), -1)
        
        # Encode categories
        unique_categories = list(set(categories))
        print(f"Found {len(unique_categories)} unique categories: {unique_categories}")
        
        self.label_encoder.fit(unique_categories)
        y = self.label_encoder.transform(categories)
        
        return X, y, np.array(categories)
    
    def create_pytorch_datasets(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create PyTorch DataLoaders for training and validation."""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42, stratify=y
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        return train_loader, val_loader
    
    def train_model(self, train_loader: torch.utils.data.DataLoader, 
                   val_loader: torch.utils.data.DataLoader,
                   input_size: int, num_classes: int) -> Tuple[PeriodValidationCNN, List[float], List[float]]:
        """Train the CNN model."""
        print(f"Training CNN model with input size {input_size} and {num_classes} classes...")
        
        # Create model
        model = PeriodValidationCNN(input_size=input_size, num_classes=num_classes)
        model.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Training history
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"Starting training for {self.n_epochs} epochs...")
        
        for epoch in range(self.n_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # Calculate average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.n_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'input_size': input_size,
                    'num_classes': num_classes,
                    'label_encoder': self.label_encoder,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, self.model_save_path)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return model, train_losses, val_losses
    
    def train_from_csv(self, stars_to_extract: Union[str, List[int], None] = None,
                      csv_file_path: str = None) -> ModelTrainingResult:
        """
        Train model using CSV data.
        
        Args:
            stars_to_extract: Star range specification
            csv_file_path: Path to CSV file (overrides Config.CSV_TRAINING_DATA_PATH)
            
        Returns:
            ModelTrainingResult with training information
        """
        try:
            print("Loading training data from CSV...")
            training_data = self.load_training_data(stars_to_extract, csv_file_path)
            
            if len(training_data) == 0:
                return ModelTrainingResult(
                    success=False,
                    error="No training data loaded",
                    epochs_trained=0,
                    final_loss=0.0,
                    model_path="",
                    training_samples=0
                )
            
            print(f"Loaded {len(training_data)} training examples")
            
            # Prepare training samples
            X, y, categories = self.prepare_training_samples(training_data)
            
            # Create PyTorch datasets
            train_loader, val_loader = self.create_pytorch_datasets(X, y)
            
            # Train model
            input_size = X.shape[1]
            num_classes = len(self.label_encoder.classes_)
            
            model, train_losses, val_losses = self.train_model(
                train_loader, val_loader, input_size, num_classes
            )
            
            # Save final model with metadata
            training_metadata = {
                'training_samples': len(training_data),
                'valid_samples': len(X),
                'num_classes': num_classes,
                'input_size': input_size,
                'epochs_trained': len(train_losses),
                'final_loss': val_losses[-1] if val_losses else 0.0,
                'class_names': self.label_encoder.classes_.tolist(),
                'stars_used': stars_to_extract,
                'csv_file_path': csv_file_path or Config.CSV_TRAINING_DATA_PATH
            }
            
            # Save complete model
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'num_classes': num_classes,
                'label_encoder': self.label_encoder,
                'training_metadata': training_metadata,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, self.model_save_path)
            
            print(f"Model saved to: {self.model_save_path}")
            
            return ModelTrainingResult(
                success=True,
                epochs_trained=len(train_losses),
                final_loss=val_losses[-1] if val_losses else 0.0,
                model_path=self.model_save_path,
                training_samples=len(training_data)
            )
            
        except Exception as e:
            print(f"Error during training: {e}")
            return ModelTrainingResult(
                success=False,
                error=str(e),
                epochs_trained=0,
                final_loss=0.0,
                model_path="",
                training_samples=0
            )


def load_trained_model(model_path: str = None) -> Tuple[PeriodValidationCNN, LabelEncoder, Dict]:
    """Load a trained model from file."""
    model_path = model_path or Config.MODEL_SAVE_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    
    # Create model
    model = PeriodValidationCNN(
        input_size=checkpoint['input_size'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load label encoder
    label_encoder = checkpoint['label_encoder']
    
    # Get metadata
    metadata = checkpoint.get('training_metadata', {})
    
    return model, label_encoder, metadata


def get_model_info(model_path: str = None) -> Dict[str, Any]:
    """Get information about a trained model."""
    model_path = model_path or Config.MODEL_SAVE_PATH
    
    if not os.path.exists(model_path):
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get file size
        file_size = os.path.getsize(model_path)
        
        # Get modification time
        import datetime
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(model_path))
        
        info = {
            'model_path': model_path,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'last_modified': mod_time.isoformat(),
            'input_size': checkpoint.get('input_size', 'unknown'),
            'num_classes': checkpoint.get('num_classes', 'unknown'),
            'training_metadata': checkpoint.get('training_metadata', {}),
            'has_label_encoder': 'label_encoder' in checkpoint
        }
        
        return info
        
    except Exception as e:
        print(f"Error reading model info: {e}")
        return None


def model_exists(model_path: str = None) -> bool:
    """Check if a trained model exists."""
    model_path = model_path or Config.MODEL_SAVE_PATH
    return os.path.exists(model_path)


# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN model for period validation using CSV data")
    parser.add_argument('--stars', type=str, help='Star range to use for training (e.g., "30:50", "42", or comma-separated "1,5,10")')
    parser.add_argument('--csv-file', type=str, help='Path to CSV training data file')
    parser.add_argument('--model-path', type=str, help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Parse stars argument
    stars_to_extract = None
    if args.stars:
        if ',' in args.stars:
            try:
                stars_to_extract = [int(s.strip()) for s in args.stars.split(',')]
            except ValueError as e:
                print(f"Error parsing star list '{args.stars}': {e}")
                sys.exit(1)
        else:
            stars_to_extract = args.stars
    
    # Check if CSV file exists
    csv_file = args.csv_file or Config.CSV_TRAINING_DATA_PATH
    if not os.path.exists(csv_file):
        print(f"CSV training data file not found: {csv_file}")
        print("Please provide a valid CSV file using --csv-file argument or set CSV_TRAINING_DATA_PATH in config")
        sys.exit(1)
    
    # Initialize trainer
    trainer = ModelTrainer(model_save_path=args.model_path)
    trainer.n_epochs = args.epochs
    trainer.batch_size = args.batch_size
    trainer.learning_rate = args.learning_rate
    
    print(f"Training CNN model using CSV data from: {csv_file}")
    if stars_to_extract:
        print(f"Using stars: {stars_to_extract}")
    
    # Train model
    result = trainer.train_from_csv(stars_to_extract=stars_to_extract, csv_file_path=args.csv_file)
    
    if result.success:
        print(f"\n✓ Training completed successfully!")
        print(f"  Model saved to: {result.model_path}")
        print(f"  Training samples: {result.training_samples}")
        print(f"  Epochs trained: {result.epochs_trained}")
        print(f"  Final loss: {result.final_loss:.4f}")
    else:
        print(f"\n✗ Training failed: {result.error}")
        sys.exit(1)