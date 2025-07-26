"""
Model training pipeline for CNN period validation using Google Sheets data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

from config import Config
from models import TrainingDataPoint, ModelTrainingResult
from google_sheets import GoogleSheetsLoader
from period_detection import PeriodValidationCNN, phase_fold_data


class ModelTrainer:
    """Handles training of the CNN model using Google Sheets data."""
    
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
        
    def load_training_data(self) -> List[TrainingDataPoint]:
        """Load training data from Google Sheets."""
        if not Config.GOOGLE_SHEET_URL:
            raise ValueError("GOOGLE_SHEET_URL not configured")
        
        loader = GoogleSheetsLoader()
        training_data = loader.extract_training_data()
        
        if len(training_data) == 0:
            raise ValueError("No training data loaded from Google Sheets")
        
        print(f"Loaded {len(training_data)} training examples")
        return training_data
    
    def preprocess_training_data(self, training_data: List[TrainingDataPoint]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Preprocess training data into tensors suitable for CNN training.
        
        Returns:
            Tuple of (folded_curves, confidence_labels, class_labels, class_names)
        """
        folded_curves = []
        confidence_labels = []
        class_labels = []
        
        # Map classification categories to standard labels
        category_mapping = {
            'dipper': 'dipper',
            'distant_peaks': 'distant_peaks', 
            'close_peak': 'close_peak',
            'sinusoidal': 'sinusoidal',
            'other': 'other'
        }
        
        print("Processing training examples...")
        processed_count = 0
        
        # Process each training example
        for data_point in training_data:
            time_series = np.array(data_point.time_series)
            flux_series = np.array(data_point.flux_series)
            
            # Create data array for phase folding
            data_array = np.column_stack([time_series, flux_series])
            
            # Map the category to standard form
            mapped_category = category_mapping.get(data_point.lc_category, 'other')
            
            # Process each valid period for this star
            periods_to_process = []
            if data_point.period_1 is not None and data_point.period_1 > 0:
                periods_to_process.append(data_point.period_1)
            if data_point.period_2 is not None and data_point.period_2 > 0:
                periods_to_process.append(data_point.period_2)
            
            for period in periods_to_process:
                try:
                    # Phase-fold the data at this period
                    folded_curve = phase_fold_data(data_array, period, n_bins=100)
                    
                    # Skip if folding failed or resulted in all zeros
                    if np.all(folded_curve == 0) or np.all(np.isnan(folded_curve)):
                        continue
                    
                    # Generate confidence label based on period quality and category
                    confidence = self._calculate_period_confidence(data_array, period, mapped_category)
                    
                    # Store the results
                    folded_curves.append(folded_curve)
                    confidence_labels.append(confidence)
                    class_labels.append(mapped_category)
                    processed_count += 1
                    
                    if processed_count % 50 == 0:
                        print(f"Processed {processed_count} training samples...")
                    
                except Exception as e:
                    print(f"Error processing period {period} for star {data_point.star_number}: {e}")
                    continue
        
        if len(folded_curves) == 0:
            raise ValueError("No valid folded curves generated from training data")
        
        # Encode class labels
        encoded_classes = self.label_encoder.fit_transform(class_labels)
        class_names = list(self.label_encoder.classes_)
        
        print(f"Generated {len(folded_curves)} training samples")
        print(f"Class distribution: {dict(zip(*np.unique(class_labels, return_counts=True)))}")
        print(f"Model classes: {class_names}")
        
        # Convert to tensors
        folded_curves_tensor = torch.tensor(folded_curves, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        confidence_tensor = torch.tensor(confidence_labels, dtype=torch.float32)
        class_tensor = torch.tensor(encoded_classes, dtype=torch.long)
        
        return folded_curves_tensor, confidence_tensor, class_tensor, class_names
    
    def _calculate_period_confidence(self, data: np.ndarray, period: float, category: str = 'other') -> float:
        """
        Calculate a confidence score for a given period based on the folded light curve quality and known category.
        
        This incorporates both curve quality metrics and category-specific confidence adjustments.
        """
        try:
            # Calculate some basic metrics for the folded curve
            folded_curve = phase_fold_data(data, period, n_bins=50)
            
            # Measure the "structure" in the folded curve
            # A good period should show clear patterns vs random noise
            
            # 1. Smoothness metric - good periods show smooth variations
            curve_diff = np.diff(folded_curve)
            smoothness = 1.0 / (1.0 + np.std(curve_diff))
            
            # 2. Amplitude metric - real periods often show significant amplitude
            amplitude = np.std(folded_curve)
            amplitude_score = min(amplitude / 0.5, 1.0)  # Normalize
            
            # 3. Periodicity check - fold again at 2x period and compare
            try:
                folded_2x = phase_fold_data(data, period * 2, n_bins=50)
                correlation = np.corrcoef(folded_curve, folded_2x)[0, 1]
                periodicity_score = max(0, correlation)  # Positive correlation is good
            except:
                periodicity_score = 0.5
            
            # 4. Category-specific adjustments
            category_confidence = {
                'dipper': 0.85,      # High confidence for clear dip patterns
                'sinusoidal': 0.90,  # Very high confidence for smooth patterns
                'distant_peaks': 0.75, # Good confidence for double peaks
                'close_peak': 0.70,  # Moderate confidence for close peaks
                'other': 0.50        # Lower baseline for irregular
            }
            
            base_confidence = category_confidence.get(category, 0.5)
            
            # Combine metrics with category bias
            quality_score = (smoothness + amplitude_score + periodicity_score) / 3
            confidence = (quality_score * 0.7) + (base_confidence * 0.3)
            
            # Add some small randomness to simulate real-world uncertainty
            confidence += np.random.normal(0, 0.05)
            confidence = np.clip(confidence, 0.1, 0.95)
            
            return float(confidence)
            
        except Exception:
            # Default confidence based on category if calculation fails
            return {'dipper': 0.7, 'sinusoidal': 0.8, 'distant_peaks': 0.6, 
                   'close_peak': 0.6, 'other': 0.4}.get(category, 0.5)
    
    def create_data_loaders(self, folded_curves: torch.Tensor, confidence_labels: torch.Tensor, 
                           class_labels: torch.Tensor) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create training and validation data loaders."""
        # Split data
        train_curves, val_curves, train_conf, val_conf, train_classes, val_classes = train_test_split(
            folded_curves, confidence_labels, class_labels, 
            test_size=self.validation_split, random_state=42, stratify=class_labels
        )
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(train_curves, train_conf, train_classes)
        val_dataset = torch.utils.data.TensorDataset(val_curves, val_conf, val_classes)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, train_loader: torch.utils.data.DataLoader, 
                   val_loader: torch.utils.data.DataLoader, num_classes: int) -> PeriodValidationCNN:
        """Train the CNN model."""
        # Initialize model
        model = PeriodValidationCNN(input_size=100, num_classes=num_classes)
        model = model.to(self.device)
        
        # Set up optimization
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        confidence_criterion = nn.MSELoss()
        classification_criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        print(f"Training model for {self.n_epochs} epochs...")
        
        for epoch in range(self.n_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_curves, batch_conf, batch_classes in train_loader:
                batch_curves = batch_curves.to(self.device)
                batch_conf = batch_conf.to(self.device)
                batch_classes = batch_classes.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                pred_confidence, pred_classes = model(batch_curves)
                
                # Calculate losses
                conf_loss = confidence_criterion(pred_confidence, batch_conf)
                class_loss = classification_criterion(pred_classes, batch_classes)
                
                # Combined loss
                total_loss = conf_loss + class_loss
                train_loss += total_loss.item()
                train_batches += 1
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_curves, batch_conf, batch_classes in val_loader:
                    batch_curves = batch_curves.to(self.device)
                    batch_conf = batch_conf.to(self.device)
                    batch_classes = batch_classes.to(self.device)
                    
                    pred_confidence, pred_classes = model(batch_curves)
                    
                    conf_loss = confidence_criterion(pred_confidence, batch_conf)
                    class_loss = classification_criterion(pred_classes, batch_classes)
                    total_loss = conf_loss + class_loss
                    
                    val_loss += total_loss.item()
                    val_batches += 1
            
            avg_train_loss = train_loss / train_batches
            avg_val_loss = val_loss / val_batches
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= patience:
                print(f"Epoch {epoch}/{self.n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model
    
    def save_model(self, model: PeriodValidationCNN, class_names: List[str], final_loss: float, 
                   epochs_trained: int, training_samples: int) -> str:
        """Save the trained model and metadata."""
        # Save model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': model.input_size,
                'num_classes': model.num_classes
            },
            'class_names': class_names,
            'label_encoder': self.label_encoder,
            'training_metadata': {
                'final_loss': final_loss,
                'epochs_trained': epochs_trained,
                'training_samples': training_samples
            }
        }, self.model_save_path)
        
        print(f"Model saved to: {self.model_save_path}")
        return self.model_save_path
    
    def train_from_google_sheets(self) -> ModelTrainingResult:
        """Complete training pipeline using Google Sheets data."""
        try:
            # Load and preprocess data
            print("Loading training data from Google Sheets...")
            training_data = self.load_training_data()
            
            print("Preprocessing training data...")
            folded_curves, confidence_labels, class_labels, class_names = self.preprocess_training_data(training_data)
            
            print("Creating data loaders...")
            train_loader, val_loader = self.create_data_loaders(folded_curves, confidence_labels, class_labels)
            
            print("Training model...")
            model = self.train_model(train_loader, val_loader, len(class_names))
            
            # Calculate final validation loss
            model.eval()
            total_loss = 0.0
            total_batches = 0
            
            with torch.no_grad():
                for batch_curves, batch_conf, batch_classes in val_loader:
                    batch_curves = batch_curves.to(self.device)
                    batch_conf = batch_conf.to(self.device)
                    batch_classes = batch_classes.to(self.device)
                    
                    pred_confidence, pred_classes = model(batch_curves)
                    
                    conf_loss = nn.MSELoss()(pred_confidence, batch_conf)
                    class_loss = nn.CrossEntropyLoss()(pred_classes, batch_classes)
                    total_loss += (conf_loss + class_loss).item()
                    total_batches += 1
            
            final_loss = total_loss / total_batches if total_batches > 0 else 0.0
            
            # Save model
            model_path = self.save_model(model, class_names, final_loss, self.n_epochs, len(folded_curves))
            
            return ModelTrainingResult(
                success=True,
                epochs_trained=self.n_epochs,
                final_loss=final_loss,
                model_path=model_path,
                training_samples=len(folded_curves)
            )
            
        except Exception as e:
            print(f"Training failed: {e}")
            return ModelTrainingResult(
                success=False,
                epochs_trained=0,
                final_loss=float('inf'),
                model_path="",
                training_samples=0
            )


def load_trained_model(model_path: str = None) -> Tuple[PeriodValidationCNN, List[str]]:
    """
    Load a trained model from disk.
    
    Returns:
        Tuple of (model, class_names)
    """
    model_path = model_path or Config.MODEL_SAVE_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    
    # Reconstruct model
    model_config = checkpoint['model_config']
    model = PeriodValidationCNN(
        input_size=model_config['input_size'],
        num_classes=model_config['num_classes']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    class_names = checkpoint['class_names']
    
    print(f"Loaded trained model from: {model_path}")
    print(f"Model classes: {class_names}")
    
    return model, class_names


# Example usage and training script
if __name__ == "__main__":
    # Check if Google Sheets URL is configured
    if not Config.validate():
        print("Configuration invalid. Please set GOOGLE_SHEET_URL in .env file")
        exit(1)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train model
    print("Starting model training from Google Sheets data...")
    result = trainer.train_from_google_sheets()
    
    if result.success:
        print(f"Training completed successfully!")
        print(f"Epochs trained: {result.epochs_trained}")
        print(f"Final loss: {result.final_loss:.4f}")
        print(f"Training samples: {result.training_samples}")
        print(f"Model saved to: {result.model_path}")
        
        # Test loading the model
        try:
            model, class_names = load_trained_model(result.model_path)
            print("Model loading test successful!")
        except Exception as e:
            print(f"Model loading test failed: {e}")
    else:
        print("Training failed. Check the logs for errors.")