#!/usr/bin/env python3
"""
Training script for Better Impuls Viewer StarClassifier model.
Trains the model in periodizer.py using the sample data.

This script:
1. Loads light curve data from sample_data/*.tbl files
2. Processes the data (removes outliers, detects periods)
3. Creates phase-folded training samples
4. Trains a CNN model for period prediction and star classification
5. Saves the trained model as 'trained_cnn_model.pth'

Usage:
    python backend/train.py

Requirements:
    - sample_data/ directory with .tbl files (relative to project root)
    - All dependencies from backend/requirements.txt

The script will create training data from all .tbl files in sample_data/,
generating multiple samples per file with different periods detected via
Lomb-Scargle periodogram analysis.

Output:
    - trained_cnn_model.pth: Saved PyTorch model with metadata
    - Console output showing training progress

Model Architecture:
    - Input: Phase-folded light curves (512 data points)
    - CNN feature extraction with 1D convolutions
    - Two heads: period regression + 14-class star classification
    - Classes: sinusoidal, double dip, eclipsing binaries, etc.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import random
from scipy.interpolate import interp1d
import glob

# Import backend modules (now we're in backend directory)
from periodizer import StarClassifier
from config import Config
from data_processing import calculate_lomb_scargle, remove_y_outliers

class LightCurveDataset(Dataset):
    """Dataset for light curve phase-folded data."""
    
    def __init__(self, phase_folded_data: List[Dict], input_length: int = 512):
        self.data = phase_folded_data
        self.input_length = input_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get phase-folded flux data
        flux = torch.tensor(item['flux'], dtype=torch.float32)
        
        # Resample to fixed length
        flux = self._resample_flux(flux, self.input_length)
        
        # Add channel dimension for 1D CNN
        flux = flux.unsqueeze(0)  # Shape: (1, input_length)
        
        # Get period and class labels
        period = torch.tensor([item['period']], dtype=torch.float32)
        star_class = torch.tensor(item['class_idx'], dtype=torch.long)
        
        return flux, period, star_class
    
    def _resample_flux(self, flux: torch.Tensor, target_length: int) -> torch.Tensor:
        """Resample flux data to target length using interpolation."""
        current_length = len(flux)
        
        if current_length == target_length:
            return flux
        
        # Create original and target indices
        orig_indices = np.linspace(0, 1, current_length)
        target_indices = np.linspace(0, 1, target_length)
        
        # Interpolate
        flux_np = flux.numpy()
        interp_func = interp1d(orig_indices, flux_np, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        resampled_flux = interp_func(target_indices)
        
        return torch.tensor(resampled_flux, dtype=torch.float32)


def load_light_curve_data(file_path: str) -> np.ndarray:
    """Load light curve data from .tbl file."""
    try:
        # Read the file, skipping the header
        data = pd.read_csv(file_path, sep='\t', comment='#', header=0)
        
        # Extract time and flux columns
        time = data.iloc[:, 0].values  # Time (days)
        flux = data.iloc[:, 1].values  # Flux
        
        # Create numpy array
        lc_data = np.column_stack([time, flux])
        
        return lc_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([])


def phase_fold_data(time: np.ndarray, flux: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    """Phase fold the light curve data."""
    phase = (time % period) / period
    
    # Sort by phase
    sort_indices = np.argsort(phase)
    phase_sorted = phase[sort_indices]
    flux_sorted = flux[sort_indices]
    
    return phase_sorted, flux_sorted


def detect_period_lomb_scargle(lc_data: np.ndarray) -> float:
    """Detect period using Lomb-Scargle periodogram."""
    try:
        frequency, power = calculate_lomb_scargle(lc_data)
        
        # Find the peak frequency
        peak_idx = np.argmax(power)
        peak_frequency = frequency[peak_idx]
        
        # Convert to period
        period = 1.0 / peak_frequency
        
        # Ensure period is in reasonable range
        period = np.clip(period, Config.MIN_PERIOD, Config.MAX_PERIOD)
        
        return period
    except Exception as e:
        print(f"Error in period detection: {e}")
        # Return a reasonable default period
        return 2.0


def assign_random_class() -> int:
    """Assign a random class for demonstration purposes."""
    return random.randint(0, len(Config.CLASS_NAMES) - 1)


def create_training_data(sample_data_dir: str, num_samples_per_star: int = 3) -> List[Dict]:
    """Create training data from sample files."""
    training_data = []
    
    # Get all .tbl files
    tbl_files = glob.glob(os.path.join(sample_data_dir, "*.tbl"))
    print(f"Found {len(tbl_files)} sample files")
    
    for file_path in tbl_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
        # Extract star number and telescope from filename
        filename = os.path.basename(file_path)
        parts = filename.replace('.tbl', '').split('-')
        if len(parts) != 2:
            continue
            
        star_num = int(parts[0])
        telescope = parts[1]
        
        # Load light curve data
        lc_data = load_light_curve_data(file_path)
        if len(lc_data) < 10:  # Skip files with too little data
            continue
        
        # Remove outliers
        lc_data = remove_y_outliers(lc_data)
        if len(lc_data) < 10:
            continue
        
        time = lc_data[:, 0]
        flux = lc_data[:, 1]
        
        # Generate multiple training samples per file
        for sample_idx in range(num_samples_per_star):
            try:
                # Detect period using Lomb-Scargle
                period = detect_period_lomb_scargle(lc_data)
                
                # Add some variation to the period for different samples
                if sample_idx > 0:
                    period_variation = 1.0 + (random.random() - 0.5) * 0.2  # ±10% variation
                    period = period * period_variation
                    period = np.clip(period, Config.MIN_PERIOD, Config.MAX_PERIOD)
                
                # Phase fold the data
                phase, folded_flux = phase_fold_data(time, flux, period)
                
                # Normalize flux (mean centering)
                folded_flux = folded_flux - np.mean(folded_flux)
                
                # Skip if we don't have enough phase-folded data
                if len(folded_flux) < 20:
                    continue
                
                # Assign a class (for demonstration, we'll use random assignment)
                # In a real scenario, this would come from labeled data
                class_idx = assign_random_class()
                
                training_sample = {
                    'star_num': star_num,
                    'telescope': telescope,
                    'period': period,
                    'phase': phase,
                    'flux': folded_flux,
                    'class_idx': class_idx,
                    'class_name': Config.CLASS_NAMES[class_idx]
                }
                
                training_data.append(training_sample)
                
            except Exception as e:
                print(f"Error processing sample {sample_idx} for {filename}: {e}")
                continue
    
    print(f"Created {len(training_data)} training samples")
    return training_data


def train_model(model: StarClassifier, train_loader: DataLoader, 
                num_epochs: int = 50, learning_rate: float = 0.001,
                device: str = 'cpu') -> StarClassifier:
    """Train the StarClassifier model."""
    
    model = model.to(device)
    model.train()
    
    # Loss functions
    period_criterion = nn.MSELoss()  # For period regression
    class_criterion = nn.CrossEntropyLoss()  # For classification
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on {device} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_period_loss = 0.0
        total_class_loss = 0.0
        num_batches = 0
        
        for batch_idx, (flux, period_target, class_target) in enumerate(train_loader):
            flux = flux.to(device)
            period_target = period_target.to(device)
            class_target = class_target.to(device)
            
            # Forward pass
            period_pred, class_pred = model(flux)
            
            # Calculate losses
            period_loss = period_criterion(period_pred, period_target)
            class_loss = class_criterion(class_pred, class_target)
            
            # Combined loss (you can adjust weights)
            total_loss = period_loss + class_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_period_loss += period_loss.item()
            total_class_loss += class_loss.item()
            num_batches += 1
        
        # Print epoch statistics
        avg_period_loss = total_period_loss / num_batches
        avg_class_loss = total_class_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Period Loss: {avg_period_loss:.4f}, "
                  f"Class Loss: {avg_class_loss:.4f}")
    
    return model


def save_model(model: StarClassifier, save_path: str):
    """Save the trained model."""
    model_state = {
        'model_state_dict': model.state_dict(),
        'num_classes': len(Config.CLASS_NAMES),
        'input_length': model.input_length,
        'class_names': Config.CLASS_NAMES
    }
    
    torch.save(model_state, save_path)
    print(f"Model saved to {save_path}")


def main():
    """Main training function."""
    print("Better Impuls Viewer - Model Training")
    print("=" * 40)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Configuration
    # Note: sample_data is relative to project root, we're in backend/ directory
    sample_data_dir = "../sample_data"
    input_length = 512
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001
    save_path = "../trained_cnn_model.pth"  # Save to project root
    
    # Check if sample data directory exists
    if not os.path.exists(sample_data_dir):
        print(f"Error: Sample data directory '{sample_data_dir}' not found!")
        print("Please run this script from the project root or ensure sample_data/ exists.")
        return
    
    # Create training data
    print("Creating training data from sample files...")
    training_data = create_training_data(sample_data_dir, num_samples_per_star=3)
    
    if len(training_data) == 0:
        print("Error: No training data created!")
        return
    
    # Create dataset and dataloader
    dataset = LightCurveDataset(training_data, input_length=input_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Batch size: {batch_size}, Number of batches: {len(train_loader)}")
    
    # Create model
    num_classes = len(Config.CLASS_NAMES)
    model = StarClassifier(num_classes=num_classes, input_length=input_length)
    
    print(f"Model created with {num_classes} classes")
    print(f"Class names: {Config.CLASS_NAMES}")
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    trained_model = train_model(model, train_loader, num_epochs=num_epochs, 
                               learning_rate=learning_rate, device=device)
    
    # Save model
    save_model(trained_model, save_path)
    
    print("Training completed successfully!")
    print(f"Trained model saved as '{save_path}'")


if __name__ == "__main__":
    main()