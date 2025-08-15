#!/usr/bin/env python3
"""
Training script for Better Impuls Viewer MultiBranchStarModelHybrid model.
Trains the multi-branch CNN+Transformer model in periodizer.py using the sample data.

This script:
1. Loads light curve data from sample_data/*.tbl files
2. Processes the data (removes outliers, detects periods)
3. Creates multi-branch training samples: raw light curve, periodogram, folded candidates
4. Trains a hybrid CNN+Transformer model for period prediction and star classification
5. Saves the trained model as 'trained_multi_branch_model.pth'

Usage:
    python backend/train.py

Requirements:
    - sample_data/ directory with .tbl files (relative to project root)
    - All dependencies from backend/requirements.txt

The script will create training data from all .tbl files in sample_data/,
generating multiple samples per file with different periods detected via
Lomb-Scargle periodogram analysis.

Output:
    - trained_multi_branch_model.pth: Saved PyTorch model with metadata
    - Console output showing training progress

Model Architecture:
    - Multi-branch inputs: raw light curve, periodogram, folded candidates with periods
    - Hybrid CNN+Transformer feature extraction per branch
    - Attention pooling over period candidates
    - Multi-task outputs: period regression + star type classification
    - Classes: 13 variability types (sinusoidal, double dip, eclipsing binaries, etc.)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import random
from scipy.interpolate import interp1d
import glob

# Import backend modules (now we're in backend directory)
from periodizer import MultiBranchStarModelHybrid, StarModelConfig, multitask_loss
from config import Config
from data_processing import calculate_lomb_scargle, remove_y_outliers

class MultiBranchDataset(Dataset):
    """Dataset for multi-branch star model training data."""
    
    def __init__(self, training_samples: List[Dict], 
                 lc_length: int = 1200, pgram_length: int = 900, folded_length: int = 200):
        self.data = training_samples
        self.lc_length = lc_length
        self.pgram_length = pgram_length
        self.folded_length = folded_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get raw light curve data
        lc = torch.tensor(item['raw_lc'], dtype=torch.float32)
        lc = self._resample_data(lc, self.lc_length)
        lc = lc.unsqueeze(0)  # Shape: (1, lc_length)
        
        # Get periodogram data
        pgram = torch.tensor(item['periodogram'], dtype=torch.float32)
        pgram = self._resample_data(pgram, self.pgram_length)
        pgram = pgram.unsqueeze(0)  # Shape: (1, pgram_length)
        
        # Get folded candidates (list of tensors)
        folded_list = []
        logP_list = []
        for folded_data, period in zip(item['folded_candidates'], item['candidate_periods']):
            folded_tensor = torch.tensor(folded_data, dtype=torch.float32)
            folded_tensor = self._resample_data(folded_tensor, self.folded_length)
            folded_tensor = folded_tensor.unsqueeze(0)  # Shape: (1, folded_length)
            folded_list.append(folded_tensor)
            
            logP_list.append(torch.tensor(np.log10(period), dtype=torch.float32))
        
        # Get labels
        true_logP = torch.tensor(np.log10(item['true_period']), dtype=torch.float32)
        star_class = torch.tensor(item['class_idx'], dtype=torch.long)
        
        return {
            'lc': lc,
            'pgram': pgram,
            'folded_list': folded_list,
            'logP_list': logP_list,
            'true_logP': true_logP,
            'star_class': star_class
        }
    
    def _resample_data(self, data: torch.Tensor, target_length: int) -> torch.Tensor:
        """Resample data to target length using interpolation."""
        current_length = len(data)
        
        if current_length == target_length:
            return data
        
        # Create original and target indices
        orig_indices = np.linspace(0, 1, current_length)
        target_indices = np.linspace(0, 1, target_length)
        
        # Interpolate
        data_np = data.numpy()
        interp_func = interp1d(orig_indices, data_np, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        resampled_data = interp_func(target_indices)
        
        return torch.tensor(resampled_data, dtype=torch.float32)


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


def detect_period_lomb_scargle(lc_data: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Detect period using Lomb-Scargle periodogram and return periodogram data."""
    try:
        frequency, power = calculate_lomb_scargle(lc_data)
        
        # Find the peak frequency
        peak_idx = np.argmax(power)
        peak_frequency = frequency[peak_idx]
        
        # Convert to period
        period = 1.0 / peak_frequency
        
        # Ensure period is in reasonable range
        period = np.clip(period, Config.MIN_PERIOD, Config.MAX_PERIOD)
        
        return period, frequency, power
    except Exception as e:
        print(f"Error in period detection: {e}")
        # Return default values
        frequency = np.linspace(0.1, 1.0, 900)
        power = np.random.normal(0.5, 0.1, 900)
        return 2.0, frequency, power


def generate_candidate_periods(true_period: float, num_candidates: int = 4) -> List[float]:
    """Generate multiple period candidates around the true period."""
    candidates = [true_period]  # Include true period
    
    # Add some variations
    for i in range(num_candidates - 1):
        # Add some noise and harmonics
        variation = np.random.uniform(0.8, 1.2)
        harmonic = np.random.choice([0.5, 2.0]) if i % 2 == 0 else 1.0
        candidate = true_period * variation * harmonic
        candidate = np.clip(candidate, Config.MIN_PERIOD, Config.MAX_PERIOD)
        candidates.append(candidate)
    
    return candidates


def create_multi_branch_sample(lc_data: np.ndarray, star_class: int) -> Dict:
    """Create a multi-branch training sample from light curve data."""
    time = lc_data[:, 0]
    flux = lc_data[:, 1]
    
    # Remove outliers
    cleaned_data = remove_y_outliers(lc_data)
    time_clean = cleaned_data[:, 0]
    flux_clean = cleaned_data[:, 1]
    
    # Detect period and get periodogram
    true_period, frequency, power = detect_period_lomb_scargle(cleaned_data)
    
    # Normalize raw light curve (subtract mean, divide by std)
    flux_norm = (flux_clean - np.mean(flux_clean)) / (np.std(flux_clean) + 1e-8)
    
    # Create periodogram (use power as the signal)
    pgram_data = power / (np.max(power) + 1e-8)  # Normalize to [0, 1]
    
    # Generate period candidates
    candidate_periods = generate_candidate_periods(true_period, num_candidates=4)
    
    # Create folded candidates
    folded_candidates = []
    for period in candidate_periods:
        phase, flux_folded = phase_fold_data(time_clean, flux_norm, period)
        # Resample folded data to consistent phase grid
        phase_grid = np.linspace(0, 1, 200)
        flux_interp = np.interp(phase_grid, phase, flux_folded)
        folded_candidates.append(flux_interp)
    
    return {
        'raw_lc': flux_norm,  # Normalized flux time series
        'periodogram': pgram_data,  # Normalized periodogram power
        'folded_candidates': folded_candidates,  # List of phase-folded curves
        'candidate_periods': candidate_periods,  # Corresponding periods
        'true_period': true_period,  # Ground truth period
        'class_idx': star_class  # Star variability class
    }


def assign_random_class() -> int:
    """Assign a random class for demonstration purposes."""
    return random.randint(0, len(Config.CLASS_NAMES) - 1)


def create_training_data(sample_data_dir: str, num_samples_per_star: int = 3) -> List[Dict]:
    """Create multi-branch training data from sample files."""
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
        
        # Generate multiple training samples per file
        for sample_idx in range(num_samples_per_star):
            try:
                # Assign a class (for demonstration, we'll use star number as proxy)
                # In real scenario, this would come from labeled data
                class_idx = (star_num - 1) % 13  # Map to 13 classes (0-12)
                
                # Create multi-branch sample
                training_sample = create_multi_branch_sample(lc_data, class_idx)
                
                # Add sample to training data
                training_data.append(training_sample)
                
            except Exception as e:
                print(f"  Error creating sample {sample_idx} for {filename}: {e}")
                continue
    
    print(f"Created {len(training_data)} training samples")
    return training_data

def collate_multi_branch(batch):
    """Custom collate function for multi-branch data."""
    lc_batch = torch.stack([item['lc'] for item in batch])
    pgram_batch = torch.stack([item['pgram'] for item in batch])
    
    # For folded_list, we need to batch each position separately
    batch_size = len(batch)
    num_candidates = len(batch[0]['folded_list'])
    
    folded_list_batch = []
    logP_list_batch = []
    
    for cand_idx in range(num_candidates):
        folded_cand_batch = torch.stack([item['folded_list'][cand_idx] for item in batch])
        logP_cand_batch = torch.stack([item['logP_list'][cand_idx] for item in batch])
        
        folded_list_batch.append(folded_cand_batch)
        logP_list_batch.append(logP_cand_batch)
    
    true_logP_batch = torch.stack([item['true_logP'] for item in batch])
    star_class_batch = torch.stack([item['star_class'] for item in batch])
    
    return {
        'lc': lc_batch,
        'pgram': pgram_batch,
        'folded_list': folded_list_batch,
        'logP_list': logP_list_batch,
        'true_logP': true_logP_batch,
        'star_class': star_class_batch
    }


def train_model(model: MultiBranchStarModelHybrid, train_loader: DataLoader, cfg: StarModelConfig,
                num_epochs: int = 50, learning_rate: float = 0.001,
                device: str = 'cpu') -> MultiBranchStarModelHybrid:
    """Train the MultiBranchStarModelHybrid model."""
    
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on {device} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_type_loss = 0.0
        total_period_loss = 0.0
        total_cand_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            lc = batch['lc'].to(device)
            pgram = batch['pgram'].to(device)
            folded_list = [f.to(device) for f in batch['folded_list']]
            logP_list = [p.to(device) for p in batch['logP_list']]
            true_logP = batch['true_logP'].to(device)
            star_class = batch['star_class'].to(device)
            
            # Forward pass
            outputs = model(lc, pgram, folded_list, logP_list)
            
            # Create candidate labels (first candidate is always correct for simplicity)
            batch_size = lc.shape[0]
            num_candidates = len(folded_list)
            cand_labels = torch.zeros(batch_size, num_candidates, device=device)
            cand_labels[:, 0] = 1.0  # First candidate is correct
            
            # Calculate loss
            loss, logs = multitask_loss(
                outputs, star_class, true_logP, cand_labels=cand_labels, cfg=cfg,
                lambda_period=1.0, lambda_cand=0.5
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += logs['loss']
            total_type_loss += logs['loss_type']
            total_period_loss += logs['loss_period']
            total_cand_loss += logs['loss_cand']
            num_batches += 1
        
        # Print epoch statistics
        avg_loss = total_loss / num_batches
        avg_type_loss = total_type_loss / num_batches
        avg_period_loss = total_period_loss / num_batches
        avg_cand_loss = total_cand_loss / num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Loss={avg_loss:.4f}, Type={avg_type_loss:.4f}, "
              f"Period={avg_period_loss:.4f}, Cand={avg_cand_loss:.4f}")
    
    return model
def save_model(model: MultiBranchStarModelHybrid, cfg: StarModelConfig, save_path: str):
    """Save the trained multi-branch model."""
    model_state = {
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'class_names': Config.CLASS_NAMES
    }
    
    torch.save(model_state, save_path)
    print(f"Model saved to {save_path}")


def main():
    """Main training function."""
    print("Better Impuls Viewer - Multi-Branch Model Training")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Configuration
    # Note: sample_data is relative to project root, we're in backend/ directory
    sample_data_dir = "../sample_data"
    batch_size = 8  # Smaller batch size for multi-branch model
    num_epochs = 30
    learning_rate = 0.0001  # Lower learning rate for complex model
    save_path = "../trained_multi_branch_model.pth"  # Save to project root
    
    # Model configuration
    n_types = 13  # Reduced from 14 to 13 as per new model
    cfg = StarModelConfig(
        n_types=n_types,
        lc_in_channels=1,
        pgram_in_channels=1,
        folded_in_channels=1,
        add_period_channel=True,
        emb_dim=128,
        merged_dim=256,
        cnn_hidden=64,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        logP_mean=0.0,  # Will be computed from data
        logP_std=1.0   # Will be computed from data
    )
    
    # Check if sample data directory exists
    if not os.path.exists(sample_data_dir):
        print(f"Error: Sample data directory '{sample_data_dir}' not found!")
        print("Please run this script from the project root or ensure sample_data/ exists.")
        return
    
    # Create training data
    print("Creating multi-branch training data from sample files...")
    training_data = create_training_data(sample_data_dir, num_samples_per_star=3)
    
    if len(training_data) == 0:
        print("Error: No training data created!")
        return
    
    # Compute period statistics for normalization
    periods = [sample['true_period'] for sample in training_data]
    log_periods = np.log10(periods)
    cfg.logP_mean = np.mean(log_periods)
    cfg.logP_std = np.std(log_periods)
    
    print(f"Period statistics: logP_mean={cfg.logP_mean:.3f}, logP_std={cfg.logP_std:.3f}")
    
    # Create dataset and dataloader
    dataset = MultiBranchDataset(training_data, lc_length=1200, pgram_length=900, folded_length=200)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_multi_branch)
    
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Batch size: {batch_size}, Number of batches: {len(train_loader)}")
    
    # Create model
    model = MultiBranchStarModelHybrid(cfg)
    
    print(f"Model created with {n_types} classes")
    print(f"Class names: {Config.CLASS_NAMES[:n_types]}")
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    trained_model = train_model(model, train_loader, cfg, num_epochs=num_epochs, 
                               learning_rate=learning_rate, device=device)
    
    # Save model
    save_model(trained_model, cfg, save_path)
    
    print("Training completed successfully!")
    print(f"Trained multi-branch model saved as '{save_path}'")


if __name__ == "__main__":
    main()