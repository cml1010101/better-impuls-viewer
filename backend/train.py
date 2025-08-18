#!/usr/bin/env python3
"""
Training script for the MultiBranchStarModelHybrid model.
Uses generator.py to create synthetic training data and trains the periodizer model.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import glob
from pathlib import Path

# Import project modules
from generator import generate_light_curve, LC_GENERATORS, SyntheticLightCurve
from periodizer import (
    MultiBranchStarModelHybrid, 
    StarModelConfig, 
    ModelInput, 
    ModelOutput,
    MultiTaskLoss
)
from data_processing import (
    detect_period_lomb_scargle, 
    phase_fold_data, 
    remove_y_outliers,
    generate_candidate_periods
)
from config import Config

def load_light_curve_data(file_path: str) -> np.ndarray:
    """
    Load light curve data from a .tbl file.
    Expected format: time, flux columns.
    
    Args:
        file_path: Path to the .tbl file
        
    Returns:
        np.ndarray: Array of shape (N, 2) with [time, flux] columns
    """
    try:
        # Try loading as space-separated values
        data = np.loadtxt(file_path, usecols=(0, 1))
        
        # Ensure we have the right shape
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        elif data.shape[1] < 2:
            raise ValueError(f"File {file_path} must have at least 2 columns (time, flux)")
        
        # Take only first two columns if more exist
        data = data[:, :2]
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(data).all(axis=1)
        data = data[valid_mask]
        
        if len(data) == 0:
            raise ValueError(f"No valid data found in {file_path}")
            
        # Sort by time
        data = data[np.argsort(data[:, 0])]
        
        return data
        
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")


def create_multi_branch_sample(
    lc_data: np.ndarray, 
    star_class: int,
    lc_length: int = 1024,
    pgram_length: int = 512,
    folded_length: int = 200,
    num_candidates: int = 4
) -> Dict:
    """
    Create a multi-branch training sample from light curve data.
    
    Args:
        lc_data: Light curve data array (N, 2) with [time, flux]
        star_class: Integer class label (0-12)
        lc_length: Length for raw light curve data
        pgram_length: Length for periodogram data
        folded_length: Length for each folded candidate
        num_candidates: Number of period candidates to generate
        
    Returns:
        Dict containing all required data for training
    """
    time = lc_data[:, 0]
    flux = lc_data[:, 1]
    
    # Remove outliers and normalize flux
    cleaned_data = remove_y_outliers(lc_data)
    time_clean = cleaned_data[:, 0]
    flux_clean = cleaned_data[:, 1]
    
    # Normalize flux
    flux_norm = (flux_clean - np.mean(flux_clean)) / (np.std(flux_clean) + 1e-8)
    
    # Ensure normalization is reasonable
    if np.isnan(flux_norm).any() or np.isinf(flux_norm).any():
        flux_norm = np.zeros_like(flux_clean)  # Fallback to zeros
    
    # Detect period and generate periodogram
    try:
        true_period, frequency, power = detect_period_lomb_scargle(cleaned_data)
    except:
        # Fallback if period detection fails
        true_period = 1.0
        frequency = np.linspace(0.01, 5.0, pgram_length)
        power = np.random.rand(pgram_length)
    
    # Normalize periodogram
    pgram_data = power / (np.max(power) + 1e-8)
    
    # Ensure periodogram is reasonable
    if np.isnan(pgram_data).any() or np.isinf(pgram_data).any():
        pgram_data = np.random.rand(len(power)) * 0.1  # Fallback to low-level noise
    
    # Resample/interpolate to target lengths
    if len(flux_norm) > lc_length:
        # Downsample
        indices = np.linspace(0, len(flux_norm)-1, lc_length, dtype=int)
        flux_resampled = flux_norm[indices]
    else:
        # Upsample/pad
        interp_indices = np.linspace(0, len(flux_norm)-1, lc_length)
        flux_resampled = np.interp(interp_indices, np.arange(len(flux_norm)), flux_norm)
    
    # Resample periodogram
    if len(pgram_data) > pgram_length:
        indices = np.linspace(0, len(pgram_data)-1, pgram_length, dtype=int)
        pgram_resampled = pgram_data[indices]
    else:
        interp_indices = np.linspace(0, len(pgram_data)-1, pgram_length)
        pgram_resampled = np.interp(interp_indices, np.arange(len(pgram_data)), pgram_data)
    
    # Generate period candidates around the true period
    try:
        candidate_periods = generate_candidate_periods(frequency, power, num_candidates)
        # Ensure we have enough candidates
        if len(candidate_periods) < num_candidates:
            # Fill with variations around true period
            base_periods = [true_period * (0.8 + 0.4 * i / num_candidates) for i in range(num_candidates)]
            candidate_periods.extend(base_periods[len(candidate_periods):])
    except:
        # Fallback if candidate generation fails
        candidate_periods = [true_period * (0.8 + 0.4 * i / num_candidates) for i in range(num_candidates)]
    
    # Ensure we have exactly num_candidates
    candidate_periods = candidate_periods[:num_candidates]
    if len(candidate_periods) < num_candidates:
        # Pad with variations of the true period
        while len(candidate_periods) < num_candidates:
            candidate_periods.append(true_period * (0.5 + np.random.rand()))
    
    # Create folded candidates
    folded_candidates = []
    for period in candidate_periods:
        try:
            phase, flux_folded = phase_fold_data(time_clean, flux_norm, period)
            # Resample to consistent length
            if len(flux_folded) > folded_length:
                indices = np.linspace(0, len(flux_folded)-1, folded_length, dtype=int)
                flux_folded_resampled = flux_folded[indices]
            else:
                interp_indices = np.linspace(0, len(flux_folded)-1, folded_length)
                flux_folded_resampled = np.interp(interp_indices, np.arange(len(flux_folded)), flux_folded)
            folded_candidates.append(flux_folded_resampled)
        except:
            # Fallback if folding fails
            folded_candidates.append(np.random.randn(folded_length) * 0.1)
    
    # Ensure we have exactly num_candidates folded curves
    while len(folded_candidates) < num_candidates:
        folded_candidates.append(np.random.randn(folded_length) * 0.1)
    
    return {
        'raw_lc': flux_resampled,
        'periodogram': pgram_resampled,
        'folded_candidates': folded_candidates,
        'candidate_periods': candidate_periods,
        'true_period': true_period,
        'class_idx': star_class
    }


class SyntheticLightCurveDataset(Dataset):
    """
    Dataset for training using synthetic light curves from generator.py
    """
    
    def __init__(
        self,
        n_per_class: int = 100,
        lc_length: int = 1024,
        pgram_length: int = 512,
        folded_length: int = 200,
        num_candidates: int = 4,
        noise_level: float = 0.02,
        min_days: float = 5.0,
        max_days: float = 30.0,
        seed: Optional[int] = None
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            n_per_class: Number of samples per light curve class
            lc_length: Length of raw light curve
            pgram_length: Length of periodogram
            folded_length: Length of each folded candidate
            num_candidates: Number of period candidates
            noise_level: Noise level for synthetic light curves
            min_days: Minimum observation duration
            max_days: Maximum observation duration
            seed: Random seed for reproducibility
        """
        self.n_per_class = n_per_class
        self.lc_length = lc_length
        self.pgram_length = pgram_length
        self.folded_length = folded_length
        self.num_candidates = num_candidates
        self.noise_level = noise_level
        self.min_days = min_days
        self.max_days = max_days
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Generate all synthetic light curves
        self.samples = []
        self.class_names = list(LC_GENERATORS.keys())
        
        print(f"Generating {n_per_class} samples per class for {len(self.class_names)} classes...")
        
        for class_idx, class_name in enumerate(self.class_names):
            for sample_idx in range(n_per_class):
                # Generate synthetic light curve
                synth_lc = generate_light_curve(
                    class_type=class_idx,
                    class_type_str=class_name,
                    noise_level=noise_level,
                    min_days=min_days,
                    max_days=max_days
                )
                
                # Convert to light curve data format
                lc_data = np.column_stack([synth_lc.time, synth_lc.flux])
                
                # Create multi-branch sample
                sample = create_multi_branch_sample(
                    lc_data=lc_data,
                    star_class=class_idx,
                    lc_length=lc_length,
                    pgram_length=pgram_length,
                    folded_length=folded_length,
                    num_candidates=num_candidates
                )
                
                # Add ground truth periods from synthetic generation
                sample['true_primary_period'] = synth_lc.primary_period
                sample['true_secondary_period'] = synth_lc.secondary_period
                
                self.samples.append(sample)
        
        print(f"Generated {len(self.samples)} total samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        raw_lc = torch.tensor(sample['raw_lc'], dtype=torch.float32).unsqueeze(0)  # (1, L)
        periodogram = torch.tensor(sample['periodogram'], dtype=torch.float32).unsqueeze(0)  # (1, L)
        
        # Stack folded candidates
        if len(sample['folded_candidates']) == 0:
            # Fallback if no candidates 
            folded_candidates = torch.randn(4, folded_length)  # Default 4 candidates
            candidate_periods = torch.randn(4)
        else:
            folded_candidates = torch.stack([
                torch.tensor(fc, dtype=torch.float32) 
                for fc in sample['folded_candidates']
            ])  # (N, L)
            
            # Candidate periods (not log-transformed - model will handle this)
            candidate_periods = torch.tensor(
                [max(p, 0.1) for p in sample['candidate_periods']], 
                dtype=torch.float32
            )  # (N,)
        
        # Ground truth
        class_label = torch.tensor(sample['class_idx'], dtype=torch.long)
        
        # Primary and secondary periods (not log-transformed - model will handle this)
        primary_period = sample['true_primary_period'] if sample['true_primary_period'] is not None else 1.0
        secondary_period = sample['true_secondary_period'] if sample['true_secondary_period'] is not None else 1.0
        
        # Ensure periods are positive and reasonable
        primary_period = max(primary_period, 0.1)
        secondary_period = max(secondary_period, 0.1)
        
        target_periods = torch.tensor([primary_period, secondary_period], dtype=torch.float32)
        
        return {
            'raw_lc': raw_lc,
            'periodogram': periodogram,
            'folded_candidates': folded_candidates,
            'candidate_periods': candidate_periods,
            'class_label': class_label,
            'target_periods': target_periods
        }


def collate_fn(batch):
    """
    Custom collate function for variable-length folded candidates.
    """
    # Stack fixed-size tensors
    raw_lc = torch.stack([item['raw_lc'] for item in batch])
    periodogram = torch.stack([item['periodogram'] for item in batch])
    class_labels = torch.stack([item['class_label'] for item in batch])
    target_periods = torch.stack([item['target_periods'] for item in batch])
    
    # Handle variable-length folded candidates by padding to max length
    max_candidates = max(item['folded_candidates'].shape[0] for item in batch)
    folded_length = batch[0]['folded_candidates'].shape[1]
    
    folded_data = torch.zeros(len(batch), max_candidates, folded_length)
    candidate_periods = torch.zeros(len(batch), max_candidates)
    
    for i, item in enumerate(batch):
        n_cands = item['folded_candidates'].shape[0]
        folded_data[i, :n_cands] = item['folded_candidates']
        candidate_periods[i, :n_cands] = item['candidate_periods']
    
    return {
        'raw_lc': raw_lc,
        'periodogram': periodogram,
        'folded_data': folded_data,
        'candidate_periods': candidate_periods,
        'class_labels': class_labels,
        'target_periods': target_periods
    }


def train_model(
    model_config: Optional[StarModelConfig] = None,
    n_per_class: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    num_epochs: int = 50,
    device: str = 'cpu',
    save_path: str = 'model.pth',
    seed: Optional[int] = 42
) -> MultiBranchStarModelHybrid:
    """
    Train the MultiBranchStarModelHybrid model on synthetic data.
    
    Args:
        model_config: Model configuration, uses default if None
        n_per_class: Number of synthetic samples per class
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        save_path: Path to save trained model
        seed: Random seed for reproducibility
        
    Returns:
        Trained model
    """
    # Set random seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Default model configuration
    if model_config is None:
        model_config = StarModelConfig(
            n_types=len(LC_GENERATORS),  # 13 classes
            lc_in_channels=1,
            pgram_in_channels=1, 
            folded_in_channels=1,
            emb_dim=128,
            merged_dim=256,
            cnn_hidden=64,
            d_model=128,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            logP_mean=0.0,
            logP_std=1.0,
        )
    
    # Create dataset and dataloader
    print("Creating synthetic dataset...")
    dataset = SyntheticLightCurveDataset(
        n_per_class=n_per_class,
        seed=seed
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Create model
    model = MultiBranchStarModelHybrid(model_config)
    model = model.to(device)
    
    # Loss function and optimizer
    loss_fn = MultiTaskLoss(cls_weight=1.0, period_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    print(f"Starting training for {num_epochs} epochs on {device}...")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_period_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device and validate inputs
            raw_lc = batch['raw_lc'].to(device)
            pgram = batch['periodogram'].to(device) 
            folded_data = batch['folded_data'].to(device)
            folded_periods = batch['candidate_periods'].to(device)
            
            # Check for invalid inputs (NaN or Inf)
            if (torch.isnan(raw_lc).any() or torch.isinf(raw_lc).any() or
                torch.isnan(pgram).any() or torch.isinf(pgram).any() or
                torch.isnan(folded_data).any() or torch.isinf(folded_data).any() or
                torch.isnan(folded_periods).any() or torch.isinf(folded_periods).any()):
                print(f"Invalid input detected at batch {batch_idx}, skipping...")
                continue
            
            model_input = ModelInput(
                lc=raw_lc,
                pgram=pgram,
                folded_data=folded_data,
                folded_periods=folded_periods
            )
            
            target_types = batch['class_labels'].to(device)
            target_periods = batch['target_periods'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(model_input)
            
            # Calculate loss
            loss_output = loss_fn(outputs, target_types, target_periods)
            
            # Check for NaN loss and debug if needed
            if torch.isnan(loss_output.total):
                print(f"NaN detected at batch {batch_idx}")
                print(f"Output logits range: {outputs.type_logits.min():.4f} to {outputs.type_logits.max():.4f}")
                print(f"Period predictions: P1 range {outputs.logP1_pred.min():.4f} to {outputs.logP1_pred.max():.4f}")
                print(f"Period predictions: P2 range {outputs.logP2_pred.min():.4f} to {outputs.logP2_pred.max():.4f}")
                print(f"Target periods range: {target_periods.min():.4f} to {target_periods.max():.4f}")
                # Skip this batch
                continue
            
            # Backward pass
            loss_output.total.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss_output.total.item()
            epoch_cls_loss += loss_output.cls.item()
            epoch_period_loss += loss_output.period.item()
        
        # Calculate average losses
        avg_loss = epoch_loss / len(dataloader)
        avg_cls_loss = epoch_cls_loss / len(dataloader)
        avg_period_loss = epoch_period_loss / len(dataloader)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  Classification Loss: {avg_cls_loss:.4f}")
            print(f"  Period Loss: {avg_period_loss:.4f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save model
    model_state = {
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'class_names': list(LC_GENERATORS.keys()),
        'training_info': {
            'n_per_class': n_per_class,
            'num_epochs': num_epochs,
            'final_loss': avg_loss
        }
    }
    
    torch.save(model_state, save_path)
    print(f"Model saved to {save_path}")
    
    return model


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MultiBranchStarModelHybrid')
    parser.add_argument('--n-per-class', type=int, default=100, help='Samples per class')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--save-path', type=str, default='model.pth', help='Save path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Train model
    model = train_model(
        n_per_class=args.n_per_class,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        device=args.device,
        save_path=args.save_path,
        seed=args.seed
    )
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()