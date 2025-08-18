#!/usr/bin/env python3
"""
Enhanced training script with improvements:
- Better model configurations
- Learning rate scheduling
- Early stopping
- Improved data augmentation
- Better validation tracking
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import argparse

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from periodizer import MultiBranchStarModelHybrid, StarModelConfig, ModelInput, MultiTaskLoss
from train import SyntheticLightCurveDataset, collate_fn, save_model_checkpoint

class ImprovedModelConfig:
    """Enhanced model configurations with proven architectures."""
    
    @staticmethod
    def get_compact_config() -> StarModelConfig:
        """Compact but effective configuration for fast training."""
        return StarModelConfig(
            n_types=14,
            lc_in_channels=1,
            pgram_in_channels=1,
            folded_in_channels=1,
            emb_dim=128,
            merged_dim=256,
            cnn_hidden=64,
            d_model=128,
            n_heads=4,
            n_layers=2,
            dropout=0.2,  # Increased dropout for better generalization
            logP_mean=0.0,
            logP_std=1.0,
        )
    
    @staticmethod
    def get_large_config() -> StarModelConfig:
        """Larger configuration for better accuracy."""
        return StarModelConfig(
            n_types=14,
            lc_in_channels=1,
            pgram_in_channels=1,
            folded_in_channels=1,
            emb_dim=256,
            merged_dim=512,
            cnn_hidden=128,
            d_model=256,
            n_heads=8,
            n_layers=4,
            dropout=0.15,
            logP_mean=0.0,
            logP_std=1.0,
        )
    
    @staticmethod
    def get_optimal_config() -> StarModelConfig:
        """Balanced configuration optimized for astronomical data."""
        return StarModelConfig(
            n_types=14,
            lc_in_channels=1,
            pgram_in_channels=1,
            folded_in_channels=1,
            emb_dim=192,
            merged_dim=384,
            cnn_hidden=96,
            d_model=192,
            n_heads=6,
            n_layers=3,
            dropout=0.1,
            logP_mean=0.0,
            logP_std=1.0,
        )

class ImprovedTrainer:
    """Enhanced training class with better optimization."""
    
    def __init__(
        self,
        model: MultiBranchStarModelHybrid,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_scheduler: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = MultiTaskLoss(cls_weight=1.0, period_weight=1.0)
        
        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        else:
            self.scheduler = None
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_period_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            raw_lc = batch['raw_lc'].to(self.device)
            pgram = batch['periodogram'].to(self.device)
            folded_data = batch['folded_data'].to(self.device)
            folded_periods = batch['candidate_periods'].to(self.device)
            class_labels = batch['class_labels'].to(self.device)
            target_periods = batch['target_periods'].to(self.device)
            
            # Check for invalid inputs
            if (torch.isnan(raw_lc).any() or torch.isinf(raw_lc).any() or
                torch.isnan(pgram).any() or torch.isinf(pgram).any() or
                torch.isnan(folded_data).any() or torch.isinf(folded_data).any()):
                print(f"Skipping batch {batch_idx} due to invalid data")
                continue
            
            # Forward pass
            model_input = ModelInput(
                lc=raw_lc,
                pgram=pgram,
                folded_data=folded_data,
                folded_periods=folded_periods
            )
            
            outputs = self.model(model_input)
            loss_output = self.loss_fn(outputs, class_labels, target_periods)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_output.total.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss_output.total.item()
            epoch_cls_loss += loss_output.cls.item()
            epoch_period_loss += loss_output.period.item()
            num_batches += 1
        
        return {
            'total_loss': epoch_loss / num_batches if num_batches > 0 else 0,
            'cls_loss': epoch_cls_loss / num_batches if num_batches > 0 else 0,
            'period_loss': epoch_period_loss / num_batches if num_batches > 0 else 0,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_period_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                raw_lc = batch['raw_lc'].to(self.device)
                pgram = batch['periodogram'].to(self.device)
                folded_data = batch['folded_data'].to(self.device)
                folded_periods = batch['candidate_periods'].to(self.device)
                class_labels = batch['class_labels'].to(self.device)
                target_periods = batch['target_periods'].to(self.device)
                
                # Forward pass
                model_input = ModelInput(
                    lc=raw_lc,
                    pgram=pgram,
                    folded_data=folded_data,
                    folded_periods=folded_periods
                )
                
                outputs = self.model(model_input)
                loss_output = self.loss_fn(outputs, class_labels, target_periods)
                
                # Accumulate losses
                val_loss += loss_output.total.item()
                val_cls_loss += loss_output.cls.item()
                val_period_loss += loss_output.period.item()
                num_batches += 1
                
                # Classification accuracy
                predictions = torch.argmax(outputs.type_logits, dim=1)
                correct_predictions += (predictions == class_labels).sum().item()
                total_predictions += class_labels.size(0)
        
        return {
            'total_loss': val_loss / num_batches if num_batches > 0 else 0,
            'cls_loss': val_cls_loss / num_batches if num_batches > 0 else 0,
            'period_loss': val_period_loss / num_batches if num_batches > 0 else 0,
            'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_path: str,
        save_every: int = 5,
        early_stopping_patience: int = 15
    ):
        """Train the model with validation and early stopping."""
        print(f"Starting enhanced training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {early_stopping_patience}")
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['total_loss'])
            
            # Validation
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['total_loss'])
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_metrics['total_loss'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"Cls: {train_metrics['cls_loss']:.4f}, "
                  f"Period: {train_metrics['period_loss']:.4f}")
            print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.1f}%)")
            print(f"  LR: {train_metrics['lr']:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
                training_info = {
                    'completed_epochs': epoch + 1,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['total_loss'],
                    'val_accuracy': val_metrics['accuracy']
                }
                save_model_checkpoint(
                    self.model, 
                    self.model.config if hasattr(self.model, 'config') else None,
                    checkpoint_path, 
                    training_info, 
                    epoch, 
                    val_metrics['total_loss']
                )
            
            # Early stopping check
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
                
                # Save best model
                best_path = save_path.replace('.pth', '_best.pth')
                training_info = {
                    'completed_epochs': epoch + 1,
                    'best_epoch': epoch + 1,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['total_loss'],
                    'val_accuracy': val_metrics['accuracy']
                }
                save_model_checkpoint(
                    self.model, 
                    self.model.config if hasattr(self.model, 'config') else None,
                    best_path, 
                    training_info, 
                    epoch, 
                    val_metrics['total_loss']
                )
                print(f"  💾 Saved new best model with val_loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save final model
        final_training_info = {
            'completed_epochs': epoch + 1,
            'final_train_loss': train_metrics['total_loss'],
            'final_val_loss': val_metrics['total_loss'],
            'final_val_accuracy': val_metrics['accuracy'],
            'best_val_loss': self.best_val_loss
        }
        save_model_checkpoint(
            self.model, 
            self.model.config if hasattr(self.model, 'config') else None,
            save_path, 
            final_training_info, 
            epoch, 
            val_metrics['total_loss']
        )
        
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

def main():
    """Enhanced training main function."""
    parser = argparse.ArgumentParser(description='Enhanced MultiBranchStarModelHybrid Training')
    parser.add_argument('--config', choices=['compact', 'large', 'optimal'], 
                       default='optimal', help='Model configuration')
    parser.add_argument('--n-per-class', type=int, default=200, help='Samples per class')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--save-path', type=str, default='enhanced_model.pth', help='Save path')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Model Training")
    print("=" * 50)
    print(f"Configuration: {args.config}")
    print(f"Samples per class: {args.n_per_class}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    
    # Get model configuration
    config_map = {
        'compact': ImprovedModelConfig.get_compact_config(),
        'large': ImprovedModelConfig.get_large_config(),
        'optimal': ImprovedModelConfig.get_optimal_config()
    }
    model_config = config_map[args.config]
    
    # Create model
    model = MultiBranchStarModelHybrid(model_config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create datasets
    print("Creating datasets...")
    full_dataset = SyntheticLightCurveDataset(n_per_class=args.n_per_class, seed=42)
    
    # Split into train/val
    dataset_size = len(full_dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create trainer
    trainer = ImprovedTrainer(
        model=model,
        device=args.device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_scheduler=True
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_path=args.save_path,
        save_every=args.save_every,
        early_stopping_patience=args.patience
    )
    
    print("✅ Enhanced training completed!")

if __name__ == "__main__":
    main()