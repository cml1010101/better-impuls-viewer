# train.py
# Training script for the multi-branch stellar classification model

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from generator import generate_dataset, LC_GENERATORS
from periodizer import (
    MultiBranchStarModelHybrid, StarModelConfig, multitask_loss,
    LightCurve, Periodogram, FoldedCandidate, prepare_input
)

# ===== Dataset Class =====

class StellarDataset(Dataset):
    """Dataset class for stellar light curves with synthetic data generation."""
    
    def __init__(
        self,
        n_per_class: int = 1000,
        max_days: float = 30.0,
        noise_level: float = 0.02,
        n_candidates: int = 4,
        seed: Optional[int] = None
    ):
        self.n_per_class = n_per_class
        self.max_days = max_days
        self.noise_level = noise_level
        self.n_candidates = n_candidates
        
        # Set up RNG for reproducibility
        self.rng = np.random.default_rng(seed)
        
        # Class mapping
        self.class_names = list(LC_GENERATORS.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Generate synthetic dataset
        print(f"Generating synthetic dataset: {n_per_class} samples per class...")
        self.light_curves = generate_dataset(
            n_per_class=n_per_class,
            max_days=max_days,
            noise_level=noise_level,
            rng=self.rng
        )
        print(f"Generated {len(self.light_curves)} total light curves")
        
        # Compute period statistics for normalization
        self._compute_period_stats()
    
    def _compute_period_stats(self):
        """Compute period statistics for normalization."""
        periods = []
        for lc in self.light_curves:
            if lc.primary_period is not None:
                periods.append(lc.primary_period)
            if lc.secondary_period is not None:
                periods.append(lc.secondary_period)
        
        if periods:
            log_periods = np.log10(periods)
            self.logP_mean = float(np.mean(log_periods))
            self.logP_std = float(np.std(log_periods))
        else:
            self.logP_mean = 0.0
            self.logP_std = 1.0
        
        print(f"Period statistics: mean={self.logP_mean:.3f}, std={self.logP_std:.3f}")
    
    def __len__(self) -> int:
        return len(self.light_curves)
    
    def __getitem__(self, idx: int) -> Dict:
        lc = self.light_curves[idx]
        
        # Prepare model inputs
        lc_data = np.column_stack([lc.time, lc.flux])
        lc_input, pgram_input, folded_candidates = prepare_input([lc_data])
        
        # Ensure we have the right number of candidates
        if len(folded_candidates) < self.n_candidates:
            # Pad with copies if needed
            while len(folded_candidates) < self.n_candidates:
                folded_candidates.append(folded_candidates[-1])
        elif len(folded_candidates) > self.n_candidates:
            # Truncate if too many
            folded_candidates = folded_candidates[:self.n_candidates]
        
        # Create labels
        type_label = self.class_to_idx[lc.label]
        
        # Handle periods (use NaN for missing periods, will be handled in loss)
        primary_period = lc.primary_period if lc.primary_period is not None else float('nan')
        secondary_period = lc.secondary_period if lc.secondary_period is not None else float('nan')
        
        # Create candidate labels (simplified: first candidate is "correct")
        candidate_labels = torch.zeros(self.n_candidates)
        if not np.isnan(primary_period):
            candidate_labels[0] = 1.0  # Mark first candidate as correct
        
        return {
            'light_curve': lc_input,
            'periodogram': pgram_input,
            'folded_candidates': folded_candidates,
            'type_label': type_label,
            'primary_period': primary_period,
            'secondary_period': secondary_period,
            'candidate_labels': candidate_labels,
            'raw_data': {
                'time': lc.time,
                'flux': lc.flux,
                'label': lc.label,
                'slope': lc.slope
            }
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for variable-length sequences."""
    
    # Extract batch components
    light_curves = [item['light_curve'] for item in batch]
    periodograms = [item['periodogram'] for item in batch]
    folded_candidates_list = [item['folded_candidates'] for item in batch]
    
    # Stack fixed-size tensors
    type_labels = torch.tensor([item['type_label'] for item in batch], dtype=torch.long)
    primary_periods = torch.tensor([item['primary_period'] for item in batch], dtype=torch.float32)
    secondary_periods = torch.tensor([item['secondary_period'] for item in batch], dtype=torch.float32)
    candidate_labels = torch.stack([item['candidate_labels'] for item in batch])
    
    # Batch variable-length light curves and periodograms
    batch_size = len(batch)
    
    # For light curves - pad sequences
    lc_times = [lc.time.squeeze(0) for lc in light_curves]
    lc_fluxes = [lc.flux.squeeze(0) for lc in light_curves]
    
    max_lc_len = max(len(t) for t in lc_times)
    batched_lc_times = torch.zeros(batch_size, max_lc_len)
    batched_lc_fluxes = torch.zeros(batch_size, max_lc_len)
    
    for i, (time, flux) in enumerate(zip(lc_times, lc_fluxes)):
        length = len(time)
        batched_lc_times[i, :length] = time
        batched_lc_fluxes[i, :length] = flux
    
    batched_lc = LightCurve(time=batched_lc_times, flux=batched_lc_fluxes)
    
    # For periodograms - pad sequences
    pg_freqs = [pg.frequencies.squeeze(0) for pg in periodograms]
    pg_powers = [pg.power.squeeze(0) for pg in periodograms]
    
    max_pg_len = max(len(f) for f in pg_freqs)
    batched_pg_freqs = torch.zeros(batch_size, max_pg_len)
    batched_pg_powers = torch.zeros(batch_size, max_pg_len)
    
    for i, (freq, power) in enumerate(zip(pg_freqs, pg_powers)):
        length = len(freq)
        batched_pg_freqs[i, :length] = freq
        batched_pg_powers[i, :length] = power
    
    batched_pg = Periodogram(frequencies=batched_pg_freqs, power=batched_pg_powers)
    
    # For folded candidates - handle variable lengths
    n_candidates = len(folded_candidates_list[0])
    batched_folded = []
    
    for cand_idx in range(n_candidates):
        cand_times = [batch_folded[cand_idx].time.squeeze(0) for batch_folded in folded_candidates_list]
        cand_fluxes = [batch_folded[cand_idx].flux.squeeze(0) for batch_folded in folded_candidates_list]
        cand_periods = torch.stack([batch_folded[cand_idx].period for batch_folded in folded_candidates_list])
        
        max_cand_len = max(len(t) for t in cand_times)
        batched_cand_times = torch.zeros(batch_size, max_cand_len)
        batched_cand_fluxes = torch.zeros(batch_size, max_cand_len)
        
        for i, (time, flux) in enumerate(zip(cand_times, cand_fluxes)):
            length = len(time)
            batched_cand_times[i, :length] = time
            batched_cand_fluxes[i, :length] = flux
        
        batched_folded.append(FoldedCandidate(
            time=batched_cand_times,
            flux=batched_cand_fluxes,
            period=cand_periods
        ))
    
    return {
        'light_curve': batched_lc,
        'periodogram': batched_pg,
        'folded_candidates': batched_folded,
        'type_labels': type_labels,
        'primary_periods': primary_periods,
        'secondary_periods': secondary_periods,
        'candidate_labels': candidate_labels
    }


# ===== Training Functions =====

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: StarModelConfig,
    device: torch.device,
    lambda_period1: float = 1.0,
    lambda_period2: float = 1.0,
    lambda_cand: float = 0.5
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_losses = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        lc = LightCurve(
            time=batch['light_curve'].time.to(device),
            flux=batch['light_curve'].flux.to(device)
        )
        pg = Periodogram(
            frequencies=batch['periodogram'].frequencies.to(device),
            power=batch['periodogram'].power.to(device)
        )
        folded_list = [
            FoldedCandidate(
                time=fc.time.to(device),
                flux=fc.flux.to(device),
                period=fc.period.to(device)
            ) for fc in batch['folded_candidates']
        ]
        
        type_labels = batch['type_labels'].to(device)
        primary_periods = batch['primary_periods'].to(device)
        secondary_periods = batch['secondary_periods'].to(device)
        candidate_labels = batch['candidate_labels'].to(device)
        
        # Handle NaN periods (mask them out in loss computation)
        primary_mask = ~torch.isnan(primary_periods)
        secondary_mask = ~torch.isnan(secondary_periods)
        
        # Replace NaN with zeros for computation (will be masked in loss)
        primary_periods = torch.where(primary_mask, primary_periods, torch.zeros_like(primary_periods))
        secondary_periods = torch.where(secondary_mask, secondary_periods, torch.zeros_like(secondary_periods))
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(lc, pg, folded_list)
        
        # Compute loss
        loss_output = multitask_loss(
            output,
            type_labels,
            primary_periods,
            secondary_periods,
            candidate_labels,
            cfg,
            lambda_period1=lambda_period1,
            lambda_period2=lambda_period2,
            lambda_cand=lambda_cand
        )
        
        loss = loss_output.total_loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        batch_size = type_labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_losses.append(loss_output.loss_logs)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'type': f"{loss_output.loss_logs.loss_type:.4f}",
            'period1': f"{loss_output.loss_logs.loss_period1:.4f}",
            'period2': f"{loss_output.loss_logs.loss_period2:.4f}",
        })
    
    # Compute average losses
    avg_loss = total_loss / total_samples
    avg_type_loss = np.mean([l.loss_type for l in all_losses])
    avg_period1_loss = np.mean([l.loss_period1 for l in all_losses])
    avg_period2_loss = np.mean([l.loss_period2 for l in all_losses])
    avg_cand_loss = np.mean([l.loss_cand for l in all_losses])
    
    return {
        'loss': avg_loss,
        'type_loss': avg_type_loss,
        'period1_loss': avg_period1_loss,
        'period2_loss': avg_period2_loss,
        'cand_loss': avg_cand_loss
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: StarModelConfig,
    device: torch.device,
    lambda_period1: float = 1.0,
    lambda_period2: float = 1.0,
    lambda_cand: float = 0.5
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_losses = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            # Move to device
            lc = LightCurve(
                time=batch['light_curve'].time.to(device),
                flux=batch['light_curve'].flux.to(device)
            )
            pg = Periodogram(
                frequencies=batch['periodogram'].frequencies.to(device),
                power=batch['periodogram'].power.to(device)
            )
            folded_list = [
                FoldedCandidate(
                    time=fc.time.to(device),
                    flux=fc.flux.to(device),
                    period=fc.period.to(device)
                ) for fc in batch['folded_candidates']
            ]
            
            type_labels = batch['type_labels'].to(device)
            primary_periods = batch['primary_periods'].to(device)
            secondary_periods = batch['secondary_periods'].to(device)
            candidate_labels = batch['candidate_labels'].to(device)
            
            # Handle NaN periods
            primary_mask = ~torch.isnan(primary_periods)
            secondary_mask = ~torch.isnan(secondary_periods)
            primary_periods = torch.where(primary_mask, primary_periods, torch.zeros_like(primary_periods))
            secondary_periods = torch.where(secondary_mask, secondary_periods, torch.zeros_like(secondary_periods))
            
            # Forward pass
            output = model(lc, pg, folded_list)
            
            # Compute loss
            loss_output = multitask_loss(
                output,
                type_labels,
                primary_periods,
                secondary_periods,
                candidate_labels,
                cfg,
                lambda_period1=lambda_period1,
                lambda_period2=lambda_period2,
                lambda_cand=lambda_cand
            )
            
            loss = loss_output.total_loss
            
            # Accumulate losses and predictions
            batch_size = type_labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_losses.append(loss_output.loss_logs)
            
            # Store predictions for metrics
            preds = torch.argmax(output.type_logits, dim=1).cpu().numpy()
            labels = type_labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Compute average losses
    avg_loss = total_loss / total_samples
    avg_type_loss = np.mean([l.loss_type for l in all_losses])
    avg_period1_loss = np.mean([l.loss_period1 for l in all_losses])
    avg_period2_loss = np.mean([l.loss_period2 for l in all_losses])
    avg_cand_loss = np.mean([l.loss_cand for l in all_losses])
    
    metrics = {
        'loss': avg_loss,
        'type_loss': avg_type_loss,
        'period1_loss': avg_period1_loss,
        'period2_loss': avg_period2_loss,
        'cand_loss': avg_cand_loss
    }
    
    return metrics, np.array(all_preds), np.array(all_labels)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    cfg: StarModelConfig,
    save_path: Path
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'config': cfg,
    }
    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train stellar classification model')
    parser.add_argument('--output-dir', type=str, default='./runs', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n-per-class', type=int, default=1000, help='Samples per class')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    print("Creating dataset...")
    dataset = StellarDataset(
        n_per_class=args.n_per_class,
        max_days=30.0,
        noise_level=0.02,
        n_candidates=4,
        seed=args.seed
    )
    
    # Split dataset
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # Create model
    cfg = StarModelConfig(
        n_types=len(dataset.class_names),
        logP_mean=dataset.logP_mean,
        logP_std=dataset.logP_std,
        emb_dim=128,
        merged_dim=256,
        cnn_hidden=64,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    )
    
    model = MultiBranchStarModelHybrid(cfg).to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, cfg, device,
            lambda_period1=1.0, lambda_period2=1.0, lambda_cand=0.5
        )
        
        # Validate
        val_metrics, val_preds, val_labels = validate_epoch(
            model, val_loader, cfg, device,
            lambda_period1=1.0, lambda_period2=1.0, lambda_cand=0.5
        )
        
        scheduler.step()
        
        # Record losses
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {np.mean(val_preds == val_labels):.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_metrics, val_metrics, cfg,
                output_dir / 'best_model.pth'
            )
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_metrics, val_metrics, cfg,
                output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            )
    
    # Final evaluation
    print("\nFinal evaluation...")
    val_metrics, val_preds, val_labels = validate_epoch(
        model, val_loader, cfg, device
    )
    
    # Classification report
    class_names = [dataset.idx_to_class[i] for i in range(len(dataset.class_names))]
    report = classification_report(val_labels, val_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'class_names': class_names,
        'config': vars(args)
    }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(val_labels, val_preds)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTraining completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()