# periodizer.py
# Hybrid CNN+Transformer multi-branch, multi-task model for stellar light curves

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Named tuples for structured outputs
class AttentionOutput(NamedTuple):
    pooled: torch.Tensor
    logits: torch.Tensor
    weights: torch.Tensor


class CandidateEncodingOutput(NamedTuple):
    folded_pooled: torch.Tensor
    cand_logits: torch.Tensor
    cand_weights: torch.Tensor
    embeddings: torch.Tensor


class ModelOutput(NamedTuple):
    type_logits: torch.Tensor
    logP1_pred: torch.Tensor  # Primary period prediction
    logP2_pred: torch.Tensor  # Secondary period prediction
    cand_logits: torch.Tensor
    cand_weights: torch.Tensor
    cand_embs: torch.Tensor
    z_fused: torch.Tensor
    z_lc: torch.Tensor
    z_pg: torch.Tensor
    z_folded: torch.Tensor

class ModelInput(NamedTuple):
    lc: torch.Tensor  # Light curve data (shape: B x C x L)
    pgram: torch.Tensor  # Periodogram data (shape: B x C x L)
    folded_data: torch.Tensor  # Folded candidates data (shape: B x N x L_i)
    folded_periods: torch.Tensor  # Single input period for each candidate (shape: B x N)


class CNNTransformer1DExtractor(nn.Module):
    """Local CNN + global Transformer + global pooling -> fixed-size embedding."""
    
    def __init__(
        self,
        in_channels: int = 1,
        cnn_hidden: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        out_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv1d(cnn_hidden, d_model, kernel_size=1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout, dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.cnn(x)
        h = self.proj(h).transpose(1, 2)
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        h = h.transpose(1, 2)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


class CandidateAttention(nn.Module):
    """Attention pooling over candidate embeddings."""
    
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, E: torch.Tensor) -> AttentionOutput:
        logits = self.scorer(E).squeeze(-1)
        weights = F.softmax(logits, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), E).squeeze(1)
        return AttentionOutput(pooled=pooled, logits=logits, weights=weights)


@dataclass
class StarModelConfig:
    n_types: int
    lc_in_channels: int = 1
    pgram_in_channels: int = 1
    folded_in_channels: int = 1
    emb_dim: int = 128
    merged_dim: int = 256
    cnn_hidden: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    logP_mean: float = 1.0
    logP_std: float = 1.0

class MultiBranchStarModelHybrid(nn.Module):
    """
    Multi-branch model with:
    - Raw light curve branch (CNN+Transformer)
    - Periodogram branch (CNN+Transformer) 
    - Folded candidates branch (CNN+Transformer + attention pooling)
    
    Each folded candidate has 1 period channel appended (single input period).
    Variable number of candidates supported.
    
    Outputs:
    - Type classification
    - Primary period regression (log10 days)
    - Secondary period regression (log10 days)
    - Per-candidate logits
    """
    
    def __init__(self, cfg: StarModelConfig):
        super().__init__()
        self.cfg = cfg

        self.lc_enc = CNNTransformer1DExtractor(
            in_channels=cfg.lc_in_channels, cnn_hidden=cfg.cnn_hidden,
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            out_dim=cfg.emb_dim, dropout=cfg.dropout
        )
        self.pgram_enc = CNNTransformer1DExtractor(
            in_channels=cfg.pgram_in_channels, cnn_hidden=cfg.cnn_hidden,
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            out_dim=cfg.emb_dim, dropout=cfg.dropout
        )

        # Only add 1 period channel (+1 for single input period)
        folded_channels = cfg.folded_in_channels + 1
        self.folded_enc = CNNTransformer1DExtractor(
            in_channels=folded_channels, cnn_hidden=cfg.cnn_hidden,
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            out_dim=cfg.emb_dim, dropout=cfg.dropout
        )
        self.cand_attn = CandidateAttention(cfg.emb_dim, hidden=cfg.emb_dim)

        fusion_in = cfg.emb_dim * 3
        self.fuse = nn.Sequential(
            nn.Linear(fusion_in, cfg.merged_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.merged_dim, cfg.merged_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )

        self.type_head = nn.Linear(cfg.merged_dim, cfg.n_types)
        # Separate heads for primary and secondary periods
        self.period1_head = nn.Linear(cfg.merged_dim, 1)  # Primary period
        self.period2_head = nn.Linear(cfg.merged_dim, 1)  # Secondary period
    
    def normalize_period(self, period: torch.Tensor) -> torch.Tensor:
        """
        Normalize period to log10 days.
        """
        return (torch.log10(period) - self.cfg.logP_mean) / (self.cfg.logP_std + 1e-8)
    
    def denormalize_period(self, logP: torch.Tensor) -> torch.Tensor:
        """
        Denormalize log10 period to original scale.
        """
        return torch.pow(10, logP * (self.cfg.logP_std + 1e-8) + self.cfg.logP_mean)

    def _add_constant_channel(self, x: torch.Tensor, const: torch.Tensor) -> torch.Tensor:
        B, _, L = x.shape
        ch = const.view(B, 1, 1).expand(B, 1, L)
        return torch.cat([x, ch], dim=1)

    def encode_folded_candidates(
        self,
        folded_data: torch.Tensor,
        folded_periods: torch.Tensor
    ) -> CandidateEncodingOutput:
        """
        Encode folded candidates with single period channel.
        folded_data: Tensor of shape (B, N, L_i)
        folded_periods: Tensor of shape (B, N) - single input period for each candidate
        """
        B, N, L = folded_data.shape
        candidate_embeddings = []
        for i in range(N):
            folded_lc = folded_data[:, i, :].unsqueeze(1)
            folded_period = folded_periods[:, i].unsqueeze(1)  # Shape (B, 1)
            folded_lc = self._add_constant_channel(folded_lc, folded_period)
            folded_lc = self.folded_enc(folded_lc)
            candidate_embeddings.append(folded_lc)
        candidate_embeddings = torch.stack(candidate_embeddings, dim=1)
        # Apply attention pooling over candidate embeddings
        cand_output = self.cand_attn(candidate_embeddings)
        return CandidateEncodingOutput(
            folded_pooled=cand_output.pooled,
            cand_logits=cand_output.logits,
            cand_weights=cand_output.weights,
            embeddings=candidate_embeddings
        )

    def forward(
        self,
        input: ModelInput
    ) -> ModelOutput:
        normalized_periods = self.normalize_period(input.folded_periods)

        z_lc = self.lc_enc(input.lc)
        z_pg = self.pgram_enc(input.pgram)
        cand_output = self.encode_folded_candidates(input.folded_data, normalized_periods)

        z = torch.cat([z_lc, z_pg, cand_output.folded_pooled], dim=-1)
        z = self.fuse(z)

        type_logits = self.type_head(z)
        
        # Predict both primary and secondary periods
        logP1_pred_norm = self.period1_head(z).squeeze(-1)
        logP2_pred_norm = self.period2_head(z).squeeze(-1)
        
        # Denormalize predictions
        logP1_pred = self.denormalize_period(logP1_pred_norm)
        logP2_pred = self.denormalize_period(logP2_pred_norm)

        return ModelOutput(
            type_logits=type_logits,
            logP1_pred=logP1_pred,
            logP2_pred=logP2_pred,
            cand_logits=cand_output.cand_logits,
            cand_weights=cand_output.cand_weights,
            cand_embs=cand_output.embeddings,
            z_fused=z,
            z_lc=z_lc,
            z_pg=z_pg,
            z_folded=cand_output.folded_pooled
        )

class LossOutput(NamedTuple):
    total: torch.Tensor
    cls: torch.Tensor
    period: torch.Tensor


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        cls_weight: float = 1.0,
        period_weight: float = 1.0,
    ):
        """
        Multi-task loss for MultiBranchStarModelHybrid.
        
        Args:
            cls_weight: weight for classification (CE loss)
            period_weight: weight for period regression (MSE/Huber loss)
        """
        super().__init__()
        self.cls_weight = cls_weight
        self.period_weight = period_weight
        self.ce = nn.CrossEntropyLoss()
        self.reg = nn.SmoothL1Loss()

    def forward(
        self,
        output: ModelOutput,
        target_types: torch.Tensor,
        target_periods: torch.Tensor,  # shape: (B, 2), [P1, P2]
    ) -> LossOutput:
        # classification
        loss_cls = self.ce(output.type_logits, target_types)

        # regression for primary + secondary periods
        pred = torch.stack([output.logP1_pred, output.logP2_pred], dim=1)
        loss_period = self.reg(pred, target_periods)

        # total
        total = (
            self.cls_weight * loss_cls
            + self.period_weight * loss_period
        )

        return LossOutput(
            total=total,
            cls=loss_cls,
            period=loss_period
        )

# Function alias for backward compatibility
def multitask_loss(output: ModelOutput, target_types: torch.Tensor, target_periods: torch.Tensor) -> LossOutput:
    """Functional interface for MultiTaskLoss."""
    loss_fn = MultiTaskLoss()
    return loss_fn(output, target_types, target_periods)

import numpy as np

def interpolate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Interpolate y values to a uniform x grid.
    """
    uniform_x = np.linspace(np.min(x), np.max(x), num=len(x))
    uniform_y = np.interp(uniform_x, x, y)
    return uniform_y

def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize the input data using mean and standard deviation.
    """
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        std = 1e-8  # Avoid division by zero
    return (x - mean) / std

from data_processing import calculate_lomb_scargle, generate_candidate_periods, phase_fold_data

def create_input_data(lc: np.ndarray) -> ModelInput:
    time = lc[:, 0]
    flux = lc[:, 1]
    lc_interpolated = interpolate(time, flux)
    lc_normalized = normalize(lc_interpolated)

    frequencies, powers = calculate_lomb_scargle(time, flux)

    

if __name__ == '__main__':
    # Define model and loss hyperparameters
    n_types = 3
    batch_size = 4
    n_candidates = 5
    lc_len = 1000
    folded_len = 128
    
    config = StarModelConfig(
        n_types=n_types,
        lc_in_channels=1,
        pgram_in_channels=1,
        folded_in_channels=1,
    )
    
    # Initialize model and loss
    model = MultiBranchStarModelHybrid(config)
    loss_fn = MultiTaskLoss()
    
    print("Initializing model and loss function...")
    print(model)
    print("---------------------------------------")
    
    # Create dummy input data
    lc_data = torch.randn(batch_size, 1, lc_len)
    pgram_data = torch.randn(batch_size, 1, lc_len)
    folded_data = torch.randn(batch_size, n_candidates, folded_len)
    folded_periods = torch.rand(batch_size, n_candidates) * 10 + 1 # Periods between 1 and 11
    
    model_input = ModelInput(
        lc=lc_data,
        pgram=pgram_data,
        folded_data=folded_data,
        folded_periods=folded_periods,
    )
    
    print("Generating dummy data for testing...")
    print(f"lc_data shape: {lc_data.shape}")
    print(f"pgram_data shape: {pgram_data.shape}")
    print(f"folded_data shape: {folded_data.shape}")
    print(f"folded_periods shape: {folded_periods.shape}")
    print("---------------------------------------")
    
    # Forward pass
    print("Performing forward pass...")
    model_output = model(model_input)
    
    # Check output shapes
    print("Checking model output shapes:")
    print(f"type_logits shape: {model_output.type_logits.shape} (Expected: [{batch_size}, {n_types}])")
    assert model_output.type_logits.shape == (batch_size, n_types)
    print(f"logP1_pred shape: {model_output.logP1_pred.shape} (Expected: [{batch_size}])")
    assert model_output.logP1_pred.shape == (batch_size,)
    print(f"logP2_pred shape: {model_output.logP2_pred.shape} (Expected: [{batch_size}])")
    assert model_output.logP2_pred.shape == (batch_size,)
    print(f"cand_logits shape: {model_output.cand_logits.shape} (Expected: [{batch_size}, {n_candidates}])")
    assert model_output.cand_logits.shape == (batch_size, n_candidates)
    
    print("Forward pass successful!")
    print("---------------------------------------")
    
    # Create dummy target data
    target_types = torch.randint(0, n_types, (batch_size,))
    target_periods = torch.rand(batch_size, 2) * 10 + 1 # Two target periods between 1 and 11
    
    print("Generating dummy target data:")
    print(f"target_types shape: {target_types.shape}")
    print(f"target_periods shape: {target_periods.shape}")
    print("---------------------------------------")
    
    # Calculate loss
    print("Calculating multi-task loss...")
    loss_output = loss_fn(model_output, target_types, target_periods)
    
    print(f"Total Loss: {loss_output.total.item():.4f}")
    print(f"Classification Loss: {loss_output.cls.item():.4f}")
    print(f"Period Regression Loss: {loss_output.period.item():.4f}")
    
    # Simple backpropagation check
    loss_output.total.backward()
    print("Loss calculation and backward pass successful!")
    print("Test completed.")