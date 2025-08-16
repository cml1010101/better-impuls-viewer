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


class LossOutput(NamedTuple):
    total_loss: torch.Tensor
    loss_logs: 'LossLogs'


class LossLogs(NamedTuple):
    loss: float
    loss_type: float
    loss_period1: float
    loss_period2: float
    loss_cand: float


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
    logP_mean: float = 0.0
    logP_std: float = 1.0

from typing import NamedTuple

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
        lc: torch.Tensor,
        pgram: torch.Tensor,
        folded_data: torch.Tensor,
        folded_periods: torch.Tensor
    ) -> ModelOutput:
        normalized_periods = self.normalize_period(folded_periods)

        z_lc = self.lc_enc(lc)
        z_pg = self.pgram_enc(pgram)
        cand_output = self.encode_folded_candidates(folded_data, normalized_periods)

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


def multitask_loss(
    out: ModelOutput,
    y_type: torch.Tensor,               # (B,) class indices
    true_logP1: torch.Tensor,           # (B,) primary period log10 days
    true_logP2: torch.Tensor,           # (B,) secondary period log10 days
    cand_labels: Optional[torch.Tensor] = None,   # (B, N) binary labels for candidates
    cfg: Optional[StarModelConfig] = None,
    lambda_period1: float = 1.0,
    lambda_period2: float = 1.0,
    lambda_cand: float = 0.5,
) -> LossOutput:
    loss_type = F.cross_entropy(out.type_logits, y_type)

    if cfg is None:
        raise ValueError("Provide cfg to normalize period targets")
    
    # Normalize targets for both periods
    true_logP1_norm = (true_logP1 - cfg.logP_mean) / (cfg.logP_std + 1e-8)
    true_logP2_norm = (true_logP2 - cfg.logP_mean) / (cfg.logP_std + 1e-8)
    
    pred_logP1_norm = (out.logP1_pred - cfg.logP_mean) / (cfg.logP_std + 1e-8)
    pred_logP2_norm = (out.logP2_pred - cfg.logP_mean) / (cfg.logP_std + 1e-8)
    
    loss_period1 = F.smooth_l1_loss(pred_logP1_norm, true_logP1_norm)
    loss_period2 = F.smooth_l1_loss(pred_logP2_norm, true_logP2_norm)

    loss_cand = torch.tensor(0.0, device=out.type_logits.device)
    if cand_labels is not None:
        loss_cand = F.binary_cross_entropy_with_logits(out.cand_logits, cand_labels)

    loss = loss_type + lambda_period1 * loss_period1 + lambda_period2 * loss_period2 + lambda_cand * loss_cand
    
    logs = LossLogs(
        loss=float(loss.detach().cpu()),
        loss_type=float(loss_type.detach().cpu()),
        loss_period1=float(loss_period1.detach().cpu()),
        loss_period2=float(loss_period2.detach().cpu()),
        loss_cand=float(loss_cand.detach().cpu())
    )
    
    return LossOutput(total_loss=loss, loss_logs=logs)

def torch_interp(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor
) -> torch.Tensor:
    """
    A torch equivalent of np.interp.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
        xp: The x-coordinates of the data points, must be increasing.
        fp: The y-coordinates of the data points.

    Returns:
        The interpolated values, with the same shape as x.
    """
    # Find the indices of the two points to interpolate between
    indices = torch.searchsorted(xp, x)

    # Clamp the indices to prevent out-of-bounds access
    indices = torch.clamp(indices, 1, len(xp) - 1)

    # Gather the corresponding xp and fp values
    x0 = torch.gather(xp, 0, indices - 1)
    x1 = torch.gather(xp, 0, indices)
    y0 = torch.gather(fp, 0, indices - 1)
    y1 = torch.gather(fp, 0, indices)

    # Perform linear interpolation
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

def interpolate(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Interpolate values to a uniform grid.
    
    Args:
        x: Tensor of shape (B, L) with values to interpolate.
        y: Tensor of shape (B, L) with corresponding x-coordinates.
        
    Returns:
        Interpolated values on a uniform grid.
    """
    # Create a uniform grid based on the minimum and maximum x values
    min_x = y.min(dim=1, keepdim=True).values
    max_x = y.max(dim=1, keepdim=True).values
    uniform_x = torch.stack([torch.linspace(min_x[i].item(), max_x[i].item(), steps=x.size(1)) for i in range(x.size(0))], dim=0)

    # Interpolate y values for each sample in the batch
    interpolated_y = torch.zeros((x.size(0), uniform_x.size(1)), device=x.device)
    for i in range(x.size(0)):
        interpolated_y[i] = torch_interp(uniform_x[i], y[i], x[i])

    return interpolated_y 

def normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize values to zero mean and unit variance.
    
    Args:
        x: Tensor of shape (B, L) with values.
        
    Returns:
        Normalized values.
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-8  # Avoid division by zero
    return (x - mean) / std

class Periodogram(NamedTuple):
    frequencies: torch.Tensor  # Shape (B, L)
    power: torch.Tensor  # Shape (B, L)

    @property
    def periods(self) -> torch.Tensor:
        """Calculate periods from frequencies."""
        return 1.0 / self.frequencies
    
    def process(self) -> torch.Tensor:
        """Interpolate the periodogram to a uniform frequency grid."""
        return normalize(interpolate(self.power, self.frequencies)).unsqueeze(1)  # Add channel dimension

class LightCurve(NamedTuple):
    time: torch.Tensor  # Shape (B, L)
    flux: torch.Tensor  # Shape (B, L)

    def process(self) -> torch.Tensor:
        """Interpolate the light curve to a uniform time grid."""
        return normalize(interpolate(self.flux, self.time)).unsqueeze(1)  # Add channel dimension

class FoldedCandidate(NamedTuple):
    time: torch.Tensor  # Shape (B, L)
    flux: torch.Tensor  # Shape (B, L)
    period: torch.Tensor  # Shape (B,) - single input period

    def process(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate the folded candidate light curve to a uniform time grid."""
        return normalize(interpolate(self.flux, self.time)).unsqueeze(1), self.period.unsqueeze(1)  # Add channel dimension for period

import numpy as np
from data_processing import generate_candidate_periods, phase_fold_data, calculate_lomb_scargle

def prepare_input(lc_datas: list[np.ndarray]) -> tuple[LightCurve, Periodogram, list[FoldedCandidate]]:
    """
    Prepare model inputs from raw light curve data.
    
    Args:
        lc_datas: List of numpy arrays, each with shape (L, 2) for time and flux.
        
    Returns:
        Tuple of LightCurve, Periodogram, and list of FoldedCandidate.
    """
    lc_list = []
    pgram_list = []
    folded_list = []

    for data in lc_datas:
        time = torch.tensor(data[:, 0], dtype=torch.float32).unsqueeze(0)  # (1, L)
        flux = torch.tensor(data[:, 1], dtype=torch.float32).unsqueeze(0)  # (1, L)
        lc = LightCurve(time=time, flux=flux)
        lc_list.append(lc)

        # Calculate periodogram
        frequencies, power = calculate_lomb_scargle(data)
        frequencies = torch.tensor(frequencies, dtype=torch.float32).unsqueeze(0)  # (1, L)
        power = torch.tensor(power, dtype=torch.float32).unsqueeze(0)  # (1, L)
        pgram = Periodogram(frequencies=frequencies, power=power)
        pgram_list.append(pgram)

        # Generate candidate periods
        candidate_periods = generate_candidate_periods(frequencies.squeeze(0).numpy(), power.squeeze(0).numpy(), num_candidates=4)
        
        folded_candidates = []
        for period in candidate_periods:
            folded_time, folded_flux = phase_fold_data(time.squeeze(0).numpy(), flux.squeeze(0).numpy(), period)
            folded_time_tensor = torch.tensor(folded_time, dtype=torch.float32).unsqueeze(0)  # (1, L')
            folded_flux_tensor = torch.tensor(folded_flux, dtype=torch.float32).unsqueeze(0)  # (1, L')
            period_tensor = torch.tensor([period], dtype=torch.float32)  # (1,)
            folded_candidate = FoldedCandidate(time=folded_time_tensor, flux=folded_flux_tensor, period=period_tensor)
            folded_candidates.append(folded_candidate)

        folded_list.append(folded_candidates)
    # Add padding to make batch
    B = len(lc_list)
    lc_batch = LightCurve(
        time=torch.nn.utils.rnn.pad_sequence([lc.time.squeeze(0) for lc in lc_list], batch_first=True),
        flux=torch.nn.utils.rnn.pad_sequence([lc.flux.squeeze(0) for lc in lc_list], batch_first=True)
    )
    pgram_batch = Periodogram(
        frequencies=torch.nn.utils.rnn.pad_sequence([pg.frequencies.squeeze(0) for pg in pgram_list], batch_first=True),
        power=torch.nn.utils.rnn.pad_sequence([pg.power.squeeze(0) for pg in pgram_list], batch_first=True)
    )
    # Assume all have same number of candidates for simplicity
    N = len(folded_list[0])
    folded_batch = []
    for i in range(N):
        folded_batch.append(
            FoldedCandidate(
                time=torch.nn.utils.rnn.pad_sequence([folded_list[b][i].time.squeeze(0) for b in range(B)], batch_first=True),
                flux=torch.nn.utils.rnn.pad_sequence([folded_list[b][i].flux.squeeze(0) for b in range(B)], batch_first=True),
                period=torch.tensor([folded_list[b][i].period.item() for b in range(B)], dtype=torch.float32)
            )
        )
    return lc_batch, pgram_batch, folded_batch

if __name__ == "__main__":
    torch.manual_seed(0)

    B = 8
    N = 4  # number of folded candidates
    n_types = 13

    cfg = StarModelConfig(
        n_types=n_types,
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

    model = MultiBranchStarModelHybrid(cfg)

    # Create mock data structures
    lc = LightCurve(
        time=torch.randn(B, 1200),
        flux=torch.randn(B, 1200)
    )
    pgram = Periodogram(
        frequencies=torch.randn(B, 900),
        power=torch.randn(B, 900)
    )
    folded_list = [
        FoldedCandidate(
            time=torch.randn(B, 200 + 25 * i),
            flux=torch.randn(B, 200 + 25 * i),
            period=torch.randn(B)  # Single period per candidate
        ) for i in range(N)
    ]

    out = model(lc, pgram, folded_list)
    print("type_logits:", out.type_logits.shape)
    print("logP1_pred:", out.logP1_pred.shape)
    print("logP2_pred:", out.logP2_pred.shape)
    print("cand_logits:", out.cand_logits.shape)  # Should be (B, N)
    print("cand_weights (row sums):", out.cand_weights.sum(dim=1))

    # Fake labels
    y_type = torch.randint(0, n_types, (B,))
    true_logP1 = torch.randn(B)  # Primary period targets
    true_logP2 = torch.randn(B)  # Secondary period targets
    cand_labels = torch.zeros(B, N)  # Binary labels for N candidates
    cand_labels[:, 0] = 1.0  # First candidate is correct

    loss_output = multitask_loss(
        out, y_type, true_logP1, true_logP2, 
        cand_labels=cand_labels, cfg=cfg
    )
    print("loss logs:", loss_output.loss_logs)

    loss_output.total_loss.backward()
    print("Backward OK.")