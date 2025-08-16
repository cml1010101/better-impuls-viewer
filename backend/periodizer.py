# star_model.py
# Hybrid CNN+Transformer multi-branch, multi-task model for stellar light curves

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, NamedTuple
from collections import namedtuple

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
    logP_pred: torch.Tensor
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
    loss_period: float
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


class MultiBranchStarModelHybrid(nn.Module):
    """
    Multi-branch model with:
    - Raw light curve branch (CNN+Transformer)
    - Periodogram branch (CNN+Transformer) 
    - Folded candidates branch (CNN+Transformer + attention pooling)
    
    Each folded candidate has 2 period channels appended (for both periods).
    Variable number of candidates supported.
    
    Outputs:
    - Type classification
    - Main period regression (log10 days)
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

        # Always add 2 period channels (+2 for both periods)
        folded_channels = cfg.folded_in_channels + 2
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
        self.period_head = nn.Linear(cfg.merged_dim, 1)

    def _add_constant_channel(self, x: torch.Tensor, const: torch.Tensor) -> torch.Tensor:
        B, _, L = x.shape
        ch = const.view(B, 1, 1).expand(B, 1, L)
        return torch.cat([x, ch], dim=1)

    def _add_two_period_channels(self, x: torch.Tensor, logP1: torch.Tensor, logP2: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        logP1, logP2: (B,) -> broadcast to (B, 1, L) each, concatenated as last 2 channels
        """
        B, _, L = x.shape
        ch1 = logP1.view(B, 1, 1).expand(B, 1, L)
        ch2 = logP2.view(B, 1, 1).expand(B, 1, L)
        return torch.cat([x, ch1, ch2], dim=1)

    def encode_folded_candidates(
        self,
        folded_list: List[torch.Tensor],  # each (B, C, L_i)
        logP1: torch.Tensor,              # (B,)
        logP2: torch.Tensor,              # (B,)
    ) -> CandidateEncodingOutput:
        assert isinstance(folded_list, list) and len(folded_list) > 0
        N = len(folded_list)
        cand_embs = []
        
        # Normalize periods
        norm_logP1 = (logP1 - self.cfg.logP_mean) / (self.cfg.logP_std + 1e-8)
        norm_logP2 = (logP2 - self.cfg.logP_mean) / (self.cfg.logP_std + 1e-8)

        for i in range(N):
            xi = folded_list[i]  # (B, C, L_i)
            # Add both period channels to each folded candidate
            xi_with_periods = self._add_two_period_channels(xi, norm_logP1, norm_logP2)  # (B, C+2, L_i)
            ei = self.folded_enc(xi_with_periods)  # (B, D)
            cand_embs.append(ei)

        E = torch.stack(cand_embs, dim=1)  # (B, N, D)
        attn_output = self.cand_attn(E)
        return CandidateEncodingOutput(
            folded_pooled=attn_output.pooled,
            cand_logits=attn_output.logits,
            cand_weights=attn_output.weights,
            embeddings=E
        )

    def forward(
        self,
        lc: torch.Tensor,                 # (B, C_lc, L_lc)
        pgram: torch.Tensor,              # (B, C_pg, L_pg)
        folded_list: List[torch.Tensor],  # len N, each (B, C_folded, L_i)
        logP1: torch.Tensor,              # (B,) - first period for all candidates
        logP2: torch.Tensor,              # (B,) - second period for all candidates
    ) -> ModelOutput:
        z_lc = self.lc_enc(lc)
        z_pg = self.pgram_enc(pgram)
        cand_output = self.encode_folded_candidates(folded_list, logP1, logP2)

        z = torch.cat([z_lc, z_pg, cand_output.folded_pooled], dim=-1)
        z = self.fuse(z)

        type_logits = self.type_head(z)
        logP_pred_norm = self.period_head(z).squeeze(-1)
        logP_pred = logP_pred_norm * (self.cfg.logP_std + 1e-8) + self.cfg.logP_mean

        return ModelOutput(
            type_logits=type_logits,
            logP_pred=logP_pred,
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
    true_logP: torch.Tensor,            # (B,) log10 days
    cand_labels: Optional[torch.Tensor] = None,   # (B, N) binary labels for candidates
    cfg: Optional[StarModelConfig] = None,
    lambda_period: float = 1.0,
    lambda_cand: float = 0.5,
) -> LossOutput:
    loss_type = F.cross_entropy(out.type_logits, y_type)

    if cfg is None:
        raise ValueError("Provide cfg to normalize period targets")
    true_logP_norm = (true_logP - cfg.logP_mean) / (cfg.logP_std + 1e-8)
    pred_logP_norm = (out.logP_pred - cfg.logP_mean) / (cfg.logP_std + 1e-8)
    loss_period = F.smooth_l1_loss(pred_logP_norm, true_logP_norm)

    loss_cand = torch.tensor(0.0, device=out.type_logits.device)
    if cand_labels is not None:
        loss_cand = F.binary_cross_entropy_with_logits(out.cand_logits, cand_labels)

    loss = loss_type + lambda_period * loss_period + lambda_cand * loss_cand
    
    logs = LossLogs(
        loss=float(loss.detach().cpu()),
        loss_type=float(loss_type.detach().cpu()),
        loss_period=float(loss_period.detach().cpu()),
        loss_cand=float(loss_cand.detach().cpu())
    )
    
    return LossOutput(total_loss=loss, loss_logs=logs)


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

    # Input tensors
    lc = torch.randn(B, cfg.lc_in_channels, 1200)
    pgram = torch.randn(B, cfg.pgram_in_channels, 900)
    folded_list = [
        torch.randn(B, cfg.folded_in_channels, 200 + 25 * i) for i in range(N)
    ]
    logP1 = torch.randn(B)  # First period for all samples
    logP2 = torch.randn(B)  # Second period for all samples

    out = model(lc, pgram, folded_list, logP1, logP2)
    print("type_logits:", out.type_logits.shape)
    print("logP_pred:", out.logP_pred.shape)
    print("cand_logits:", out.cand_logits.shape)  # Should be (B, N)
    print("cand_weights (row sums):", out.cand_weights.sum(dim=1))

    # Fake labels
    y_type = torch.randint(0, n_types, (B,))
    true_logP = torch.randn(B)
    cand_labels = torch.zeros(B, N)  # Binary labels for N candidates
    cand_labels[:, 0] = 1.0  # First candidate is correct

    loss_output = multitask_loss(out, y_type, true_logP, cand_labels=cand_labels, cfg=cfg)
    print("loss logs:", loss_output.loss_logs)

    loss_output.total_loss.backward()
    print("Backward OK.")