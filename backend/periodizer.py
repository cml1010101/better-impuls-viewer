# star_model.py
# Hybrid CNN+Transformer multi-branch, multi-task model for stellar light curves.
# - Branches: raw light curve, periodogram, phase-folded candidates (+ period as constant channel)
# - Heads: type classification, main period regression (log10 days), per-candidate scoring (attention logits)
# - Variable-length safe via global pooling; folded candidates passed as a list of variable-length tensors.

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Blocks
# -----------------------------

class CNNTransformer1DExtractor(nn.Module):
    """
    Local CNN + global Transformer + global pooling -> fixed-size embedding.
    Length-agnostic: accepts (B, C, L) for any L.
    """
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
        """
        x: (B, C, L)
        src_key_padding_mask: (B, L) with True at PAD positions (optional; rarely needed if you didn't pad).
        """
        h = self.cnn(x)                    # (B, hidden, L)
        h = self.proj(h).transpose(1, 2)   # (B, L, d_model)
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)
        h = h.transpose(1, 2)              # (B, d_model, L)
        h = self.pool(h).squeeze(-1)       # (B, d_model)
        return self.fc(h)                  # (B, out_dim)


class CandidateAttention(nn.Module):
    """
    Attention pooling over candidate embeddings (B, N, D) -> (B, D) + (B, N) logits + (B, N) weights.
    """
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        E: (B, N, D)
        Returns:
          pooled: (B, D)
          logits: (B, N)
          weights: (B, N)
        """
        logits = self.scorer(E).squeeze(-1)   # (B, N)
        weights = F.softmax(logits, dim=1)    # (B, N)
        pooled = torch.bmm(weights.unsqueeze(1), E).squeeze(1)  # (B, D)
        return pooled, logits, weights


# -----------------------------
# Config
# -----------------------------

@dataclass
class StarModelConfig:
    n_types: int                                 # number of variability classes
    lc_in_channels: int = 1                      # raw light curve channels (e.g., flux or flux+delta_t)
    pgram_in_channels: int = 1                   # periodogram channels (e.g., power or power+freq)
    folded_in_channels: int = 1                  # folded flux channels BEFORE adding period channel
    add_period_channel: bool = True              # append constant logP channel for each folded candidate
    emb_dim: int = 128                           # per-branch embedding size
    merged_dim: int = 256                        # fused representation size
    cnn_hidden: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    # Normalization for period regression target (log10 days)
    logP_mean: float = 0.0
    logP_std: float = 1.0


# -----------------------------
# Model
# -----------------------------

class MultiBranchStarModelHybrid(nn.Module):
    """
    Branches:
      - raw light curve   : CNN+Transformer
      - periodogram       : CNN+Transformer
      - folded candidates : shared CNN+Transformer + attention pooling (period passed as constant channel)

    Heads:
      - type classification: (B, n_types)
      - main period regression (log10 days): (B,)
      - per-candidate logits: (B, N) (useful for supervising candidate correctness / multi-period)

    Forward expects:
      lc:           (B, C_lc, L_lc)
      pgram:        (B, C_pg, L_pg)
      folded_list:  List length N; each (B, C_folded, L_i)
      logP_list:    Optional[List length N; each (B,)] only needed if add_period_channel=True
    """
    def __init__(self, cfg: StarModelConfig):
        super().__init__()
        self.cfg = cfg

        # Encoders (hybrid per branch)
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

        folded_channels = cfg.folded_in_channels + (1 if cfg.add_period_channel else 0)
        self.folded_enc = CNNTransformer1DExtractor(
            in_channels=folded_channels, cnn_hidden=cfg.cnn_hidden,
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            out_dim=cfg.emb_dim, dropout=cfg.dropout
        )
        self.cand_attn = CandidateAttention(cfg.emb_dim, hidden=cfg.emb_dim)

        # Fusion MLP
        fusion_in = cfg.emb_dim * 3  # lc + pgram + folded
        self.fuse = nn.Sequential(
            nn.Linear(fusion_in, cfg.merged_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.merged_dim, cfg.merged_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )

        # Heads
        self.type_head = nn.Linear(cfg.merged_dim, cfg.n_types)
        self.period_head = nn.Linear(cfg.merged_dim, 1)  # predicts normalized logP; we de/normalize outside

    # ---- helpers ----

    def _add_constant_channel(self, x: torch.Tensor, const: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        const: (B,) -> broadcast to (B, 1, L), concatenated as last channel
        """
        B, _, L = x.shape
        ch = const.view(B, 1, 1).expand(B, 1, L)
        return torch.cat([x, ch], dim=1)

    def encode_folded_candidates(
        self,
        folded_list: List[torch.Tensor],               # each (B, C, L_i)
        logP_list: Optional[List[torch.Tensor]] = None # each (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          folded_pooled: (B, D)
          cand_logits:   (B, N)
          cand_weights:  (B, N)
          cand_embs:     (B, N, D)
        """
        assert isinstance(folded_list, list) and len(folded_list) > 0, "folded_list must be a non-empty list"
        B = folded_list[0].shape[0]
        N = len(folded_list)
        cand_embs = []

        for i in range(N):
            xi = folded_list[i]  # (B, C, L_i)
            if self.cfg.add_period_channel:
                assert logP_list is not None, "logP_list required when add_period_channel=True"
                norm_logP = (logP_list[i] - self.cfg.logP_mean) / (self.cfg.logP_std + 1e-8)
                xi = self._add_constant_channel(xi, norm_logP)  # (B, C+1, L_i)
            ei = self.folded_enc(xi)  # (B, D)
            cand_embs.append(ei)

        E = torch.stack(cand_embs, dim=1)                                # (B, N, D)
        folded_pooled, cand_logits, cand_weights = self.cand_attn(E)     # (B,D), (B,N), (B,N)
        return folded_pooled, cand_logits, cand_weights, E

    # ---- forward ----

    def forward(
        self,
        lc: torch.Tensor,                        # (B, C_lc, L_lc)
        pgram: torch.Tensor,                     # (B, C_pg, L_pg)
        folded_list: List[torch.Tensor],         # len N, each (B, C_folded, L_i)
        logP_list: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        z_lc = self.lc_enc(lc)                   # (B, D)
        z_pg = self.pgram_enc(pgram)             # (B, D)
        z_folded, cand_logits, cand_w, E = self.encode_folded_candidates(folded_list, logP_list)

        # fuse
        z = torch.cat([z_lc, z_pg, z_folded], dim=-1)   # (B, 3D)
        z = self.fuse(z)                                # (B, merged_dim)

        # heads
        type_logits = self.type_head(z)                 # (B, n_types)
        logP_pred_norm = self.period_head(z).squeeze(-1)  # (B,)
        logP_pred = logP_pred_norm * (self.cfg.logP_std + 1e-8) + self.cfg.logP_mean

        return {
            "type_logits": type_logits,     # (B, n_types)
            "logP_pred": logP_pred,         # (B,), log10 days
            "cand_logits": cand_logits,     # (B, N)
            "cand_weights": cand_w,         # (B, N)
            "cand_embs": E,                 # (B, N, D)
            "z_fused": z,                   # (B, merged_dim)
            "z_lc": z_lc, "z_pg": z_pg, "z_folded": z_folded
        }


# -----------------------------
# Loss utilities
# -----------------------------

def multitask_loss(
    out: Dict[str, torch.Tensor],
    y_type: torch.Tensor,               # (B,) class indices
    true_logP: torch.Tensor,            # (B,) log10 days
    cand_labels: Optional[torch.Tensor] = None,   # (B, N) multi-label (0/1)
    cfg: Optional[StarModelConfig] = None,
    lambda_period: float = 1.0,
    lambda_cand: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    cand_labels: if provided, uses BCEWithLogits on cand_logits (multi-period friendly).
    """
    # Type classification
    loss_type = F.cross_entropy(out["type_logits"], y_type)

    # Period regression on normalized targets
    if cfg is None:
        raise ValueError("Provide cfg to normalize period targets")
    true_logP_norm = (true_logP - cfg.logP_mean) / (cfg.logP_std + 1e-8)
    pred_logP_norm = (out["logP_pred"] - cfg.logP_mean) / (cfg.logP_std + 1e-8)
    loss_period = F.smooth_l1_loss(pred_logP_norm, true_logP_norm)

    # Candidate scoring
    loss_cand = torch.tensor(0.0, device=out["type_logits"].device)
    if cand_labels is not None:
        loss_cand = F.binary_cross_entropy_with_logits(out["cand_logits"], cand_labels)

    loss = loss_type + lambda_period * loss_period + lambda_cand * loss_cand
    logs = {
        "loss": float(loss.detach().cpu()),
        "loss_type": float(loss_type.detach().cpu()),
        "loss_period": float(loss_period.detach().cpu()),
        "loss_cand": float(loss_cand.detach().cpu()),
    }
    return loss, logs


# -----------------------------
# Example usage / sanity test
# -----------------------------

if __name__ == "__main__":
    # Synthetic batch with variable lengths
    torch.manual_seed(0)

    B = 8        # batch size
    N = 4        # number of candidate periods per sample
    n_types = 13

    cfg = StarModelConfig(
        n_types=n_types,
        lc_in_channels=1,
        pgram_in_channels=1,
        folded_in_channels=1,   # folded flux only (period channel will be added)
        add_period_channel=True,
        emb_dim=128,
        merged_dim=256,
        cnn_hidden=64,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        # Set these from your training set stats
        logP_mean=0.0,
        logP_std=1.0,
    )

    model = MultiBranchStarModelHybrid(cfg)

    # Raw LC and periodogram (variable lengths across batches are fine if you bucket; here fixed for simplicity)
    lc = torch.randn(B, cfg.lc_in_channels, 1200)
    pgram = torch.randn(B, cfg.pgram_in_channels, 900)

    # Folded candidates: list of N tensors, each (B, C, L_i) with different L_i
    folded_list = [
        torch.randn(B, cfg.folded_in_channels, 200 + 25 * i) for i in range(N)
    ]
    # Corresponding log10 periods per candidate: list length N, each (B,)
    logP_list = [torch.randn(B) for _ in range(N)]

    # Forward pass
    out = model(lc, pgram, folded_list, logP_list)
    print("type_logits:", out["type_logits"].shape)
    print("logP_pred:", out["logP_pred"].shape)
    print("cand_logits:", out["cand_logits"].shape)
    print("cand_weights (row sums):", out["cand_weights"].sum(dim=1))

    # Fake labels
    y_type = torch.randint(0, n_types, (B,))
    true_logP = torch.randn(B)
    cand_labels = torch.zeros(B, N)
    cand_labels[:, 0] = 1.0  # pretend candidate 0 is correct

    # Compute loss
    loss, logs = multitask_loss(out, y_type, true_logP, cand_labels=cand_labels, cfg=cfg)
    print("loss logs:", logs)

    # Backward
    loss.backward()
    print("Backward OK.")