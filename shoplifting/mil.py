"""A small multiple-instance learning head for frozen video features.

Only video labels supervise this head. Window scores are weakly supervised
ranking signals; they are not annotations or proof of a particular action.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MILHead(nn.Module):
    """Score each temporal window without collapsing the window dimension."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[0] < 1:
            raise ValueError("features must have shape [windows, input_dim] with at least one window")
        return self.network(features).squeeze(-1)


class LinearHead(nn.Module):
    """A portable export of the mean-feature logistic classifier."""

    def __init__(self, input_dim: int = 512):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[0] < 1:
            raise ValueError("features must have shape [windows, input_dim]")
        return self.linear(features).squeeze(-1)


class TemporalGRUHead(nn.Module):
    """Use ordered window context; incident labels do not identify action stages."""
    def __init__(self, input_dim=512, hidden_dim=64, dropout=.2):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.sequence = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.output = nn.Linear(2 * hidden_dim, 1)

    def forward(self, features):
        if features.ndim != 2 or features.shape[0] < 1:
            raise ValueError("features must have shape [windows, input_dim]")
        context, _ = self.sequence(self.project(features).unsqueeze(0))
        return self.output(context.squeeze(0)).squeeze(-1)


def head_from_checkpoint(checkpoint: dict) -> nn.Module:
    """Restore either supported head; older checkpoints default to MIL."""
    kind = checkpoint.get("head_type", "mil")
    if kind == "mil":
        head = MILHead(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["dropout"])
    elif kind == "temporal_gru":
        head = TemporalGRUHead(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["dropout"])
    elif kind == "mean_logistic":
        head = LinearHead(checkpoint["input_dim"])
        if checkpoint["topk_fraction"] != 1.0:
            raise ValueError("A mean-logistic checkpoint must aggregate all window logits.")
    else:
        raise ValueError(f"Unsupported model head_type: {kind}")
    head.load_state_dict(checkpoint["state_dict"])
    return head


def aggregate_scores(logits: torch.Tensor, topk_fraction: float = 0.25) -> torch.Tensor:
    """Mean of the highest ceil(fraction * windows) logits, as a scalar.

    Aggregate logits before sigmoid, identically during training and inference.
    """
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits must be a non-empty one-dimensional tensor")
    if not 0 < topk_fraction <= 1:
        raise ValueError("topk_fraction must be in (0, 1]")
    count = max(1, math.ceil(logits.numel() * topk_fraction))
    return torch.topk(logits, k=count).values.mean()
