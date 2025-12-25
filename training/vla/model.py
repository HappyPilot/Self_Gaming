"""Simple action-chunk model for VLA training stub."""
from __future__ import annotations

from typing import Iterable

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch is required for training/vla/model.py") from exc


class ActionChunkMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: Iterable[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
