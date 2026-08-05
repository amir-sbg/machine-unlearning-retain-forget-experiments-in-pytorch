from __future__ import annotations

import torch
from torch import nn


class DigitMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
