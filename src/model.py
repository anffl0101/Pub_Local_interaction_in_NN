"""The MLP: linear layers with ReLU between them.

Input width (2p) and output width (p) are read from the dataset (``MLP.for_dataset``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    n_layers: int = 3  # number of weight matrices
    width: int = 256  # hidden width
    bias: bool = True


class MLP(nn.Module):
    """Plain MLP with ReLU hidden activations; the layers are in ``self.linears``."""

    def __init__(self, cfg: ModelConfig, in_dim: int, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.in_dim = in_dim
        self.out_dim = out_dim
        dims = [in_dim] + [cfg.width] * (cfg.n_layers - 1) + [out_dim]
        self.linears = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], bias=cfg.bias)
            for i in range(cfg.n_layers)
        )
        self.act = nn.ReLU()

    @classmethod
    def for_dataset(cls, cfg: ModelConfig, data: dict) -> "MLP":
        """Build an MLP sized to a dataset from ``make_modular_addition``."""
        in_dim = int(data["x_train"].shape[1])
        out_dim = int(data["p"])
        return cls(cfg, in_dim=in_dim, out_dim=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < len(self.linears) - 1:  # no activation after the last layer
                x = self.act(x)
        return x


def orthogonal_init_(model: "MLP") -> None:
    """Set each weight matrix to a random (semi-)orthogonal matrix and each bias to zero, in place.

    Uses a QR decomposition on the CPU, so the CPU thread count changes the initial weights.
    """
    n = len(model.linears)
    for i, lin in enumerate(model.linears):
        gain = (2.0 ** 0.5) if i < n - 1 else 1.0  # sqrt(2) offsets ReLU dropping about half the signal; output layer gain 1
        nn.init.orthogonal_(lin.weight, gain=gain)
        if lin.bias is not None:
            nn.init.zeros_(lin.bias)
