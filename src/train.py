"""Settings of the two optimizers, used by the batched training in src/batched_train.py.

SGLD step: theta <- theta - eta * grad(L) + sqrt(2 * eta * T) * xi, with xi ~ N(0, I).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdamWConfig:
    steps: int = 20000
    lr: float = 1e-3
    weight_decay: float = 1.0  # high weight decay makes the network grok
    log_every: int = 50
    seed: int = 0


@dataclass
class SGLDConfig:
    eta: float = 1e-3  # step size
    T: float = 1.0  # injected temperature
    weight_decay: float = 0.01  # confining term so the potential is well-defined
    seed: int = 0
