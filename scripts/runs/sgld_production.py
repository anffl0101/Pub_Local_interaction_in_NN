"""The SGLD settings of the paper's networks (SGLDRecipe), the folder of saved states, and the SGLD config.
Used by sgld_512_runs.py.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.train import SGLDConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SGLDRecipe:
    # --- model ---
    n_layers: int = 5
    width: int = 256
    ortho_init: bool = True      # orthogonal start
    # --- SGLD ---
    eta: float = 0.1
    weight_decay: float = 1e-3
    T: float = 3e-5              # injected temperature
    init_seed: int = 42          # one shared start for every member
    log_every: int = 20


def ckpt_dir() -> Path:
    """Folder for stacked checkpoints (GB-scale): $GROKKING_CKPT, else <repo>/checkpoints."""
    env = os.environ.get("GROKKING_CKPT")
    root = Path(env) if env else (ROOT / "checkpoints")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sgld_cfg(rec: SGLDRecipe) -> SGLDConfig:
    # cfg.seed is the NOISE base for the ensemble; chunk + member index complete it.
    return SGLDConfig(eta=rec.eta, T=rec.T, weight_decay=rec.weight_decay,
                      seed=rec.init_seed)
