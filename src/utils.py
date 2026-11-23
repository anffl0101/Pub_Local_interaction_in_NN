"""Shared helpers: output folders, seeding, TF32 switch, config save/load.

Configs are written as JSON with platform, versions and git commit.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def data_dir(*parts: str) -> Path:
    """Folder for saved numbers: <repo>/data/<parts>, or under $GROKKING_RESULTS if set."""
    env = os.environ.get("GROKKING_RESULTS")
    d = (Path(env) if env else REPO / "data").joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir(*parts: str) -> Path:
    """Folder for figures: <repo>/figures/<parts>, or $GROKKING_RESULTS/figures/<parts> if set."""
    env = os.environ.get("GROKKING_RESULTS")
    d = (Path(env) / "figures" if env else REPO / "figures").joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch from one integer."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def disable_tf32() -> None:
    """Turn off TF32 matmuls on NVIDIA GPUs, because its rounding acts as extra noise on top of the SGLD noise."""
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except ImportError:
        pass


def _git_commit() -> str | None:
    """Current git commit, or None if not in a git repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def save_config(config: Any, path: str | Path) -> None:
    """Write a config (dataclass or dict) to JSON with platform, versions and git commit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(config) if is_dataclass(config) else dict(config)
    payload["_provenance"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": _git_commit(),
    }
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def load_config(path: str | Path) -> dict:
    """Read a config JSON into a dict."""
    with Path(path).open() as fh:
        return json.load(fh)


def build_config(cls, section: dict | None):
    """Build dataclass cls with defaults overridden by section; keys starting with "_" are ignored."""
    from dataclasses import fields

    section = section or {}
    valid = {f.name for f in fields(cls)}
    unknown = [k for k in section if k not in valid and not k.startswith("_")]
    if unknown:
        # an unknown key is most likely a typo, so fail instead of ignoring it
        raise ValueError(
            f"{cls.__name__}: config keys {unknown} are not fields of this config "
            f"(valid fields: {sorted(valid)})"
        )
    kwargs = {k: v for k, v in section.items() if k in valid}
    return cls(**kwargs)
