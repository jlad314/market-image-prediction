"""Seeding and run-provenance helpers."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch.

    Note: this does not guarantee bitwise determinism on MPS or CUDA. Any run that
    claims determinism must also record the accelerator and backend flags below.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_sha(short: bool = True) -> str:
    """Current commit, or 'unknown' outside a repo. Recorded with every run."""
    cmd = ["git", "rev-parse", *(["--short"] if short else []), "HEAD"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_is_dirty() -> bool:
    """True if the working tree has uncommitted changes (result is not reproducible)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607 - git resolved from PATH by design
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def provenance() -> dict[str, str | bool]:
    """Environment snapshot to attach to every artifact and MLflow run."""
    import numpy

    env: dict[str, str | bool] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
    }
    for name in ("torch", "yfinance", "polars", "pandas", "sklearn", "lightgbm"):
        try:
            env[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - version reporting is best-effort
            env[name] = "unavailable"
    return env
