"""Parquet/JSON helpers and the config-hash used to version datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_hash(payload: dict[str, Any], length: int = 12) -> str:
    """Stable hash of a config dict.

    Sorted keys and a fixed separator make this reproducible across processes, so the
    same feature configuration always resolves to the same dataset version.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:length]


def write_parquet(df: pl.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.write_parquet(path, compression="zstd")
    return path


def read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def write_json(payload: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
