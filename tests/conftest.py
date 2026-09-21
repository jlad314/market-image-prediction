"""Shared fixtures.

Tests run against the real built dataset where one exists, and against a synthetic
panel otherwise, so the suite is meaningful in CI without shipping data.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from market_image_prediction.config import Config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config()


@pytest.fixture(scope="session")
def synthetic_panel() -> pl.DataFrame:
    """A deterministic 4-ticker panel with a known geometric-random-walk price path."""
    rng = np.random.default_rng(0)
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(400)]
    frames = []
    for t in ("AAA", "BBB", "CCC", "DDD"):
        steps = rng.normal(0.0005, 0.01, len(dates))
        close = 100 * np.exp(np.cumsum(steps))
        frames.append(
            pl.DataFrame(
                {
                    "ticker": [t] * len(dates),
                    "date": dates,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "adj_close": close,
                    "volume": rng.integers(1_000_000, 9_000_000, len(dates)).astype(float),
                    "sector": [t] * len(dates),
                }
            )
        )
    panel = pl.concat(frames).sort(["ticker", "date"])
    return panel.with_columns(
        (pl.col("adj_close").log() - pl.col("adj_close").log().shift(1))
        .over("ticker")
        .alias("log_return")
    )


@pytest.fixture(scope="session")
def built(cfg: Config):
    """The real manifest and tensor, or skip if the dataset has not been built."""
    version = cfg.version_hash()
    tensor = ROOT / "data" / "tensors" / cfg.universe / version / "X_gaf.npy"
    manifest = ROOT / "data" / "manifests" / cfg.universe / f"manifest_{version}.parquet"
    if not (tensor.exists() and manifest.exists()):
        pytest.skip("dataset not built; run `market-image-prediction build`")
    return np.load(tensor, mmap_mode="r"), pl.read_parquet(manifest)
