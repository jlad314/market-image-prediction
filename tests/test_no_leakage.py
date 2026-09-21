"""Look-ahead leakage tests.

These are the tests the whole project rests on. If any of them fails, every downstream
metric is meaningless regardless of how good it looks.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from market_image_prediction.config import Config
from market_image_prediction.data.features import add_features, feature_columns
from market_image_prediction.data.gaf import encode_windows


def test_manifest_date_ordering(built) -> None:
    """input_end < execution <= target_end, for every sample."""
    _, m = built
    assert m.select((pl.col("input_start_date") <= pl.col("input_end_date")).all()).item()
    assert m.select((pl.col("input_end_date") < pl.col("execution_date")).all()).item()
    assert m.select((pl.col("execution_date") <= pl.col("target_end_date")).all()).item()


def test_no_tensor_without_manifest_row(built) -> None:
    x, m = built
    assert x.shape[0] == m.height
    assert m["sample_id"].n_unique() == m.height


def test_features_never_see_the_future(synthetic_panel: pl.DataFrame, cfg: Config) -> None:
    """Corrupting bar t must leave every feature at t-1 and earlier untouched."""
    base = add_features(synthetic_panel, cfg)
    cut = synthetic_panel["date"][300]
    shocked = add_features(
        synthetic_panel.with_columns(
            pl.when(pl.col("date") >= cut)
            .then(pl.col("adj_close") * 3.0)
            .otherwise(pl.col("adj_close"))
            .alias("adj_close"),
            pl.when(pl.col("date") >= cut)
            .then(pl.col("volume") * 7.0)
            .otherwise(pl.col("volume"))
            .alias("volume"),
        ),
        cfg,
    )
    cols = feature_columns(cfg)
    a = base.filter(pl.col("date") < cut).select(cols).to_numpy()
    b = shocked.filter(pl.col("date") < cut).select(cols).to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_encoding_is_per_sample(cfg: Config) -> None:
    """A window's image depends on that window alone -- no fitted, shared statistics."""
    rng = np.random.default_rng(3)
    w = rng.normal(size=(8, len(cfg.features.channels), cfg.features.window))
    full = encode_windows(w, cfg)
    subset = encode_windows(w[2:5], cfg)
    assert np.allclose(full[2:5], subset, atol=1e-6)


def test_input_window_precedes_label_window(built, cfg: Config) -> None:
    """No sample's input window may overlap its own label window."""
    _, m = built
    assert m.select((pl.col("input_end_date") < pl.col("execution_date")).all()).item()


def test_tensor_matches_manifest_metadata(built, cfg: Config) -> None:
    """Recompute a sample's image from the dates its manifest row claims."""
    x, m = built
    panel = add_features(pl.read_parquet(f"data/canonical/{cfg.universe}/prices.parquet"), cfg)
    cols = feature_columns(cfg)
    rng = np.random.default_rng(7)
    for k in rng.integers(0, m.height, 4):
        row = m.row(int(k), named=True)
        sub = panel.filter(
            (pl.col("ticker") == row["ticker"])
            & pl.col("date").is_between(row["input_start_date"], row["input_end_date"])
        ).sort("date")
        assert sub.height == cfg.features.window
        w = sub.select(cols).to_numpy().T[None, :, :]
        assert np.allclose(encode_windows(w, cfg)[0], x[int(k)], atol=1e-6)
