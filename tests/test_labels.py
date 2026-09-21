"""The target must be exactly reproducible from raw prices under the stated convention."""

from __future__ import annotations

import numpy as np
import polars as pl

from market_image_prediction.config import Config
from market_image_prediction.data.labels import add_forward_return, build_labels


def test_target_reproduces_from_prices(synthetic_panel: pl.DataFrame, cfg: Config) -> None:
    p = add_forward_return(synthetic_panel, cfg)
    sub = p.filter(pl.col("ticker") == "AAA").sort("date")
    lag, horizon = cfg.labels.execution_lag, cfg.labels.horizon
    for i in (10, 100, 250):
        entry = sub["adj_open"][i + lag]
        exit_ = sub["adj_close"][i + horizon]
        assert np.isclose(sub["target_return"][i], np.log(exit_ / entry), atol=1e-12)


def test_execution_dates_follow_the_convention(synthetic_panel: pl.DataFrame, cfg: Config) -> None:
    p = add_forward_return(synthetic_panel, cfg).drop_nulls("target_return")
    # The close that formed the signal can never be the price that filled the trade.
    assert (
        (p["input_end_date"] if "input_end_date" in p.columns else p["date"])
        .le(p["execution_date"])
        .all()
    )
    assert p.select((pl.col("date") < pl.col("execution_date")).all()).item()
    assert p.select((pl.col("execution_date") <= pl.col("target_end_date")).all()).item()


def test_direction_matches_sign(synthetic_panel: pl.DataFrame, cfg: Config) -> None:
    p = add_forward_return(synthetic_panel, cfg).drop_nulls("target_return")
    assert p.select(
        ((pl.col("target_return") > 0).cast(pl.Int8) == pl.col("target_direction")).all()
    ).item()


def test_cross_sectional_target_is_within_date_only(
    synthetic_panel: pl.DataFrame, cfg: Config
) -> None:
    """Perturbing one date's returns must not change any other date's target."""
    base = build_labels(synthetic_panel, cfg)
    victim = base["asof_date" if "asof_date" in base.columns else "date"][200]
    shocked = synthetic_panel.with_columns(
        pl.when(pl.col("date") == victim)
        .then(pl.col("adj_close") * 1.5)
        .otherwise(pl.col("adj_close"))
        .alias("adj_close")
    )
    after = build_labels(shocked, cfg)
    # Rows whose input, execution and label windows all avoid the shocked date.
    mask = (pl.col("target_end_date") < victim) | (pl.col("date") > victim)
    a = base.filter(mask).drop_nulls("target_cs")["target_cs"].to_numpy()
    b = after.filter(mask).drop_nulls("target_cs")["target_cs"].to_numpy()
    assert np.allclose(a, b)


def test_rank_target_is_scale_invariant_to_cross_section_width(
    synthetic_panel: pl.DataFrame, cfg: Config
) -> None:
    """Extremes are exactly -1 and +1 whether 3 or 4 names trade that day."""
    full = build_labels(
        synthetic_panel, Config.model_validate({"labels": {"min_cross_section": 3}})
    ).drop_nulls("target_cs")
    narrowed = build_labels(
        synthetic_panel.filter(pl.col("ticker") != "DDD"),
        Config.model_validate({"labels": {"min_cross_section": 3}}),
    ).drop_nulls("target_cs")
    for frame in (full, narrowed):
        by_date = frame.group_by("date").agg(
            pl.col("target_cs").min().alias("lo"), pl.col("target_cs").max().alias("hi")
        )
        assert np.allclose(by_date["lo"].to_numpy(), -1.0)
        assert np.allclose(by_date["hi"].to_numpy(), 1.0)
