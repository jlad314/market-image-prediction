"""Targets, and the single execution convention they encode.

    signal formed from data through close(t)
      -> trade filled at OPEN(t + execution_lag)
      -> position closed at CLOSE(t + horizon)

so the primary target is

    y(t) = log( close(t + horizon) / open(t + execution_lag) )

Forward shifts here are correct and deliberate: a label is *defined* by data after the
decision date. What must never happen is a forward shift leaking into a feature, or a
target that begins on or before the last bar the model saw. `execution_lag >= 1`
enforces the second condition -- the close that generated the signal is never also the
price that fills the trade.

Adjustment note: Yahoo supplies `adj_close` but no adjusted open. Mixing a raw open
with an adjusted close silently books any dividend or split inside the holding window
as return. The open is therefore rescaled by the same factor: adj_open = open *
(adj_close / close).
"""

from __future__ import annotations

import polars as pl

from market_image_prediction.config import Config


def adjusted_prices(panel: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Put open/high/low on the same adjustment basis as `adj_close`."""
    if cfg.labels.price_field == "close":
        return panel.with_columns(pl.col("open").alias("adj_open"), pl.lit(1.0).alias("adj_factor"))
    factor = (pl.col("adj_close") / pl.col("close")).alias("adj_factor")
    return panel.with_columns(factor).with_columns(
        (pl.col("open") * pl.col("adj_factor")).alias("adj_open")
    )


def add_forward_return(panel: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Attach the tradable forward return and its derived direction label."""
    lag, horizon = cfg.labels.execution_lag, cfg.labels.horizon
    if horizon < lag:
        raise ValueError(f"horizon {horizon} must be >= execution_lag {lag}")

    entry_col = "adj_open" if cfg.labels.entry_price == "open" else cfg.labels.price_field
    exit_col = cfg.labels.price_field if cfg.labels.exit_price == "close" else "adj_open"

    entry = pl.col(entry_col).shift(-lag)
    exit_ = pl.col(exit_col).shift(-horizon)

    panel = adjusted_prices(panel, cfg)
    panel = panel.sort(["ticker", "date"]).with_columns(
        (exit_ / entry).log().over("ticker", order_by="date").alias("target_return"),
        pl.col("date").shift(-lag).over("ticker", order_by="date").alias("execution_date"),
        pl.col("date").shift(-horizon).over("ticker", order_by="date").alias("target_end_date"),
    )
    panel = panel.with_columns(
        (pl.col("target_return") > 0).cast(pl.Int8).alias("target_direction")
    )
    if cfg.labels.volatility_scaled:
        # Scale by volatility known at t, never by realised volatility over the
        # holding window -- the latter would be a label built from its own outcome.
        panel = panel.with_columns(
            (pl.col("target_return") / pl.col("feat_realised_vol_20d")).alias(
                "target_return_volscaled"
            )
        )
    return panel


def cross_sectional_target(panel: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Attach `target_cs`, the same-date ranking target the model is trained against.

    All three transforms are computed strictly within a `date`, so no information
    crosses dates and the label stays as-of its own decision day.

    The cross-section widens twice in this universe (XLRE in 2015, XLC in 2018). Every
    transform below is therefore normalised by the number of names *observed on that
    date*, never by a constant, so the target's scale does not step at those dates --
    a discontinuity a model would otherwise happily learn as signal.
    """
    r = pl.col("target_return")
    # Count only rows with an observed return: a ticker not yet listed, or one whose
    # holding window runs off the end of the sample, is absent from the cross-section
    # rather than being treated as a zero.
    n_obs = r.count().over("date")

    if cfg.labels.cross_sectional_transform == "rank":
        # average ties -> [1, n]; rescaled to [-1, 1] by (n - 1) so the extremes are
        # always exactly -1 and +1 regardless of how many names traded that day.
        rank = r.rank(method="average").over("date")
        target = pl.when(n_obs > 1).then(2.0 * (rank - 1.0) / (n_obs - 1.0) - 1.0).otherwise(None)
    elif cfg.labels.cross_sectional_transform == "demean":
        target = r - r.mean().over("date")
    else:  # zscore
        sd = r.std().over("date")
        target = pl.when(sd > 0).then((r - r.mean().over("date")) / sd).otherwise(None)

    return panel.with_columns(
        n_obs.alias("cs_size"),
        # A date too thin to rank produces a null target, not a degenerate one. These
        # rows are kept so the manifest still records why they were unusable.
        pl.when(n_obs >= cfg.labels.min_cross_section)
        .then(target)
        .otherwise(None)
        .alias("target_cs"),
    )


def build_labels(panel: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    return cross_sectional_target(add_forward_return(panel, cfg), cfg)
