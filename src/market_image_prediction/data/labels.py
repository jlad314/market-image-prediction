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
    """Attach the cross-sectional target the ranking model is trained against.

    Called after `add_forward_return`. Operates within each `date`, across the names
    tradable on that date, and must add a `target_cs` column.
    """
    # TODO(human): add the `target_cs` column, computed within each date.
    #
    # Available per row: date, ticker, target_return (null where the horizon runs off
    # the end of the sample, and on dates before a ticker listed).
    #
    # Group with `.over("date")`. Nothing here may reference another date.
    raise NotImplementedError("cross_sectional_target")


def build_labels(panel: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    return cross_sectional_target(add_forward_return(panel, cfg), cfg)
