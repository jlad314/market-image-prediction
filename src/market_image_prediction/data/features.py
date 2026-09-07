"""Feature channels.

Every feature is a polars expression over the canonical panel, evaluated per ticker and
using only data available at or before its own bar. Features are registered by name so
`FeatureConfig.channels` selects them from YAML without code changes, and so the audit
can enumerate exactly what a dataset version contains.

Causality rule for this module: no expression may reference a positive shift. Rolling
windows end at the current bar, which is legitimate -- the bar's own close is known at
the close of that bar. It is the *label* that must look forward, never the feature.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from market_image_prediction.config import Config

FeatureFn = Callable[[Config], pl.Expr]

REGISTRY: dict[str, FeatureFn] = {}


def register(name: str) -> Callable[[FeatureFn], FeatureFn]:
    def wrap(fn: FeatureFn) -> FeatureFn:
        REGISTRY[name] = fn
        return fn

    return wrap


# --- baseline channels -------------------------------------------------------------


@register("log_return")
def _log_return(cfg: Config) -> pl.Expr:
    """Close-to-close log return. Already materialised by `cleaning.add_returns`."""
    return pl.col("log_return")


@register("log_volume_change")
def _log_volume_change(cfg: Config) -> pl.Expr:
    """log(1+V_t) - log(1+V_{t-1}); the log1p keeps zero-volume bars finite."""
    v = (pl.col("volume").cast(pl.Float64) + 1.0).log()
    return v - v.shift(1)


@register("realised_vol_20d")
def _realised_vol_20d(cfg: Config) -> pl.Expr:
    """Trailing realised volatility, window ending at t."""
    w = cfg.features.realised_vol_lookback
    return pl.col("log_return").rolling_std(w, min_samples=w)


# --- extended channels (opt-in via config, for the ablation table) -----------------


@register("intraday_range")
def _intraday_range(cfg: Config) -> pl.Expr:
    """log(H/L): a Parkinson-style volatility proxy from a single bar."""
    return (pl.col("high") / pl.col("low")).log()


@register("overnight_return")
def _overnight_return(cfg: Config) -> pl.Expr:
    """log(O_t / C_{t-1}): the gap, which behaves differently from intraday drift."""
    return (pl.col("open") / pl.col("close").shift(1)).log()


@register("intraday_return")
def _intraday_return(cfg: Config) -> pl.Expr:
    """log(C_t / O_t): the session move, complement of the overnight gap."""
    return (pl.col("close") / pl.col("open")).log()


@register("momentum_20d")
def _momentum_20d(cfg: Config) -> pl.Expr:
    return pl.col("log_return").rolling_sum(20, min_samples=20)


@register("momentum_60d")
def _momentum_60d(cfg: Config) -> pl.Expr:
    return pl.col("log_return").rolling_sum(60, min_samples=60)


@register("dist_to_ma20")
def _dist_to_ma20(cfg: Config) -> pl.Expr:
    """log price relative to its own 20-day mean; a normalised level, not a level."""
    p = pl.col(cfg.labels.price_field)
    return (p / p.rolling_mean(20, min_samples=20)).log()


@register("dollar_volume")
def _dollar_volume(cfg: Config) -> pl.Expr:
    """log dollar volume: a liquidity proxy, and a capacity diagnostic later."""
    return (pl.col("close") * pl.col("volume") + 1.0).log()


# --- assembly ----------------------------------------------------------------------


def available() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def add_features(panel: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Attach every configured channel, computed per ticker in date order."""
    unknown = set(cfg.features.channels) - set(REGISTRY)
    if unknown:
        raise KeyError(f"unknown channels {sorted(unknown)}; available: {available()}")

    exprs = [
        REGISTRY[name](cfg).over("ticker", order_by="date").alias(f"feat_{name}")
        for name in cfg.features.channels
    ]
    return panel.sort(["ticker", "date"]).with_columns(exprs)


def feature_columns(cfg: Config) -> list[str]:
    return [f"feat_{name}" for name in cfg.features.channels]
