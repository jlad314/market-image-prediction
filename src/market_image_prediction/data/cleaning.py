"""Raw -> canonical price panel.

Responsibilities, in order:

1. align every symbol to the official NYSE trading calendar (never infer trading days
   from the data itself, or a symbol-wide outage becomes an invisible date gap);
2. enforce structural integrity (unique keys, OHLC consistency, positive prices);
3. flag suspicious bars for the audit step, without silently deleting them.

Nothing here may look forward. Every check is computed from data available on or
before the bar's own date.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas_market_calendars as mcal
import polars as pl

from market_image_prediction.config import Config
from market_image_prediction.data.universe import get_universe
from market_image_prediction.utils.io import write_parquet
from market_image_prediction.utils.logging import get_logger

log = get_logger(__name__)

QUALITY_FLAGS = (
    "flag_nonpositive_price",
    "flag_ohlc_inconsistent",
    "flag_zero_volume",
    "flag_extreme_return",
    "flag_stale_price",
)


def trading_days(start: dt.date, end: dt.date, calendar: str = "XNYS") -> pl.Series:
    """Official session dates from the exchange calendar.

    Goes via `to_numpy()` rather than the DatetimeIndex `.date` accessor: the latter
    yields an object-dtype array of `datetime.date`, which polars imports as Object and
    then refuses to cast. `to_numpy()` gives datetime64, which casts cleanly.
    """
    sched = mcal.get_calendar(calendar).schedule(start_date=start, end_date=end)
    return pl.Series("date", sched.index.to_numpy()).cast(pl.Date)


def align_to_calendar(prices: pl.DataFrame, sessions: pl.Series) -> pl.DataFrame:
    """Reindex each ticker onto the exchange calendar from its own first observation.

    Rows before a symbol's first bar are not created: an ETF that did not exist is
    absent, not missing. Gaps *after* listing become explicit nulls so the audit can
    count them.
    """
    out = []
    for (ticker,), group in prices.group_by(["ticker"], maintain_order=True):
        first, last = group["date"].min(), group["date"].max()
        spine = (
            sessions.to_frame()
            .filter((pl.col("date") >= first) & (pl.col("date") <= last))
            .with_columns(pl.lit(ticker).alias("ticker"))
        )
        out.append(spine.join(group, on=["ticker", "date"], how="left"))
    return pl.concat(out, how="vertical").sort(["ticker", "date"])


def add_returns(prices: pl.DataFrame, price_field: str = "adj_close") -> pl.DataFrame:
    """Close-to-close log return, per ticker. Uses only the current and prior bar."""
    return prices.with_columns(
        (pl.col(price_field).log() - pl.col(price_field).log().shift(1))
        .over("ticker")
        .alias("log_return")
    )


def flag_suspicious_bars(prices: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Attach boolean data-quality flags named in ``QUALITY_FLAGS``.

    Flags are diagnostic, not destructive: the audit reports them and the sample
    builder decides what to exclude. This keeps every exclusion visible and reversible.
    """
    q = cfg.cleaning
    ret = pl.col("log_return")
    w = q.extreme_return_lookback

    scale = (ret - ret.rolling_median(w, min_samples=w)).abs().rolling_median(
        w, min_samples=w
    ).shift(1) * 1.4826

    robust_z = ret.abs() / scale.clip(lower_bound=q.mad_epsilon)

    run_of_flat = (
        ret.eq(0)
        .fill_null(False)
        .cast(pl.Int32)
        .rolling_sum(q.stale_price_run, min_samples=q.stale_price_run)
    )

    ohlc = ("open", "high", "low", "close")

    return prices.with_columns(
        # Retained, not just thresholded. This is a causal regime-shift score -- it is
        # what actually separates March 2020 from October 2008 -- and it feeds the
        # by-volatility-regime diagnostics the backtest reports.
        robust_z.over("ticker", order_by="date").alias("robust_z_return"),
        pl.any_horizontal(pl.col(c) <= 0 for c in (*ohlc, "adj_close"))
        .fill_null(False)
        .alias("flag_nonpositive_price"),
        pl.any_horizontal(
            pl.col("high") < pl.col("low"),
            pl.col("high") < pl.col("open"),
            pl.col("high") < pl.col("close"),
            pl.col("low") > pl.col("open"),
            pl.col("low") > pl.col("close"),
        )
        .fill_null(False)
        .alias("flag_ohlc_inconsistent"),
        (pl.col("volume") <= 0).fill_null(False).alias("flag_zero_volume"),
        ((ret.abs() > q.extreme_return_abs) | (robust_z > q.extreme_return_sigma))
        .over("ticker", order_by="date")
        .fill_null(False)
        .alias("flag_extreme_return"),
        (run_of_flat == q.stale_price_run)
        .over("ticker", order_by="date")
        .fill_null(False)
        .alias("flag_stale_price"),
    )


def build_canonical(cfg: Config, force: bool = False) -> Path:
    """Produce the canonical panel consumed by every downstream stage."""
    universe = get_universe(cfg.universe)
    raw_path = Path(cfg.paths.raw) / cfg.universe / "prices.parquet"
    out_path = Path(cfg.paths.canonical) / cfg.universe / "prices.parquet"
    if out_path.exists() and not force:
        log.info("canonical panel exists at %s; skipping", out_path)
        return out_path
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found; run `download` first")

    prices = pl.read_parquet(raw_path)
    sessions = trading_days(cfg.download.start, cfg.download.end)

    tradable = prices.filter(pl.col("ticker").is_in(universe.tickers))
    tradable = align_to_calendar(tradable, sessions)
    tradable = add_returns(tradable, cfg.labels.price_field)
    tradable = flag_suspicious_bars(tradable, cfg)

    sector_map = {a.ticker: a.sector for a in universe.assets}
    tradable = tradable.with_columns(
        pl.col("ticker").replace_strict(sector_map, default=None).alias("sector")
    )

    write_parquet(tradable, out_path)
    log.info(
        "wrote canonical panel: %d rows, %d tickers -> %s",
        tradable.height,
        tradable["ticker"].n_unique(),
        out_path,
    )
    return out_path
