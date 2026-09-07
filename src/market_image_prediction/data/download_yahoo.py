"""Yahoo Finance acquisition.

Raw files are immutable snapshots. Yahoo restates history whenever a corporate action
occurs, so re-downloading silently rewrites the past and makes earlier results
irreproducible. Every file is therefore stamped with a download date and source
versions, and overwriting requires an explicit flag.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import polars as pl
import yfinance as yf

from market_image_prediction.config import Config
from market_image_prediction.data.universe import get_universe
from market_image_prediction.utils.io import ensure_dir, write_json, write_parquet
from market_image_prediction.utils.logging import get_logger
from market_image_prediction.utils.seed import provenance

log = get_logger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _download_one(ticker: str, cfg: Config) -> pl.DataFrame:
    """Fetch a single symbol with retry/backoff, normalised to a tidy frame."""
    last_error: Exception | None = None
    for attempt in range(1, cfg.download.max_retries + 1):
        try:
            raw = yf.download(
                ticker,
                start=cfg.download.start,
                end=cfg.download.end,
                auto_adjust=cfg.download.auto_adjust,
                progress=False,
                multi_level_index=False,
                threads=False,
            )
            if raw is None or raw.empty:
                raise ValueError(f"empty response for {ticker}")
            break
        except Exception as exc:  # noqa: BLE001 - network layer is genuinely broad
            last_error = exc
            wait = cfg.download.retry_backoff_seconds * (2 ** (attempt - 1))
            log.warning(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                ticker,
                attempt,
                cfg.download.max_retries,
                exc,
                wait,
            )
            time.sleep(wait)
    else:
        raise RuntimeError(f"download failed for {ticker}") from last_error

    # Index carries the trading date; indices such as ^VIX have no Adj Close/Volume.
    raw = raw.reset_index()
    present = [c for c in OHLCV_COLUMNS if c in raw.columns]
    frame: pl.DataFrame = pl.from_pandas(raw[["Date", *present]])  # type: ignore[assignment]

    rename = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    frame = frame.rename({k: v for k, v in rename.items() if k in frame.columns})
    frame = frame.with_columns(
        pl.col("date").cast(pl.Date),
        pl.lit(ticker).alias("ticker"),
    )
    for missing in {"open", "high", "low", "close", "adj_close", "volume"} - set(frame.columns):
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(missing))

    return frame.select(
        "ticker", "date", "open", "high", "low", "close", "adj_close", "volume"
    ).sort("date")


def download_universe(cfg: Config, force: bool = False) -> Path:
    """Download every symbol the universe needs and write one immutable snapshot."""
    universe = get_universe(cfg.universe)
    out_dir = ensure_dir(Path(cfg.paths.raw) / cfg.universe)
    prices_path = out_dir / "prices.parquet"

    if prices_path.exists() and not (force or cfg.download.allow_overwrite):
        log.info(
            "raw snapshot already exists at %s; skipping (pass force=True to refresh)",
            prices_path,
        )
        return prices_path

    frames: list[pl.DataFrame] = []
    for ticker in universe.download_tickers:
        log.info("downloading %s", ticker)
        frames.append(_download_one(ticker, cfg))

    prices = pl.concat(frames, how="vertical").sort(["ticker", "date"])
    write_parquet(prices, prices_path)

    write_json(
        {
            "universe": cfg.universe,
            "tickers": list(universe.download_tickers),
            "start": cfg.download.start,
            "end": cfg.download.end,
            "auto_adjust": cfg.download.auto_adjust,
            "download_date": dt.datetime.now(dt.UTC).date(),
            "n_rows": prices.height,
            "environment": provenance(),
        },
        out_dir / "prices.meta.json",
    )
    log.info(
        "wrote %d rows for %d symbols to %s",
        prices.height,
        len(universe.download_tickers),
        prices_path,
    )
    return prices_path


def download_metadata(cfg: Config, force: bool = False) -> Path:
    """Cache sector/industry metadata.

    Treated as a convenience source, never as point-in-time truth. For the sector ETF
    universe the mapping is already known statically, so this is mainly a cross-check.
    """
    universe = get_universe(cfg.universe)
    out_dir = ensure_dir(Path(cfg.paths.raw) / cfg.universe)
    path = out_dir / "metadata.parquet"
    if path.exists() and not force:
        log.info("metadata cache exists at %s; skipping", path)
        return path

    rows = []
    for asset in universe.assets:
        info: dict = {}
        try:
            info = yf.Ticker(asset.ticker).info or {}
        except Exception as exc:  # noqa: BLE001
            log.warning("metadata fetch failed for %s: %s", asset.ticker, exc)
        rows.append(
            {
                "ticker": asset.ticker,
                "name": asset.name,
                "sector_static": asset.sector,
                "sector_yahoo": info.get("sector"),
                "industry_yahoo": info.get("industry"),
                "quote_type": info.get("quoteType"),
                "inception": asset.inception,
                "fetched_on": dt.datetime.now(dt.UTC).date(),
            }
        )
    write_parquet(pl.DataFrame(rows), path)
    log.info("wrote metadata for %d assets to %s", len(rows), path)
    return path
