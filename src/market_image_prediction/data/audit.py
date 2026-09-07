"""Data audit.

The gate between the canonical panel and any modelling. It answers, with evidence:
is the panel complete, internally consistent, and wide enough at every date to support
a cross-sectional study?

Everything here is descriptive. It never mutates the panel.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from market_image_prediction.config import Config
from market_image_prediction.data.cleaning import QUALITY_FLAGS
from market_image_prediction.data.universe import get_universe
from market_image_prediction.utils.io import ensure_dir, write_parquet
from market_image_prediction.utils.logging import console, get_logger

log = get_logger(__name__)

PRICE_FIELDS = ("open", "high", "low", "close", "adj_close", "volume")


def _as_int(value: object) -> int:
    """Polars scalar aggregations are typed as a broad union; pin one to int.

    Returns 0 for null (empty frame) rather than raising, so a summary line can still
    render when a table legitimately has no rows.
    """
    return 0 if value is None else int(value)  # type: ignore[arg-type]


def coverage(panel: pl.DataFrame) -> pl.DataFrame:
    """Per-ticker span, bar count and calendar gaps."""
    return (
        panel.group_by("ticker")
        .agg(
            pl.len().alias("bars"),
            pl.col("date").min().alias("first"),
            pl.col("date").max().alias("last"),
            pl.col("close").null_count().alias("null_close"),
            pl.col("log_return").null_count().alias("null_return"),
        )
        .sort("first", "ticker")
    )


def missingness(panel: pl.DataFrame) -> pl.DataFrame:
    """Null counts per field per ticker; only non-clean rows are returned."""
    out = panel.group_by("ticker").agg([pl.col(f).null_count().alias(f) for f in PRICE_FIELDS])
    total = pl.sum_horizontal([pl.col(f) for f in PRICE_FIELDS]).alias("total_nulls")
    return out.with_columns(total).filter(pl.col("total_nulls") > 0).sort("ticker")


def duplicates(panel: pl.DataFrame) -> pl.DataFrame:
    """Any (ticker, date) appearing more than once. Must be empty."""
    return (
        panel.group_by(["ticker", "date"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .sort(["ticker", "date"])
    )


def flag_summary(panel: pl.DataFrame) -> pl.DataFrame:
    """Quality-flag counts per ticker, wide -> long."""
    counts = panel.group_by("ticker").agg([pl.col(f).sum().alias(f) for f in QUALITY_FLAGS])
    return (
        counts.unpivot(index="ticker", variable_name="flag", value_name="n")
        .filter(pl.col("n") > 0)
        .sort(["flag", "n"], descending=[False, True])
    )


def cross_section_width(panel: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Names available per date, once a full feature window of history exists.

    This is the binding constraint on a cross-sectional study: a long-short book needs
    enough names on *every* rebalance date, not on average.
    """
    eligible = (
        panel.filter(pl.col("close").is_not_null())
        .sort(["ticker", "date"])
        .with_columns(pl.int_range(pl.len()).over("ticker").alias("bar_index"))
        .filter(pl.col("bar_index") >= cfg.features.window - 1)
    )
    return eligible.group_by("date").agg(pl.len().alias("n_names")).sort("date")


def usable_windows(panel: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Samples each ticker can contribute, after window warm-up and label horizon."""
    need_future = cfg.labels.execution_lag + cfg.labels.horizon
    return (
        panel.filter(pl.col("close").is_not_null())
        .group_by("ticker")
        .agg(pl.len().alias("bars"))
        .with_columns(
            (pl.col("bars") - (cfg.features.window - 1) - need_future)
            .clip(lower_bound=0)
            .alias("usable_samples")
        )
        .sort("usable_samples", descending=True)
    )


def run_audit(cfg: Config, save: bool = True) -> dict[str, pl.DataFrame]:
    """Compute every audit table, print a summary, optionally persist to Parquet."""
    panel_path = Path(cfg.paths.canonical) / cfg.universe / "prices.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} not found; run `clean` first")
    panel = pl.read_parquet(panel_path)
    universe = get_universe(cfg.universe)

    tables = {
        "coverage": coverage(panel),
        "missingness": missingness(panel),
        "duplicates": duplicates(panel),
        "flags": flag_summary(panel),
        "cross_section_width": cross_section_width(panel, cfg),
        "usable_windows": usable_windows(panel, cfg),
    }

    width = tables["cross_section_width"]
    min_leg = cfg.backtest.min_names_per_leg * 2
    thin = width.filter(pl.col("n_names") < min_leg)

    dup_status = "  [green]OK[/green]" if tables["duplicates"].is_empty() else "  [red]FAIL[/red]"
    flagged = _as_int(tables["flags"]["n"].sum()) if not tables["flags"].is_empty() else 0

    console.rule("[bold]Data audit")
    console.print(
        f"universe            {universe.name} "
        f"(survivorship-bias-free={universe.survivorship_bias_free})"
    )
    console.print(f"rows / tickers      {panel.height:,} / {panel['ticker'].n_unique()}")
    console.print(f"date span           {panel['date'].min()} -> {panel['date'].max()}")
    console.print(f"duplicate keys      {tables['duplicates'].height}{dup_status}")
    console.print(f"tickers with nulls  {tables['missingness'].height}")
    console.print(f"flagged bars        {flagged}")
    total_samples = _as_int(tables["usable_windows"]["usable_samples"].sum())
    w_min = _as_int(width["n_names"].min())
    w_med = _as_int(width["n_names"].median())
    w_max = _as_int(width["n_names"].max())
    console.print(f"total usable samples {total_samples:,}")
    console.print(f"cross-section width  min={w_min} median={w_med} max={w_max}")
    console.print(
        f"dates below {min_leg} names  {thin.height} "
        f"({100 * thin.height / max(width.height, 1):.1f}% of dates)"
    )

    if save:
        out = ensure_dir(Path(cfg.paths.outputs) / "audit" / cfg.universe)
        for name, table in tables.items():
            write_parquet(table, out / f"{name}.parquet")
        log.info("audit tables written to %s", out)

    return tables
