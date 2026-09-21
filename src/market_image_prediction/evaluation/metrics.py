"""Predictive metrics for overlapping, cross-sectional panel forecasts.

Two statistical hazards dominate here and both are handled explicitly.

**Overlap.** With `horizon = 5`, consecutive daily IC observations share four days of
their label window, so the IC series is autocorrelated by construction. A plain t-test
on its mean is anticonservative -- it will call noise significant. Newey-West (HAC)
standard errors with a lag at least as long as the overlap correct for this.

**Width.** A rank correlation computed across 9 names is extremely noisy. Nothing fixes
that, so `n_names` is reported alongside every IC and dates too thin to score are
dropped rather than silently averaged in.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats


def _spearman(pred: np.ndarray, actual: np.ndarray) -> float:
    if pred.size < 2 or np.all(pred == pred[0]) or np.all(actual == actual[0]):
        return np.nan
    return float(stats.spearmanr(pred, actual).statistic)


def _pearson(pred: np.ndarray, actual: np.ndarray) -> float:
    if pred.size < 2 or np.all(pred == pred[0]) or np.all(actual == actual[0]):
        return np.nan
    return float(np.corrcoef(pred, actual)[0, 1])


def information_coefficient(
    predictions: pl.DataFrame,
    pred_col: str = "prediction",
    target_col: str = "target_return",
    min_names: int = 4,
) -> pl.DataFrame:
    """Per-date rank IC and Pearson IC.

    One row per `asof_date`. Dates with fewer than `min_names` scored names are
    excluded: a rank correlation over three points is not a measurement.
    """
    rows = []
    for (date,), group in predictions.group_by(["asof_date"], maintain_order=True):
        sub = group.drop_nulls([pred_col, target_col])
        if sub.height < min_names:
            continue
        p = sub[pred_col].to_numpy()
        a = sub[target_col].to_numpy()
        rows.append(
            {
                "asof_date": date,
                "rank_ic": _spearman(p, a),
                "ic": _pearson(p, a),
                "n_names": sub.height,
            }
        )
    return (
        pl.DataFrame(rows).sort("asof_date")
        if rows
        else pl.DataFrame(
            schema={
                "asof_date": pl.Date,
                "rank_ic": pl.Float64,
                "ic": pl.Float64,
                "n_names": pl.Int64,
            }
        )
    )


def newey_west_tstat(series: np.ndarray, lags: int) -> tuple[float, float]:
    """Mean and HAC t-statistic of a serially correlated series.

    Regressing the series on a constant makes the intercept its mean; the HAC
    covariance then supplies a standard error valid under autocorrelation up to `lags`.
    """
    x = series[~np.isnan(series)]
    if x.size < 3:
        return float("nan"), float("nan")
    model = sm.OLS(x, np.ones(x.size)).fit(cov_type="HAC", cov_kwds={"maxlags": max(int(lags), 1)})
    return float(model.params[0]), float(model.tvalues[0])


def summarise_ic(ic_frame: pl.DataFrame, horizon: int, periods_per_year: int = 252) -> dict:
    """Headline IC statistics with overlap-aware inference.

    `ic_ir` is the mean/std ratio of the IC series. Annualising it assumes independent
    observations, which overlapping labels violate -- so the HAC t-statistic, not the
    annualised IR, is the number to trust.
    """
    out: dict[str, float] = {}
    for col in ("rank_ic", "ic"):
        if col not in ic_frame.columns or ic_frame.is_empty():
            continue
        v = ic_frame[col].to_numpy().astype(float)
        mean, tstat = newey_west_tstat(v, lags=horizon)
        sd = float(np.nanstd(v, ddof=1))
        out[f"{col}_mean"] = mean
        out[f"{col}_std"] = sd
        out[f"{col}_ir"] = mean / sd if sd > 0 else float("nan")
        out[f"{col}_tstat_hac"] = tstat
        out[f"{col}_hit_rate"] = float(np.nanmean(v > 0))
    out["n_dates"] = float(ic_frame.height)
    mean_names = ic_frame["n_names"].mean() if not ic_frame.is_empty() else 0
    out["mean_names"] = float(mean_names) if isinstance(mean_names, (int, float)) else 0.0
    # Independent observations, not overlapping ones. The honest sample size.
    out["n_effective"] = out["n_dates"] / max(horizon, 1)
    return out


def directional_accuracy(
    predictions: pl.DataFrame, pred_col: str = "prediction", target_col: str = "target_return"
) -> dict:
    """Sign agreement, reported against the base rate it must beat.

    Equity drift makes "always long" a ~55% classifier here, so raw accuracy alone is
    not evidence of skill.
    """
    sub = predictions.drop_nulls([pred_col, target_col])
    if sub.is_empty():
        return {"accuracy": float("nan"), "base_rate": float("nan"), "n": 0.0}
    p = sub[pred_col].to_numpy() > 0
    a = sub[target_col].to_numpy() > 0
    return {
        "accuracy": float(np.mean(p == a)),
        "base_rate": float(max(np.mean(a), 1 - np.mean(a))),
        "n": float(sub.height),
    }


def quantile_profile(
    predictions: pl.DataFrame,
    n_quantiles: int = 10,
    pred_col: str = "prediction",
    target_col: str = "target_return",
    min_names: int = 20,
) -> pl.DataFrame:
    """Mean forward return by predicted quantile.

    Full-sample IC weights every pair equally, so a swap between ranks 249 and 251
    counts as much as one between ranks 1 and 500 -- yet only the extremes are ever
    traded. This is the metric that matches the strategy: sort by *prediction* each
    date, bucket, and measure what each bucket actually earned.

    Sorting on the prediction is legitimate -- it is known at the decision date.
    Sorting on the realised return would be look-ahead.

    Returns one row per quantile (1 = lowest predicted), with the mean realised return
    and the number of name-dates behind it.
    """
    rows = []
    for (date,), group in predictions.group_by(["asof_date"], maintain_order=True):
        sub = group.drop_nulls([pred_col, target_col])
        if sub.height < min_names:
            continue
        ranked = sub.with_columns(
            ((pl.col(pred_col).rank(method="ordinal") - 1) * n_quantiles // pl.len() + 1).alias(
                "quantile"
            )
        )
        rows.append(
            ranked.group_by("quantile")
            .agg(pl.col(target_col).mean().alias("ret"), pl.len().alias("n"))
            .with_columns(pl.lit(date).alias("asof_date"))
        )
    if not rows:
        return pl.DataFrame(schema={"quantile": pl.Int64, "mean_return": pl.Float64})
    per_date = pl.concat(rows, how="vertical")
    return (
        per_date.group_by("quantile")
        .agg(
            pl.col("ret").mean().alias("mean_return"),
            pl.col("ret").std().alias("std_return"),
            pl.col("n").sum().alias("name_dates"),
        )
        .sort("quantile")
    )


def spread_series(
    predictions: pl.DataFrame,
    quantile: float = 0.25,
    pred_col: str = "prediction",
    target_col: str = "target_return",
    min_names: int = 20,
) -> pl.DataFrame:
    """Per-date top-minus-bottom return, the quantity the long-short book earns.

    This is the tails-only measure that full-sample IC does not give you: the *level*
    difference between the traded groups, rather than the ordering within them.
    """
    rows = []
    for (date,), group in predictions.group_by(["asof_date"], maintain_order=True):
        sub = group.drop_nulls([pred_col, target_col])
        if sub.height < min_names:
            continue
        k = max(1, round(sub.height * quantile))
        ordered = sub.sort(pred_col)
        lo, hi = ordered.head(k)[target_col].mean(), ordered.tail(k)[target_col].mean()
        if lo is None or hi is None:
            continue
        rows.append(
            {"asof_date": date, "spread": float(hi) - float(lo), "n_per_leg": k}  # type: ignore[arg-type]
        )
    return pl.DataFrame(rows).sort("asof_date") if rows else pl.DataFrame()


def tail_ic(
    predictions: pl.DataFrame,
    quantile: float = 0.25,
    pred_col: str = "prediction",
    target_col: str = "target_return",
    min_names: int = 20,
) -> pl.DataFrame:
    """Rank IC computed only within the predicted top and bottom quantiles.

    Provided for comparison, but read it with care. Conditioning on the predictor
    compresses its spread inside each leg, which attenuates correlation mechanically --
    so a low tail IC does not imply a worse model, and it is not comparable to the
    full-sample figure. A model that says "these 125 all rise" and is right earns the
    full spread while scoring near zero here.
    """
    rows = []
    for (date,), group in predictions.group_by(["asof_date"], maintain_order=True):
        sub = group.drop_nulls([pred_col, target_col])
        if sub.height < min_names:
            continue
        k = max(2, round(sub.height * quantile))
        ordered = sub.sort(pred_col)
        tails = pl.concat([ordered.head(k), ordered.tail(k)], how="vertical")
        rows.append(
            {
                "asof_date": date,
                "tail_rank_ic": _spearman(tails[pred_col].to_numpy(), tails[target_col].to_numpy()),
                "n_names": tails.height,
            }
        )
    return pl.DataFrame(rows).sort("asof_date") if rows else pl.DataFrame()
