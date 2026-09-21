"""Cross-sectional long-short backtest.

Runs entirely from a saved out-of-fold prediction file. It never touches a model, so a
result can be reproduced months later from two files on disk.

Execution timing is explicit and conservative:

* predictions are formed from data through the close of `asof_date`;
* the book is rebalanced on `execution_date`, one session later;
* the position is held `horizon` sessions and the realised return is the manifest's
  `target_return`, which was built under exactly that convention.

Because the label already encodes the fill convention, the backtest cannot silently
disagree with the target the model was trained on -- a common source of backtests that
look better than the signal.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from market_image_prediction.config import Config
from market_image_prediction.evaluation.costs import turnover
from market_image_prediction.utils.logging import get_logger

log = get_logger(__name__)


SAMPLE_SPACING_DAYS = {"daily": 1, "weekly": 5, "monthly": 21}


def rebalance_dates(dates: list, frequency: str, sample_frequency: str = "daily") -> set:
    """Thin decision dates down to the rebalance schedule.

    The step is the *ratio* of the two frequencies, not the rebalance frequency alone.
    Decision dates already arrive thinned to `sample_frequency`, so treating a monthly
    date list as if it were daily and stepping by 21 rebalances every 21 months.

    That was a real bug: with monthly samples and a "weekly" rebalance setting the
    backtest traded 47 times in 19 years while annualising as though it traded 12 times a
    year, overstating annualised return 4.9x and Sharpe 2.2x.
    """
    have = SAMPLE_SPACING_DAYS.get(sample_frequency, 1)
    want = SAMPLE_SPACING_DAYS.get(frequency, 1)
    step = max(1, round(want / have))
    return set(dates[::step])


def observed_periods_per_year(dates: list) -> float:
    """Annualisation factor measured from the traded dates themselves.

    Derived rather than assumed. A factor inferred from config silently disagrees with
    reality whenever the schedule and the holding period drift apart; measuring the actual
    spacing makes that impossible.
    """
    if len(dates) < 2:
        return float("nan")
    span_days = (dates[-1] - dates[0]).days
    if span_days <= 0:
        return float("nan")
    return (len(dates) - 1) / (span_days / 365.25)


def _leg_weights(scores: np.ndarray, pivot: float, scheme: str, gross: float) -> np.ndarray:
    """Weights within one leg, summing in absolute value to `gross`.

    `pivot` is the cross-sectional median prediction for the date. Conviction is measured
    as distance from it, which is monotonic within each leg -- the best name in the long
    leg and the worst in the short leg both sit furthest from the pivot. Measuring
    distance from the *leg's own* centre instead would produce a U-shape, paying as much
    for the marginal name as for the strongest.

    Equal weighting spends the same capital on the strongest conviction as on the 125th.
    """
    n = len(scores)
    if n == 0:
        return scores
    if scheme == "equal":
        raw = np.ones(n)
    elif scheme == "rank":
        # Rank of |distance from pivot| -- robust to the scale of the prediction, which
        # differs by orders of magnitude between a ridge and a CNN.
        raw = stats.rankdata(np.abs(scores - pivot))
    else:  # signal
        raw = np.abs(scores - pivot)
        if not np.any(raw > 0):
            raw = np.ones(n)
    return gross * raw / raw.sum()


def build_weights(predictions: pl.DataFrame, cfg: Config) -> tuple[np.ndarray, list[str], list]:
    """Long-short quantile weights per rebalance date.

    Returns (weights, tickers, dates) where weights is (n_dates, n_tickers). Each leg is
    scaled to 0.5 gross so total gross exposure is 1.0 and the book is dollar-neutral by
    construction, whatever the weighting scheme.
    """
    dates = sorted(predictions["asof_date"].unique().to_list())
    schedule = rebalance_dates(dates, cfg.backtest.rebalance, cfg.features.sample_frequency)
    tickers = sorted(predictions["ticker"].unique().to_list())
    index = {t: i for i, t in enumerate(tickers)}
    scheme = cfg.backtest.weighting

    kept_dates, rows = [], []
    for date in dates:
        if date not in schedule:
            continue
        day = predictions.filter(pl.col("asof_date") == date).drop_nulls("prediction")
        n = day.height
        if n < 2 * cfg.backtest.min_names_per_leg:
            continue
        n_long = max(cfg.backtest.min_names_per_leg, round(n * cfg.backtest.long_quantile))
        n_short = max(cfg.backtest.min_names_per_leg, round(n * cfg.backtest.short_quantile))
        if n_long + n_short > n:
            # Too few names to form disjoint legs; split what exists.
            n_long = n_short = n // 2
        ranked = day.sort("prediction", descending=True)
        scores = ranked["prediction"].to_numpy()
        pivot = float(np.median(scores))
        w = np.zeros(len(tickers))

        long_names = ranked["ticker"][:n_long]
        long_w = _leg_weights(scores[:n_long], pivot, scheme, 0.5)
        for t, wi in zip(long_names, long_w, strict=True):
            w[index[t]] = wi

        short_names = ranked["ticker"][-n_short:]
        short_w = _leg_weights(scores[-n_short:], pivot, scheme, 0.5)
        for t, wi in zip(short_names, short_w, strict=True):
            w[index[t]] = -wi

        rows.append(w)
        kept_dates.append(date)
    return (np.array(rows) if rows else np.zeros((0, len(tickers)))), tickers, kept_dates


def period_returns(
    predictions: pl.DataFrame, weights: np.ndarray, tickers: list[str], dates: list
) -> np.ndarray:
    """Realised gross return of each rebalance, using the label's own fill convention."""
    index = {t: i for i, t in enumerate(tickers)}
    out = np.zeros(len(dates))
    for k, date in enumerate(dates):
        day = predictions.filter(pl.col("asof_date") == date).drop_nulls("target_return")
        r = np.zeros(len(tickers))
        for ticker, value in zip(day["ticker"], day["target_return"], strict=True):
            r[index[ticker]] = value
        out[k] = float(weights[k] @ r)
    return out


def performance_stats(returns: np.ndarray, periods_per_year: float) -> dict:
    """Annualised performance of a period-return series.

    `periods_per_year` must reflect the *rebalance* frequency, not trading days, or the
    annualisation silently inflates Sharpe by sqrt(5) at weekly frequency.
    """
    r = returns[~np.isnan(returns)]
    if r.size < 2:
        return {
            k: float("nan")
            for k in ("mean", "ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate", "n")
        }
    ann_return = float(np.mean(r) * periods_per_year)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(periods_per_year))
    equity = np.cumsum(r)
    drawdown = equity - np.maximum.accumulate(equity)
    return {
        "mean": float(np.mean(r)),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float(np.mean(r > 0)),
        "n": float(r.size),
    }


def _per_name_cost_matrix(liquidity: pl.DataFrame, tickers: list[str], dates: list) -> np.ndarray:
    """(n_dates, n_tickers) relative-cost multipliers from dollar volume.

    Returns multipliers rather than absolute rates so the caller can still sweep a cost
    grid: each grid point sets the median name's rate and this scales around it.
    """
    from market_image_prediction.evaluation.costs import liquidity_scaled_bps

    index = {t: i for i, t in enumerate(tickers)}
    out = np.ones((len(dates), len(tickers)))
    for di, date in enumerate(dates):
        day = liquidity.filter(pl.col("asof_date") == date)
        if day.is_empty():
            continue
        cols = np.array([index.get(t, -1) for t in day["ticker"].to_list()])
        mult = liquidity_scaled_bps(day["dollar_volume"].to_numpy(), 1.0)
        keep = cols >= 0
        out[di, cols[keep]] = mult[keep]
    return out


def run_backtest(
    predictions: pl.DataFrame, cfg: Config, liquidity: pl.DataFrame | None = None
) -> dict:
    """Full backtest of one model's predictions, gross and net across the cost grid.

    `liquidity` is an optional (ticker, asof_date, dollar_volume) frame. When supplied and
    `cfg.backtest.cost_model` is "liquidity", each name is charged at its own rate rather
    than a universe-wide one -- which is the honest way to cost a wide universe.
    """
    weights, tickers, dates = build_weights(predictions, cfg)
    if weights.shape[0] == 0:
        return {"error": "no rebalance dates with enough names"}
    gross = period_returns(predictions, weights, tickers, dates)
    turns = turnover(weights)
    # Measured from the dates actually traded, not inferred from config. The two agree
    # only when the rebalance spacing matches the holding period; when they diverge the
    # measured figure is the honest one, and the warning below says which case applies.
    periods_per_year = observed_periods_per_year(dates)
    implied = 252.0 / max(cfg.labels.horizon, 1)
    if np.isfinite(periods_per_year) and abs(periods_per_year - implied) / implied > 0.2:
        log.warning(
            "rebalance spacing (%.1f/yr) disagrees with the %d-day holding period "
            "(%.1f/yr): positions %s. Annualising on the measured spacing.",
            periods_per_year,
            cfg.labels.horizon,
            implied,
            "overlap" if periods_per_year > implied else "leave idle gaps",
        )

    from market_image_prediction.evaluation.costs import cost_sensitivity

    per_name_bps = None
    if cfg.backtest.cost_model == "liquidity":
        if liquidity is None:
            log.warning(
                "cost_model='liquidity' but no liquidity frame supplied; "
                "falling back to a flat rate"
            )
        else:
            per_name_bps = _per_name_cost_matrix(liquidity, tickers, dates)

    return {
        "dates": dates,
        "gross_returns": gross,
        "turnover": turns,
        "weights": weights,
        "tickers": tickers,
        "gross": performance_stats(gross, periods_per_year),
        "cost_grid": cost_sensitivity(
            gross,
            turns,
            cfg.backtest.cost_bps_grid,
            periods_per_year,
            weights=weights,
            per_name_bps=per_name_bps,
        ),
    }
