"""Metric correctness, especially the overlap-aware inference."""

from __future__ import annotations

import numpy as np
import polars as pl

from market_image_prediction.evaluation.metrics import (
    information_coefficient,
    newey_west_tstat,
    summarise_ic,
)


def _panel(pred: np.ndarray, actual: np.ndarray, n_dates: int, n_names: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "asof_date": np.repeat(np.arange(n_dates), n_names),
            "prediction": pred,
            "target_return": actual,
        }
    )


def test_perfect_predictor_scores_one() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=300 * 9)
    ic = information_coefficient(_panel(a, a, 300, 9))
    assert np.allclose(ic["rank_ic"].to_numpy(), 1.0)


def test_inverted_predictor_scores_minus_one() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=300 * 9)
    ic = information_coefficient(_panel(-a, a, 300, 9))
    assert np.allclose(ic["rank_ic"].to_numpy(), -1.0)


def test_noise_scores_about_zero() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    ic = information_coefficient(_panel(rng.normal(size=n * 9), rng.normal(size=n * 9), n, 9))
    mean_ic = ic["rank_ic"].mean()
    assert isinstance(mean_ic, float) and abs(mean_ic) < 0.02


def test_hac_is_more_conservative_under_autocorrelation() -> None:
    """The whole reason HAC is used: overlapping labels inflate a naive t-statistic."""
    rng = np.random.default_rng(5)
    x = np.zeros(3000)
    for i in range(1, x.size):
        x[i] = 0.85 * x[i - 1] + rng.normal(0, 0.05)
    x += 0.02
    naive_t = x.mean() / (x.std(ddof=1) / np.sqrt(x.size))
    _, hac_t = newey_west_tstat(x, lags=5)
    assert abs(hac_t) < abs(naive_t)


def test_hac_matches_naive_when_iid() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(0.01, 1.0, 5000)
    naive_t = x.mean() / (x.std(ddof=1) / np.sqrt(x.size))
    _, hac_t = newey_west_tstat(x, lags=1)
    assert abs(hac_t - naive_t) < 0.25


def test_thin_dates_are_dropped() -> None:
    df = pl.DataFrame(
        {
            "asof_date": [1, 1, 2, 2, 2, 2, 2],
            "prediction": [0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5],
            "target_return": [0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    ic = information_coefficient(df, min_names=4)
    assert ic.height == 1
    assert ic["asof_date"][0] == 2


def test_effective_sample_accounts_for_overlap() -> None:
    rng = np.random.default_rng(7)
    n = 500
    ic = information_coefficient(_panel(rng.normal(size=n * 9), rng.normal(size=n * 9), n, 9))
    s = summarise_ic(ic, horizon=5)
    assert np.isclose(s["n_effective"], s["n_dates"] / 5)
