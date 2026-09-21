"""Backtest mechanics: weights, turnover, costs, annualisation."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from market_image_prediction.config import Config
from market_image_prediction.evaluation.backtest import (
    build_weights,
    performance_stats,
    period_returns,
)
from market_image_prediction.evaluation.costs import apply_costs, turnover


@pytest.fixture
def toy_predictions() -> pl.DataFrame:
    """8 names over 40 dates; prediction equals the realised return (perfect foresight)."""
    rng = np.random.default_rng(4)
    dates = [dt.date(2020, 1, 6) + dt.timedelta(days=i) for i in range(40)]
    rows = []
    for d in dates:
        for t in "ABCDEFGH":
            r = float(rng.normal(0, 0.02))
            rows.append({"asof_date": d, "ticker": t, "prediction": r, "target_return": r})
    return pl.DataFrame(rows)


def test_weights_are_dollar_neutral_and_unit_gross(toy_predictions, cfg: Config) -> None:
    w, _, _ = build_weights(toy_predictions, cfg)
    assert w.shape[0] > 0
    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-12)
    assert np.allclose(np.abs(w).sum(axis=1), 1.0, atol=1e-12)


def test_legs_are_equal_weighted(toy_predictions, cfg: Config) -> None:
    w, _, _ = build_weights(toy_predictions, cfg)
    for row in w:
        longs, shorts = row[row > 0], row[row < 0]
        assert np.allclose(longs, longs[0])
        assert np.allclose(shorts, shorts[0])
        assert np.isclose(longs.sum(), 0.5)
        assert np.isclose(shorts.sum(), -0.5)


def test_perfect_foresight_earns_positive_gross(toy_predictions, cfg: Config) -> None:
    """A sanity floor: if ranking by the realised return loses money, the wiring is wrong."""
    w, tickers, dates = build_weights(toy_predictions, cfg)
    r = period_returns(toy_predictions, w, tickers, dates)
    assert r.mean() > 0
    assert (r > 0).all()


def test_inverted_predictions_flip_the_sign(toy_predictions, cfg: Config) -> None:
    flipped = toy_predictions.with_columns(-pl.col("prediction"))
    w, tickers, dates = build_weights(flipped, cfg)
    r = period_returns(flipped, w, tickers, dates)
    assert (r < 0).all()


def test_turnover_is_one_way() -> None:
    """Rotating a whole 2-name book is 100% turnover, not 200%."""
    w = np.array([[0.5, -0.5], [-0.5, 0.5]])
    t = turnover(w)
    assert np.isclose(t[0], 0.5)  # first period builds the book from empty
    assert np.isclose(t[1], 1.0)


def test_unchanged_weights_cost_nothing() -> None:
    w = np.tile(np.array([0.5, -0.5]), (4, 1))
    t = turnover(w)
    assert np.allclose(t[1:], 0.0)


def test_costs_reduce_returns_monotonically() -> None:
    gross = np.full(10, 0.01)
    turns = np.full(10, 0.5)
    prev = gross.mean()
    for bps in (0.0, 5.0, 10.0, 20.0, 50.0):
        net = apply_costs(gross, turns, bps).mean()
        assert net <= prev
        prev = net


def test_cost_arithmetic_is_exact() -> None:
    """10 bps on 50% turnover costs exactly 5 bps."""
    net = apply_costs(np.array([0.0]), np.array([0.5]), 10.0)
    assert np.isclose(net[0], -0.0005)


def test_sharpe_uses_rebalance_frequency() -> None:
    """Annualising by trading days instead of holding periods inflates Sharpe by sqrt(5)."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 500)
    weekly = performance_stats(r, 252 / 5)
    daily = performance_stats(r, 252)
    assert np.isclose(daily["sharpe"] / weekly["sharpe"], np.sqrt(5), rtol=1e-6)


def test_max_drawdown_is_non_positive() -> None:
    rng = np.random.default_rng(1)
    stats = performance_stats(rng.normal(0, 0.01, 200), 252 / 5)
    assert stats["max_drawdown"] <= 0


def test_rebalance_step_accounts_for_sample_frequency() -> None:
    """Decision dates arrive already thinned; stepping again double-thins them.

    Regression guard. With monthly samples and a "weekly" rebalance setting this traded
    47 times in 19 years while annualising as though it traded 12 times a year --
    overstating annualised return 4.9x and Sharpe 2.2x.
    """
    import datetime as dt

    from market_image_prediction.evaluation.backtest import rebalance_dates

    monthly = [dt.date(2000, 1, 1) + dt.timedelta(days=30 * i) for i in range(24)]
    # Samples are monthly and we want monthly rebalancing: every date is traded.
    assert len(rebalance_dates(monthly, "monthly", "monthly")) == len(monthly)
    # Samples are monthly, rebalance weekly: cannot trade faster than the samples arrive.
    assert len(rebalance_dates(monthly, "weekly", "monthly")) == len(monthly)
    # Daily samples, monthly rebalance: step by 21.
    daily = [dt.date(2000, 1, 1) + dt.timedelta(days=i) for i in range(210)]
    assert len(rebalance_dates(daily, "monthly", "daily")) == 10


def test_periods_per_year_is_measured_not_assumed() -> None:
    """The annualisation factor comes from the traded dates themselves."""
    import datetime as dt

    from market_image_prediction.evaluation.backtest import observed_periods_per_year

    monthly = [dt.date(2000, 1, 1) + dt.timedelta(days=30 * i) for i in range(25)]
    assert observed_periods_per_year(monthly) == pytest.approx(12.2, abs=0.3)
    weekly = [dt.date(2000, 1, 1) + dt.timedelta(days=7 * i) for i in range(53)]
    assert observed_periods_per_year(weekly) == pytest.approx(52.2, abs=1.0)
    assert not np.isfinite(observed_periods_per_year([dt.date(2000, 1, 1)]))


def test_liquidity_scaling_charges_illiquid_names_more() -> None:
    """A thin name must cost more to trade than a heavily traded one."""
    from market_image_prediction.evaluation.costs import liquidity_scaled_bps

    dv = np.array([1e9, 1e8, 1e7, 1e6])
    bps = liquidity_scaled_bps(dv, 10.0)
    assert (np.diff(bps) > 0).all(), "cost must rise as volume falls"
    # The median name keeps the stated rate, within the bounds.
    assert bps.min() >= 10.0 * 0.4 - 1e-9
    assert bps.max() <= 10.0 * 6.0 + 1e-9


def test_liquidity_scaling_is_normalised_at_the_median() -> None:
    """`cost_bps` must keep meaning something: the typical name pays roughly it."""
    from market_image_prediction.evaluation.costs import liquidity_scaled_bps

    dv = np.array([1e8, 1e8, 1e8, 1e8])
    assert np.allclose(liquidity_scaled_bps(dv, 12.0), 12.0)


def test_weighted_cost_matches_flat_when_rates_are_equal() -> None:
    """The two models must agree when every name has the same rate."""
    from market_image_prediction.evaluation.costs import turnover, weighted_turnover_cost

    w = np.array([[0.5, -0.5, 0.0], [0.0, 0.5, -0.5]])
    flat = turnover(w) * (10 / 1e4)
    scaled = weighted_turnover_cost(w, np.full(w.shape, 10.0))
    assert np.allclose(flat, scaled)


def test_weighted_cost_penalises_trading_the_illiquid_leg() -> None:
    from market_image_prediction.evaluation.costs import weighted_turnover_cost

    w = np.array([[0.5, -0.5]])
    cheap = weighted_turnover_cost(w, np.array([[5.0, 5.0]]))
    dear = weighted_turnover_cost(w, np.array([[5.0, 50.0]]))
    assert dear[0] > cheap[0]


def test_degenerate_volume_falls_back_to_the_flat_rate() -> None:
    """All-missing volume must not produce zero or infinite costs."""
    from market_image_prediction.evaluation.costs import liquidity_scaled_bps

    assert np.allclose(liquidity_scaled_bps(np.array([0.0, np.nan]), 9.0), 9.0)
