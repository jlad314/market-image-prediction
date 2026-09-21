"""Transaction costs and turnover.

Turnover is defined the standard way, as *one-way* traded fraction of the book:

    turnover_t = 0.5 * sum_i |w_it - w_i,t-1|

The one-half matters. Selling one name and buying another moves |dw| by 2 in total but
trades only 1 unit of the portfolio, so omitting the half doubles every cost estimate.

Costs are applied to that turnover in basis points. Two models are available:

* **flat** -- one rate for every name. Fair for a handful of mega-cap ETFs; optimistic
  the moment a universe includes small caps, whose spreads are several times wider.
* **liquidity-scaled** -- the rate varies per name as the inverse square root of dollar
  volume, a standard empirical regularity for effective spread. The stated `cost_bps`
  then anchors the *median-liquidity* name rather than every name.

The distinction matters whenever the universe widens. Adding 1,500 smaller names to a
500-name book raises breadth, which flatters Sharpe, while also raising the average cost
of trading -- and a flat model books the first effect without the second.

Neither model captures market impact, which grows with position size rather than with
the security's liquidity. Both therefore understate costs at scale.
"""

from __future__ import annotations

import numpy as np


def turnover(weights: np.ndarray) -> np.ndarray:
    """One-way turnover per rebalance for a (n_periods, n_assets) weight matrix.

    The first period trades from an empty book, so its turnover is the gross exposure
    put on. That is a real cost and is not discounted.
    """
    prev = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    return 0.5 * np.abs(weights - prev).sum(axis=1)


def apply_costs(gross_returns: np.ndarray, turnovers: np.ndarray, cost_bps: float) -> np.ndarray:
    """Deduct linear costs from gross period returns."""
    return gross_returns - turnovers * (cost_bps / 1e4)


def cost_sensitivity(
    gross_returns: np.ndarray,
    turnovers: np.ndarray,
    cost_grid: tuple[float, ...],
    periods_per_year: float,
    weights: np.ndarray | None = None,
    per_name_bps: np.ndarray | None = None,
) -> list[dict]:
    """Net performance across a grid of cost assumptions.

    Reporting a grid rather than a single number is the point: it exposes the cost level
    at which the strategy stops working, which is the number that decides whether a
    signal is tradable.
    """
    from market_image_prediction.evaluation.backtest import performance_stats

    rows = []
    for bps in cost_grid:
        if per_name_bps is not None and weights is not None:
            # per_name_bps holds relative multipliers; `bps` sets the median name's rate.
            net = gross_returns - weighted_turnover_cost(weights, per_name_bps * bps)
        else:
            net = apply_costs(gross_returns, turnovers, bps)
        stats = performance_stats(net, periods_per_year)
        stats["cost_bps"] = bps
        stats["mean_turnover"] = float(np.mean(turnovers))
        rows.append(stats)
    return rows


def liquidity_scaled_bps(
    dollar_volume: np.ndarray,
    base_bps: float,
    exponent: float = 0.5,
    floor_mult: float = 0.4,
    cap_mult: float = 6.0,
) -> np.ndarray:
    """Per-name cost in basis points, scaled by relative liquidity.

    Effective spread scales roughly as the inverse square root of volume, so a name
    trading a tenth of the median volume costs about three times as much to trade. The
    scaling is normalised at the cross-sectional median, so `base_bps` keeps its usual
    interpretation for a typical name.

    Bounds are applied because the relationship breaks down in both tails: the very
    largest names do not get arbitrarily cheap, and the smallest are rarely as expensive
    as an unbounded power law implies.
    """
    v = np.asarray(dollar_volume, dtype=float)
    valid = np.isfinite(v) & (v > 0)
    if not valid.any():
        return np.full(v.shape, base_bps)
    median = float(np.median(v[valid]))
    ratio = np.where(valid, median / np.maximum(v, 1e-9), 1.0)
    mult = np.clip(ratio**exponent, floor_mult, cap_mult)
    return base_bps * mult


def weighted_turnover_cost(weights: np.ndarray, cost_bps_per_name: np.ndarray) -> np.ndarray:
    """Per-period cost when each name has its own rate.

    The flat model multiplies aggregate turnover by one rate; this one charges each
    name's traded fraction at its own rate, which is the whole point of modelling
    liquidity. `cost_bps_per_name` is (n_periods, n_assets) so a name's cost can vary
    over time as its volume does.
    """
    prev = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    traded = 0.5 * np.abs(weights - prev)
    return (traded * (cost_bps_per_name / 1e4)).sum(axis=1)
