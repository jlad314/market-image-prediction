"""Tabular summary features over the same windows the image models receive.

The point of these baselines is a *fair* comparison: they must be given the identical
information set, differing only in representation. So every column here is a statistic
of `X_seq` -- the exact raw windows the GAF encoder consumes -- and nothing is read
from the panel separately.

All statistics are computed within a window, so they inherit the window's causality:
no fitting, no cross-sample state, nothing to leak.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# Index of each feature channel within X_seq, mirroring FeatureConfig.channels order.
RET, VOL_CHG, RVOL = 0, 1, 2


def _max_drawdown(cum: np.ndarray) -> np.ndarray:
    """Worst peak-to-trough of the cumulative log-return path, per sample."""
    running_max = np.maximum.accumulate(cum, axis=1)
    return (cum - running_max).min(axis=1)


def tabular_features(seq: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Summarise (n_samples, n_features, window) windows into a design matrix.

    Returns (X, names) with X of shape (n_samples, n_columns), float64.
    """
    r = seq[:, RET, :].astype(np.float64)
    dv = seq[:, VOL_CHG, :].astype(np.float64)
    rv = seq[:, RVOL, :].astype(np.float64)
    window = r.shape[1]
    cum = np.cumsum(r, axis=1)

    cols: dict[str, np.ndarray] = {
        # Distributional shape of the return window.
        "ret_mean": r.mean(axis=1),
        "ret_std": r.std(axis=1, ddof=1),
        "ret_min": r.min(axis=1),
        "ret_max": r.max(axis=1),
        "ret_skew": stats.skew(r, axis=1),
        "ret_kurt": stats.kurtosis(r, axis=1),
        # Momentum at several horizons: the classic cross-sectional equity predictors.
        "cumret_5": r[:, -5:].sum(axis=1),
        "cumret_20": r[:, -20:].sum(axis=1),
        "cumret_60": r[:, -min(60, window) :].sum(axis=1),
        # Realised volatility at matching horizons.
        "rvol_5": r[:, -5:].std(axis=1, ddof=1),
        "rvol_20": r[:, -20:].std(axis=1, ddof=1),
        "rvol_60": r[:, -min(60, window) :].std(axis=1, ddof=1),
        # Path shape, not just endpoints.
        "max_drawdown": _max_drawdown(cum),
        "range": cum.max(axis=1) - cum.min(axis=1),
        # Volume and volatility dynamics.
        "dvol_mean": dv.mean(axis=1),
        "dvol_std": dv.std(axis=1, ddof=1),
        "rvol_last": rv[:, -1],
        "rvol_mean": rv.mean(axis=1),
    }
    # Volatility-scaled momentum: the single most robust of the naive signals, so it is
    # given to the linear models explicitly rather than hoping they reconstruct a ratio.
    cols["mom20_scaled"] = cols["cumret_20"] / np.maximum(cols["rvol_20"], 1e-8)
    cols["rvol_ratio"] = cols["rvol_5"] / np.maximum(cols["rvol_20"], 1e-8)

    names = list(cols)
    x = np.column_stack([cols[n] for n in names])
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), names
