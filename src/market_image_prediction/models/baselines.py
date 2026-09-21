"""Baseline models, in ascending order of capacity.

The brief's discipline: a GAF CNN is only interesting if it beats these on identical
folds, targets and information. Each model exposes the same `fit`/`predict` interface so
the walk-forward runner treats them uniformly.

Every learned object -- scalers included -- is fitted inside `fit`, which the runner
calls on training data only. Nothing is fitted at module level or across folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_image_prediction.models.base import FoldData, Model
from market_image_prediction.models.tabular import tabular_features


@dataclass
class NaiveSignal:
    """Model A: a fixed formula, nothing learned.

    `fit` is a genuine no-op -- that is the point. These establish the return available
    from well-known effects before any estimation, so a learned model that fails to beat
    them has demonstrated nothing.
    """

    signal: str = "momentum_20"
    name: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.name = f"naive_{self.signal}"

    def fit(self, data: FoldData) -> None:
        return None

    def predict(self, data: FoldData) -> np.ndarray:
        r = data.seq_test[:, 0, :].astype(np.float64)
        if self.signal == "momentum_20":
            return r[:, -20:].sum(axis=1)
        if self.signal == "momentum_60":
            return r.sum(axis=1)
        if self.signal == "reversal_5":
            # Short-horizon reversal: recent winners underperform, hence the sign.
            return -r[:, -5:].sum(axis=1)
        if self.signal == "vol_scaled_momentum":
            mom = r[:, -20:].sum(axis=1)
            vol = r[:, -20:].std(axis=1, ddof=1)
            return mom / np.maximum(vol, 1e-8)
        raise KeyError(f"unknown naive signal {self.signal!r}")


@dataclass
class RidgeTabular:
    """Model B: linear regression on window summary statistics.

    RidgeCV picks the penalty by internal cross-validation *within the training block*,
    so no validation or test observation influences the choice.
    """

    name: str = "ridge_tabular"
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            [("scale", StandardScaler()), ("model", RidgeCV(alphas=self.alphas))]
        )

    def fit(self, data: FoldData) -> None:
        x, _ = tabular_features(data.seq_train)
        self.pipeline.fit(x, data.y_train)

    def predict(self, data: FoldData) -> np.ndarray:
        x, _ = tabular_features(data.seq_test)
        return np.asarray(self.pipeline.predict(x))


@dataclass
class LogisticTabular:
    """Model B': direction classification. Emits P(up) - 0.5 so it ranks like the others."""

    name: str = "logistic_tabular"
    C: float = 1.0

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=self.C, max_iter=2000)),
            ]
        )

    def fit(self, data: FoldData) -> None:
        x, _ = tabular_features(data.seq_train)
        self.pipeline.fit(x, (data.y_train > 0).astype(int))

    def predict(self, data: FoldData) -> np.ndarray:
        x, _ = tabular_features(data.seq_test)
        return np.asarray(self.pipeline.predict_proba(x))[:, 1] - 0.5


@dataclass
class GBMTabular:
    """Model C: gradient-boosted trees on the same summary statistics.

    Uses scikit-learn's histogram-based implementation rather than LightGBM. LightGBM's
    macOS wheel links a second OpenMP runtime alongside the one PyTorch bundles, and
    importing both in either order can segfault the interpreter -- an accident waiting
    for the first refactor that reorders imports. `HistGradientBoostingRegressor` is the
    same algorithm family (histogram binning, leaf-wise growth) with no native
    dependency outside scikit-learn, which keeps the environment reproducible on any
    machine.

    Deliberately small and heavily regularised. The effective sample is a few thousand
    independent dates, not 60k overlapping rows, so a large ensemble would fit the
    overlap rather than the signal.
    """

    name: str = "gbm_tabular"
    max_iter: int = 300
    learning_rate: float = 0.03
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 200
    l2_regularization: float = 1.0
    seed: int = 17

    def __post_init__(self) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,  # type: ignore[arg-type]  stub types this as str
            random_state=self.seed,
        )

    def fit(self, data: FoldData) -> None:
        x, _ = tabular_features(data.seq_train)
        self.model.fit(x, data.y_train)

    def predict(self, data: FoldData) -> np.ndarray:
        x, _ = tabular_features(data.seq_test)
        return np.asarray(self.model.predict(x))


def default_baselines() -> list[Model]:
    return [
        NaiveSignal("momentum_20"),
        NaiveSignal("momentum_60"),
        NaiveSignal("reversal_5"),
        NaiveSignal("vol_scaled_momentum"),
        RidgeTabular(),
        LogisticTabular(),
        GBMTabular(),
    ]


@dataclass
class SurfaceRidge:
    """Ridge on the flattened surface.

    Separates two questions the CNN alone cannot: does the *surface* carry cross-sectional
    information, and can a convolution extract more of it than a linear read of the same
    cells? If this scores and the CNN does not, the limit is the architecture (or the
    sample size); if neither scores, the surface itself is uninformative at this horizon.

    Each cell is standardised within its own column across the training fold, because raw
    implied volatilities differ by an order of magnitude between a calm mega-cap and a
    distressed name -- without it the fit is dominated by level rather than shape.
    """

    name: str = "surface_ridge"
    alpha: float = 100.0
    pipeline: Pipeline = field(init=False)

    def __post_init__(self) -> None:
        self.pipeline = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=self.alpha))])

    @staticmethod
    def _flat(x) -> np.ndarray:
        arr = np.asarray(x[: len(x)]) if not isinstance(x, np.ndarray) else x
        return arr.reshape(len(arr), -1).astype(np.float64)

    def fit(self, data: FoldData) -> None:
        self.pipeline.fit(self._flat(data.gaf_train), data.y_train)

    def predict(self, data: FoldData) -> np.ndarray:
        return np.asarray(self.pipeline.predict(self._flat(data.gaf_test)))
