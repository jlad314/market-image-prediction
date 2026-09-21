"""Risk attribution against standard equity factors.

A long-short book can post a respectable Sharpe while owning nothing but well-known
factor exposures. Shorting small, unprofitable, high-volatility names has paid for
decades; a model that rediscovers that is not producing alpha, it is producing a
repackaged risk premium with a crash tail.

The regression is

    r_t = alpha + sum_k beta_k * f_kt + e_t

so `alpha` is the part of the return the factors cannot explain. Standard errors are
Newey-West, because monthly strategy returns are autocorrelated and overlapping label
windows make an ordinary t-statistic anticonservative.

Factors come from Ken French's data library -- free, no registration, and the reference
source that published results are benchmarked against.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import polars as pl
import statsmodels.api as sm

from market_image_prediction.utils.io import ensure_dir, write_parquet
from market_image_prediction.utils.logging import get_logger

log = get_logger(__name__)

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FIVE_FACTOR = "F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOMENTUM = "F-F_Momentum_Factor_CSV.zip"


def _parse_french_csv(raw: str, n_cols: int) -> pl.DataFrame:
    """Extract the monthly block from a Ken French CSV.

    The files carry a prose header, then monthly rows keyed `YYYYMM`, then an annual
    block keyed `YYYY`. Selecting on a six-digit key takes the monthly section and stops
    cleanly at the annual one without relying on line offsets that change between
    vintages.
    """
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != n_cols + 1:
            continue
        key = parts[0]
        if not (key.isdigit() and len(key) == 6):
            continue
        try:
            values = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        rows.append([int(key), *values])
    if not rows:
        raise RuntimeError("no monthly rows parsed from the French file")
    return pl.DataFrame(
        {
            "yyyymm": [r[0] for r in rows],
            **{f"c{i}": [r[i + 1] for r in rows] for i in range(n_cols)},
        }
    )


def download_factors(cache_dir: Path) -> pl.DataFrame:
    """Fetch and cache the five factors plus momentum, as monthly decimals."""
    ensure_dir(cache_dir)
    path = cache_dir / "ff_factors.parquet"
    if path.exists():
        return pl.read_parquet(path)

    def fetch(name: str, n_cols: int) -> pl.DataFrame:
        log.info("downloading %s", name)
        with urlopen(FRENCH_BASE + name, timeout=60) as resp:  # noqa: S310 - fixed host
            blob = resp.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            raw = zf.read(zf.namelist()[0]).decode("latin-1")
        return _parse_french_csv(raw, n_cols)

    five = fetch(FIVE_FACTOR, 6).rename(
        {"c0": "mkt_rf", "c1": "smb", "c2": "hml", "c3": "rmw", "c4": "cma", "c5": "rf"}
    )
    mom = fetch(MOMENTUM, 1).rename({"c0": "mom"})
    # French reports percent; the strategy returns are decimals.
    factors = five.join(mom, on="yyyymm", how="inner").with_columns(
        [pl.col(c) / 100.0 for c in ("mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom")]
    )
    write_parquet(factors, path)
    log.info("cached %d monthly factor rows to %s", factors.height, path)
    return factors


def attribute(
    returns: np.ndarray,
    dates: list,
    factors: pl.DataFrame,
    lags: int = 3,
) -> dict:
    """Regress strategy returns on the factors and report alpha with HAC errors.

    Returns annualised alpha, its t-statistic, each factor loading, and the R-squared --
    the last being the share of the strategy that is simply known risk premia.
    """
    key = np.array([d.year * 100 + d.month for d in dates])
    fmap = {int(k): i for i, k in enumerate(factors["yyyymm"].to_list())}
    keep = np.array([k in fmap for k in key])
    if keep.sum() < 24:
        raise RuntimeError("too few overlapping months to attribute")
    idx = [fmap[k] for k in key[keep]]
    names = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    x = np.column_stack([factors[n].to_numpy()[idx] for n in names])
    y = returns[keep]

    model = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {
        "n_months": int(keep.sum()),
        "alpha_monthly": float(model.params[0]),
        "alpha_annual": float(model.params[0]) * 12,
        "alpha_t": float(model.tvalues[0]),
        "r_squared": float(model.rsquared),
        "loadings": {
            n: {"beta": float(model.params[i + 1]), "t": float(model.tvalues[i + 1])}
            for i, n in enumerate(names)
        },
    }
    # How much of the raw mean return survives the factor exposures?
    out["explained_share"] = (
        1.0 - out["alpha_monthly"] / float(np.mean(y)) if np.mean(y) != 0 else float("nan")
    )
    return out
