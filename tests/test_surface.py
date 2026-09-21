"""IV-surface pivot, completeness and alignment.

The pivot turns long vendor rows into a delta x maturity image. Two things must hold or
every downstream number is meaningless: a cell must land at the grid position its
(delta, days) says it does, and a surface must not be emitted unless the grid is whole.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from market_image_prediction.config import Config
from market_image_prediction.data.surface import (
    SURFACE_CHANNELS,
    link_secid_to_permno,
    resolve_grid,
    surfaces_to_tensor,
)

DELTAS = [-50, -25, 25, 50]
DAYS = [30, 91, 365]
DATE_A, DATE_B = dt.date(2015, 6, 30), dt.date(2015, 7, 31)


def _surface_rows(secid: int, date: dt.date, base: float, drop: int = 0) -> list[dict]:
    """A full grid for one security-date, optionally missing `drop` cells."""
    rows = []
    for d in DELTAS:
        for m in DAYS:
            rows.append(
                {
                    "secid": secid,
                    "date": date,
                    "delta": d,
                    "days": m,
                    # Encodes its own coordinates so a misplaced cell is detectable.
                    "impl_volatility": base + d / 1000.0 + m / 10000.0,
                    "dispersion": 0.01,
                }
            )
    return rows[: len(rows) - drop] if drop else rows


@pytest.fixture
def surfaces() -> pl.DataFrame:
    rows = (
        _surface_rows(1, DATE_A, 0.20)
        + _surface_rows(1, DATE_B, 0.30)
        + _surface_rows(2, DATE_A, 0.40)
        + _surface_rows(2, DATE_B, 0.50, drop=2)  # incomplete on purpose
    )
    return pl.DataFrame(rows)


@pytest.fixture
def keys() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "date": [DATE_A, DATE_B, DATE_A, DATE_B],
            "secid": [1, 1, 2, 2],
        }
    )


def test_grid_axes_are_sorted(surfaces: pl.DataFrame) -> None:
    days, deltas = resolve_grid(surfaces)
    assert days == sorted(DAYS)
    assert deltas == sorted(DELTAS)


def test_incomplete_surface_is_dropped(surfaces: pl.DataFrame, keys: pl.DataFrame) -> None:
    """A partial grid must be excluded, not zero-filled.

    Zero is a legitimate volatility reading to a network; inventing one would teach it
    that missing data means calm.
    """
    _, manifest, meta = surfaces_to_tensor(surfaces, keys, Config())
    pairs = set(zip(manifest["ticker"].to_list(), manifest["date"].to_list(), strict=True))
    assert ("BBB", DATE_B) not in pairs
    assert len(pairs) == 3
    assert meta["dropped_incomplete"] == 1


def test_tensor_shape_matches_grid(surfaces: pl.DataFrame, keys: pl.DataFrame) -> None:
    tensor, manifest, meta = surfaces_to_tensor(surfaces, keys, Config())
    assert tensor.shape == (manifest.height, len(SURFACE_CHANNELS), len(DELTAS), len(DAYS))
    assert tensor.dtype == np.float32
    assert meta["grid_deltas"] == sorted(DELTAS)
    assert meta["grid_days"] == sorted(DAYS)


def test_cells_land_at_their_grid_coordinates(surfaces: pl.DataFrame, keys: pl.DataFrame) -> None:
    """Each cell carries its own (delta, days) in its value, so a transpose is visible."""
    tensor, manifest, meta = surfaces_to_tensor(surfaces, keys, Config())
    deltas, days = meta["grid_deltas"], meta["grid_days"]
    row = next(
        i
        for i, (t, d) in enumerate(
            zip(manifest["ticker"].to_list(), manifest["date"].to_list(), strict=True)
        )
        if (t, d) == ("AAA", DATE_A)
    )
    for di, delta in enumerate(deltas):
        for mi, maturity in enumerate(days):
            expected = 0.20 + delta / 1000.0 + maturity / 10000.0
            assert tensor[row, 0, di, mi] == pytest.approx(expected, abs=1e-6)


def test_manifest_order_matches_tensor(surfaces: pl.DataFrame, keys: pl.DataFrame) -> None:
    """Row i of the manifest must describe row i of the tensor."""
    tensor, manifest, _ = surfaces_to_tensor(surfaces, keys, Config())
    assert tensor.shape[0] == manifest.height
    dates = manifest["date"].to_list()
    assert dates == sorted(dates)


def test_link_uses_the_effective_cusip_at_the_date() -> None:
    """secnmd is a history table; a stale CUSIP must not attribute another firm's surface."""
    link = pl.DataFrame(
        {
            "secid": [10, 11],
            "cusip": ["AAAA1111", "AAAA1111"],
            "effect_date": [dt.date(2000, 1, 1), dt.date(2015, 1, 1)],
        }
    )
    panel = pl.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": [dt.date(2010, 6, 30), dt.date(2020, 6, 30)],
            "cusip": ["AAAA1111", "AAAA1111"],
        }
    )
    out = link_secid_to_permno(link, panel).sort("date")
    assert out["secid"].to_list() == [10, 11]


def test_link_excludes_dates_before_any_effective_row() -> None:
    link = pl.DataFrame(
        {"secid": [10], "cusip": ["AAAA1111"], "effect_date": [dt.date(2015, 1, 1)]}
    )
    panel = pl.DataFrame({"ticker": ["AAA"], "date": [dt.date(2010, 6, 30)], "cusip": ["AAAA1111"]})
    assert link_secid_to_permno(link, panel).is_empty()


def test_null_values_do_not_count_as_observed(keys: pl.DataFrame) -> None:
    """A row can exist with a null value; completeness must count values, not rows.

    The vendor emits a row for every grid point and leaves the value null when it could
    not fit one, so a row-count check passes surfaces that are full of holes.
    """
    rows = _surface_rows(1, DATE_A, 0.20)
    for r in rows[:3]:
        r["impl_volatility"] = None
    surfaces = pl.DataFrame(rows + _surface_rows(2, DATE_A, 0.40))
    keys_a = keys.filter(pl.col("date") == DATE_A)
    _, manifest, meta = surfaces_to_tensor(surfaces, keys_a, Config())
    assert manifest["ticker"].to_list() == ["BBB"]
    assert meta["dropped_incomplete"] == 1


def test_sparse_maturity_is_dropped_not_zero_filled() -> None:
    """A tenor that is mostly missing is removed from the grid entirely.

    Otherwise the choice is between discarding most surfaces and inventing a whole row.
    """
    from market_image_prediction.data.surface import drop_sparse_maturities

    rows = []
    for date in (DATE_A, DATE_B):
        for d in DELTAS:
            for m in [10, *DAYS]:
                # The 10-day tenor is observed only on the first date, mirroring the
                # real grid where weekly options are absent for most of the sample.
                value = None if (m == 10 and date == DATE_B) else 0.2
                rows.append(
                    {
                        "secid": 1,
                        "date": date,
                        "delta": d,
                        "days": m,
                        "impl_volatility": value,
                        "dispersion": 0.01,
                    }
                )
    out = drop_sparse_maturities(pl.DataFrame(rows), min_coverage=0.6)
    assert sorted(out["days"].unique().to_list()) == sorted(DAYS)


def test_dense_maturities_survive_the_filter() -> None:
    from market_image_prediction.data.surface import drop_sparse_maturities

    surfaces = pl.DataFrame(_surface_rows(1, DATE_A, 0.2) + _surface_rows(1, DATE_B, 0.3))
    out = drop_sparse_maturities(surfaces)
    assert sorted(out["days"].unique().to_list()) == sorted(DAYS)


def test_demean_removes_level_and_keeps_shape() -> None:
    """Two surfaces with the same geometry at different levels must become identical.

    This is the whole point: the Phase 2 CNN ranked on volatility level (rho -0.40) and
    ignored skew (-0.02), because level is the dominant direction of variance.
    """
    from market_image_prediction.data.surface import normalise_surfaces

    rng = np.random.default_rng(0)
    shape = rng.normal(size=(34, 10)) * 0.02
    disp = np.full((34, 10), 0.01)
    t = np.stack([np.stack([shape + 0.20, disp]), np.stack([shape + 0.60, disp])]).astype(
        np.float32
    )
    out = normalise_surfaces(t, "demean")
    assert np.allclose(out[0, 0], out[1, 0], atol=1e-5)
    assert not np.allclose(t[0, 0], t[1, 0])


def test_normalisation_leaves_dispersion_alone() -> None:
    """Dispersion is a fit-quality measure; its absolute level carries the information."""
    from market_image_prediction.data.surface import normalise_surfaces

    rng = np.random.default_rng(1)
    t = rng.normal(size=(4, 2, 34, 10)).astype(np.float32)
    for scheme in ("demean", "standardise"):
        assert np.allclose(normalise_surfaces(t, scheme)[:, 1], t[:, 1])


def test_raw_normalisation_is_identity() -> None:
    from market_image_prediction.data.surface import normalise_surfaces

    rng = np.random.default_rng(2)
    t = rng.normal(size=(3, 2, 34, 10)).astype(np.float32)
    assert normalise_surfaces(t, "raw") is t


def test_normalisation_is_per_sample_not_across_the_panel() -> None:
    """Causality: a sample's scaling must depend only on that sample."""
    from market_image_prediction.data.surface import normalise_surfaces

    rng = np.random.default_rng(3)
    t = rng.normal(size=(6, 2, 34, 10)).astype(np.float32)
    full = normalise_surfaces(t, "standardise")
    subset = normalise_surfaces(t[2:4], "standardise")
    assert np.allclose(full[2:4], subset, atol=1e-6)


def test_normalisation_changes_the_dataset_version() -> None:
    a = Config.model_validate({"features": {"surface_normalisation": "raw"}})
    b = Config.model_validate({"features": {"surface_normalisation": "demean"}})
    assert a.version_hash() != b.version_hash()
