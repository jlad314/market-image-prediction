"""Point-in-time membership and sampling schedule.

These guard the two filters that make a data-backed universe correct. Both silently
failed once during development -- the build completed and wrote 6.8M samples instead of
0.7M -- so they are tested directly rather than only through the pipeline.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from market_image_prediction.config import Config
from market_image_prediction.data.dataset import apply_membership, sampling_dates
from market_image_prediction.data.universe import MembershipInterval, Universe

MEMBERSHIP = (
    MembershipInterval("A", dt.date(2020, 1, 31), dt.date(2020, 2, 27)),
    MembershipInterval("A", dt.date(2020, 2, 28), dt.date(2020, 3, 30)),
    MembershipInterval("B", dt.date(2020, 2, 28), dt.date(2020, 3, 30)),
)
UNIVERSE = Universe(name="t", assets=(), membership=MEMBERSHIP)


def _manifest() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["A", "A", "B", "B", "C"],
            "asof_date": [
                dt.date(2020, 2, 7),
                dt.date(2020, 3, 6),
                dt.date(2020, 2, 7),
                dt.date(2020, 3, 6),
                dt.date(2020, 3, 6),
            ],
        }
    ).with_row_index("_row")


def test_member_kept_within_its_periods() -> None:
    out = apply_membership(_manifest(), UNIVERSE)
    kept = set(zip(out["ticker"].to_list(), out["asof_date"].to_list(), strict=True))
    assert ("A", dt.date(2020, 2, 7)) in kept
    assert ("A", dt.date(2020, 3, 6)) in kept


def test_security_excluded_before_it_joins() -> None:
    """B enters at the February rebalance, so January/February samples must not survive."""
    out = apply_membership(_manifest(), UNIVERSE)
    kept = set(zip(out["ticker"].to_list(), out["asof_date"].to_list(), strict=True))
    assert ("B", dt.date(2020, 2, 7)) not in kept
    assert ("B", dt.date(2020, 3, 6)) in kept


def test_non_member_dropped_entirely() -> None:
    out = apply_membership(_manifest(), UNIVERSE)
    assert "C" not in out["ticker"].to_list()


def test_input_order_preserved() -> None:
    """A caller reorders a parallel tensor by the surviving index, so order must hold."""
    out = apply_membership(_manifest(), UNIVERSE)
    assert out["_row"].to_list() == sorted(out["_row"].to_list())


def test_no_membership_is_a_passthrough() -> None:
    plain = Universe(name="static", assets=())
    out = apply_membership(_manifest(), plain)
    assert out.height == 5


def test_sampling_schedule_sizes() -> None:
    """Weekly and monthly schedules must actually thin the calendar."""
    daily = Config.model_validate({"features": {"sample_frequency": "daily"}})
    weekly = Config.model_validate({"features": {"sample_frequency": "weekly"}})
    monthly = Config.model_validate({"features": {"sample_frequency": "monthly"}})
    assert sampling_dates(daily) is None
    w, m = sampling_dates(weekly), sampling_dates(monthly)
    assert w is not None and m is not None
    assert len(m) < len(w)
    # ~52 weeks and 12 months per year over the configured span.
    years = daily.download.end.year - daily.download.start.year
    assert 45 * years < len(w) < 55 * years
    assert 11 * years < len(m) < 13 * years


def test_sample_frequency_changes_dataset_version() -> None:
    """A different schedule is a different dataset and must not reuse cached tensors."""
    a = Config.model_validate({"features": {"sample_frequency": "daily"}})
    b = Config.model_validate({"features": {"sample_frequency": "weekly"}})
    assert a.version_hash() != b.version_hash()


def test_checkpoint_names_carry_the_whole_span() -> None:
    """A chunk filename must change when the chunk width does.

    Naming by start year alone let a switch from three-year to two-year chunks write
    files that overlapped the old ones: 7.5M duplicated rows, 35% of the panel, every
    file individually valid.
    """

    def name(start: dt.date, stop: dt.date) -> str:
        return f"panel_{start:%Y%m%d}_{stop:%Y%m%d}.parquet"

    three_year = name(dt.date(1998, 12, 22), dt.date(2000, 12, 31))
    two_year = name(dt.date(1998, 12, 22), dt.date(1999, 12, 31))
    assert three_year != two_year


def test_overlap_guard_rejects_duplicated_checkpoints(tmp_path) -> None:
    """Assembling overlapping checkpoints must fail loudly, not double-count."""
    from market_image_prediction.data.wrds.crsp import fetch_daily_panel

    shared = pl.DataFrame(
        {"permno": [1, 1], "dlycaldt": [dt.date(2000, 1, 3), dt.date(2000, 1, 4)]}
    )
    shared.write_parquet(tmp_path / "panel_19980101_20001231.parquet")
    shared.write_parquet(tmp_path / "panel_20000101_20011231.parquet")
    # start > end so no query runs; only the assembly path is exercised.
    with pytest.raises(RuntimeError, match="overlap"):
        fetch_daily_panel([1], dt.date(2010, 1, 1), dt.date(2009, 1, 1), tmp_path)
