"""Fold construction, purging and embargo."""

from __future__ import annotations

import datetime as dt
from itertools import pairwise

import polars as pl

from market_image_prediction.config import Config
from market_image_prediction.data.splits import assign_fold, make_folds

FIRST, LAST = dt.date(1998, 12, 22), dt.date(2025, 12, 31)


def test_folds_move_strictly_forward(cfg: Config) -> None:
    for f in make_folds(cfg, FIRST, LAST):
        assert f.train_start <= f.train_end < f.val_start <= f.val_end < f.test_start
        assert f.test_start <= f.test_end


def test_test_periods_do_not_overlap(cfg: Config) -> None:
    folds = make_folds(cfg, FIRST, LAST)
    for a, b in pairwise(folds):
        assert a.test_end < b.test_start


def test_expanding_train_start_is_fixed(cfg: Config) -> None:
    folds = make_folds(cfg, FIRST, LAST)
    assert len({f.train_start for f in folds}) == 1
    assert folds[0].train_end < folds[-1].train_end


def test_rolling_train_window_has_constant_width() -> None:
    cfg = Config.model_validate({"splits": {"expanding": False}})
    folds = make_folds(cfg, FIRST, LAST)
    widths = {(f.train_end - f.train_start).days // 365 for f in folds[1:]}
    assert len(widths) == 1


def test_no_train_label_resolves_inside_validation(built, cfg: Config) -> None:
    """The defining property of purging."""
    _, m = built
    for fold in make_folds(cfg, FIRST, LAST)[:5]:
        tagged = assign_fold(m, fold, cfg)
        train = tagged.filter(pl.col("split") == "train")
        assert train.select((pl.col("target_end_date") < fold.val_start).all()).item()


def test_no_validation_label_resolves_inside_test(built, cfg: Config) -> None:
    _, m = built
    for fold in make_folds(cfg, FIRST, LAST)[:5]:
        tagged = assign_fold(m, fold, cfg)
        val = tagged.filter(pl.col("split") == "val")
        assert val.select((pl.col("target_end_date") < fold.test_start).all()).item()


def test_embargo_creates_a_gap_at_every_boundary(built, cfg: Config) -> None:
    """Kept training samples must stop short of the validation start by the embargo."""
    _, m = built
    embargo = cfg.splits.resolved_embargo(cfg.labels, cfg.features)
    for fold in make_folds(cfg, FIRST, LAST)[:5]:
        tagged = assign_fold(m, fold, cfg)
        last_train = tagged.filter(pl.col("split") == "train")["asof_date"].max()
        assert isinstance(last_train, dt.date)
        assert (fold.val_start - last_train).days >= 1
        assert last_train <= fold.val_start - dt.timedelta(days=embargo)


def test_splits_are_disjoint(built, cfg: Config) -> None:
    _, m = built
    for fold in make_folds(cfg, FIRST, LAST)[:3]:
        tagged = assign_fold(m, fold, cfg)
        counts = tagged.group_by("split").agg(pl.len().alias("n"))
        assert counts["n"].sum() == m.height
        for a, b in (("train", "val"), ("val", "test"), ("train", "test")):
            da = set(tagged.filter(pl.col("split") == a)["asof_date"].to_list())
            db = set(tagged.filter(pl.col("split") == b)["asof_date"].to_list())
            assert not (da & db)


def test_trading_days_returns_a_date_series() -> None:
    """Regression guard: the DatetimeIndex `.date` accessor yields Object, not Date."""
    from market_image_prediction.data.cleaning import trading_days

    s = trading_days(dt.date(2020, 1, 1), dt.date(2020, 3, 1))
    assert s.dtype == pl.Date
    assert s.len() > 0
    # 2020-01-01 was a holiday; 2020-01-02 was the first session.
    assert s[0] == dt.date(2020, 1, 2)
