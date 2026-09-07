"""Typed research configuration.

Every parameter that changes the dataset lives here, so a dataset version is fully
described by `DataConfig.version_hash()`. Nothing that affects features, labels or
execution timing may be hard-coded elsewhere.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo

from market_image_prediction.utils.io import config_hash

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PathsConfig(BaseModel):
    root: Path = PROJECT_ROOT
    raw: Path = PROJECT_ROOT / "data" / "raw"
    canonical: Path = PROJECT_ROOT / "data" / "canonical"
    tensors: Path = PROJECT_ROOT / "data" / "tensors"
    manifests: Path = PROJECT_ROOT / "data" / "manifests"
    predictions: Path = PROJECT_ROOT / "data" / "predictions"
    outputs: Path = PROJECT_ROOT / "outputs"


class DownloadConfig(BaseModel):
    """Raw acquisition settings.

    `auto_adjust=False` is deliberate: Yahoo defaults it to True, which silently
    replaces Close with the adjusted series. Keeping both columns leaves the price
    convention reversible at build time instead of baking it in at download time.
    """

    source: Literal["yahoo"] = "yahoo"
    start: dt.date = dt.date(1998, 12, 22)
    end: dt.date = dt.date(2026, 1, 1)
    auto_adjust: bool = False
    max_retries: int = 4
    retry_backoff_seconds: float = 2.0
    # Yahoo restates history when corporate actions occur, so raw files are immutable
    # snapshots. Re-downloading changes the past; it must be an explicit act.
    allow_overwrite: bool = False


class FeatureConfig(BaseModel):
    window: int = Field(default=60, ge=8, description="trading days of history per sample")
    image_size: int = Field(default=32, ge=8, description="GAF resolution after PAA")
    channels: tuple[str, ...] = ("log_return", "log_volume_change", "realised_vol_20d")
    encodings: tuple[Literal["gasf", "gadf", "recurrence", "mtf"], ...] = (
        "gasf",
        "gadf",
    )
    realised_vol_lookback: int = 20
    # Per-window robust scaling: (x - median) / (1.4826 * MAD), clipped to +/- clip_sigma
    # then mapped to [-1, 1]. Causal by construction: uses only the input window.
    mad_epsilon: float = 1e-8
    clip_sigma: float = 5.0

    @field_validator("image_size")
    @classmethod
    def _size_le_window(cls, v: int, info: ValidationInfo) -> int:
        window = info.data.get("window")
        if window is not None and v > window:
            raise ValueError(f"image_size {v} cannot exceed window {window}")
        return v

    @property
    def n_channels(self) -> int:
        return len(self.channels) * len(self.encodings)


class LabelConfig(BaseModel):
    """Execution convention. Fixed project-wide; see README.

    Signal is formed from data through the close of `t`. The trade is executed at the
    open of `t+1`. The position is held to the close of `t+horizon`. The target is
    therefore log(close[t+horizon] / open[t+1]) and is unobservable at `t`, by design.
    """

    horizon: int = Field(default=5, ge=1, description="trading days held")
    execution_lag: int = Field(default=1, ge=1, description="days from signal to fill")
    entry_price: Literal["open", "close"] = "open"
    exit_price: Literal["open", "close"] = "close"
    price_field: Literal["adj_close", "close"] = "adj_close"
    volatility_scaled: bool = False


class CleaningConfig(BaseModel):
    """Data-quality thresholds for `flag_suspicious_bars`.

    These are *diagnostic* thresholds, not filters: crossing one marks a bar for the
    audit, it never removes it. They live here because a flag that the sample builder
    acts on changes the dataset, and so must be part of `version_hash()`.
    """

    # A move this large in a diversified sector ETF is a bad print, not a market event.
    # Acts as the warm-up rule too, before the trailing scale below has enough history.
    extreme_return_abs: float = Field(default=0.25, gt=0, description="hard |log return| cap")
    # Regime-aware companion: 2008 and 2020 make any fixed cap either deaf or noisy.
    #
    # Calibrated empirically, not guessed. Across all 65,650 sector-ETF bars the robust
    # z tops out at 20.3 (XLE, 2020-03-09) and its whole right tail above 10 is March
    # 2020 -- real crash moves, not bad prints. A trailing-scale rule is structurally a
    # regime-shift detector: 2008's volatility ramped slowly enough that the trailing
    # MAD had already adapted, so -18% days scored unremarkably. 25.0 therefore sits
    # above everything this dataset contains, leaving the flag to fire only on
    # something genuinely unprecedented while `robust_z_return` keeps the regime
    # information as a diagnostic column.
    extreme_return_sigma: float = Field(default=25.0, gt=0, description="robust z cut-off")
    extreme_return_lookback: int = Field(default=60, ge=10, description="bars of trailing scale")
    mad_epsilon: float = 1e-8
    # Consecutive unchanged closes that constitute a stale quote. 3 keeps genuine
    # low-activity days (which do happen around holidays) out of the flag.
    stale_price_run: int = Field(default=3, ge=2)


class SplitConfig(BaseModel):
    scheme: Literal["fixed", "rolling_origin"] = "rolling_origin"
    train_years: int = 6
    validate_years: int = 1
    test_years: int = 1
    step_years: int = 1
    expanding: bool = True
    # Embargo must cover the label horizon; overlapping input windows argue for more.
    embargo_days: int | None = None

    def resolved_embargo(self, labels: LabelConfig, features: FeatureConfig) -> int:
        """Default embargo spans the label horizon *and* the input window overlap."""
        if self.embargo_days is not None:
            return self.embargo_days
        return labels.horizon + labels.execution_lag + features.window


class BacktestConfig(BaseModel):
    rebalance: Literal["weekly", "monthly"] = "weekly"
    long_quantile: float = Field(default=0.25, gt=0, lt=0.5)
    short_quantile: float = Field(default=0.25, gt=0, lt=0.5)
    min_names_per_leg: int = 2
    weighting: Literal["equal"] = "equal"
    sector_neutral: bool = False
    cost_bps_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 50.0)
    baseline_cost_bps: float = 10.0


class Config(BaseModel):
    universe: str = "sector_etf"
    seed: int = 17
    paths: PathsConfig = PathsConfig()
    download: DownloadConfig = DownloadConfig()
    features: FeatureConfig = FeatureConfig()
    labels: LabelConfig = LabelConfig()
    cleaning: CleaningConfig = CleaningConfig()
    splits: SplitConfig = SplitConfig()
    backtest: BacktestConfig = BacktestConfig()

    @model_validator(mode="after")
    def _check_dates(self) -> Config:
        if self.download.start >= self.download.end:
            raise ValueError("download.start must precede download.end")
        return self

    def version_hash(self) -> str:
        """Identity of the *dataset*, ignoring downstream modelling choices.

        Paths and backtest settings are excluded: moving the output directory or
        re-costing a backtest must not invalidate cached tensors.
        """
        payload = self.model_dump(mode="json", exclude={"paths", "backtest", "seed"})
        return config_hash(payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))
        return p
