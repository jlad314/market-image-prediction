"""Universe definitions.

The default universe is the SPDR Select Sector ETF suite. It is chosen because it is
*survivorship-bias free*: the suite is a complete partition of the S&P 500 by GICS
sector, so membership is determined by classification structure rather than by
performance, and no constituent has ever closed or been delisted.

The suite is nonetheless **time-varying**. XLRE was carved out of XLF in 2015 and XLC
out of XLK/XLY/XLP in 2018. Those are genuine structural events, not missing data, and
the cross-section must widen at those dates rather than assume a fixed panel.

A secondary mega-cap equity universe is provided for architecture comparison only. It
carries obvious survivorship and selection bias and must never back a headline claim.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

BENCHMARK = "SPY"
VIX = "^VIX"
RISK_FREE = "^IRX"  # 13-week T-bill discount rate, in percent


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    sector: str
    # First date the ETF traded. Used to build a time-varying universe; it is not a
    # substitute for requiring `window` bars of history before a sample is eligible.
    inception: dt.date


SECTOR_ETFS: tuple[Asset, ...] = (
    Asset("XLB", "Materials Select Sector SPDR", "Materials", dt.date(1998, 12, 22)),
    Asset("XLE", "Energy Select Sector SPDR", "Energy", dt.date(1998, 12, 22)),
    Asset("XLF", "Financial Select Sector SPDR", "Financials", dt.date(1998, 12, 22)),
    Asset("XLI", "Industrial Select Sector SPDR", "Industrials", dt.date(1998, 12, 22)),
    Asset(
        "XLK",
        "Technology Select Sector SPDR",
        "Information Technology",
        dt.date(1998, 12, 22),
    ),
    Asset(
        "XLP",
        "Consumer Staples Select Sector SPDR",
        "Consumer Staples",
        dt.date(1998, 12, 22),
    ),
    Asset("XLU", "Utilities Select Sector SPDR", "Utilities", dt.date(1998, 12, 22)),
    Asset("XLV", "Health Care Select Sector SPDR", "Health Care", dt.date(1998, 12, 22)),
    Asset(
        "XLY",
        "Consumer Discretionary Select Sector SPDR",
        "Consumer Discretionary",
        dt.date(1998, 12, 22),
    ),
    Asset("XLRE", "Real Estate Select Sector SPDR", "Real Estate", dt.date(2015, 10, 8)),
    Asset(
        "XLC",
        "Communication Services Select Sector SPDR",
        "Communication Services",
        dt.date(2018, 6, 19),
    ),
)

# Survivorship- and selection-biased. Secondary use only; see module docstring.
MEGACAP_EQUITIES: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "XOM",
    "LLY",
    "UNH",
    "HD",
    "COST",
    "PG",
    "JNJ",
    "V",
    "MA",
    "ABBV",
    "MRK",
    "PEP",
    "KO",
    "CVX",
    "BAC",
    "WMT",
    "DIS",
)


@dataclass(frozen=True)
class MembershipInterval:
    """One contiguous spell of index membership for one security.

    Membership is not a single span: a company can leave the index and rejoin years
    later, so a security is represented by a *list* of intervals rather than one
    inception date.
    """

    identifier: str
    start: dt.date
    end: dt.date


@dataclass(frozen=True)
class Universe:
    name: str
    assets: tuple[Asset, ...]
    benchmark: str = BENCHMARK
    context: tuple[str, ...] = field(default_factory=lambda: (VIX, RISK_FREE))
    survivorship_bias_free: bool = False
    # Populated for data-backed universes (CRSP). When present it, not `assets`,
    # decides who is tradable on a given date.
    membership: tuple[MembershipInterval, ...] = ()
    # Where the panel is loaded from, for universes not sourced from Yahoo.
    source: str = "yahoo"

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(a.ticker for a in self.assets)

    @property
    def download_tickers(self) -> tuple[str, ...]:
        """Every symbol needed from the data source, tradable and contextual alike."""
        return tuple(dict.fromkeys([*self.tickers, self.benchmark, *self.context]))

    def sector_of(self, ticker: str) -> str | None:
        return next((a.sector for a in self.assets if a.ticker == ticker), None)

    def active_on(self, date: dt.date) -> tuple[str, ...]:
        """Securities eligible to be held on `date`.

        For a data-backed universe this is genuine point-in-time index membership, so a
        firm that was a member in 2005 and delisted in 2008 is present for 2005-2008 and
        absent afterwards -- which is exactly what removes survivorship bias.

        This is a membership filter only. Sample eligibility additionally requires a full
        feature window of history, which the sample builder enforces.
        """
        if self.membership:
            return tuple(
                sorted({m.identifier for m in self.membership if m.start <= date <= m.end})
            )
        return tuple(a.ticker for a in self.assets if a.inception <= date)


SECTOR_UNIVERSE = Universe(
    name="sector_etf",
    assets=SECTOR_ETFS,
    survivorship_bias_free=True,
)

MEGACAP_UNIVERSE = Universe(
    name="megacap_equity",
    assets=tuple(Asset(t, t, "unknown", dt.date(1990, 1, 1)) for t in MEGACAP_EQUITIES),
    survivorship_bias_free=False,
)

UNIVERSES: dict[str, Universe] = {
    SECTOR_UNIVERSE.name: SECTOR_UNIVERSE,
    MEGACAP_UNIVERSE.name: MEGACAP_UNIVERSE,
}


CRSP_MEMBERSHIP_PATH = Path("data") / "raw" / "wrds" / "crsp_largecap" / "membership.parquet"
CRSP_TOPN_PATTERN = re.compile(r"^crsp_top(\d+)$")


def crsp_snapshot_dir(universe_name: str) -> Path:
    """Where a CRSP universe's raw snapshot lives.

    `crsp_largecap` is the original top-500 pull. `crsp_topN` names a pull of a different
    width and gets its own directory, so widening the universe never overwrites a
    completed download or silently mixes two membership definitions.
    """
    if universe_name == "crsp_largecap":
        return Path("data") / "raw" / "wrds" / "crsp_largecap"
    return Path("data") / "raw" / "wrds" / universe_name


def get_universe(name: str) -> Universe:
    """Resolve a universe by name.

    `crsp_largecap` is data-backed: its membership is a Parquet file produced by
    `crsp-download`, so it is constructed lazily rather than being a module constant.
    """
    if name == "crsp_largecap" or CRSP_TOPN_PATTERN.match(name):
        return load_crsp_universe(crsp_snapshot_dir(name) / "membership.parquet", name)
    if name not in UNIVERSES:
        available = [*sorted(UNIVERSES), "crsp_largecap", "crsp_top<N>"]
        raise KeyError(f"unknown universe {name!r}; available: {available}")
    return UNIVERSES[name]


def load_crsp_universe(membership_path: Path, name: str = "crsp_largecap") -> Universe:
    """Build a point-in-time large-cap universe from a saved CRSP membership table.

    Membership comes from ranking every eligible US common stock by market cap at each
    month-end and keeping the top N, so it reflects only information available on the
    ranking date. Securities are keyed by PERMNO rather than ticker because tickers are
    reused and reassigned between companies -- a ticker's history is not a firm's history.
    """
    import polars as pl

    if not membership_path.exists():
        raise FileNotFoundError(f"{membership_path} not found; run `crsp-download` first")
    frame = pl.read_parquet(membership_path)
    intervals = tuple(
        MembershipInterval(identifier=str(r["permno"]), start=r["start"], end=r["ending"])
        for r in frame.iter_rows(named=True)
    )
    return Universe(
        name=name,
        assets=(),
        benchmark=BENCHMARK,
        context=(),
        survivorship_bias_free=True,
        membership=intervals,
        source="crsp",
    )
