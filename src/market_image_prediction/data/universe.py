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
from dataclasses import dataclass, field

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
class Universe:
    name: str
    assets: tuple[Asset, ...]
    benchmark: str = BENCHMARK
    context: tuple[str, ...] = field(default_factory=lambda: (VIX, RISK_FREE))
    survivorship_bias_free: bool = False

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
        """Tickers already trading on `date`.

        This is a listing filter only. Sample eligibility additionally requires a full
        feature window of history, which the sample builder enforces.
        """
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


def get_universe(name: str) -> Universe:
    if name not in UNIVERSES:
        raise KeyError(f"unknown universe {name!r}; available: {sorted(UNIVERSES)}")
    return UNIVERSES[name]
