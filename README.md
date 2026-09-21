# Market Image Prediction

Can financial market state be encoded as an image and predicted with convolutional
networks, and does the result survive realistic validation and trading costs?

Two representations were tested against non-image baselines on identical data:

1. **Gramian Angular Fields** — a 60-day price/volume/volatility window folded into a
   32×32 image.
2. **Implied volatility surfaces** — the option market's price of risk across moneyness
   and maturity, which is natively two-dimensional.

The first failed. The second works, but only after removing the volatility level from
each surface and widening the universe fourfold.

---

## Headline result

Out-of-fold, 2006–2025, 235 monthly decision dates, ~1,856 US equities per date.
20 rolling-origin folds with purging and embargo. Signal-weighted quartile long/short.

| model | input | rank IC | HAC t | net Sharpe @10bps | @50bps | annual alpha | alpha t |
|---|---|---|---|---|---|---|---|
| **surface CNN** | IV surface | +0.0244 | 3.70 | **1.40** | **1.02** | **+10.4%** | **6.29** |
| surface ridge | IV surface | +0.0147 | 3.35 | 1.25 | 0.55 | +6.0% | 5.52 |
| gradient boosting | price | +0.0252 | 4.26 | 0.98 | 0.51 | +7.4% | 4.83 |
| ridge | price | +0.0324 | 3.40 | 0.90 | 0.63 | +8.6% | 4.19 |
| 5-day reversal | price | +0.0071 | 0.98 | −0.06 | −0.59 | +0.1% | 0.05 |

Alpha is the intercept from a regression on the Fama–French five factors plus momentum,
with Newey–West standard errors. R² is 0.097 — the factors explain almost none of it.

Full table: [`results/main_results.csv`](results/main_results.csv).

---

## What the data is

| | |
|---|---|
| Prices | CRSP daily (CIZ schema), via WRDS |
| Options | OptionMetrics standardised volatility surfaces, via WRDS |
| Universe | top 2,000 US common stocks by market cap, re-ranked monthly |
| Period | 1998-12 to 2025-12 |
| Samples | 558,065 (security × decision date) |

**The universe is point-in-time.** It is rebuilt at each month-end from market
capitalisation as of that date, so a company enters when it grows large enough and leaves
when it shrinks, is acquired, or fails. Lehman Brothers is present at $39.5bn in June
2007 and $10.9bn in June 2008, then stops — 490 training samples ending ten days before
the bankruptcy filing.

CRSP Indexes is not in the subscription used here, and Compustat's US S&P constituent
tables turn out to be *current-membership snapshots* — 500 firms, zero closed spells.
Using them would have introduced a worse survivorship bias than the problem they were
meant to solve. Building the universe from market cap avoids this and has no
index-committee selection effect.

---

## Method

### Labels and execution

Signal is formed from data through the close of `t`. The trade fills at the open of
`t+1` and is held to the close of `t+21`, so the target is

```
y = log( close(t+21) / open(t+1) )
```

The label is therefore unobservable at the decision date by construction, and the close
that generated the signal is never the price that fills the trade.

Yahoo-style data has no adjusted open, so the open is rescaled by `adj_close / close`.
Pairing a raw open with an adjusted close books every dividend inside the holding window
as return — worth 34% of cumulative return on a 27-year AAPL series.

### Validation

Random splits are invalid here for two compounding reasons: adjacent samples share 59 of
60 input days, and a label spans 21 days *after* its decision date.

Splitting by date fixes the first, not the second. So at every fold boundary:

* **purge** — drop training samples whose label resolves inside the next block;
* **embargo** — drop samples within 82 days of the boundary, whose input window still
  overlaps it.

Both are applied at every boundary of every fold. 1.33% of fold-sample pairs are removed.

### Metrics

**Rank IC** is the Spearman correlation between predicted and realised ranks, computed
*within* a single date and averaged across dates. One number per date, 235 dates.

A rank IC of 0.02 is a normal effect size in equity cross-section. What turns it into
money is breadth: the same weak edge applied across ~1,850 names, 12 times a year.

**t-statistics are Newey–West.** With a 21-day label and monthly decisions the IC series
is autocorrelated by construction; an ordinary standard error treats correlated
observations as independent and inflates t by roughly √(overlap). Every t on this page
is HAC-corrected, and `n_effective` is reported alongside `n_dates`.

**The annualisation factor is measured from the traded dates**, not inferred from config.
This is not a detail: an earlier version stepped the rebalance schedule twice, traded 47
times in 19 years while annualising as 12/year, and overstated every Sharpe by 2.2×. The
factor is now derived from the data and warns when it disagrees with the holding period.

---

## What worked, and what the evidence was

### Demeaning the surface — the decisive change

Implied volatility varies far more *across firms* than across the grid of one firm's
surface. Fed raw values, the network found that level and stopped: it ranked stocks by
volatility (ρ = −0.40 with mean IV) and ignored the shape entirely (ρ = −0.02 with skew).

Subtracting each surface's own mean leaves only geometry — skew, term structure,
curvature.

| | rank IC | t | spread | t |
|---|---|---|---|---|
| raw surfaces | +0.0038 | 0.31 | +0.489% | 1.34 |
| demeaned | **+0.0185** | **2.34** | **+0.669%** | **2.67** |

Correlation with level fell to zero; correlation with skew rose sixfold. The level was
not carrying the signal — it was crowding it out.

Scaling is per surface, so it depends on nothing but that sample. There is no fitted
state and therefore no leakage channel.

### Breadth — arithmetic, then confirmed

The fundamental law of active management says IR ≈ IC × √breadth. At 500 names and
monthly rebalancing the theoretical ceiling is 1.43 and the realised figure was 0.71.

Widening 500 → 2,000 names predicted a 2.00× gain. Measured:

| model | 500 names | 2,000 names | ratio |
|---|---|---|---|
| ridge | 0.500 | 0.983 | **1.97×** |
| logistic | 0.430 | 0.963 | 2.24× |
| gradient boosting | 0.291 | 1.022 | 3.51× |
| 5-day reversal | 0.132 | 0.065 | 0.49× |

Turnover *fell* (0.687 → 0.562): quartile boundaries are more stable in a wider
cross-section, so fewer names cross the cut.

Reversal went backwards, which is informative — short-horizon reversal is a small-cap
effect, and diluting it across larger names weakened it. Breadth multiplies a signal that
generalises; it is not a free lever.

### Signal weighting

Equal weighting spends the same capital on the strongest conviction as on the 125th.
Weighting by distance from the cross-sectional median is worth **+0.105 mean Sharpe**,
consistent across all 30 model × quantile cells of a 90-configuration sweep.

Quantile width, swept over the same grid, was mostly noise: ridge ran 0.516–0.534 across
five widths.

---

## What did not work

Recorded because the failures constrain the conclusions.

**GAF images.** On price data the GAF CNN lost to a 1D convolutional network given
provably identical information — same windows, same features, same folds, same resolution.
Rank IC +0.0078 against +0.0126, net Sharpe +0.18 against +0.45. Folding a
one-dimensional series into a square costs information rather than exposing it.

**Sector specialisation.** Per-sector IC appeared to vary 4.3× (0.0088 to 0.0382). But
capping every sector to the same cross-section width and comparing against *arbitrary*
splits of the same data:

| | IC spread (sd) |
|---|---|
| real sectors, capped to 30 names | 0.0193 |
| arbitrary halves, 30 names | 0.0171 |

A ratio of 1.13×. The apparent heterogeneity was cross-section width, not sector.
Separately, 99% of the signal survives sector-neutralisation, so the model was never
making sector bets in the first place.

**Regime gating.** IC by dispersion tercile looked strong (0.0115 / 0.0388 / 0.0465) —
but that sorted dates by their *own realised* dispersion, which is unobservable at the
decision date. Using prior-month dispersion, the tradable version, the spread collapses
from 0.0184 to 0.0060 and beats only 11% of random date regroupings.

**Signal blending.** Surface and price signals correlate only 0.21 per date (two price
models correlate 0.57), so diversification looked promising. A fixed 50/50 blend produces
a better *predictor* — highest rank IC (0.0355), highest spread (1.331%), highest alpha
(+11.7%) — but a worse *strategy*: Sharpe 1.21 against the surface CNN's 1.40. Equal
weighting a strong signal with a weaker one pulls toward the average. Fitting the weights
would fix that and would also be peeking.

---

## Interpreting the strategy

**Costs are not the binding constraint.** At 10bps the drag is 0.68% of NAV per year and
takes Sharpe from 1.05 to 0.98. The constraint is gross signal.

**It earns most of its return in stressed markets.** By cross-sectional dispersion
tercile, annualised: 2.6% / 7.6% / 18.6%. 2008 was the best year at +43.7%.

**Its failure mode is junk rallies.** Worst drawdown −23.0%, from January 2009 to March
2010; worst years 2020 (−22.3%) and 2009 (−10.8%). The `rmw` loading of +0.29 says it is
long profitable and short unprofitable — a position that is destroyed when low-quality
names rip off a bottom.

---

## Caveats

* **Flat basis-point costs.** A liquidity-scaled model (cost ∝ 1/√volume) is implemented
  but the headline figures use a flat rate, which is optimistic for the smaller names in a
  1,856-stock cross-section.
* **No market impact and no borrow costs.** Both bite hardest on the short leg, which is
  where most of the spread comes from.
* **235 monthly observations.** Enough for the headline result, not enough to resolve
  conditional effects — which is precisely what the sector and regime tests demonstrated.
* **Vendor data.** CRSP and OptionMetrics under academic licence. No vendor rows are
  redistributed here; `data/` is gitignored and only derived statistics are committed.

---

## Layout

```
src/market_image_prediction/
├── config.py            typed config; a dataset version is its own hash
├── data/
│   ├── universe.py      point-in-time membership, leave-and-rejoin aware
│   ├── cleaning.py      calendar alignment, quality flags
│   ├── features.py      feature registry, causal by construction
│   ├── labels.py        the execution convention, enforced in one place
│   ├── gaf.py           Gramian angular fields, recurrence plots
│   ├── surface.py       IV surface pivot and normalisation
│   ├── splits.py        rolling-origin folds with purge and embargo
│   ├── dataset.py       sample builder; the manifest is the alignment layer
│   └── wrds/            CRSP and OptionMetrics loaders
├── models/              baselines, networks, torch training
├── evaluation/          metrics, backtest, costs, factor attribution
└── experiments/         walk-forward runner
```

91 tests. The ones that matter test alignment and leakage by **perturbation**: corrupt
every bar from date *t* onward and assert every feature before *t* is unchanged;
reconstruct a tensor from the dates its manifest row claims and compare.

## Reproducing

Requires WRDS access with CRSP and OptionMetrics.

```bash
uv sync
cp .env.example .env          # add WRDS_USERNAME

uv run market-image-prediction crsp-download -c configs/crsp_top2000.yaml --top-n 2000
uv run market-image-prediction clean        -c configs/iv_wide_demean.yaml
uv run market-image-prediction build        -c configs/iv_wide_demean.yaml
uv run market-image-prediction iv-download  -c configs/iv_wide_demean.yaml
uv run market-image-prediction iv-build     -c configs/iv_wide_demean.yaml
uv run market-image-prediction iv-train     -c configs/iv_wide_demean.yaml
```

Downloads are checkpointed per chunk and resume after a dropped connection.
[`notebooks/results.ipynb`](notebooks/results.ipynb) renders the analysis from the
committed CSVs and needs no data access.
