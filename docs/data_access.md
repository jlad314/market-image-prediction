# Data Access Playbook

Status legend: **[HAVE]** obtainable today, free, no approval · **[APPLY]** free with
university credentials, requires an application · **[PAID]** costs money.

The rule for this project: never let a data-acquisition dependency block a phase.
Every builder is written against a *schema*, not a vendor, so a better source can be
swapped in without touching modelling code.

---

## Phase 1 — US equity daily OHLCV  **[HAVE]**

| Item | Source | Notes |
|---|---|---|
| Daily OHLCV, 24 tickers, 2018-01-01 → 2026-01-01 | `yfinance` | ~48k rows |
| Sector / industry | `yfinance.Ticker().info` | Cache to disk on first fetch. Not point-in-time. |
| Benchmark | SPY | benchmark-relative return features |
| Sector ETFs | XLK XLF XLE XLV XLY XLP XLI XLB XLU XLRE XLC | sector-relative features |
| Regime | `^VIX` | regime-bucketed diagnostics |
| Risk-free | `^IRX`, or FRED `DGS3MO` | required for a defensible Sharpe |
| Trading calendar | `pandas-market-calendars` (`XNYS`) | validate coverage; never infer trading days from the data |

### Known limitations of the Phase 1 source

1. **Adjustment convention.** `yfinance` defaults to `auto_adjust=True`, silently
   overwriting `Close` with the split/dividend-adjusted series. Set this explicitly
   and record the choice in the dataset config.
2. **Adjustments are not point-in-time.** A split applied today retroactively rewrites
   history. Therefore: download once, snapshot raw Parquet with a `download_date`,
   and treat `data/raw/` as immutable. Re-downloading silently changes past results.
3. **Survivorship bias.** The 24-ticker list is 100% survivors selected with hindsight.
   Acceptable for pipeline validation. Not acceptable for any cross-sectional claim.
   → resolved by WRDS/CRSP below.

---

## Point-in-time universe — removing survivorship bias

### Route A: WRDS + CRSP  **[APPLY]** — the one to pursue

WRDS (Wharton Research Data Services) is the standard academic gateway. If your
university subscribes, a student account is free.

**How to get it**
1. Check institutional access: <https://wrds-www.wharton.upenn.edu/> → *Register* →
   select your institution. If it appears in the dropdown, you have a subscription.
2. Register with your **university email**. Choose account type *Student*.
3. A faculty sponsor must approve — typically a finance/econ supervisor, or your
   dissertation supervisor. Approval is usually days, occasionally weeks.
4. Access is via web query forms, or programmatically over PostgreSQL
   (`pip install wrds`, or psycopg2 against `wrds-pgdata.wharton.upenn.edu:9737`).

**What CRSP gives you that Yahoo cannot**
- Delisted securities with delisting returns — the actual fix for survivorship bias
- Point-in-time S&P 500 constituent history (`crsp.msp500list`)
- PERMNO identifiers stable across ticker changes and mergers
- Clean, research-grade corporate action adjustments

**Licence constraint:** academic research only. Do not redistribute the raw data;
commit derived aggregates and code, never vendor rows.

### Route B: free approximations **[HAVE]**
- Wikipedia's *List of S&P 500 companies* has a "Selected changes" table; scraping and
  replaying it backwards reconstructs an approximate PIT membership series. Imperfect
  (misses some historical edits) but free and better than nothing.
- Some universities license **Refinitiv Eikon / LSEG Workspace** or **Bloomberg**
  terminals in the business library. Bloomberg allows limited data export
  (~500k data points/month per user) and has historical index membership.
  Check your library's database A–Z list.

---

## Phase 2 — Implied volatility surfaces

### Route A: OptionMetrics IvyDB via WRDS  **[APPLY]** — strongly preferred

This is the exact dataset used by the IV-surface papers cited in the research report
(Oxford, Höfler). Institutional price is five figures; via WRDS it is free to you.

**How to get it**
1. Obtain the WRDS account above first.
2. IvyDB is a *separately licensed* WRDS product — your institution may subscribe to
   WRDS but not to OptionMetrics. Check the WRDS product list once logged in.
3. If not subscribed, ask your supervisor or library data librarian whether a trial or
   departmental subscription is possible. Many universities will add a product on
   request if a researcher justifies it.

**Key tables**
- `optionm.opprcd<YYYY>` — daily option prices, IV, greeks, open interest, volume
- `optionm.secprd` — underlying security prices
- `optionm.vsurfd` — **pre-computed standardised volatility surfaces**
  (interpolated at fixed deltas × fixed maturities)

`vsurfd` is the shortcut: it is already the grid the project needs, which removes the
SVI-fitting problem from the critical path. Fit your own surfaces later as a
methodology comparison, not as a prerequisite.

### Route B: start recording now  **[HAVE]** — do this regardless
`yfinance` exposes *current* option chains but no history. A daily cron job snapshotting
chains for the universe costs nothing and accumulates real history from today forward.
Even if WRDS comes through, this is a useful independent cross-check.

### Route C: paid fallbacks  **[PAID]**
| Source | Cost | Note |
|---|---|---|
| Alpha Vantage `HISTORICAL_OPTIONS` | ~$50/mo | IV + greeks, history to 2008. Best solo-researcher value. |
| Polygon.io Options | $29–199/mo | Quotes + aggregates, bulk flat files. You compute IV. |
| ORATS | ~$1–2k/yr | Ships pre-smoothed surfaces. |
| CBOE DataShop | per-dataset | À la carte historical EOD quotes. |

---

## Phase 3 — Limit order book

### FI-2010  **[HAVE]**
The benchmark used by DeepLOB. Free download, no registration.
Hosted on the Etsin/Fairdata research repository (Ntakaris et al., 2018).
Pre-normalised, 5 Nasdaq Nordic stocks. Benchmark replication only — not a
tradable-strategy dataset.

### LOBSTER  **[APPLY]**
- Free sample data (a single day, several Nasdaq tickers, all depth levels) with no
  registration: <https://lobsterdata.com/info/DataSamples.php>
- **Academic Data Programme**: free/heavily discounted access for students and
  researchers at subscribing universities. Apply with your university email and a
  short research description. Check whether your institution already subscribes first.

### Free full-depth alternative  **[HAVE]**
Binance publishes complete historical order book snapshots and trades as public S3
dumps. Not US equities, but genuinely deep, free, and adequate for the
representation-learning questions in Phase 3.

### Paid  **[PAID]**
Databento (Nasdaq TotalView-ITCH, MBP-10/MBO), pay-as-you-go. Excellent API.
Databento also runs a **student/academic credit programme** — worth an email.

---

## Other free-for-students sources worth knowing

| Source | How | Gives you |
|---|---|---|
| **FRED** (St. Louis Fed) | free API key, instant | risk-free rates, macro regime series |
| **Ken French Data Library** | free download, no registration | Fama-French factors — essential for risk-adjusting any alpha claim |
| **SEC EDGAR full-text + XBRL** | free API, no key | fundamentals, point-in-time by filing date |
| **Nasdaq Data Link (Quandl)** | free tier + academic discounts | Sharadar fundamentals (paid), many free tables |
| **GitHub Student Developer Pack** | verify with university email | occasional data/compute credits |
| **Google Cloud / AWS research credits** | apply via supervisor | GPU hours for the Stage 3 diffusion work |

---

## Recommended sequencing

1. **This week** — Phase 1 on `yfinance`. Nothing is blocked.
2. **This week, in parallel** — start the WRDS application. It has the longest lead time
   (needs a faculty sponsor) and unlocks both the survivorship fix *and* IvyDB.
3. **This week, 30 minutes** — stand up the daily `yfinance` option-chain recorder.
   It only accumulates value while you wait.
4. **When WRDS lands** — re-run Phase 1 on a point-in-time CRSP universe. That converts
   the proof of concept into a defensible cross-sectional result.
5. **Phase 2** — starts the day `optionm.vsurfd` is reachable.
