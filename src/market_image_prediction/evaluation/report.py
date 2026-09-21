"""Model comparison report.

Reads a saved prediction file and produces the two tables that decide whether a signal
is real: predictive statistics with overlap-aware inference, and net-of-cost portfolio
performance across a grid of cost assumptions.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from market_image_prediction.config import Config
from market_image_prediction.evaluation.backtest import run_backtest
from market_image_prediction.evaluation.metrics import (
    directional_accuracy,
    information_coefficient,
    summarise_ic,
)
from market_image_prediction.utils.io import ensure_dir, write_parquet
from market_image_prediction.utils.logging import console


def predictions_path(cfg: Config, name: str = "baselines") -> Path:
    return Path(cfg.paths.predictions) / cfg.universe / f"{name}_{cfg.version_hash()}.parquet"


def load_predictions(cfg: Config, names: list[str]) -> pl.DataFrame:
    """Concatenate several prediction files so every model lands in one table.

    Files are only comparable if they were produced from the same dataset version and
    the same folds; the version is part of each filename, so a mismatch fails loudly
    here rather than silently producing an unfair comparison.
    """
    frames = []
    for name in names:
        path = predictions_path(cfg, name)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `{name}` first")
        frames.append(pl.read_parquet(path))
    common = set(frames[0].columns).intersection(*(set(f.columns) for f in frames[1:]))
    ordered = [c for c in frames[0].columns if c in common]
    return pl.concat([f.select(ordered) for f in frames], how="vertical")


def compare_models(cfg: Config, names: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (predictive table, cost-sensitivity table) across every model present."""
    oof = load_predictions(cfg, names)

    pred_rows, cost_rows = [], []
    for (model,), group in oof.group_by(["model"], maintain_order=True):
        ic = information_coefficient(group)
        stats = summarise_ic(ic, horizon=cfg.labels.horizon)
        acc = directional_accuracy(group)
        pred_rows.append(
            {
                "model": model,
                "rank_ic": stats.get("rank_ic_mean", float("nan")),
                "rank_ic_ir": stats.get("rank_ic_ir", float("nan")),
                "rank_ic_t_hac": stats.get("rank_ic_tstat_hac", float("nan")),
                "ic_hit_rate": stats.get("rank_ic_hit_rate", float("nan")),
                "dir_accuracy": acc["accuracy"],
                "dir_base_rate": acc["base_rate"],
                "n_dates": stats.get("n_dates", 0.0),
                "n_effective": stats.get("n_effective", 0.0),
            }
        )

        bt = run_backtest(group, cfg)
        if "error" in bt:
            continue
        for row in bt["cost_grid"]:
            cost_rows.append(
                {
                    "model": model,
                    "cost_bps": row["cost_bps"],
                    "ann_return": row["ann_return"],
                    "ann_vol": row["ann_vol"],
                    "sharpe": row["sharpe"],
                    "max_drawdown": row["max_drawdown"],
                    "turnover": row["mean_turnover"],
                    "n_rebalances": row["n"],
                }
            )

    predictive = pl.DataFrame(pred_rows).sort("rank_ic", descending=True)
    costs = pl.DataFrame(cost_rows)
    return predictive, costs


def print_report(cfg: Config, names: list[str], save: bool = True) -> None:
    predictive, costs = compare_models(cfg, names)
    stem = "_".join(names)

    console.rule("[bold]Predictive statistics (out-of-fold)")
    console.print(predictive.to_pandas().to_string(index=False, float_format="%.4f"))
    console.print(
        "\n[dim]rank_ic_t_hac uses Newey-West errors at the label horizon; "
        "n_effective divides dates by the horizon to reflect overlap.[/dim]"
    )

    console.rule("[bold]Long-short portfolio, net of costs")
    baseline = costs.filter(pl.col("cost_bps") == cfg.backtest.baseline_cost_bps)
    console.print(baseline.to_pandas().to_string(index=False, float_format="%.4f"))

    console.rule("[bold]Sharpe by cost assumption")
    pivot = costs.pivot(on="cost_bps", index="model", values="sharpe").sort("model")
    console.print(pivot.to_pandas().to_string(index=False, float_format="%.3f"))

    if save:
        out = ensure_dir(Path(cfg.paths.outputs) / "reports" / cfg.universe)
        write_parquet(predictive, out / f"{stem}_predictive.parquet")
        write_parquet(costs, out / f"{stem}_costs.parquet")
        console.print(f"\n[dim]tables written to {out}[/dim]")
