"""Command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from market_image_prediction.config import Config
from market_image_prediction.utils.logging import console, setup_logging

app = typer.Typer(add_completion=False, help="Market image prediction research CLI.")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Path to a YAML config.")]


def _load(config: Path | None) -> Config:
    return Config.from_yaml(config) if config else Config()


@app.command()
def download(
    config: ConfigOpt = None,
    force: Annotated[bool, typer.Option(help="Overwrite the raw snapshot.")] = False,
) -> None:
    """Fetch raw prices and metadata into an immutable snapshot."""
    setup_logging()
    from market_image_prediction.data.download_yahoo import (
        download_metadata,
        download_universe,
    )

    cfg = _load(config)
    download_universe(cfg, force=force)
    download_metadata(cfg, force=force)


@app.command()
def clean(
    config: ConfigOpt = None,
    force: Annotated[bool, typer.Option(help="Rebuild the canonical panel.")] = False,
) -> None:
    """Build the canonical, calendar-aligned, quality-flagged price panel."""
    setup_logging()

    cfg = _load(config)
    if cfg.universe.startswith("crsp"):
        from market_image_prediction.data.crsp_canonical import build_canonical_crsp

        build_canonical_crsp(cfg, force=force)
    else:
        from market_image_prediction.data.cleaning import build_canonical

        build_canonical(cfg, force=force)


@app.command()
def audit(config: ConfigOpt = None) -> None:
    """Report coverage, integrity and cross-sectional width. Gate before modelling."""
    setup_logging()
    from market_image_prediction.data.audit import run_audit

    run_audit(_load(config))


@app.command()
def build(
    config: ConfigOpt = None,
    force: Annotated[bool, typer.Option(help="Rebuild tensors and manifest.")] = False,
) -> None:
    """Build GAF tensors and the sample manifest."""
    setup_logging()
    from market_image_prediction.data.dataset import build_samples

    build_samples(_load(config), force=force)


@app.command()
def baselines(
    config: ConfigOpt = None,
    max_folds: Annotated[int | None, typer.Option(help="Limit folds (for a smoke run).")] = None,
) -> None:
    """Fit the baseline ladder walk-forward and write out-of-fold predictions."""
    setup_logging()
    from market_image_prediction.experiments.run_baselines import run_walk_forward

    run_walk_forward(_load(config), max_folds=max_folds)


@app.command()
def neural(
    config: ConfigOpt = None,
    max_folds: Annotated[int | None, typer.Option(help="Limit folds (for a smoke run).")] = None,
    epochs: Annotated[int, typer.Option(help="Max epochs per fold.")] = 40,
) -> None:
    """Train the 1D sequence CNN and the 2D GAF CNN walk-forward."""
    setup_logging()
    from market_image_prediction.experiments.run_baselines import run_walk_forward
    from market_image_prediction.models.neural import Conv1dModel, GafCnnModel

    cfg = _load(config)
    models = [
        Conv1dModel(epochs=epochs, seed=cfg.seed),
        GafCnnModel(epochs=epochs, seed=cfg.seed),
    ]
    run_walk_forward(cfg, models=models, max_folds=max_folds, name="neural")


@app.command()
def report(
    config: ConfigOpt = None,
    name: Annotated[
        list[str] | None,
        typer.Option("--name", "-n", help="Prediction file stems; repeat to combine."),
    ] = None,
) -> None:
    """Compare models on out-of-fold predictions, with cost sensitivity."""
    setup_logging()
    from market_image_prediction.evaluation.report import print_report

    print_report(_load(config), names=name or ["baselines"])


@app.command(name="wrds-discover")
def wrds_discover(
    username: Annotated[
        str | None, typer.Option(help="WRDS username; defaults to $WRDS_USERNAME.")
    ] = None,
) -> None:
    """Catalogue the CRSP/OptionMetrics tables this WRDS account can see.

    First run is interactive: it prompts for the WRDS password and offers to create
    ~/.pgpass so later runs are silent.
    """
    setup_logging()
    from market_image_prediction.data.wrds.discover import discover

    discover(username=username)


@app.command(name="crsp-download")
def crsp_download(
    config: ConfigOpt = None,
    top_n: Annotated[int, typer.Option(help="Securities kept at each month-end.")] = 500,
    force: Annotated[bool, typer.Option(help="Re-download the CRSP snapshot.")] = False,
) -> None:
    """Download a point-in-time large-cap US equity panel from CRSP (CIZ)."""
    setup_logging()
    from market_image_prediction.data.wrds.crsp import download_crsp_universe

    download_crsp_universe(_load(config), top_n=top_n, force=force)


@app.command(name="iv-download")
def iv_download(
    config: ConfigOpt = None,
    force: Annotated[bool, typer.Option(help="Re-download the surface snapshot.")] = False,
) -> None:
    """Download OptionMetrics IV surfaces for the CRSP universe."""
    setup_logging()
    from market_image_prediction.data.wrds.optionmetrics import download_iv_surfaces

    download_iv_surfaces(_load(config), force=force)


@app.command(name="iv-build")
def iv_build(
    config: ConfigOpt = None,
    force: Annotated[bool, typer.Option(help="Rebuild surface tensors.")] = False,
) -> None:
    """Build IV-surface tensors and manifest, labelled from the CRSP pipeline."""
    setup_logging()
    from market_image_prediction.data.surface import build_surface_dataset

    build_surface_dataset(_load(config), force=force)


@app.command(name="iv-train")
def iv_train(
    config: ConfigOpt = None,
    max_folds: Annotated[int | None, typer.Option(help="Limit folds (smoke run).")] = None,
    epochs: Annotated[int, typer.Option(help="Max epochs per fold.")] = 25,
) -> None:
    """Train the IV-surface CNN walk-forward, on the surface manifest."""
    setup_logging()
    from market_image_prediction.experiments.run_baselines import run_walk_forward
    from market_image_prediction.models.baselines import SurfaceRidge
    from market_image_prediction.models.neural import SurfaceCnnModel

    cfg = _load(config)
    run_walk_forward(
        cfg,
        # Ridge on the flattened surface runs alongside the CNN so that "the surface has
        # information" and "a convolution can exploit it" stay separable.
        models=[SurfaceRidge(), SurfaceCnnModel(epochs=epochs, seed=cfg.seed)],
        max_folds=max_folds,
        name="surface",
        modality="surface",
    )


@app.command(name="iv-baselines")
def iv_baselines(
    config: ConfigOpt = None,
    max_folds: Annotated[int | None, typer.Option(help="Limit folds (smoke run).")] = None,
) -> None:
    """Run the price-based ladder on exactly the samples that have a surface."""
    setup_logging()
    from market_image_prediction.experiments.run_baselines import run_walk_forward

    run_walk_forward(
        _load(config), max_folds=max_folds, name="surface_baselines", modality="surface"
    )


@app.command()
def info(config: ConfigOpt = None) -> None:
    """Show the resolved configuration and dataset version."""
    setup_logging()
    from market_image_prediction.data.universe import get_universe

    cfg = _load(config)
    universe = get_universe(cfg.universe)
    console.print(
        f"[bold]universe[/bold]        {universe.name} "
        f"({len(universe.assets)} assets, "
        f"survivorship-bias-free={universe.survivorship_bias_free})"
    )
    console.print(f"[bold]dataset version[/bold] {cfg.version_hash()}")
    console.print(
        f"[bold]tensor shape[/bold]    "
        f"[{cfg.features.n_channels}, {cfg.features.image_size}, "
        f"{cfg.features.image_size}]"
    )
    console.print(
        f"[bold]execution[/bold]       signal close(t) -> "
        f"{cfg.labels.entry_price}(t+{cfg.labels.execution_lag}) -> "
        f"{cfg.labels.exit_price}(t+{cfg.labels.horizon})"
    )
    console.print(
        f"[bold]embargo[/bold]         "
        f"{cfg.splits.resolved_embargo(cfg.labels, cfg.features)} trading days"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
