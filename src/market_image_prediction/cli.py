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
    from market_image_prediction.data.cleaning import build_canonical

    build_canonical(_load(config), force=force)


@app.command()
def audit(config: ConfigOpt = None) -> None:
    """Report coverage, integrity and cross-sectional width. Gate before modelling."""
    setup_logging()
    from market_image_prediction.data.audit import run_audit

    run_audit(_load(config))


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
