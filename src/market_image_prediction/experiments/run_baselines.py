"""Walk-forward baseline runner.

Produces one out-of-fold prediction file per model. Every downstream evaluation --
IC, backtest, diagnostics -- reads that file and never the training process, so results
cannot depend on notebook state and can be recomputed months later from disk.

The loop's discipline, per fold:

  1. slice train / val / test by the fold's own `split` labels (already purged and
     embargoed);
  2. fit on train **only**, including every scaler;
  3. predict the test block;
  4. append to the out-of-fold table with the fold recorded.

Because test blocks are disjoint across folds, concatenating them yields a genuine
out-of-sample series spanning 2005-2025 with no sample scored by a model that saw it.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import polars as pl

from market_image_prediction.config import Config
from market_image_prediction.data.splits import assign_fold, make_folds
from market_image_prediction.models.base import FoldData, LazyRows, Model
from market_image_prediction.models.baselines import default_baselines
from market_image_prediction.utils.io import ensure_dir, write_json, write_parquet
from market_image_prediction.utils.logging import get_logger
from market_image_prediction.utils.seed import provenance, set_seed

log = get_logger(__name__)


def load_dataset(cfg: Config, modality: str = "gaf") -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Memory-map the image and sequence arrays and read the manifest indexing them.

    `modality` selects which picture the image slot carries:

    * "gaf"     -- Phase 1 GAF images, under `tensors/<universe>/`
    * "surface" -- Phase 2 IV surfaces, under `tensors/iv_surface/`

    Both modalities also carry the raw price window in the sequence slot, so a price
    model and a surface model can be compared on one fold.
    """
    version = cfg.version_hash()
    seq_path = Path(cfg.paths.tensors) / cfg.universe / version / "X_seq.npy"
    if modality == "surface":
        image_path = (
            Path(cfg.paths.tensors) / f"iv_surface_{cfg.universe}" / version / "X_surface.npy"
        )
        manifest_path = (
            Path(cfg.paths.manifests) / f"iv_surface_{cfg.universe}" / f"manifest_{version}.parquet"
        )
    else:
        image_path = Path(cfg.paths.tensors) / cfg.universe / version / "X_gaf.npy"
        manifest_path = Path(cfg.paths.manifests) / cfg.universe / f"manifest_{version}.parquet"

    missing_files = [p for p in (seq_path, image_path, manifest_path) if not p.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"{modality} dataset {version} incomplete; missing: "
            + ", ".join(str(m) for m in missing_files)
        )

    seq = np.load(seq_path, mmap_mode="r")
    image = np.load(image_path, mmap_mode="r")
    manifest = pl.read_parquet(manifest_path)

    if modality == "surface":
        # X_seq is indexed by the *price* manifest; the surface manifest is a different
        # set, so row i of one is not row i of the other and the two must be intersected
        # before anything consumes them.
        #
        # They disagree for a real reason, not a config error: a security can have an
        # options surface before it has a full window of price features -- a recent
        # listing, or the feature warm-up at the start of the sample. Keeping only the
        # rows present in both is also what makes the comparison controlled, since every
        # model then sees exactly the same samples.
        price_manifest = pl.read_parquet(
            Path(cfg.paths.manifests) / cfg.universe / f"manifest_{version}.parquet",
            columns=["ticker", "asof_date"],
        ).with_row_index("_seq_row")
        aligned = (
            manifest.select("ticker", "asof_date")
            .with_row_index("_surface_row")
            .join(price_manifest, on=["ticker", "asof_date"], how="left")
            .sort("_surface_row")
        )
        keep = aligned.filter(pl.col("_seq_row").is_not_null())
        dropped = aligned.height - keep.height
        if dropped:
            log.info(
                "dropped %d surface samples (%.1f%%) with no price counterpart; "
                "comparing on the %d samples common to both",
                dropped,
                100 * dropped / aligned.height,
                keep.height,
            )
        if keep.is_empty():
            raise RuntimeError("no samples common to the surface and price manifests")
        surface_rows = keep["_surface_row"].to_numpy().astype(np.intp)
        seq = np.asarray(seq[keep["_seq_row"].to_numpy().astype(np.intp)])
        image = np.asarray(image[surface_rows])
        manifest = manifest[surface_rows.tolist()]

    return seq, image, manifest


def run_walk_forward(
    cfg: Config,
    models: list[Model] | None = None,
    target_col: str = "target_cs",
    max_folds: int | None = None,
    name: str = "baselines",
    modality: str = "gaf",
) -> Path:
    """Fit every model on every fold and write the pooled out-of-fold predictions."""
    set_seed(cfg.seed)
    seq, gaf, manifest = load_dataset(cfg, modality=modality)
    models = models or default_baselines()

    first, last = manifest["asof_date"].min(), manifest["asof_date"].max()
    if not (isinstance(first, dt.date) and isinstance(last, dt.date)):
        raise TypeError("manifest asof_date is not a date column")
    folds = make_folds(cfg, first, last)
    if max_folds is not None:
        folds = folds[:max_folds]

    # Row position in the manifest is row position in the array; keep that explicit.
    manifest = manifest.with_row_index("row")
    records: list[pl.DataFrame] = []
    timings: dict[str, float] = {}

    for fold in folds:
        tagged = assign_fold(manifest, fold, cfg)
        train = tagged.filter(pl.col("split") == "train")
        val = tagged.filter(pl.col("split") == "val")
        test = tagged.filter(pl.col("split") == "test")
        if train.height < 500 or test.is_empty() or val.is_empty():
            log.warning(
                "fold %d skipped: train=%d val=%d test=%d",
                fold.index,
                train.height,
                val.height,
                test.height,
            )
            continue

        # Materialised once per fold: every model in the ladder sees the same arrays,
        # which is what makes "same information set" a fact rather than a claim.
        rows_tr, rows_va, rows_te = (
            train["row"].to_numpy(),
            val["row"].to_numpy(),
            test["row"].to_numpy(),
        )
        # Sequences are small (~0.4 GB for the whole dataset) and every model touches
        # them, so they are materialised. The GAF tensor is ~17 GB and a single fold's
        # training slice is ~12.7 GB, so it stays a lazy view and is read in batches.
        data = FoldData(
            seq_train=np.asarray(seq[rows_tr]),
            gaf_train=LazyRows(gaf, rows_tr),
            y_train=train[target_col].to_numpy().astype(np.float64),
            seq_val=np.asarray(seq[rows_va]),
            gaf_val=LazyRows(gaf, rows_va),
            y_val=val[target_col].to_numpy().astype(np.float64),
            seq_test=np.asarray(seq[rows_te]),
            gaf_test=LazyRows(gaf, rows_te),
            dates_val=val["asof_date"].to_numpy(),
        )

        for model in models:
            start = time.perf_counter()
            model.fit(data)
            preds = model.predict(data)
            timings[model.name] = timings.get(model.name, 0.0) + time.perf_counter() - start
            records.append(
                test.select(
                    "asof_date",
                    "ticker",
                    "sector",
                    "execution_date",
                    "target_end_date",
                    "target_return",
                    "target_direction",
                    "target_cs",
                    "cs_size",
                ).with_columns(
                    pl.lit(model.name).alias("model"),
                    pl.lit(fold.index).alias("fold"),
                    pl.Series("prediction", preds.astype(np.float64)),
                )
            )
        log.info(
            "fold %2d: train=%6d val=%5d test=%5d (%s -> %s)",
            fold.index,
            train.height,
            val.height,
            test.height,
            fold.test_start,
            fold.test_end,
        )

    if not records:
        raise RuntimeError("no folds produced predictions")

    oof = pl.concat(records, how="vertical").sort(["model", "asof_date", "ticker"])
    out_dir = ensure_dir(Path(cfg.paths.predictions) / cfg.universe)
    path = out_dir / f"{name}_{cfg.version_hash()}.parquet"
    write_parquet(oof, path)
    write_json(
        {
            "dataset_version": cfg.version_hash(),
            "target": target_col,
            "n_folds": len(folds),
            "models": [m.name for m in models],
            "fit_seconds": timings,
            "config": cfg.model_dump(mode="json", exclude={"paths"}),
            "environment": provenance(),
        },
        out_dir / f"{name}_{cfg.version_hash()}.meta.json",
    )
    log.info("wrote %d out-of-fold rows for %d models -> %s", oof.height, len(models), path)
    return path
