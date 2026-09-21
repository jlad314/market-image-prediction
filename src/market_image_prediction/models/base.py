"""Shared model interface.

Every model receives the same `FoldData`: the raw sequence windows, the GAF tensor, and
the targets, already split into train/val/test by the purged, embargoed fold. This is
what makes the comparison fair -- a model cannot quietly source extra information,
because there is nowhere else to get it from.

The validation block exists so neural models can early-stop. Tabular models ignore it.
Neither may touch the test block during `fit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class LazyRows:
    """A row selection of a memmapped tensor, materialised only when it is worth it.

    The GAF tensor for the CRSP universe is ~17 GB, so a fold's training slice is ~12.7 GB
    in float32 -- more than fits alongside a torch copy on a 24 GB machine. Holding the
    memmap plus an index array lets a training loop pull one batch at a time.

    Streaming batches is far slower than it looks, though. Measured on this dataset:
    random memmap gather sustains ~4,000 samples/s while the network computes at ~16,000
    samples/s, so the accelerator idles ~75% of the time. Two properties fix that:

    * **fold rows are contiguous.** The manifest is sorted by (date, ticker) and a fold is
      a date range, so a fold's rows form a solid block. Reading a slice is sequential I/O
      rather than a scattered gather.
    * **float16 is ample here.** GAF channels live in [-1, 1], where float16 resolves to
      ~5e-4. Halving the bytes brings the largest fold to ~6.4 GB, which fits in RAM, so
      the read is paid once per fold instead of once per epoch.
    """

    def __init__(self, source: np.ndarray, rows: np.ndarray) -> None:
        self.source = source
        self.rows = np.asarray(rows, dtype=np.intp)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.rows), *self.source.shape[1:])

    @property
    def is_contiguous(self) -> bool:
        """True when the selection is a solid ascending block."""
        if self.rows.size < 2:
            return True
        return bool(self.rows[-1] - self.rows[0] + 1 == self.rows.size) and bool(
            np.all(np.diff(self.rows) == 1)
        )

    def __getitem__(self, index: slice | np.ndarray) -> np.ndarray:
        """Materialise the selected rows. Accepts a slice or an index array."""
        return np.ascontiguousarray(self.source[self.rows[index]])

    def materialise(self, dtype: np.dtype | type | None = None) -> np.ndarray:
        """Realise every row, using a sequential read when the selection allows it."""
        if self.is_contiguous:
            block = self.source[self.rows[0] : self.rows[-1] + 1]
        else:
            block = self.source[self.rows]
        out = np.ascontiguousarray(block)
        return out.astype(dtype, copy=False) if dtype is not None else out

    def nbytes(self, dtype: np.dtype | type) -> int:
        return int(len(self.rows) * np.prod(self.source.shape[1:]) * np.dtype(dtype).itemsize)


def cache_if_it_fits(
    x: ArrayLike, dtype: np.dtype | type = np.float16, budget_bytes: int = 8_000_000_000
) -> ArrayLike:
    """Pull a lazy selection into RAM at reduced precision when it fits the budget.

    Returns the input untouched when it is already an array, or when materialising it
    would exceed `budget_bytes` -- in which case the caller keeps streaming.
    """
    if not isinstance(x, LazyRows):
        return x
    if x.nbytes(dtype) > budget_bytes:
        return x
    return x.materialise(dtype)


ArrayLike = np.ndarray | LazyRows


@dataclass
class FoldData:
    """One fold's arrays. `seq` is (n, n_features, window); `gaf` is (n, C, S, S).

    `gaf_*` may be a `LazyRows` view rather than a materialised array; consumers must
    index it rather than assume `np.ndarray` semantics beyond slicing.
    """

    seq_train: np.ndarray
    gaf_train: ArrayLike
    y_train: np.ndarray

    seq_val: np.ndarray
    gaf_val: ArrayLike
    y_val: np.ndarray

    seq_test: np.ndarray
    gaf_test: ArrayLike

    # Validation decision dates, so a model can early-stop on cross-sectional rank IC --
    # the quantity the project is judged on -- rather than on pooled MSE, which rewards
    # predicting the market-wide level the ranking target has already removed.
    dates_val: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.seq_train) != len(self.y_train):
            raise ValueError("train sequence and target lengths disagree")
        if len(self.gaf_train) != len(self.y_train):
            raise ValueError("train tensor and target lengths disagree")


class Model(Protocol):
    """Fit on train (early-stopping on val if useful), then score test."""

    name: str

    def fit(self, data: FoldData) -> None: ...
    def predict(self, data: FoldData) -> np.ndarray: ...
