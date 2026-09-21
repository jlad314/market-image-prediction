"""Torch training wrapper for the neural baselines.

Shared training discipline for models D and E:

* **No fitted preprocessing.** Inputs arrive already in [-1, 1] -- GAF images by
  construction, sequences via the same per-window robust scale the encoder uses. There
  is no scaler carrying training statistics into validation or test, so this class of
  leakage is absent by design rather than by care.
* **Early stopping on validation rank IC**, not loss. Pooled MSE rewards predicting the
  market-wide level, which the cross-sectional target has already removed; rank IC is
  what the backtest actually consumes.
* **Best-epoch weights are restored** before scoring the test block, so the reported
  result is the model validation selected, not wherever training happened to stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import stats
from torch import nn

from market_image_prediction.data.gaf import paa, robust_scale
from market_image_prediction.models.base import ArrayLike, FoldData, cache_if_it_fits
from market_image_prediction.models.nets import (
    Conv1dNet,
    GafCnn,
    SurfaceCnn,
    count_parameters,
)
from market_image_prediction.utils.logging import get_logger

log = get_logger(__name__)


def pick_device(prefer: str = "auto") -> torch.device:
    """MPS when available, else CPU.

    MPS is not bitwise deterministic, so any run claiming reproducibility must record
    the device alongside the seed; `provenance()` does.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _rank_ic(pred: np.ndarray, actual: np.ndarray, dates: np.ndarray) -> float:
    """Mean per-date Spearman correlation. Dates with under 4 names are skipped."""
    out = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 4:
            continue
        p, a = pred[m], actual[m]
        if np.all(p == p[0]) or np.all(a == a[0]):
            continue
        out.append(float(stats.spearmanr(p, a).statistic))  # type: ignore[attr-defined]
    return float(np.nanmean(out)) if out else float("nan")


def prepare_sequence(seq: np.ndarray, image_size: int | None, clip_sigma: float) -> np.ndarray:
    """Robust-scale each window, optionally PAA to the GAF's resolution.

    Downsampling to `image_size` is the default so the sequence model sees *exactly* the
    information the GAF encoder sees. Without it a win for the 1D model could simply be
    extra resolution rather than a better representation.
    """
    n, f, w = seq.shape
    flat = robust_scale(seq.reshape(n * f, w).astype(np.float64), clip_sigma=clip_sigma)
    if image_size is not None and image_size < w:
        flat = np.clip(paa(flat, image_size), -1.0, 1.0)
    return flat.reshape(n, f, -1).astype(np.float32)


@dataclass
class TorchModel:
    """Common fit/predict loop; subclasses supply the network and choose the input."""

    name: str = "torch_model"
    epochs: int = 40
    patience: int = 6
    # 256 measured at 2,341 samples/s on MPS against 15,970 at 1024 -- small batches
    # leave the accelerator mostly idle in kernel-launch overhead.
    batch_size: int = 1024
    # Scaled with sqrt(batch ratio) from the 1e-3 used at batch 256, so the per-update
    # step size stays comparable despite 4x fewer updates per epoch.
    lr: float = 2e-3
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    width: int = 32
    dropout: float = 0.3
    seed: int = 17
    device: str = "auto"
    match_gaf_resolution: int | None = 32
    infer_batch_size: int = 8192
    # Fold tensors are pulled into RAM at this precision when they fit the budget.
    # float16 resolves GAF's [-1, 1] range to ~5e-4, far finer than the signal.
    cache_dtype: str = "float16"
    cache_budget_bytes: int = 9_000_000_000
    clip_sigma: float = 5.0
    history: list[dict] = field(default_factory=list)

    def _build(self, in_channels: int) -> nn.Module:
        raise NotImplementedError

    def _inputs(self, data: FoldData) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Train / val / test inputs. May be arrays or lazy memmap views."""
        raise NotImplementedError

    def fit(self, data: FoldData) -> None:
        # The runner reuses one model instance across folds, so per-fold state must be
        # cleared here. The network itself is rebuilt below, so weights never carry
        # over; only this log would otherwise accumulate.
        self.history = []
        torch.manual_seed(self.seed)
        device = pick_device(self.device)
        x_train, x_val, _ = self._inputs(data)
        # Paid once per fold rather than once per epoch: streaming from the memmap runs
        # at ~4,000 samples/s while the network computes at ~16,000.
        dtype = np.dtype(self.cache_dtype)
        x_train = cache_if_it_fits(x_train, dtype, self.cache_budget_bytes)
        x_val = cache_if_it_fits(x_val, dtype, self.cache_budget_bytes)
        log.info(
            "%s fold inputs: train %s%s, val %s",
            self.name,
            type(x_train).__name__,
            f" ({len(x_train)} x {dtype.name})" if isinstance(x_train, np.ndarray) else "",
            type(x_val).__name__,
        )

        net = self._build(int(x_train.shape[1])).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.HuberLoss(delta=self.huber_delta)

        y_train = torch.from_numpy(data.y_train.astype(np.float32))
        y_val = data.y_val.astype(np.float64)
        dates_val = (
            data.dates_val if data.dates_val is not None else np.zeros(len(y_val), dtype=int)
        )

        best_score, best_state, bad_epochs = -np.inf, None, 0
        generator = torch.Generator().manual_seed(self.seed)
        n_train = len(x_train)

        for epoch in range(self.epochs):
            net.train()
            order = torch.randperm(n_train, generator=generator).numpy()
            total = 0.0
            for start in range(0, n_train, self.batch_size):
                idx = np.sort(order[start : start + self.batch_size])
                # Batches are pulled from the (possibly memmapped) source one at a time,
                # so peak memory is one batch rather than the whole fold. Indices are
                # sorted first because reading a memmap in ascending order is markedly
                # faster than random access.
                xb = torch.from_numpy(np.asarray(x_train[idx])).to(device).float()
                yb = y_train[idx].to(device)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                # Financial batches contain occasional extreme windows; clipping keeps a
                # single crash day from destabilising the whole run.
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                total += float(loss.detach()) * len(idx)

            preds = self._infer(net, x_val, device)
            score = _rank_ic(preds, y_val, dates_val)
            self.history.append(
                {"epoch": epoch, "train_loss": total / n_train, "val_rank_ic": score}
            )

            if np.isfinite(score) and score > best_score:
                best_score, bad_epochs = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net.eval()
        self.device_used = device
        self.best_val_rank_ic = best_score
        log.info(
            "%s: %d params, best val rank IC %.4f after %d epochs",
            self.name,
            count_parameters(net),
            best_score,
            len(self.history),
        )

    def _infer(self, net: nn.Module, x, device: torch.device) -> np.ndarray:
        """Score in batches, so inference also never materialises a whole fold."""
        net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), self.infer_batch_size):
                chunk = np.asarray(x[start : start + self.infer_batch_size])
                out.append(net(torch.from_numpy(chunk).to(device).float()).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def predict(self, data: FoldData) -> np.ndarray:
        _, _, x_test = self._inputs(data)
        return self._infer(self.net, x_test, self.device_used)


@dataclass
class Conv1dModel(TorchModel):
    """Model D: temporal CNN on the raw sequence."""

    name: str = "conv1d_seq"

    def _build(self, in_channels: int) -> nn.Module:
        return Conv1dNet(in_channels, width=self.width, dropout=self.dropout)

    def _inputs(self, data: FoldData) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        def prep(a: np.ndarray) -> np.ndarray:
            return prepare_sequence(a, self.match_gaf_resolution, self.clip_sigma)

        return prep(data.seq_train), prep(data.seq_val), prep(data.seq_test)


@dataclass
class GafCnnModel(TorchModel):
    """Model E: 2D CNN on the GAF images."""

    name: str = "gaf_cnn"

    def _build(self, in_channels: int) -> nn.Module:
        return GafCnn(in_channels, width=self.width, dropout=self.dropout)

    def _inputs(self, data: FoldData) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        # Returned as-is: these may be LazyRows views onto a 17 GB memmap, and
        # materialising them here would defeat the batching above.
        return data.gaf_train, data.gaf_val, data.gaf_test


@dataclass
class SurfaceCnnModel(TorchModel):
    """Model F: CNN on the IV surface.

    Surfaces arrive in the `gaf_*` slots of `FoldData` -- the field names are historical;
    what matters is that this is the image channel and the sequence channel still carries
    the price window, so surface and price models remain directly comparable on one fold.
    """

    name: str = "surface_cnn"
    width: int = 24

    def _build(self, in_channels: int) -> nn.Module:
        return SurfaceCnn(in_channels, width=self.width, dropout=self.dropout)

    def _inputs(self, data: FoldData) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        return data.gaf_train, data.gaf_val, data.gaf_test
