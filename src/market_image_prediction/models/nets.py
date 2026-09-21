"""Network definitions.

Both networks are deliberately small. The effective sample is roughly a thousand
independent dates, not sixty thousand overlapping rows, and financial signal-to-noise is
low enough that capacity mostly buys memorisation. The GBM result from the baseline run
-- worst of seven models -- is the empirical warning.

GroupNorm rather than BatchNorm throughout. BatchNorm's running statistics are estimated
across the batch, which on a cross-sectional panel mixes names from the same date and
makes a prediction depend on which other assets happened to be batched with it.
GroupNorm normalises within a sample, so each prediction stays independent.
"""

from __future__ import annotations

import torch
from torch import nn


def _groups(channels: int, target: int = 8) -> int:
    """Largest divisor of `channels` no greater than `target`."""
    for g in range(min(target, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class Conv1dNet(nn.Module):
    """Model D: temporal CNN over the raw (n_features, window) sequence.

    Dilations 1/2/4 widen the receptive field geometrically, so a 3-layer stack sees the
    whole window without the parameter cost of a deeper net or the sequential cost of an
    RNN.
    """

    def __init__(self, in_channels: int, width: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=3, padding=1, dilation=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            nn.Conv1d(width, width * 2, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(_groups(width * 2), width * 2),
            nn.GELU(),
            nn.Conv1d(width * 2, width * 4, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


class GafCnn(nn.Module):
    """Model E: the compact 2D CNN from the brief, over (C, 32, 32) GAF images.

    Global average pooling instead of a flatten-then-dense head: a 128x8x8 flatten would
    add ~1M parameters to a problem with about a thousand independent observations.
    """

    def __init__(self, in_channels: int, width: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(width * 2), width * 2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SurfaceCnn(nn.Module):
    """Model F: 2D CNN over an implied-volatility surface, (C, n_deltas, n_maturities).

    The surface is not square -- 34 deltas against 11 maturities -- and the two axes mean
    different things, so the geometry differs from `GafCnn` in two ways.

    * **Pooling is asymmetric.** Three rounds of 2x2 pooling would take the maturity axis
      from 11 to 1 and throw the term structure away, so pooling halves the delta axis
      only. Moneyness is finely sampled (every 5 delta points) and genuinely smooth;
      maturity is coarse and every tenor carries distinct information.
    * **Kernels are 3x3 with padding**, keeping the maturity axis intact through the
      stack rather than eroding it at the edges.

    Capacity is deliberately smaller than GafCnn's. Monthly sampling leaves ~325 decision
    dates, and the Phase 1 evidence is that networks starve at that scale.
    """

    def __init__(self, in_channels: int, width: int = 24, dropout: float = 0.3) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            # (delta, maturity) -> halve delta only.
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(width * 2), width * 2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)
