"""Encoding correctness: the images must be what the formulae say they are."""

from __future__ import annotations

import numpy as np
import pytest

from market_image_prediction.config import Config
from market_image_prediction.data import gaf


@pytest.fixture
def x() -> np.ndarray:
    rng = np.random.default_rng(11)
    return np.clip(rng.normal(size=(6, 24)) / 3.0, -1.0, 1.0)


def test_gasf_matches_trig_definition(x: np.ndarray) -> None:
    phi = np.arccos(x)
    assert np.allclose(gaf.gasf(x), np.cos(phi[:, :, None] + phi[:, None, :]))


def test_gadf_matches_trig_definition(x: np.ndarray) -> None:
    phi = np.arccos(x)
    assert np.allclose(gaf.gadf(x), np.sin(phi[:, :, None] - phi[:, None, :]))


def test_gasf_diagonal_recovers_series(x: np.ndarray) -> None:
    """The main diagonal is 2x^2 - 1, so the series is recoverable up to sign."""
    assert np.allclose(np.diagonal(gaf.gasf(x), axis1=1, axis2=2), 2 * x**2 - 1)


def test_symmetries(x: np.ndarray) -> None:
    assert np.allclose(gaf.gasf(x), gaf.gasf(x).transpose(0, 2, 1))
    assert np.allclose(gaf.gadf(x), -gaf.gadf(x).transpose(0, 2, 1))
    assert np.allclose(np.diagonal(gaf.gadf(x), axis1=1, axis2=2), 0.0)


def test_all_encodings_bounded(x: np.ndarray) -> None:
    for name in gaf.ENCODINGS:
        out = gaf._ENCODERS[name](x)
        assert out.min() >= -1.0 - 1e-9 and out.max() <= 1.0 + 1e-9, name


def test_paa_weights_form_an_average() -> None:
    w = gaf.paa_weights(60, 32)
    assert w.shape == (32, 60)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w >= 0).all()


def test_paa_preserves_a_constant() -> None:
    x = np.full((3, 60), 0.37)
    assert np.allclose(gaf.paa(x, 32), 0.37)


def test_paa_preserves_the_mean() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 64))
    assert np.allclose(gaf.paa(x, 32).mean(axis=1), x.mean(axis=1))


def test_robust_scale_is_outlier_resistant() -> None:
    """One extreme bar must not squash the rest of the window toward zero."""
    rng = np.random.default_rng(9)
    clean = rng.normal(size=(1, 60)) * 0.01
    dirty = clean.copy()
    dirty[0, 30] = 5.0
    a, b = gaf.robust_scale(clean), gaf.robust_scale(dirty)
    others = [i for i in range(60) if i != 30]
    assert np.allclose(a[0, others], b[0, others], atol=1e-9)


def test_encode_shape_and_dtype() -> None:
    cfg = Config()
    rng = np.random.default_rng(2)
    w = rng.normal(size=(4, 3, 60))
    out = gaf.encode_windows(w, cfg)
    assert out.shape == (4, 6, 32, 32)
    assert out.dtype == np.float32
    assert len(gaf.channel_names(cfg)) == 6
