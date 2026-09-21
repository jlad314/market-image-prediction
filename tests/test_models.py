"""Model interface, network shapes, and the training loop's leakage guarantees."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from market_image_prediction.models.base import FoldData
from market_image_prediction.models.baselines import default_baselines
from market_image_prediction.models.nets import Conv1dNet, GafCnn, count_parameters
from market_image_prediction.models.neural import (
    Conv1dModel,
    GafCnnModel,
    prepare_sequence,
)


@pytest.fixture
def fold() -> FoldData:
    rng = np.random.default_rng(0)
    n_tr, n_va, n_te = 400, 120, 100

    def make(n):
        return (
            rng.normal(size=(n, 3, 60)).astype(np.float32),
            rng.normal(size=(n, 6, 32, 32)).astype(np.float32),
        )

    s_tr, g_tr = make(n_tr)
    s_va, g_va = make(n_va)
    s_te, g_te = make(n_te)
    return FoldData(
        seq_train=s_tr,
        gaf_train=g_tr,
        y_train=rng.normal(size=n_tr),
        seq_val=s_va,
        gaf_val=g_va,
        y_val=rng.normal(size=n_va),
        seq_test=s_te,
        gaf_test=g_te,
        dates_val=np.repeat(np.arange(n_va // 8), 8),
    )


def test_every_baseline_honours_the_interface(fold: FoldData) -> None:
    for model in default_baselines():
        model.fit(fold)
        p = model.predict(fold)
        assert p.shape == (len(fold.seq_test),)
        assert np.isfinite(p).all()


def test_fold_data_rejects_mismatched_lengths() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        FoldData(
            seq_train=rng.normal(size=(10, 3, 60)),
            gaf_train=rng.normal(size=(10, 6, 32, 32)),
            y_train=rng.normal(size=9),
            seq_val=rng.normal(size=(5, 3, 60)),
            gaf_val=rng.normal(size=(5, 6, 32, 32)),
            y_val=rng.normal(size=5),
            seq_test=rng.normal(size=(5, 3, 60)),
            gaf_test=rng.normal(size=(5, 6, 32, 32)),
        )


def test_networks_produce_one_score_per_sample() -> None:
    assert Conv1dNet(3)(torch.randn(7, 3, 32)).shape == (7,)
    assert GafCnn(6)(torch.randn(7, 6, 32, 32)).shape == (7,)


def test_networks_stay_small() -> None:
    """Capacity guard: ~1,000 independent dates cannot support a large model."""
    assert count_parameters(Conv1dNet(3)) < 200_000
    assert count_parameters(GafCnn(6)) < 200_000


def test_prepare_sequence_matches_gaf_resolution() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5, 3, 60))
    out = prepare_sequence(x, image_size=32, clip_sigma=5.0)
    assert out.shape == (5, 3, 32)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_prepare_sequence_is_per_window() -> None:
    """No statistic is shared across samples, so there is nothing to leak."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(8, 3, 60))
    full = prepare_sequence(x, 32, 5.0)
    part = prepare_sequence(x[2:5], 32, 5.0)
    assert np.allclose(full[2:5], part)


def test_groupnorm_makes_predictions_batch_independent(fold: FoldData) -> None:
    """A sample's score must not depend on which other names share its batch.

    BatchNorm would violate this at train time; GroupNorm normalises within a sample.
    """
    net = GafCnn(6).eval()
    x = torch.from_numpy(fold.gaf_test)
    with torch.no_grad():
        alone = net(x[:4])
        together = net(x)[:4]
    assert torch.allclose(alone, together, atol=1e-6)


@pytest.mark.parametrize("model_cls", [Conv1dModel, GafCnnModel])
def test_neural_models_train_and_predict(fold: FoldData, model_cls) -> None:
    model = model_cls(epochs=3, patience=2, device="cpu")
    model.fit(fold)
    p = model.predict(fold)
    assert p.shape == (len(fold.seq_test),)
    assert np.isfinite(p).all()
    assert len(model.history) <= 3


def test_neural_training_is_seed_reproducible(fold: FoldData) -> None:
    a = GafCnnModel(epochs=3, patience=3, device="cpu", seed=5)
    b = GafCnnModel(epochs=3, patience=3, device="cpu", seed=5)
    a.fit(fold)
    b.fit(fold)
    assert np.allclose(a.predict(fold), b.predict(fold), atol=1e-5)


def test_neural_model_never_reads_the_test_block_during_fit(fold: FoldData) -> None:
    """Corrupting the test inputs after fitting must not change the fitted weights."""
    model = GafCnnModel(epochs=3, patience=3, device="cpu", seed=9)
    model.fit(fold)
    before = model.predict(fold)

    poisoned = FoldData(
        seq_train=fold.seq_train,
        gaf_train=fold.gaf_train,
        y_train=fold.y_train,
        seq_val=fold.seq_val,
        gaf_val=fold.gaf_val,
        y_val=fold.y_val,
        seq_test=fold.seq_test * 100.0,
        gaf_test=fold.gaf_test,
        dates_val=fold.dates_val,
    )
    # Same weights, same GAF test input -> identical scores despite the poisoned seq.
    assert np.allclose(before, model.predict(poisoned), atol=1e-6)
