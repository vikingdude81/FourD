"""
Basic smoke tests for the FourD pipeline.
"""

import numpy as np
import pytest

from src.features.windows import sliding_windows
from src.features.basic_stats import compute_basic_stats
from src.features.entropy import spectral_entropy, permutation_entropy
from src.features.complexity import lz_complexity
from src.features.autocorr import autocorr_lag
from src.anomaly.scoring import score_windows
from src.anomaly.standardize import zscore_dataframe
from src.latent.basins import (
    initialize_basin_attractors,
    basin_similarities,
    attractor_pull,
    basin_switch_event,
)
from src.latent.mapping import feature_row_to_drives
from src.latent.coordinator import run_coordinator
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Feature tests
# ──────────────────────────────────────────────────────────────────────────────

def test_sliding_windows_basic():
    bits = np.zeros(10000, dtype=np.uint8)
    windows = list(sliding_windows(bits, window_size=1000, overlap=0.5))
    assert len(windows) > 0
    for start, w in windows:
        assert len(w) == 1000


def test_basic_stats():
    bits = np.array([0, 1, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
    stats = compute_basic_stats(bits)
    assert "bias" in stats
    assert "runs_count" in stats
    assert "longest_run" in stats
    assert 0.0 <= stats["bias"] <= 1.0
    assert stats["longest_run"] >= 1


def test_spectral_entropy_uniform():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=4096).astype(np.uint8)
    se = spectral_entropy(bits)
    assert 0.0 <= se <= 1.0


def test_permutation_entropy_range():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=512).astype(np.uint8)
    pe = permutation_entropy(bits, order=3)
    assert 0.0 <= pe <= 1.0


def test_lz_complexity_range():
    bits = np.array([0, 1] * 512, dtype=np.uint8)
    lz = lz_complexity(bits)
    assert 0.0 <= lz <= 1.0


def test_autocorr_lag():
    bits = np.array([0, 1] * 256, dtype=np.uint8)
    ac = autocorr_lag(bits, lag=1)
    assert -1.0 <= ac <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Anomaly tests
# ──────────────────────────────────────────────────────────────────────────────

def _make_feature_df(n=20):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "bias": rng.uniform(0.4, 0.6, n),
        "runs_count": rng.integers(100, 300, n).astype(float),
        "spectral_entropy": rng.uniform(0.9, 1.0, n),
        "permutation_entropy": rng.uniform(0.8, 1.0, n),
        "lz_complexity": rng.uniform(0.7, 0.9, n),
        "autocorr_lag1": rng.uniform(-0.05, 0.05, n),
        "sample_entropy": rng.uniform(1.5, 2.0, n),
    })


def test_zscore_dataframe():
    df = _make_feature_df()
    scaled = zscore_dataframe(df)
    for col in df.select_dtypes(include=[np.number]).columns:
        assert abs(scaled[col].mean()) < 0.1


def test_score_windows():
    df = _make_feature_df()
    scored = score_windows(df, method="zscore")
    assert "anomaly_score" in scored.columns
    assert len(scored) == len(df)
    assert (scored["anomaly_score"] >= 0).all()


# ──────────────────────────────────────────────────────────────────────────────
# Latent tests
# ──────────────────────────────────────────────────────────────────────────────

def test_initialize_basin_attractors():
    attractors = initialize_basin_attractors(5, 4)
    assert len(attractors) == 5
    for a in attractors:
        assert a.shape == (4,)


def test_basin_similarities_shape():
    attractors = initialize_basin_attractors(5, 4, noise_scale=0.0)
    state = np.array([1.0, 0.0, 0.0, 0.0])
    sims = basin_similarities(state, attractors)
    assert sims.shape == (5,)
    assert np.all(sims >= -1.0) and np.all(sims <= 1.0)


def test_attractor_pull_converges():
    rng = np.random.default_rng(7)
    state = rng.standard_normal(4)
    target = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(500):
        state = attractor_pull(state, target, learning_rate=0.1, noise_level=0.0)
    assert np.linalg.norm(state - target) < 0.5


def test_basin_switch_event():
    attractors = initialize_basin_attractors(5, 4)
    state = attractors[2].copy()
    result = basin_switch_event(state, attractors, previous_index=None)
    assert 0 <= result.chosen_index < 5
    assert isinstance(result.switched, bool)


def test_feature_row_to_drives():
    features = {
        "bias": 0.5,
        "runs_count": 200.0,
        "longest_run": 8.0,
        "autocorr_lag1": -0.01,
        "spectral_entropy": 0.98,
    }
    drives = feature_row_to_drives(features)
    for name, drive in drives.items():
        assert drive.shape == (4,)


def test_run_coordinator():
    rng = np.random.default_rng(0)
    n_steps = 10
    drive_sequence = [
        {"perception": rng.standard_normal(4), "planning": rng.standard_normal(4)}
        for _ in range(n_steps)
    ]
    df = run_coordinator(drive_sequence, n_basins=3, n_dims=4, random_seed=0)
    assert len(df) == n_steps
    assert "coord_0" in df.columns
    assert "chosen_basin" in df.columns
