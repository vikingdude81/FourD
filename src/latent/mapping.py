"""
Latent subsystem mapping.

Maps feature windows produced by the QRNG analysis pipeline into per-subsystem
drive vectors that feed the 4D latent coordinator.

Subsystem architecture:
- perception: Basic statistics (bias, runs, alternation rate)
- planning: Autocorrelation and predictability features  
- emotion: Entropy measures (spectral, permutation, sample entropy)
- memory: Complexity metrics (Lempel-Ziv, block entropy)
- attention: Change-point detection signals
- executive: Anomaly scores and composite metrics
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Default feature → subsystem weights (expanded)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SUBSYSTEM_WEIGHTS: Dict[str, Dict[str, float]] = {
    "perception": {
        # Basic statistics - raw bit properties
        "bias": 0.35,
        "runs_count": 0.25,
        "longest_run": 0.25,
        "alternating_rate": 0.15,
    },
    "planning": {
        # Autocorrelation - predictability and temporal structure
        "autocorr_lag1": 0.4,
        "autocorr_lag3": 0.3,
        "max_autocorrelation": 0.3,
    },
    "emotion": {
        # Entropy measures - uncertainty and randomness
        "permutation_entropy": 0.35,
        "spectral_entropy": 0.30,
        "sample_entropy": 0.20,
        "shannon_entropy": 0.15,
    },
    "memory": {
        # Complexity - algorithmic structure and compressibility
        "lz_complexity": 0.40,
        "block_entropy_4": 0.30,
        "run_length_variance": 0.20,
        "ks_entropy_estimate": 0.10,
    },
    "attention": {
        # Change-point detection - structural breaks and regime shifts
        "cusum_detected": 0.40,
        "moving_variance_ratio": 0.30,
        "sliding_mean_deviation": 0.20,
        "bayesian_segmentation_score": 0.10,
    },
    "executive": {
        # Composite/anomaly metrics - overall quality assessment
        "autocorr_lag1": 0.30,  # Also feeds into planning
        "min_entropy": 0.35,
        "sum_abs_autocorrelations": 0.20,
        "entropy_rate_change": 0.15,
    },
}

# Dimensionality of the output drive vectors (must match the coordinator)
DRIVE_DIMS = 4


def feature_row_to_drives(
    features: Dict[str, float],
    subsystem_weights: Optional[Dict[str, Dict[str, float]]] = None,
    dims: int = DRIVE_DIMS,
) -> Dict[str, np.ndarray]:
    """
    Convert a single feature window into per-subsystem drive vectors.

    Parameters
    ----------
    features:
        Dictionary mapping feature name to scalar value (one window).
    subsystem_weights:
        Mapping of subsystem name → {feature_name: weight}.  Uses
        :data:`DEFAULT_SUBSYSTEM_WEIGHTS` when ``None``.
    dims:
        Dimensionality of each drive vector.

    Returns
    -------
    Dict mapping subsystem name to a drive vector of shape ``(dims,)``.
    """
    if subsystem_weights is None:
        subsystem_weights = DEFAULT_SUBSYSTEM_WEIGHTS

    drives: Dict[str, np.ndarray] = {}
    for subsystem, weights in subsystem_weights.items():
        drive = np.zeros(dims, dtype=float)
        for feat, w in weights.items():
            val = features.get(feat, 0.0)
            if not np.isfinite(val):
                val = 0.0
            # Distribute the scalar influence across dimensions using the
            # subsystem index as a phase offset so each subsystem projects
            # onto a distinct region of the latent space.
            idx = list(subsystem_weights.keys()).index(subsystem)
            for d in range(dims):
                phase = 2.0 * np.pi * (d + idx) / dims
                drive[d] += w * val * np.cos(phase)
        drives[subsystem] = drive
    return drives


def dataframe_to_drives(
    df: pd.DataFrame,
    subsystem_weights: Dict[str, Dict[str, float]] | None = None,
    dims: int = DRIVE_DIMS,
) -> List[Dict[str, np.ndarray]]:
    """
    Convert a feature DataFrame (one row per window) into a list of
    subsystem drive dicts, one entry per row.

    Parameters
    ----------
    df:
        DataFrame where each row is a feature window.  Column names must
        include the feature names referenced by *subsystem_weights*.
    subsystem_weights:
        Same as :func:`feature_row_to_drives`.
    dims:
        Dimensionality of each drive vector.

    Returns
    -------
    List of dicts (one per row), each mapping subsystem name → drive vector.
    """
    result: List[Dict[str, np.ndarray]] = []
    for _, row in df.iterrows():
        features = row.to_dict()
        result.append(feature_row_to_drives(features, subsystem_weights, dims))
    return result
