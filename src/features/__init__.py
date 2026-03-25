"""
Feature extraction modules for QRNG bitstream analysis.

This package provides comprehensive feature extraction capabilities:
- Basic statistics (bias, runs, etc.)
- Entropy measures (spectral, permutation, sample)
- Complexity metrics (Lempel-Ziv, block entropy)
- Autocorrelation analysis
- Change-point detection
"""

from __future__ import annotations

# Re-export key functions from submodules
from .basic_stats import (
    compute_bias,
    compute_runs_count,
    compute_longest_run,
    compute_zero_one_ratio,
    compute_alternating_rate,
    compute_feature_vector as basic_compute_feature_vector,
)

from .entropy import (
    spectral_entropy,
    permutation_entropy,
    sample_entropy,
    approximate_entropy,
    shannon_entropy,
    conditional_entropy,
    compute_feature_vector as entropy_compute_feature_vector,
)

from .complexity import (
    lempel_ziv_complexity,
    lempel_ziv_76_complexity,
    block_entropy,
    kolmogorov_sinai_estimate,
    run_length_variance,
    run_length_mean,
    compute_feature_vector as complexity_compute_feature_vector,
)

from .autocorr import (
    autocorr_lag1,
    autocorr_lagk,
    autocorr_decay_rate,
    spectral_density_estimate,
    compute_feature_vector as autocorr_compute_feature_vector,
)

from .changepoint import (
    cusum_change_point,
    moving_variance_ratio,
    sliding_mean_deviation,
    bayesian_segmentation_score,
    compute_feature_vector as changepoint_compute_feature_vector,
)

from .windows import (
    sliding_windows,
    window_metadata,
)


# Comprehensive feature extraction that combines all modules
def extract_all_features(bits: np.ndarray, 
                         block_size: int = 8,
                         m: int = 2,
                         max_lag: int = 20,
                         window_size: int = 64) -> dict:
    """
    Extract all available features from a bit window.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    block_size : int
        Block size for entropy/min-entropy computation.
    m : int
        Template length for sample/approximate entropy.
    max_lag : int
        Maximum lag for autocorrelation tests.
    window_size : int
        Window size for change-point detection.
        
    Returns
    -------
    dict
        Dictionary containing all computed features.
    """
    import numpy as np
    
    return {
        # Basic stats
        **basic_compute_feature_vector(bits),
        
        # Entropy features
        **entropy_compute_feature_vector(bits, m=m),
        
        # Complexity features
        **complexity_compute_feature_vector(bits, block_size=block_size),
        
        # Autocorrelation features
        **autocorr_compute_feature_vector(bits, max_lag=max_lag),
        
        # Change-point features
        **changepoint_compute_feature_vector(bits, window_size=window_size),
    }


__all__ = [
    # Basic stats
    "compute_bias",
    "compute_runs_count", 
    "compute_longest_run",
    "compute_zero_one_ratio",
    "compute_alternating_rate",
    "basic_compute_feature_vector",
    
    # Entropy
    "spectral_entropy",
    "permutation_entropy",
    "sample_entropy",
    "approximate_entropy",
    "shannon_entropy",
    "conditional_entropy",
    "entropy_compute_feature_vector",
    
    # Complexity
    "lempel_ziv_complexity",
    "lempel_ziv_76_complexity",
    "block_entropy",
    "kolmogorov_sinai_estimate",
    "run_length_variance",
    "run_length_mean",
    "complexity_compute_feature_vector",
    
    # Autocorrelation
    "autocorr_lag1",
    "autocorr_lagk",
    "autocorr_decay_rate",
    "spectral_density_estimate",
    "autocorr_compute_feature_vector",
    
    # Change-point detection
    "cusum_change_point",
    "moving_variance_ratio",
    "sliding_mean_deviation",
    "bayesian_segmentation_score",
    "changepoint_compute_feature_vector",
    
    # Windows
    "sliding_windows",
    "window_metadata",
    
    # Combined extraction
    "extract_all_features",
]