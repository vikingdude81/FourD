"""
Anomaly scoring for QRNG bitstream analysis.

Provides methods for computing anomaly scores by comparing features
against null distributions (PRNG, shuffled data, etc.).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


class NullDistribution:
    """Stores statistics of a null distribution for comparison."""
    
    def __init__(self, name: str):
        self.name = name
        self.mean: Dict[str, float] = {}
        self.std: Dict[str, float] = {}
        self.min_val: Dict[str, float] = {}
        self.max_val: Dict[str, float] = {}
        self.count: int = 0
    
    def update(self, feature_values: Dict[str, float]) -> None:
        """Update the null distribution with new feature values."""
        for feat_name, value in feature_values.items():
            if np.isfinite(value):
                if feat_name not in self.mean:
                    self.mean[feat_name] = 0.0
                    self.std[feat_name] = 0.0
                    self.min_val[feat_name] = float('inf')
                    self.max_val[feat_name] = float('-inf')
                
                # Online mean/std computation (Welford's algorithm)
                n = self.count + 1
                delta = value - self.mean[feat_name]
                self.mean[feat_name] += delta / n
                delta2 = value - self.mean[feat_name]
                self.std[feat_name] += delta * delta2
                
                self.min_val[feat_name] = min(self.min_val[feat_name], value)
                self.max_val[feat_name] = max(self.max_val[feat_name], value)
        
        self.count += 1
    
    def finalize(self) -> None:
        """Finalize the standard deviation computation."""
        if self.count > 1:
            for feat_name in self.mean:
                self.std[feat_name] = np.sqrt(self.std[feat_name] / (self.count - 1))
    
    def z_score(self, feature_values: Dict[str, float]) -> Dict[str, float]:
        """Compute z-scores for a set of features against this null distribution."""
        z_scores = {}
        for feat_name, value in feature_values.items():
            if not np.isfinite(value):
                z_scores[feat_name] = 0.0
                continue
            
            std = self.std.get(feat_name, 1.0)
            if std < 1e-10:
                std = 1.0
            
            z_scores[feat_name] = float((value - self.mean[feat_name]) / std)
        
        return z_scores
    
    def percentile(self, feature_values: Dict[str, float], p: float = 50.0) -> Dict[str, float]:
        """Estimate percentiles for features (requires storing raw values)."""
        # This would require storing all values - simplified version returns mean
        return {k: self.mean.get(k, 0.0) for k in feature_values.keys()}


class AnomalyScorer:
    """Computes anomaly scores by comparing features against null distributions."""
    
    def __init__(self):
        # Store null distributions for different control types
        self.null_distributions: Dict[str, NullDistribution] = {}
        
        # Feature weights for composite scoring
        self.feature_weights: Dict[str, float] = {
            "bias": 1.0,
            "runs_count": 1.0,
            "longest_run": 1.5,  # Weight longest run higher (more sensitive to structure)
            "autocorr_lag1": 2.0,  # Autocorrelation is key for randomness
            "spectral_entropy": 1.5,
            "permutation_entropy": 1.5,
            "sample_entropy": 1.5,
            "lz_complexity": 1.5,
        }
    
    def add_null_distribution(self, name: str) -> NullDistribution:
        """Add a new null distribution for comparison."""
        if name not in self.null_distributions:
            self.null_distributions[name] = NullDistribution(name)
        return self.null_distributions[name]
    
    def update_nulls(self, feature_values: Dict[str, float], 
                     control_type: str = "prng") -> None:
        """Update null distributions with new feature values."""
        if control_type not in self.null_distributions:
            self.add_null_distribution(control_type)
        
        self.null_distributions[control_type].update(feature_values)
    
    def finalize_all_nulls(self) -> None:
        """Finalize all null distribution computations."""
        for dist in self.null_distributions.values():
            dist.finalize()
    
    def compute_z_scores(self, feature_values: Dict[str, float], 
                         control_type: str = "prng") -> Dict[str, float]:
        """Compute z-scores against a specific null distribution."""
        if control_type not in self.null_distributions:
            raise ValueError(f"No null distribution found for '{control_type}'")
        
        return self.null_distributions[control_type].z_score(feature_values)
    
    def compute_composite_anomaly_score(self, feature_values: Dict[str, float],
                                        control_type: str = "prng",
                                        method: str = "mahalanobis") -> float:
        """
        Compute a composite anomaly score.
        
        Parameters
        ----------
        feature_values : dict
            Dictionary of feature values to score.
        control_type : str
            Which null distribution to use for comparison.
        method : str
            Scoring method: "mahalanobis", "sum_abs_z", or "max_z".
            
        Returns
        -------
        float
            Composite anomaly score (higher = more anomalous).
        """
        z_scores = self.compute_z_scores(feature_values, control_type)
        
        if method == "mahalanobis":
            # Sum of weighted absolute z-scores (simplified Mahalanobis)
            total_score = 0.0
            for feat_name, z in z_scores.items():
                weight = self.feature_weights.get(feat_name, 1.0)
                total_score += weight * abs(z)
            
            # Normalize by number of features
            return float(total_score / max(1, len(feature_values)))
        
        elif method == "sum_abs_z":
            # Simple sum of absolute z-scores
            return float(np.sum([abs(z) for z in z_scores.values()]))
        
        elif method == "max_z":
            # Maximum absolute z-score (most extreme feature)
            return float(max(abs(z) for z in z_scores.values())) if z_scores else 0.0
        
        else:
            raise ValueError(f"Unknown scoring method: {method}")
    
    def compute_anomaly_time_series(self, 
                                    feature_sequences: List[Dict[str, float]],
                                    control_type: str = "prng",
                                    method: str = "mahalanobis") -> np.ndarray:
        """
        Compute anomaly scores for a sequence of feature windows.
        
        Parameters
        ----------
        feature_sequences : list
            List of feature dictionaries (one per window).
        control_type : str
            Which null distribution to use.
        method : str
            Scoring method.
            
        Returns
        -------
        np.ndarray
            Array of anomaly scores, one per window.
        """
        scores = []
        for features in feature_sequences:
            score = self.compute_composite_anomaly_score(
                features, control_type=control_type, method=method
            )
            scores.append(score)
        
        return np.array(scores)


def compute_min_entropy(bits: np.ndarray, block_size: int = 8) -> float:
    """
    Estimate min-entropy from observed block frequencies.
    
    Min-entropy is the worst-case entropy, based on the most probable outcome.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    block_size : int
        Size of blocks to analyze.
        
    Returns
    -------
    float
        Estimated min-entropy in bits per bit.
    """
    if len(bits) < block_size:
        return 1.0
    
    # Count block frequencies
    from collections import Counter
    blocks = []
    
    for i in range(0, len(bits) - block_size + 1, block_size):
        block = tuple(bits[i:i+block_size])
        blocks.append(block)
    
    if not blocks:
        return 1.0
    
    counts = Counter(blocks)
    total = len(blocks)
    
    # Find most frequent block
    max_count = max(counts.values())
    p_max = max_count / total
    
    # Min-entropy = -log2(p_max)
    if p_max <= 0:
        return float(block_size)
    
    min_entropy = -np.log2(p_max)
    
    # Normalize by block size to get bits per bit
    return float(min_entropy / block_size)


def compute_sample_entropy_fast(bits: np.ndarray, m: int = 2, 
                                tolerance_factor: float = 0.2) -> float:
    """
    Fast approximation of sample entropy using numpy operations.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    m : int
        Template length.
    tolerance_factor : float
        Tolerance as fraction of std.
        
    Returns
    -------
    float
        Sample entropy value.
    """
    if len(bits) < m + 5:
        return 0.0
    
    signal = bits.astype(float)
    n = len(signal)
    
    # Convert to bipolar for distance computation
    bipolar = 2 * signal - 1
    
    std = np.std(bipolar)
    if std < 1e-10:
        return 0.0
    
    r = tolerance_factor * std
    
    def count_matches_fast(seq, m_val):
        """Count matches using vectorized operations."""
        n_m = n - m_val
        
        # Create matrix of all pairs
        X = np.array([seq[i:i+m_val] for i in range(n_m)])
        
        # Compute pairwise distances
        diff = X[:, None, :] - X[None, :, :]
        max_diff = np.max(np.abs(diff), axis=2)
        
        matches = (max_diff <= r).sum(axis=1) - 1  # Exclude self-matches
        
        return matches
    
    B_matches = count_matches_fast(bipolar, m)
    A_matches = count_matches_fast(bipolar, m + 1)
    
    B = np.mean(B_matches > 0) if n_m > 1 else 0.0
    A = np.mean(A_matches > 0) if n_m > 2 else 0.0
    
    if B <= 0 or A <= 0:
        return float('inf')
    
    return float(-np.log(A / B))


def compute_feature_vector(bits: np.ndarray, 
                           block_size: int = 8,
                           m: int = 2) -> dict:
    """
    Compute all anomaly-relevant features for a bit window.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    block_size : int
        Block size for min-entropy computation.
    m : int
        Template length for sample entropy.
        
    Returns
    -------
    dict
        Dictionary of feature names to computed values.
    """
    return {
        "bias": float(np.mean(bits)),
        "runs_count": int(np.sum(np.abs(np.diff(bits))) + 1),
        "longest_run": compute_longest_run(bits),
        "autocorr_lag1": autocorr_lag1(bits),
        "spectral_entropy": spectral_entropy(bits),
        "permutation_entropy": permutation_entropy(bits),
        "sample_entropy_fast": sample_entropy_fast(bits, m=m),
        "lz_complexity": lempel_ziv_complexity(bits),
        "min_entropy": min_entropy(bits, block_size=block_size),
    }


# Import helper functions from other modules
from src.features.basic_stats import compute_longest_run
from src.features.autocorr import autocorr_lag1
from src.features.entropy import spectral_entropy, permutation_entropy
from src.features.complexity import lempel_ziv_complexity