"""
Change-point detection features for QRNG bitstream analysis.

Computes measures of structural breaks and regime shifts:
- Cumulative sum (CUSUM) based change points
- Bayesian information criterion (BIC) based segmentation
- Moving window variance ratio
"""

from __future__ import annotations

import numpy as np


def cusum_change_point(bits: np.ndarray, threshold: float = 3.0) -> int:
    """
    Detect the most significant change point using CUSUM method.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    threshold : float
        Threshold for detecting change points (in standard deviations).
        
    Returns
    -------
    int
        Index of the most significant change point, or -1 if none detected.
    """
    if len(bits) < 10:
        return -1
    
    # Convert to bipolar (-1, 1)
    signal = 2 * bits.astype(float) - 1
    
    n = len(signal)
    
    # Compute cumulative sum deviations from mean
    cumsum = np.cumsum(signal - np.mean(signal))
    
    # Find the point with maximum deviation
    max_idx = np.argmax(np.abs(cumsum))
    max_deviation = abs(cumsum[max_idx])
    
    # Normalize by standard deviation and sample size
    std = np.std(signal)
    if std < 1e-10:
        return -1
    
    normalized_deviation = max_deviation / (std * np.sqrt(n))
    
    if normalized_deviation > threshold:
        return int(max_idx)
    
    return -1


def moving_variance_ratio(bits: np.ndarray, window_size: int = 64) -> float:
    """
    Compute the maximum variance ratio across sliding windows.
    
    High variance ratios indicate potential change points or regime shifts.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    window_size : int
        Size of sliding windows for variance computation.
        
    Returns
    -------
    float
        Maximum ratio of adjacent window variances.
    """
    if len(bits) < 2 * window_size:
        return 1.0
    
    # Convert to bipolar
    signal = bits.astype(float)
    
    max_ratio = 1.0
    
    for i in range(0, len(signal) - 2 * window_size + 1, window_size // 2):
        window1 = signal[i:i + window_size]
        window2 = signal[i + window_size:i + 2 * window_size]
        
        var1 = np.var(window1)
        var2 = np.var(window2)
        
        if var1 > 1e-10 and var2 > 1e-10:
            ratio = max(var1 / var2, var2 / var1)
            max_ratio = max(max_ratio, ratio)
    
    return float(max_ratio)


def sliding_mean_deviation(bits: np.ndarray, window_size: int = 64) -> float:
    """
    Compute the maximum deviation of local means from global mean.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    window_size : int
        Size of sliding windows for mean computation.
        
    Returns
    -------
    float
        Maximum absolute deviation of local means, normalized by global std.
    """
    if len(bits) < 2 * window_size:
        return 0.0
    
    # Convert to bipolar
    signal = bits.astype(float)
    
    global_mean = np.mean(signal)
    global_std = np.std(signal)
    
    if global_std < 1e-10:
        return 0.0
    
    max_deviation = 0.0
    
    for i in range(0, len(signal) - window_size + 1, window_size // 2):
        local_mean = np.mean(signal[i:i + window_size])
        deviation = abs(local_mean - global_mean) / global_std
        max_deviation = max(max_deviation, deviation)
    
    return float(max_deviation)


def run_length_change(bits: np.ndarray, window_size: int = 64) -> dict:
    """
    Analyze changes in run length statistics across windows.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    window_size : int
        Size of sliding windows for analysis.
        
    Returns
    -------
    dict
        Dictionary containing change point metrics.
    """
    if len(bits) < 2 * window_size:
        return {
            "max_run_length_ratio": 1.0,
            "mean_run_length_change": 0.0,
            "run_length_variance_ratio": 1.0,
        }
    
    # Compute run lengths for first and second half
    def compute_run_stats(window):
        if len(window) < 2:
            return {"mean": np.mean(window), "var": np.var(window)}
        
        runs = []
        current_run = 1
        
        for i in range(1, len(window)):
            if window[i] == window[i - 1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        return {"mean": np.mean(runs), "var": np.var(runs)}
    
    first_half = compute_run_stats(bits[:window_size])
    second_half = compute_run_stats(bits[-window_size:])
    
    # Compute ratios and differences
    run_length_ratio = max(first_half["mean"] / (second_half["mean"] + 1e-10), 
                           second_half["mean"] / (first_half["mean"] + 1e-10))
    
    mean_change = abs(first_half["mean"] - second_half["mean"]) / (np.mean([first_half["mean"], second_half["mean"]]) + 1e-10)
    
    var_ratio = max(first_half["var"] / (second_half["var"] + 1e-10), 
                    second_half["var"] / (first_half["var"] + 1e-10))
    
    return {
        "max_run_length_ratio": float(run_length_ratio),
        "mean_run_length_change": float(mean_change),
        "run_length_variance_ratio": float(var_ratio),
    }


def bayesian_segmentation_score(bits: np.ndarray, max_segments: int = 3) -> float:
    """
    Compute a BIC-based score for segmenting the bit stream.
    
    Higher scores indicate more evidence for structural breaks.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    max_segments : int
        Maximum number of segments to consider.
        
    Returns
    -------
    float
        BIC improvement score for optimal segmentation.
    """
    if len(bits) < 20:
        return 0.0
    
    # Convert to bipolar
    signal = bits.astype(float)
    
    n = len(signal)
    
    # Compute global variance (null model)
    global_var = np.var(signal)
    log_likelihood_null = -n / 2 * np.log(global_var + 1e-10)
    
    best_bic_improvement = 0.0
    
    # Try different segmentation points
    for k in range(1, min(max_segments, n // 5)):
        # Simple grid search over possible segment boundaries
        step = n // (k + 1)
        
        total_var = 0
        params = 0
        
        for seg in range(k + 1):
            start = seg * step
            end = min((seg + 1) * step, n) if seg < k else n
            
            segment = signal[start:end]
            seg_var = np.var(segment)
            total_var += len(segment) * seg_var
            params += 2  # mean and variance per segment
        
        log_likelihood_seg = -n / 2 * np.log((total_var + 1e-10) / n)
        
        # BIC = -2*logL + k*log(n), we want improvement over null
        bic_null = -2 * log_likelihood_null + 2 * np.log(n)
        bic_seg = -2 * log_likelihood_seg + params * np.log(n)
        
        improvement = (bic_null - bic_seg) / n
        
        best_bic_improvement = max(best_bic_improvement, float(improvement))
    
    return best_bic_improvement


def entropy_rate_change(bits: np.ndarray, window_size: int = 64) -> dict:
    """
    Detect changes in local entropy rate.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    window_size : int
        Size of sliding windows for entropy computation.
        
    Returns
    -------
    dict
        Dictionary containing entropy change metrics.
    """
    if len(bits) < 2 * window_size:
        return {
            "max_entropy_change": 0.0,
            "entropy_variance_ratio": 1.0,
        }
    
    def compute_block_entropy(window, block_size=4):
        """Simple block entropy computation."""
        if len(window) < block_size:
            return 1.0
        
        # Count block frequencies
        from collections import Counter
        blocks = []
        for i in range(len(window) - block_size + 1):
            block = tuple(window[i:i+block_size])
            blocks.append(block)
        
        counts = Counter(blocks)
        total = len(blocks)
        
        probs = np.array(list(counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize by max possible entropy
        max_entropy = np.log2(2 ** block_size)
        
        return float(entropy / max_entropy) if max_entropy > 0 else 1.0
    
    first_half_entropy = compute_block_entropy(bits[:window_size])
    second_half_entropy = compute_block_entropy(bits[-window_size:])
    
    entropy_change = abs(first_half_entropy - second_half_entropy)
    entropy_ratio = max(first_half_entropy / (second_half_entropy + 1e-10), 
                        second_half_entropy / (first_half_entropy + 1e-10))
    
    return {
        "max_entropy_change": float(entropy_change),
        "entropy_variance_ratio": float(entropy_ratio),
    }


def compute_feature_vector(bits: np.ndarray, window_size: int = 64) -> dict:
    """
    Compute all change-point-based features for a bit window.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    window_size : int
        Window size for sliding computations.
        
    Returns
    -------
    dict
        Dictionary of feature names to computed values.
    """
    cp = cusum_change_point(bits)
    rl_stats = run_length_change(bits, window_size=window_size)
    entropy_stats = entropy_rate_change(bits, window_size=window_size)
    
    return {
        "cusum_change_point": float(cp),  # -1 if no change point detected
        "cusum_detected": bool(cp >= 0),
        "moving_variance_ratio": moving_variance_ratio(bits, window_size=window_size),
        "sliding_mean_deviation": sliding_mean_deviation(bits, window_size=window_size),
        "max_run_length_ratio": rl_stats["max_run_length_ratio"],
        "mean_run_length_change": rl_stats["mean_run_length_change"],
        "run_length_variance_ratio": rl_stats["run_length_variance_ratio"],
        "bayesian_segmentation_score": bayesian_segmentation_score(bits),
        "entropy_rate_change": entropy_stats["max_entropy_change"],
    }