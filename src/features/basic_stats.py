"""
Basic statistical features for QRNG bitstream analysis.

Computes fundamental statistics like bias, run counts, longest runs, etc.
"""

from __future__ import annotations

import numpy as np


def compute_bias(bits: np.ndarray) -> float:
    """
    Compute the proportion of 1s in the bit array (bias from 0.5).
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    float
        Bias value between 0 and 1, where 0.5 indicates perfect balance.
    """
    return float(np.mean(bits))


def compute_runs_count(bits: np.ndarray) -> int:
    """
    Count the number of runs (consecutive identical bits).
    
    A run is a maximal sequence of identical values. For example,
    [0, 0, 1, 1, 1, 0] has 3 runs: two 0s, three 1s, one 0.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    int
        Number of runs in the bit array.
    """
    if len(bits) < 2:
        return 1
    
    # Count transitions + 1 for the initial run
    transitions = np.sum(np.abs(np.diff(bits)))
    return int(transitions + 1)


def compute_longest_run(bits: np.ndarray) -> int:
    """
    Find the length of the longest consecutive sequence of identical bits.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    int
        Length of the longest run.
    """
    if len(bits) == 0:
        return 0
    
    max_run = 1
    current_run = 1
    
    for i in range(1, len(bits)):
        if bits[i] == bits[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    
    return int(max_run)


def compute_runs_distribution(bits: np.ndarray) -> dict:
    """
    Compute the distribution of run lengths.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    dict
        Mapping from run length to count of runs with that length.
    """
    if len(bits) < 2:
        return {1: 1}
    
    runs = {}
    current_run = 1
    
    for i in range(1, len(bits)):
        if bits[i] == bits[i - 1]:
            current_run += 1
        else:
            runs[current_run] = runs.get(current_run, 0) + 1
            current_run = 1
    
    # Don't forget the last run
    runs[current_run] = runs.get(current_run, 0) + 1
    
    return {k: int(v) for k, v in runs.items()}


def compute_zero_one_ratio(bits: np.ndarray) -> float:
    """
    Compute the ratio of zeros to ones.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    float
        Ratio of zeros to ones. Values > 1 indicate more zeros, < 1 indicate more ones.
    """
    n_zeros = np.sum(bits == 0)
    n_ones = np.sum(bits == 1)
    
    if n_ones == 0:
        return float('inf') if n_zeros > 0 else 1.0
    
    return float(n_zeros / n_ones)


def compute_alternating_rate(bits: np.ndarray) -> float:
    """
    Compute the rate of bit alternation (0→1 or 1→0 transitions).
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    float
        Fraction of positions where adjacent bits differ.
    """
    if len(bits) < 2:
        return 0.0
    
    transitions = np.sum(np.abs(np.diff(bits)))
    return float(transitions / (len(bits) - 1))


def compute_feature_vector(bits: np.ndarray) -> dict:
    """
    Compute all basic statistical features for a bit window.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    dict
        Dictionary of feature names to computed values.
    """
    return {
        "bias": compute_bias(bits),
        "runs_count": compute_runs_count(bits),
        "longest_run": compute_longest_run(bits),
        "zero_one_ratio": compute_zero_one_ratio(bits),
        "alternating_rate": compute_alternating_rate(bits),
    }


# Alias for backward compatibility
compute_basic_stats = compute_feature_vector