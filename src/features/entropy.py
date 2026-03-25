"""
Entropy-based features for QRNG bitstream analysis.

Computes various entropy measures to quantify randomness and unpredictability:
- Spectral entropy (frequency domain)
- Permutation entropy (ordinal patterns)
- Sample entropy (regularity measure)
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft, rfft


def spectral_entropy(bits: np.ndarray, normalize: bool = True) -> float:
    """
    Compute the spectral entropy of a bit stream.
    
    Spectral entropy measures the uniformity of power distribution across
    frequency components. A perfectly random signal has high spectral entropy.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1). Can be converted to bipolar (-1, 1) internally.
    normalize : bool
        If True, return normalized entropy in [0, 1].
        
    Returns
    -------
    float
        Spectral entropy value. Higher = more random/uniform spectrum.
    """
    # Convert to bipolar for FFT analysis
    signal = 2 * bits.astype(float) - 1
    
    # Compute FFT
    n = len(signal)
    fft_result = rfft(signal)
    
    # Compute power spectrum
    power = np.abs(fft_result) ** 2
    
    # Avoid zeros in power spectrum
    power = power + 1e-10
    
    # Normalize to get probability distribution
    if normalize:
        prob = power / np.sum(power)
    else:
        prob = power
    
    # Compute entropy
    entropy = -np.sum(prob * np.log2(prob))
    
    # Normalize by maximum possible entropy (log2 of number of frequency bins)
    max_entropy = np.log2(len(prob)) if len(prob) > 1 else 1.0
    
    return float(entropy / max_entropy) if normalize else float(entropy)


def permutation_entropy(bits: np.ndarray, embedding_dim: int = 3, delay: int = 1) -> float:
    """
    Compute the permutation entropy of a bit stream.
    
    Permutation entropy analyzes the order relations between values in
    time series segments. It's robust to noise and captures structural patterns.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    embedding_dim : int
        Dimension of the phase space (number of points per pattern).
    delay : int
        Time delay between consecutive elements in each pattern.
        
    Returns
    -------
    float
        Permutation entropy normalized to [0, 1]. Higher = more random.
    """
    if len(bits) < embedding_dim:
        return 1.0
    
    # Convert to float for ordinal analysis
    signal = bits.astype(float)
    
    # Create embedded vectors and compute their ordinal patterns
    n_patterns = len(signal) - (embedding_dim - 1) * delay
    if n_patterns <= 0:
        return 1.0
    
    patterns = []
    for i in range(n_patterns):
        segment = signal[i:i + embedding_dim * delay:delay]
        # Get the rank order pattern (permutation of indices sorted by value)
        pattern = tuple(np.argsort(segment))
        patterns.append(pattern)
    
    # Count frequency of each pattern
    pattern_counts = {}
    for p in patterns:
        pattern_counts[p] = pattern_counts.get(p, 0) + 1
    
    # Compute probability distribution
    total = len(patterns)
    probs = np.array(list(pattern_counts.values())) / total
    
    # Compute entropy
    entropy = -np.sum(probs * np.log2(probs))
    
    # Normalize by maximum possible entropy (log2 of m! permutations)
    max_entropy = np.log2(np.math.factorial(embedding_dim)) if embedding_dim > 1 else 1.0
    
    return float(entropy / max_entropy) if max_entropy > 0 else 1.0


def sample_entropy(bits: np.ndarray, m: int = 2, tolerance_factor: float = 0.2) -> float:
    """
    Compute the sample entropy of a bit stream.
    
    Sample entropy measures the likelihood that similar patterns remain
    similar when extended by one point. Lower values indicate more regularity.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    m : int
        Length of compared run lengths (template length).
    tolerance_factor : float
        Tolerance as a fraction of the standard deviation.
        
    Returns
    -------
    float
        Sample entropy value. Lower = more regular/predictable.
    """
    if len(bits) < m + 1:
        return float('inf')
    
    # Convert to float
    signal = bits.astype(float)
    n = len(signal)
    
    # Compute standard deviation for tolerance
    std = np.std(signal)
    if std < 1e-10:
        return 0.0
    
    r = tolerance_factor * std
    
    def _count_matches(seq, m_val):
        """Count pairs of vectors within tolerance r."""
        count = 0
        for i in range(n - m_val):
            for j in range(i + 1, n - m_val):
                # Check if all elements are within tolerance
                if np.all(np.abs(seq[i:i+m_val] - seq[j:j+m_val]) <= r):
                    count += 1
        return count
    
    # Count matches for length m and m+1
    B = _count_matches(signal, m) / ((n - m) * (n - m - 1))
    A = _count_matches(signal, m + 1) / ((n - m - 1) * (n - m - 2))
    
    if B <= 0 or A <= 0:
        return float('inf')
    
    # Sample entropy is -ln(A/B)
    return float(-np.log(A / B))


def approximate_entropy(bits: np.ndarray, m: int = 2, tolerance_factor: float = 0.2) -> float:
    """
    Compute the approximate entropy of a bit stream.
    
    Similar to sample entropy but uses self-matches (including i=j).
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    m : int
        Length of compared run lengths.
    tolerance_factor : float
        Tolerance as a fraction of the standard deviation.
        
    Returns
    -------
    float
        Approximate entropy value.
    """
    if len(bits) < m + 1:
        return float('inf')
    
    signal = bits.astype(float)
    n = len(signal)
    
    std = np.std(signal)
    if std < 1e-10:
        return 0.0
    
    r = tolerance_factor * std
    
    def _count_matches(seq, m_val):
        """Count pairs of vectors within tolerance r (including self-matches)."""
        count = np.zeros(n - m_val)
        for i in range(n - m_val):
            for j in range(n - m_val):
                if np.all(np.abs(seq[i:i+m_val] - seq[j:j+m_val]) <= r):
                    count[i] += 1
        return count
    
    # Count matches for length m and m+1
    Cm = _count_matches(signal, m) / (n - m)
    Cm1 = _count_matches(signal, m + 1) / (n - m - 1)
    
    phi_m = np.mean(np.log(Cm))
    phi_m1 = np.mean(np.log(Cm1))
    
    return float(phi_m - phi_m1)


def shannon_entropy(bits: np.ndarray) -> float:
    """
    Compute the Shannon entropy of bit values.
    
    This is a simple measure of randomness based on the frequency
    distribution of 0s and 1s.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    float
        Shannon entropy in bits. Maximum is 1.0 for perfectly balanced bits.
    """
    n = len(bits)
    if n == 0:
        return 0.0
    
    # Count occurrences
    counts = np.bincount(bits, minlength=2)
    probs = counts / n
    
    # Compute entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return float(entropy)


def conditional_entropy(bits: np.ndarray, lag: int = 1) -> float:
    """
    Compute the conditional entropy H(X_t | X_{t-lag}).
    
    Measures how much knowing past bits reduces uncertainty about current bit.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    lag : int
        Lag for conditioning variable.
        
    Returns
    -------
    float
        Conditional entropy in bits.
    """
    if len(bits) <= lag:
        return 1.0
    
    # Count joint and marginal frequencies
    joint_counts = np.zeros((2, 2))
    
    for i in range(lag, len(bits)):
        prev_idx = int(bits[i - lag])
        curr_idx = int(bits[i])
        joint_counts[prev_idx, curr_idx] += 1
    
    # Marginal probabilities
    p_prev = joint_counts.sum(axis=1) / joint_counts.sum()
    p_curr = joint_counts.sum(axis=0) / joint_counts.sum()
    
    # Joint probability
    p_joint = joint_counts / joint_counts.sum()
    
    # Conditional entropy H(X_t | X_{t-lag}) = sum p(x,y) * log(p(y|x))
    cond_entropy = 0.0
    for x in range(2):
        for y in range(2):
            if p_joint[x, y] > 0 and p_prev[x] > 0:
                p_y_given_x = p_joint[x, y] / p_prev[x]
                cond_entropy -= p_joint[x, y] * np.log2(p_y_given_x)
    
    return float(cond_entropy)


def compute_feature_vector(bits: np.ndarray, m: int = 2, tolerance_factor: float = 0.2) -> dict:
    """
    Compute all entropy-based features for a bit window.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    m : int
        Parameter for sample/approximate entropy.
    tolerance_factor : float
        Tolerance factor for sample/approximate entropy.
        
    Returns
    -------
    dict
        Dictionary of feature names to computed values.
    """
    return {
        "spectral_entropy": spectral_entropy(bits),
        "permutation_entropy": permutation_entropy(bits),
        "sample_entropy": sample_entropy(bits, m=m, tolerance_factor=tolerance_factor),
        "approximate_entropy": approximate_entropy(bits, m=m, tolerance_factor=tolerance_factor),
        "shannon_entropy": shannon_entropy(bits),
        "conditional_entropy_lag1": conditional_entropy(bits, lag=1),
    }