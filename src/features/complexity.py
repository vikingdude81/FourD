"""
Complexity-based features for QRNG bitstream analysis.

Computes measures of algorithmic complexity and structural patterns:
- Lempel-Ziv complexity
- Block complexity
- Kolmogorov-Sinai entropy estimate
"""

from __future__ import annotations

import numpy as np


def lempel_ziv_complexity(bits: np.ndarray) -> float:
    """
    Compute the Lempel-Ziv complexity of a bit stream.
    
    LZ complexity measures the number of distinct substrings encountered
    when scanning the sequence from left to right. Higher values indicate
    more randomness/complexity.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    float
        Normalized Lempel-Ziv complexity in [0, 1].
    """
    if len(bits) == 0:
        return 0.0
    
    # Convert to string for easier processing
    s = ''.join(str(b) for b in bits)
    n = len(s)
    
    if n <= 1:
        return 1.0
    
    c = 1  # Complexity counter
    i = 0
    
    while i < n - 1:
        c += 1
        k = i + 1
        l = i
        
        # Find the longest match
        while True:
            if s[l] == s[k]:
                l += 1
                k += 1
                if k >= n or l <= i:
                    break
            else:
                break
        
        i = l
    
    # Normalize by theoretical maximum (n / log2(n))
    if n <= 1:
        return 1.0
    
    max_c = n / np.log2(n) if n > 1 else 1.0
    normalized = c / max_c
    
    return float(min(1.0, normalized))


def lempel_ziv_76_complexity(bits: np.ndarray) -> float:
    """
    Compute Lempel-Ziv '76 complexity (original algorithm).
    
    This version uses a slightly different parsing strategy than the
    simplified LZ complexity above.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    float
        Normalized Lempel-Ziv '76 complexity in [0, 1].
    """
    if len(bits) == 0:
        return 0.0
    
    s = bits.tolist()
    n = len(s)
    
    if n <= 1:
        return 1.0
    
    c = 1
    k = 2
    
    while k <= n:
        found = False
        for j in range(1, c + 1):
            # Check if substring starting at position j matches prefix of length (k-j)
            match_len = min(k - j, c)
            if s[j:j+match_len] == s[k-match_len:k]:
                found = True
                break
        
        if not found:
            c += 1
        
        k += 1
    
    # Normalize
    max_c = n / np.log2(n) if n > 1 else 1.0
    return float(min(1.0, c / max_c))


def block_entropy(bits: np.ndarray, block_size: int = 4) -> float:
    """
    Compute the block entropy for a given block size.
    
    Block entropy measures the uncertainty in predicting the next bit
    given the previous (block_size - 1) bits.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    block_size : int
        Size of blocks to consider.
        
    Returns
    -------
    float
        Block entropy normalized by log2(block_size).
    """
    if len(bits) < block_size:
        return 1.0
    
    # Count block frequencies
    block_counts = {}
    
    for i in range(len(bits) - block_size + 1):
        block = tuple(bits[i:i+block_size])
        block_counts[block] = block_counts.get(block, 0) + 1
    
    total = sum(block_counts.values())
    
    # Compute entropy
    entropy = 0.0
    for count in block_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    # Normalize by maximum possible entropy (log2 of number of possible blocks)
    max_entropy = np.log2(2 ** block_size) if block_size > 0 else 1.0
    
    return float(entropy / max_entropy)


def kolmogorov_sinai_estimate(bits: np.ndarray, m: int = 3) -> float:
    """
    Estimate Kolmogorov-Sinai entropy using permutation patterns.
    
    KS entropy measures the rate of information production in a dynamical system.
    This is an approximation based on permutation entropy differences.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    m : int
        Embedding dimension for pattern analysis.
        
    Returns
    -------
    float
        KS entropy estimate in bits per symbol.
    """
    if len(bits) < m + 1:
        return 0.0
    
    # Compute permutation entropies for different embedding dimensions
    pe_m = _permutation_entropy_simple(bits, m)
    pe_m1 = _permutation_entropy_simple(bits, m - 1) if m > 1 else 1.0
    
    # KS entropy estimate is the difference in permutation entropies
    ks_estimate = max(0.0, pe_m - pe_m1)
    
    return float(ks_estimate)


def _permutation_entropy_simple(bits: np.ndarray, embedding_dim: int) -> float:
    """Simplified permutation entropy computation."""
    if len(bits) < embedding_dim:
        return 1.0
    
    signal = bits.astype(float)
    n_patterns = len(signal) - (embedding_dim - 1)
    
    patterns = []
    for i in range(n_patterns):
        segment = signal[i:i + embedding_dim]
        pattern = tuple(np.argsort(segment))
        patterns.append(pattern)
    
    # Count frequencies
    from collections import Counter
    counts = Counter(patterns)
    total = len(patterns)
    
    # Compute entropy
    probs = np.array(list(counts.values())) / total
    entropy = -np.sum(probs * np.log2(probs))
    
    # Normalize
    max_entropy = np.log2(np.math.factorial(embedding_dim)) if embedding_dim > 1 else 1.0
    
    return float(entropy / max_entropy) if max_entropy > 0 else 1.0


def run_length_variance(bits: np.ndarray) -> float:
    """
    Compute the variance of run lengths in a bit stream.
    
    Random sequences have exponentially distributed run lengths,
    while structured sequences show different patterns.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    float
        Variance of run lengths.
    """
    if len(bits) < 2:
        return 0.0
    
    # Compute run lengths
    runs = []
    current_run = 1
    
    for i in range(1, len(bits)):
        if bits[i] == bits[i - 1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    
    runs.append(current_run)  # Don't forget the last run
    
    return float(np.var(runs))


def run_length_mean(bits: np.ndarray) -> float:
    """
    Compute the mean run length in a bit stream.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    float
        Mean run length.
    """
    if len(bits) < 2:
        return 1.0
    
    runs = []
    current_run = 1
    
    for i in range(1, len(bits)):
        if bits[i] == bits[i - 1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    
    runs.append(current_run)
    
    return float(np.mean(runs))


def compressibility_ratio(bits: np.ndarray, compression_algo: str = 'lz77') -> float:
    """
    Estimate the compressibility of a bit stream.
    
    More random sequences are harder to compress. This provides an
    alternative measure of randomness based on actual compression.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    compression_algo : str
        Compression algorithm to use ('lz77', 'gzip', etc.).
        
    Returns
    -------
    float
        Ratio of compressed size to original size. Lower = more compressible.
    """
    import zlib
    
    # Convert bits to bytes for compression
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        if len(chunk) < 8:
            chunk = np.pad(chunk, (0, 8 - len(chunk)), constant_values=0)
        
        byte_val = 0
        for j, b in enumerate(chunk[:8]):
            byte_val |= int(b) << (7 - j)
        byte_array.append(byte_val)
    
    original_size = len(byte_array)
    
    # Compress using zlib (DEFLATE algorithm)
    compressed = zlib.compress(bytes(byte_array), level=9)
    compressed_size = len(compressed)
    
    return float(compressed_size / original_size)


def compute_feature_vector(bits: np.ndarray, block_size: int = 4) -> dict:
    """
    Compute all complexity-based features for a bit window.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    block_size : int
        Block size for block entropy computation.
        
    Returns
    -------
    dict
        Dictionary of feature names to computed values.
    """
    return {
        "lz_complexity": lempel_ziv_complexity(bits),
        "lz76_complexity": lempel_ziv_76_complexity(bits),
        "block_entropy_4": block_entropy(bits, block_size=4),
        "ks_entropy_estimate": kolmogorov_sinai_estimate(bits, m=3),
        "run_length_variance": run_length_variance(bits),
        "run_length_mean": run_length_mean(bits),
    }