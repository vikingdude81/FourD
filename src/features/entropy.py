"""
Entropy-based features for a bitstream window.
"""

from __future__ import annotations

import math

import numpy as np


def spectral_entropy(window: np.ndarray) -> float:
    """
    Normalized spectral entropy of the bitstream.

    Computes the power spectral density via FFT and then calculates the
    Shannon entropy of the normalized PSD.
    """
    w = window.astype(float)
    n = len(w)
    if n == 0:
        return 0.0
    fft_vals = np.fft.rfft(w - w.mean())
    psd = np.abs(fft_vals) ** 2
    psd_sum = psd.sum()
    if psd_sum < 1e-12:
        return 0.0
    p = psd / psd_sum
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(len(p) + 1))


def permutation_entropy(window: np.ndarray, order: int = 3) -> float:
    """
    Normalized permutation entropy of order *order*.

    Parameters
    ----------
    window:
        1-D bit array.
    order:
        Embedding dimension (permutation length).  Default is 3.
    """
    n = len(window)
    if n < order:
        return 0.0
    w = window.astype(float)

    # Count ordinal patterns
    counts: dict = {}
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(w[i : i + order]))
        counts[pattern] = counts.get(pattern, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    probs = np.array(list(counts.values()), dtype=float) / total
    probs = probs[probs > 0]
    max_entropy = np.log2(math.factorial(order))
    if max_entropy < 1e-12:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)) / max_entropy)


def sample_entropy(window: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """
    Approximate sample entropy of the bitstream.

    Parameters
    ----------
    window:
        1-D bit array.
    m:
        Template length.
    r:
        Tolerance.  Defaults to ``0.2 * std(window)``.
    """
    w = window.astype(float)
    n = len(w)
    if n < m + 2:
        return 0.0
    if r is None:
        std = float(np.std(w))
        r = 0.2 * std if std > 0 else 0.1

    def _count_matches(length: int) -> int:
        count = 0
        for i in range(n - length):
            template = w[i : i + length]
            for j in range(i + 1, n - length + 1):
                if np.max(np.abs(template - w[j : j + length])) < r:
                    count += 1
        return count

    a = _count_matches(m + 1)
    b = _count_matches(m)
    if b == 0:
        return 0.0
    return float(-np.log(a / b)) if a > 0 else 0.0


def compute_entropy_features(window: np.ndarray) -> dict:
    """
    Return a dict with all entropy features for one window.

    Keys: ``spectral_entropy``, ``permutation_entropy``, ``sample_entropy``.
    """
    return {
        "spectral_entropy": spectral_entropy(window),
        "permutation_entropy": permutation_entropy(window),
        "sample_entropy": sample_entropy(window),
    }
