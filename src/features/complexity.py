"""
Complexity features for a bitstream window.
"""

from __future__ import annotations

import numpy as np


def lz_complexity(window: np.ndarray) -> float:
    """
    Normalized Lempel-Ziv complexity of a binary sequence.

    Returns a value in ``[0, 1]`` where 1 indicates maximum complexity.
    """
    bits = window.astype(int)
    n = len(bits)
    if n == 0:
        return 0.0

    sequence = "".join(map(str, bits))

    # Lempel-Ziv 1976 phrase count
    c = _lz76(sequence)

    # Normalise against the theoretical maximum for a random binary sequence
    if n <= 1:
        return 0.0
    max_c = n / np.log2(n + 1)
    return float(min(c / max_c, 1.0))


def _lz76(sequence: str) -> int:
    """Count LZ76 phrases for a binary string."""
    n = len(sequence)
    if n == 0:
        return 0
    c = 1
    i = 0
    l = 1
    while i + l <= n:
        substring = sequence[i : i + l]
        if substring[:-1] in sequence[:i] or i == 0:
            l += 1
        else:
            c += 1
            i += l
            l = 1
        if i + l > n:
            if i < n:
                c += 1
            break
    return c


def compute_complexity_features(window: np.ndarray) -> dict:
    """
    Return a dict with all complexity features for one window.

    Keys: ``lz_complexity``.
    """
    return {
        "lz_complexity": lz_complexity(window),
    }
