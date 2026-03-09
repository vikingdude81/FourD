"""
Basic statistical features for a bitstream window.
"""

from __future__ import annotations

import numpy as np


def bias(window: np.ndarray) -> float:
    """Mean bit value (proportion of 1s)."""
    return float(np.mean(window))


def runs_count(window: np.ndarray) -> int:
    """Number of consecutive runs of identical bits."""
    if len(window) == 0:
        return 0
    changes = np.count_nonzero(np.diff(window.astype(int)))
    return int(changes + 1)


def longest_run(window: np.ndarray) -> int:
    """Length of the longest consecutive run of a single bit value."""
    if len(window) == 0:
        return 0
    max_run = 1
    current_run = 1
    bits = window.astype(int)
    for i in range(1, len(bits)):
        if bits[i] == bits[i - 1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    return max_run


def compute_basic_stats(window: np.ndarray) -> dict:
    """
    Return a dict with all basic statistics for one window.

    Keys: ``bias``, ``runs_count``, ``longest_run``.
    """
    return {
        "bias": bias(window),
        "runs_count": runs_count(window),
        "longest_run": longest_run(window),
    }
