"""
Change-point detection features for a bitstream window.

Identifies abrupt shifts in the local mean of sub-segments of a window,
providing a measure of non-stationarity.
"""

from __future__ import annotations

from typing import List

import numpy as np


def mean_shift_score(window: np.ndarray, n_segments: int = 4) -> float:
    """
    Score the magnitude of mean shifts across *n_segments* equal sub-segments.

    Returns the standard deviation of sub-segment means, normalised by the
    overall mean absolute deviation of the window.

    Parameters
    ----------
    window:
        1-D bit array.
    n_segments:
        Number of equal sub-segments to split the window into.
    """
    n = len(window)
    if n < n_segments:
        return 0.0

    w = window.astype(float)
    segment_size = n // n_segments
    means = [w[i * segment_size : (i + 1) * segment_size].mean() for i in range(n_segments)]
    shift = float(np.std(means))

    overall_mad = float(np.mean(np.abs(w - w.mean())))
    if overall_mad < 1e-12:
        return 0.0
    return shift / overall_mad


def cusum_range(window: np.ndarray) -> float:
    """
    Range of the CUSUM (cumulative sum of deviations from the mean).

    A large range indicates a sustained mean shift somewhere in the window.
    """
    w = window.astype(float)
    if len(w) == 0:
        return 0.0
    cusum = np.cumsum(w - w.mean())
    return float(cusum.max() - cusum.min())


def detect_changepoints(window: np.ndarray, n_segments: int = 4) -> List[int]:
    """
    Heuristic change-point locations (indices) within *window*.

    Splits the window into *n_segments* and returns the segment boundaries
    where the absolute mean difference exceeds one standard deviation of all
    segment means.

    Returns
    -------
    List of sample indices within the window where a change is detected.
    """
    n = len(window)
    if n < n_segments * 2:
        return []

    w = window.astype(float)
    segment_size = n // n_segments
    means = np.array([w[i * segment_size : (i + 1) * segment_size].mean() for i in range(n_segments)])
    threshold = float(np.std(means))
    changepoints = []
    for i in range(1, len(means)):
        if abs(means[i] - means[i - 1]) > threshold:
            changepoints.append(i * segment_size)
    return changepoints


def compute_changepoint_features(window: np.ndarray) -> dict:
    """
    Return a dict with change-point features for one window.

    Keys: ``mean_shift_score``, ``cusum_range``, ``n_changepoints``.
    """
    cps = detect_changepoints(window)
    return {
        "mean_shift_score": mean_shift_score(window),
        "cusum_range": cusum_range(window),
        "n_changepoints": len(cps),
    }
