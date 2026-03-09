"""
Time-delay embedding of a 1-D time series.

Used to prepare anomaly-score trajectories for recurrence analysis.
"""

from __future__ import annotations

import numpy as np


def time_delay_embedding(
    signal: np.ndarray,
    dimension: int = 3,
    delay: int = 1,
) -> np.ndarray:
    """
    Construct a Takens time-delay embedding of a 1-D signal.

    Parameters
    ----------
    signal:
        1-D array of scalar values (e.g., anomaly scores over time).
    dimension:
        Embedding dimension (number of coordinates per point).
    delay:
        Time delay in samples between successive coordinates.

    Returns
    -------
    2-D array of shape ``(n_points, dimension)`` where
    ``n_points = len(signal) - (dimension - 1) * delay``.
    """
    n = len(signal)
    n_points = n - (dimension - 1) * delay
    if n_points <= 0:
        raise ValueError(
            f"Signal too short ({n}) for dimension={dimension}, delay={delay}."
        )
    embedded = np.empty((n_points, dimension), dtype=float)
    for d in range(dimension):
        embedded[:, d] = signal[d * delay : d * delay + n_points]
    return embedded


def estimate_delay(signal: np.ndarray, max_lag: int = 20) -> int:
    """
    Estimate a suitable time delay using the first minimum of autocorrelation.

    Parameters
    ----------
    signal:
        1-D time series.
    max_lag:
        Maximum lag to search.

    Returns
    -------
    Lag at the first local minimum of autocorrelation, or 1 if none found.
    """
    n = len(signal)
    s = signal.astype(float)
    mean = s.mean()
    var = s.var()
    if var < 1e-12 or n <= max_lag:
        return 1

    acf = []
    for lag in range(1, min(max_lag + 1, n)):
        a = s[: n - lag] - mean
        b = s[lag:] - mean
        acf.append(float(np.dot(a, b) / ((n - lag) * var)))

    # First local minimum
    for i in range(1, len(acf) - 1):
        if acf[i] < acf[i - 1] and acf[i] < acf[i + 1]:
            return i + 1
    return 1
