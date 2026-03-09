"""
Autocorrelation features for a bitstream window.
"""

from __future__ import annotations

from typing import List

import numpy as np


def autocorr_lag(window: np.ndarray, lag: int = 1) -> float:
    """
    Pearson autocorrelation at the specified *lag*.

    Parameters
    ----------
    window:
        1-D bit array.
    lag:
        Lag in samples.

    Returns
    -------
    Autocorrelation coefficient in ``[-1, 1]``.
    """
    w = window.astype(float)
    n = len(w)
    if n <= lag:
        return 0.0
    mean = w.mean()
    var = w.var()
    if var < 1e-12:
        return 0.0
    a = w[: n - lag] - mean
    b = w[lag:] - mean
    return float(np.dot(a, b) / ((n - lag) * var))


def autocorr_profile(window: np.ndarray, max_lag: int = 10) -> List[float]:
    """
    Autocorrelation values for lags 1 through *max_lag*.
    """
    return [autocorr_lag(window, lag) for lag in range(1, max_lag + 1)]


def compute_autocorr_features(window: np.ndarray, max_lag: int = 1) -> dict:
    """
    Return a dict with autocorrelation features for one window.

    Keys: ``autocorr_lag1`` (and higher lags if *max_lag* > 1).
    """
    result = {}
    for lag in range(1, max_lag + 1):
        result[f"autocorr_lag{lag}"] = autocorr_lag(window, lag)
    return result
