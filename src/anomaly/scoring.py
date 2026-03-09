"""
Anomaly scoring: aggregate feature windows into scalar anomaly scores.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def mahalanobis_scores(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute Mahalanobis distance from the mean for each row in *df*.

    Parameters
    ----------
    df:
        DataFrame of (already standardised) feature values.
    columns:
        Subset of columns to use.  Defaults to all numeric columns.

    Returns
    -------
    1-D numpy array of non-negative anomaly scores.
    """
    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)
    X = df[columns].to_numpy(dtype=float)
    mean = X.mean(axis=0)
    X_centred = X - mean
    cov = np.cov(X_centred, rowvar=False)

    # Regularise in case the covariance is singular
    cov += np.eye(cov.shape[0]) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    scores = np.sqrt(np.sum(X_centred @ cov_inv * X_centred, axis=1))
    return scores


def zscore_combined_score(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Combine columns into a single anomaly score as the RMS of z-scored values.

    Parameters
    ----------
    df:
        DataFrame (columns should ideally already be standardised).
    columns:
        Subset of columns to include.  Defaults to all numeric columns.

    Returns
    -------
    1-D numpy array of anomaly scores.
    """
    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)
    X = df[columns].to_numpy(dtype=float)

    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds < 1e-8] = 1.0

    Z = (X - means) / stds
    return np.sqrt(np.mean(Z ** 2, axis=1))


def score_windows(
    df: pd.DataFrame,
    method: str = "zscore",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Append an ``anomaly_score`` column to *df*.

    Parameters
    ----------
    df:
        Feature DataFrame (one row per window).
    method:
        Scoring method: ``"zscore"`` (default) or ``"mahalanobis"``.
    columns:
        Feature columns to use for scoring.

    Returns
    -------
    Copy of *df* with an additional ``anomaly_score`` column.
    """
    if method == "mahalanobis":
        scores = mahalanobis_scores(df, columns)
    else:
        scores = zscore_combined_score(df, columns)

    out = df.copy()
    out["anomaly_score"] = scores
    return out
