"""
Feature standardization (z-score normalization) for anomaly scoring.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def zscore_dataframe(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Z-score normalize numeric columns in *df*.

    Parameters
    ----------
    df:
        Input DataFrame of feature windows.
    columns:
        Subset of columns to normalize.  When ``None`` all numeric columns
        are normalized.
    eps:
        Small value added to the standard deviation to avoid division by zero.

    Returns
    -------
    A new DataFrame with the selected columns standardized; other columns
    are carried through unchanged.
    """
    out = df.copy()
    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)
    for col in columns:
        mean = out[col].mean()
        std = out[col].std(ddof=1)
        out[col] = (out[col] - mean) / (std + eps)
    return out


def robust_scale_dataframe(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Median / IQR robust scaling for numeric columns in *df*.

    More resistant to outliers than z-score normalization.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to scale.  Defaults to all numeric columns.
    eps:
        Small value added to the IQR.

    Returns
    -------
    Scaled copy of *df*.
    """
    out = df.copy()
    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)
    for col in columns:
        median = out[col].median()
        q75 = out[col].quantile(0.75)
        q25 = out[col].quantile(0.25)
        iqr = q75 - q25
        out[col] = (out[col] - median) / (iqr + eps)
    return out
