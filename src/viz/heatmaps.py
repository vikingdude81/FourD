"""
Heatmap utilities for visualizing feature matrices.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def feature_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Feature Correlation",
) -> plt.Axes:
    """
    Draw a Pearson correlation heatmap for numeric columns of *df*.

    Parameters
    ----------
    df:
        DataFrame of feature windows.
    columns:
        Subset of columns to include.  Defaults to all numeric columns.
    ax:
        Matplotlib axes to draw on.  Creates a new figure if ``None``.
    title:
        Axes title.

    Returns
    -------
    The matplotlib Axes containing the heatmap.
    """
    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)

    corr = df[columns].corr().to_numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, len(columns)), max(4, len(columns))))

    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="equal")
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(columns, fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def anomaly_score_heatmap(
    scores: np.ndarray,
    n_cols: int = 20,
    ax: Optional[plt.Axes] = None,
    title: str = "Anomaly Score Heatmap",
) -> plt.Axes:
    """
    Reshape a 1-D anomaly-score array into a 2-D grid and display it as a heatmap.

    Parameters
    ----------
    scores:
        1-D array of anomaly scores (one per window).
    n_cols:
        Number of columns in the display grid.
    ax:
        Matplotlib axes to draw on.  Creates a new figure if ``None``.
    title:
        Axes title.

    Returns
    -------
    The matplotlib Axes containing the heatmap.
    """
    n = len(scores)
    n_rows = int(np.ceil(n / n_cols))
    padded = np.full(n_rows * n_cols, np.nan)
    padded[:n] = scores
    grid = padded.reshape(n_rows, n_cols)

    if ax is None:
        _, ax = plt.subplots(figsize=(min(14, n_cols * 0.5), max(3, n_rows * 0.5)))

    im = ax.imshow(grid, cmap="hot_r", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Window (column)")
    ax.set_ylabel("Block (row)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Anomaly score")
    return ax
