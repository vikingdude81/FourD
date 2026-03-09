"""
Recurrence plot construction from an embedded trajectory.
"""

from __future__ import annotations

import numpy as np


def distance_matrix(embedded: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix for an embedded trajectory.

    Parameters
    ----------
    embedded:
        2-D array of shape ``(n_points, dimension)``.

    Returns
    -------
    Symmetric 2-D array of shape ``(n_points, n_points)``.
    """
    n = len(embedded)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        diff = embedded - embedded[i]
        dist[i] = np.sqrt((diff ** 2).sum(axis=1))
    return dist


def recurrence_matrix(
    embedded: np.ndarray,
    threshold: float | None = None,
    threshold_percentile: float = 10.0,
) -> np.ndarray:
    """
    Build a binary recurrence matrix from an embedded trajectory.

    Parameters
    ----------
    embedded:
        2-D array from :func:`~src.recurrence.embedding.time_delay_embedding`.
    threshold:
        Distance threshold ε.  When ``None``, *threshold_percentile* of all
        pairwise distances is used.
    threshold_percentile:
        Percentile of the distance distribution to use as ε when *threshold*
        is ``None``.

    Returns
    -------
    Boolean 2-D array of shape ``(n, n)``, where ``True`` indicates
    recurrence (distance ≤ ε).
    """
    D = distance_matrix(embedded)
    if threshold is None:
        threshold = float(np.percentile(D, threshold_percentile))
    return D <= threshold


def plot_recurrence(
    rmat: np.ndarray,
    ax=None,
    title: str = "Recurrence Plot",
) -> None:
    """
    Display a recurrence matrix as an image using matplotlib.

    Parameters
    ----------
    rmat:
        Boolean recurrence matrix from :func:`recurrence_matrix`.
    ax:
        Matplotlib axes to draw on.  A new figure is created when ``None``.
    title:
        Plot title.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(rmat, cmap="binary", origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Time")
