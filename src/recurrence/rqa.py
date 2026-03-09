"""
Recurrence Quantification Analysis (RQA) measures.

Computes standard RQA statistics from a binary recurrence matrix.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def recurrence_rate(rmat: np.ndarray) -> float:
    """Fraction of recurrent points (excluding the main diagonal)."""
    n = len(rmat)
    if n <= 1:
        return 0.0
    off_diag = rmat.astype(int).sum() - n  # exclude main diagonal
    return float(off_diag) / float(n * (n - 1))


def determinism(rmat: np.ndarray, min_length: int = 2) -> float:
    """
    Fraction of recurrent points that lie on diagonal lines of length
    >= *min_length*.
    """
    n = len(rmat)
    if n == 0:
        return 0.0

    diagonal_points = 0
    total_recurrent = 0

    for k in range(-(n - 1), n):
        diag = np.diag(rmat, k)
        total_recurrent += int(diag.sum())
        # Count runs of True of length >= min_length
        runs = _run_lengths(diag.astype(bool))
        diagonal_points += sum(r for r in runs if r >= min_length)

    # Exclude main diagonal from total
    total_recurrent -= int(np.trace(rmat))

    if total_recurrent <= 0:
        return 0.0
    return float(diagonal_points) / float(total_recurrent)


def avg_diagonal_line(rmat: np.ndarray, min_length: int = 2) -> float:
    """Average length of diagonal lines of length >= *min_length*."""
    n = len(rmat)
    if n == 0:
        return 0.0

    all_runs = []
    for k in range(-(n - 1), n):
        if k == 0:
            continue
        diag = np.diag(rmat, k).astype(bool)
        all_runs.extend(r for r in _run_lengths(diag) if r >= min_length)

    return float(np.mean(all_runs)) if all_runs else 0.0


def laminarity(rmat: np.ndarray, min_length: int = 2) -> float:
    """
    Fraction of recurrent points forming vertical lines of length
    >= *min_length*.
    """
    n = len(rmat)
    if n == 0:
        return 0.0

    vertical_points = 0
    total_recurrent = rmat.astype(int).sum() - int(np.trace(rmat))

    for col in range(n):
        runs = _run_lengths(rmat[:, col].astype(bool))
        vertical_points += sum(r for r in runs if r >= min_length)

    if total_recurrent <= 0:
        return 0.0
    return float(vertical_points) / float(total_recurrent)


def _run_lengths(arr: np.ndarray) -> list:
    """Return list of run lengths of ``True`` values in a 1-D boolean array."""
    runs = []
    current = 0
    for val in arr:
        if val:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return runs


def compute_rqa(rmat: np.ndarray, min_length: int = 2) -> Dict[str, float]:
    """
    Compute a standard set of RQA measures.

    Parameters
    ----------
    rmat:
        Binary recurrence matrix.
    min_length:
        Minimum line length for DET and LAM calculations.

    Returns
    -------
    Dict with keys: ``rr``, ``det``, ``avg_diag``, ``lam``.
    """
    return {
        "rr": recurrence_rate(rmat),
        "det": determinism(rmat, min_length),
        "avg_diag": avg_diagonal_line(rmat, min_length),
        "lam": laminarity(rmat, min_length),
    }
