"""
Sliding-window segmentation of QRNG bitstreams.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np


def sliding_windows(
    bits: np.ndarray,
    window_size: int = 4096,
    overlap: float = 0.5,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Yield ``(start_index, window)`` pairs for a 1-D bit array.

    Parameters
    ----------
    bits:
        1-D array of integer bits (0 or 1).
    window_size:
        Number of bits per window.
    overlap:
        Fraction of the window that overlaps with the previous window
        (0 = no overlap, 0.5 = 50 % overlap).

    Yields
    ------
    ``(start_index, window_array)`` where *window_array* has length
    *window_size*.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    step = max(1, int(window_size * (1.0 - overlap)))
    n = len(bits)
    start = 0
    while start + window_size <= n:
        yield start, bits[start : start + window_size]
        start += step


def window_metadata(
    bits: np.ndarray,
    window_size: int = 4096,
    overlap: float = 0.5,
    stream_id: str = "",
) -> List[dict]:
    """
    Return a list of metadata dicts describing every window.

    Each dict contains: ``stream_id``, ``window_index``, ``start_bit``,
    ``end_bit``, ``window_size``, ``overlap``.
    """
    meta = []
    for idx, (start, _) in enumerate(sliding_windows(bits, window_size, overlap)):
        meta.append(
            {
                "stream_id": stream_id,
                "window_index": idx,
                "start_bit": start,
                "end_bit": start + window_size,
                "window_size": window_size,
                "overlap": overlap,
            }
        )
    return meta
