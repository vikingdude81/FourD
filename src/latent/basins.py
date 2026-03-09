"""
Latent-state basin dynamics.

Provides attractor initialization, similarity scoring, attractor pull,
and basin-switch logic for the 4D latent coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BasinSwitchResult:
    """Result of a basin-switch evaluation."""
    chosen_index: int
    switched: bool
    similarities: np.ndarray


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def initialize_basin_attractors(
    n_basins: int,
    n_dimensions: int,
    noise_scale: float = 0.05,
) -> List[np.ndarray]:
    """
    Create a set of random unit-normalized basin attractors in *n_dimensions*.

    Parameters
    ----------
    n_basins:
        Number of attractor basins.
    n_dimensions:
        Dimensionality of the latent coordinator state.
    noise_scale:
        Magnitude of the Gaussian noise added to each attractor.

    Returns
    -------
    List of numpy arrays, each of shape ``(n_dimensions,)``.
    """
    attractors: List[np.ndarray] = []
    for _ in range(n_basins):
        vec = np.random.randn(n_dimensions)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        vec += np.random.randn(n_dimensions) * noise_scale
        attractors.append(vec)
    return attractors


def basin_similarities(
    state: np.ndarray,
    basin_attractors: List[np.ndarray],
) -> np.ndarray:
    """
    Compute cosine similarity between *state* and each basin attractor.

    Parameters
    ----------
    state:
        Current coordinator state vector, shape ``(n_dimensions,)``.
    basin_attractors:
        List of attractor vectors, each shape ``(n_dimensions,)``.

    Returns
    -------
    1-D numpy array of cosine similarities, one per basin.
    """
    state_norm = np.linalg.norm(state)
    sims = np.empty(len(basin_attractors), dtype=float)
    for i, attractor in enumerate(basin_attractors):
        att_norm = np.linalg.norm(attractor)
        if state_norm < 1e-8 or att_norm < 1e-8:
            sims[i] = 0.0
        else:
            sims[i] = float(np.dot(state, attractor) / (state_norm * att_norm))
    return sims


def attractor_pull(
    state: np.ndarray,
    basin_attractor: np.ndarray,
    learning_rate: float = 0.02,
    noise_level: float = 0.01,
) -> np.ndarray:
    """
    Nudge *state* toward *basin_attractor* with a small random perturbation.

    Parameters
    ----------
    state:
        Current coordinator state vector.
    basin_attractor:
        Target attractor vector.
    learning_rate:
        Step size toward the attractor (0 < lr < 1).
    noise_level:
        Standard deviation of added Gaussian noise.

    Returns
    -------
    Updated state vector of the same shape as *state*.
    """
    noise = np.random.randn(*state.shape) * noise_level
    updated = state + learning_rate * (basin_attractor - state) + noise
    return updated


def basin_switch_event(
    state: np.ndarray,
    basin_attractors: List[np.ndarray],
    ambiguity_threshold: float = 0.05,
    previous_index: Optional[int] = None,
) -> BasinSwitchResult:
    """
    Decide whether the coordinator should switch to a new basin.

    The highest-similarity basin wins.  A switch is declared when the new
    basin differs from *previous_index*.  If the two highest similarities
    are within *ambiguity_threshold* of each other, the previous basin is
    retained to avoid rapid flickering.

    Parameters
    ----------
    state:
        Current coordinator state vector.
    basin_attractors:
        List of attractor vectors.
    ambiguity_threshold:
        Minimum gap between the top two similarity scores required to
        accept a switch.
    previous_index:
        Index of the basin chosen at the previous time step, or ``None``
        if this is the first call.

    Returns
    -------
    :class:`BasinSwitchResult` with the chosen basin index, a flag
    indicating whether a switch occurred, and the full similarity array.
    """
    sims = basin_similarities(state, basin_attractors)
    sorted_indices = np.argsort(sims)[::-1]
    best_idx = int(sorted_indices[0])

    # Ambiguity guard: if top-2 are too close, stay with the previous basin
    if len(sorted_indices) >= 2:
        second_idx = int(sorted_indices[1])
        gap = sims[best_idx] - sims[second_idx]
        if gap < ambiguity_threshold and previous_index is not None:
            chosen_index = previous_index
        else:
            chosen_index = best_idx
    else:
        chosen_index = best_idx

    switched = (previous_index is not None) and (chosen_index != previous_index)
    return BasinSwitchResult(
        chosen_index=chosen_index,
        switched=switched,
        similarities=sims,
    )
