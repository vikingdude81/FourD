"""
Latent coordinator dynamics.

Runs the 4D coordinator update loop given a sequence of per-subsystem
drive vectors and basin attractors, producing a latent trajectory.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.latent.basins import (
    attractor_pull,
    basin_switch_event,
    initialize_basin_attractors,
)


# ──────────────────────────────────────────────────────────────────────────────
# Coordinator update
# ──────────────────────────────────────────────────────────────────────────────

def coordinator_step(
    coordinator: np.ndarray,
    drives: Dict[str, np.ndarray],
    basin_attractor: np.ndarray,
    learning_rate: float = 0.05,
    noise_level: float = 0.02,
    basin_pull_strength: float = 0.02,
    max_norm: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advance the coordinator by one step.

    The coordinator is nudged toward the weighted average of active
    subsystem drive vectors and pulled toward its current basin attractor.

    Parameters
    ----------
    coordinator:
        Current coordinator state, shape ``(n_dims,)``.
    drives:
        Dict mapping subsystem name → drive vector.
    basin_attractor:
        Current basin attractor vector.
    learning_rate:
        Step size toward the weighted drive input.
    noise_level:
        Standard deviation of additive Gaussian noise.
    basin_pull_strength:
        Step size toward the basin attractor.
    max_norm:
        Hard clip applied to the output state.

    Returns
    -------
    ``(updated_coordinator, weighted_input)``
    """
    n_dims = len(coordinator)
    drive_list = list(drives.values())

    if drive_list:
        norms = np.array([np.linalg.norm(d) for d in drive_list], dtype=float)
        total_norm = norms.sum() + 1e-8
        weighted_input = sum(d * (norms[i] / total_norm) for i, d in enumerate(drive_list))
    else:
        weighted_input = np.zeros(n_dims, dtype=float)

    noise = np.random.randn(n_dims) * noise_level
    updated = (
        (1.0 - learning_rate) * coordinator
        + learning_rate * weighted_input
        + noise
    )
    updated = attractor_pull(updated, basin_attractor, learning_rate=basin_pull_strength, noise_level=0.0)

    # Hard-clip to prevent runaway growth
    norm = np.linalg.norm(updated)
    if norm > max_norm:
        updated = updated * (max_norm / norm)

    return updated, weighted_input


# ──────────────────────────────────────────────────────────────────────────────
# Full trajectory runner
# ──────────────────────────────────────────────────────────────────────────────

def run_coordinator(
    drive_sequence: List[Dict[str, np.ndarray]],
    n_basins: int = 5,
    n_dims: int = 4,
    learning_rate: float = 0.05,
    noise_level: float = 0.02,
    basin_pull_strength: float = 0.02,
    ambiguity_threshold: float = 0.05,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the coordinator over a sequence of drive inputs.

    Parameters
    ----------
    drive_sequence:
        List of subsystem drive dicts (one per time step / window).
    n_basins:
        Number of latent basin attractors to initialise.
    n_dims:
        Dimensionality of the coordinator state.
    learning_rate:
        Step size toward the weighted drive input.
    noise_level:
        Standard deviation of additive noise.
    basin_pull_strength:
        Step size toward the basin attractor.
    ambiguity_threshold:
        Gap required between top-2 similarities before a switch is accepted.
    random_seed:
        Optional seed for reproducibility.

    Returns
    -------
    DataFrame with columns ``coord_0 … coord_{n_dims-1}``,
    ``chosen_basin``, ``basin_similarity_0 … basin_similarity_{n_basins-1}``,
    and ``weighted_input_0 … weighted_input_{n_dims-1}``.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    attractors = initialize_basin_attractors(n_basins, n_dims)
    coordinator = np.random.randn(n_dims) * 0.5
    current_basin_attractor = attractors[0]
    current_basin_idx: Optional[int] = None

    records = []
    for drives in drive_sequence:
        result = basin_switch_event(
            coordinator,
            attractors,
            ambiguity_threshold=ambiguity_threshold,
            previous_index=current_basin_idx,
        )
        current_basin_idx = result.chosen_index
        current_basin_attractor = attractors[current_basin_idx]

        coordinator, weighted_input = coordinator_step(
            coordinator,
            drives,
            current_basin_attractor,
            learning_rate=learning_rate,
            noise_level=noise_level,
            basin_pull_strength=basin_pull_strength,
        )

        row: Dict[str, object] = {}
        for d in range(n_dims):
            row[f"coord_{d}"] = coordinator[d]
        row["chosen_basin"] = current_basin_idx
        for b, sim in enumerate(result.similarities):
            row[f"basin_similarity_{b}"] = sim
        for d in range(n_dims):
            row[f"weighted_input_{d}"] = weighted_input[d]
        records.append(row)

    return pd.DataFrame(records)
