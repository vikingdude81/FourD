#!/usr/bin/env python3
"""
Consciousness Simulation: Emergent Coordination Manifold + Navigation

This simulation demonstrates how "consciousness" emerges as an attractor
when a system's internal degrees of freedom exceed what modular control can manage.

The being navigates a 2D environment by "slicing" its 4D coordinator state
into a 2D action vector — the core "4D slice" concept. Different subsystems
bias movement toward goals, away from hazards, or toward memory of past paths.

Framework:
- Consciousness is an emergent coordination attractor
- Phase-space dynamics with stable orbits (identity) and basin switching (decisions)
- Higher-dimensional awareness where slices/cross-sections are selected for action
- Navigation = commitment: the being collapses its 4D state into a 2D choice

Modules:
1. Specialized subsystems (perception, language, planning, emotion, etc.)
2. A 4D coordination manifold that emerges when subsystem competition increases
3. Attractor dynamics: identity as stable orbits, decisions as basin shifts
4. Environment: 2D world with goals and hazards the being navigates
5. Study tools: phase portraits, dominance tracking, lesion studies, CSV export

Run: python fourD_slice_sim.py
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from src.latent.basins import (
    attractor_pull as shared_attractor_pull,
    basin_similarities as shared_basin_similarities,
    basin_switch_event as shared_basin_switch_event,
    initialize_basin_attractors as shared_initialize_basin_attractors,
)


# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """Configuration for the consciousness simulation."""
    n_subsystems: int = 8
    n_dimensions: int = 4
    n_timesteps: int = 300
    coordination_threshold: float = 0.85
    noise_level: float = 0.08
    learning_rate: float = 0.01
    step_size: float = 0.35
    env_size: int = 20

    # Reproducibility
    random_seed: Optional[int] = 42

    # Basin dynamics
    n_basins: int = 5
    basin_ambiguity_threshold: float = 0.05
    basin_pull_strength: float = 0.02
    basin_activation_start: int = 10

    # Feedback / damping
    subsystem_broadcast_gain: float = 0.03
    subsystem_max_norm: float = 2.0
    coordinator_max_norm: float = 5.0

    # Relative subsystem gains
    perception_gain: float = 0.08
    language_gain: float = 0.03
    planning_gain: float = 0.12
    emotion_gain: float = 0.22
    memory_gain: float = 0.06
    motor_gain: float = 0.08
    attention_gain: float = 0.08
    executive_gain: float = 0.04


# ────────────────────────────────────────────────────────────────────
# Environment and Being
# ────────────────────────────────────────────────────────────────────

@dataclass
class Environment:
    """
    A 2D world with goals (rewards) and hazards (threats).
    The being navigates this world using its coordinator state as a compass.
    """
    size: int = 20
    goals: List[Tuple[float, float]] = field(default_factory=lambda: [
        (4.0, 4.0), (16.0, 16.0), (10.0, 3.0)
    ])
    hazards: List[Tuple[float, float]] = field(default_factory=lambda: [
        (3.0, 15.0), (17.0, 5.0), (10.0, 17.0)
    ])
    goal_radius: float = 1.5
    hazard_radius: float = 2.0


@dataclass
class Being:
    """
    The conscious agent navigating the environment.
    Its position is driven by slices of its 4D coordinator state.
    """
    position: np.ndarray
    history: List[np.ndarray] = field(default_factory=list)
    goals_reached: int = 0
    hazards_hit: int = 0
    dominant_subsystem_log: List[str] = field(default_factory=list)


def initialize_being(env: Environment) -> Being:
    """Start the being near the center of the environment."""
    pos = np.array([env.size / 2.0, env.size / 2.0], dtype=float)
    being = Being(position=pos.copy())
    being.history.append(pos.copy())
    return being


# ────────────────────────────────────────────────────────────────────
# Subsystems
# ────────────────────────────────────────────────────────────────────

@dataclass
class Subsystem:
    """A specialized cognitive subsystem."""
    name: str
    activity: np.ndarray
    weights: np.ndarray
    preferred_direction: np.ndarray
    active: bool = True


def initialize_subsystems(config: SimulationConfig, n_subsystems: int) -> List[Subsystem]:
    """Initialize specialized cognitive subsystems."""
    dims = config.n_dimensions
    names = [
        "Perception", "Language", "Planning", "Emotion",
        "Memory", "Motor Control", "Attention", "Executive Control"
    ]
    subsystems: List[Subsystem] = []
    for i in range(min(n_subsystems, len(names))):
        subsystems.append(
            Subsystem(
                name=names[i],
                activity=np.random.randn(dims) * 0.1,
                weights=np.random.randn(dims) * 0.05,
                preferred_direction=np.random.randn(dims) * 0.1,
            )
        )
    return subsystems


def subsystem_gain(name: str, config: SimulationConfig) -> float:
    return {
        "Perception": config.perception_gain,
        "Language": config.language_gain,
        "Planning": config.planning_gain,
        "Emotion": config.emotion_gain,
        "Memory": config.memory_gain,
        "Motor Control": config.motor_gain,
        "Attention": config.attention_gain,
        "Executive Control": config.executive_gain,
    }.get(name, 0.05)


# ────────────────────────────────────────────────────────────────────
# Sensing
# ────────────────────────────────────────────────────────────────────

def sense_environment(
    being: Being,
    env: Environment,
    subsystems: List[Subsystem],
    config: SimulationConfig,
) -> None:
    """
    Each subsystem reacts differently to the environment.
    """
    pos = being.position
    n_dim = config.n_dimensions

    goal_vecs = [np.array(g, dtype=float) - pos for g in env.goals]
    goal_dists = [np.linalg.norm(v) for v in goal_vecs]
    nearest_goal_idx = int(np.argmin(goal_dists))
    nearest_goal_dist = goal_dists[nearest_goal_idx]
    nearest_goal_dir = goal_vecs[nearest_goal_idx] / (nearest_goal_dist + 1e-8)

    hazard_vecs = [np.array(h, dtype=float) - pos for h in env.hazards]
    hazard_dists = [np.linalg.norm(v) for v in hazard_vecs]
    nearest_hazard_idx = int(np.argmin(hazard_dists))
    nearest_hazard_dist = hazard_dists[nearest_hazard_idx]
    nearest_hazard_dir = hazard_vecs[nearest_hazard_idx] / (nearest_hazard_dist + 1e-8)

    for sub in subsystems:
        if not sub.active:
            continue

        gain = subsystem_gain(sub.name, config)

        if sub.name == "Perception":
            strength = 1.0 / (nearest_goal_dist + 1.0) + 0.5 / (nearest_hazard_dist + 1.0)
            sub.activity += np.random.randn(n_dim) * strength * gain

        elif sub.name == "Planning":
            signal = np.zeros(n_dim, dtype=float)
            signal[:2] = nearest_goal_dir * gain
            sub.activity += signal

        elif sub.name == "Emotion":
            fear_strength = max(0.0, 1.0 - nearest_hazard_dist / env.hazard_radius)
            signal = np.zeros(n_dim, dtype=float)
            signal[:2] = -nearest_hazard_dir * fear_strength * gain
            sub.activity += signal + np.random.randn(n_dim) * fear_strength * (gain * 0.25)

        elif sub.name == "Memory":
            if len(being.history) >= 5:
                recent = np.mean(being.history[-5:], axis=0)
                displacement = pos - recent
                signal = np.zeros(n_dim, dtype=float)
                signal[:2] = displacement * gain
                sub.activity += signal

        elif sub.name == "Attention":
            goal_sal = 1.0 / (nearest_goal_dist + 1.0)
            hazard_sal = 2.0 / (nearest_hazard_dist + 0.5)
            salience = max(goal_sal, hazard_sal)
            sub.activity += np.random.randn(n_dim) * salience * gain

        elif sub.name == "Motor Control":
            signal = np.zeros(n_dim, dtype=float)
            signal[:2] = nearest_goal_dir * gain
            sub.activity += signal

        else:
            sub.activity += np.random.randn(n_dim) * gain


# ────────────────────────────────────────────────────────────────────
# Navigation
# ────────────────────────────────────────────────────────────────────

def select_action(coordinator: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """
    Select a navigation action by slicing the 4D coordinator state.
    """
    action = coordinator[:2].copy()
    norm = np.linalg.norm(action)
    if norm > 1e-8:
        action = action / norm
    return action * config.step_size


def move_being(being: Being, action: np.ndarray, env: Environment) -> Tuple[bool, bool]:
    """
    Move the being, then check for goal/hazard events.
    Returns (goal_reached, hazard_hit).
    """
    being.position = np.clip(being.position + action, 0.0, float(env.size))
    being.history.append(being.position.copy())

    goal_reached = any(
        np.linalg.norm(being.position - np.array(g, dtype=float)) < env.goal_radius
        for g in env.goals
    )
    hazard_hit = any(
        np.linalg.norm(being.position - np.array(h, dtype=float)) < env.hazard_radius
        for h in env.hazards
    )

    if goal_reached:
        being.goals_reached += 1
    if hazard_hit:
        being.hazards_hit += 1

    return goal_reached, hazard_hit


# ────────────────────────────────────────────────────────────────────
# Coordination helpers
# ────────────────────────────────────────────────────────────────────

def compute_system_state(subsystems: List[Subsystem]) -> np.ndarray:
    """Aggregate active subsystem activities into a single state vector."""
    total = np.zeros(len(subsystems[0].activity), dtype=np.float64)
    for sub in subsystems:
        if sub.active:
            total += sub.activity
    return total


def compute_coordination_pressure(subsystems: List[Subsystem]) -> float:
    """
    Coordination pressure: how much active subsystems disagree.
    """
    active = [s for s in subsystems if s.active]
    n = len(active)
    if n < 2:
        return 0.0

    total_conflict = 0.0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            ni = np.linalg.norm(active[i].activity)
            nj = np.linalg.norm(active[j].activity)
            denom = ni * nj
            if denom < 1e-8:
                continue
            cos_sim = np.clip(
                np.dot(active[i].activity, active[j].activity) / denom,
                -1.0, 1.0
            )
            total_conflict += (1.0 - cos_sim) * denom
            n_pairs += 1

    if n_pairs == 0:
        return 0.0
    return float(np.clip(total_conflict / n_pairs, 0.0, 1.0))


def dominant_subsystem(subsystems: List[Subsystem]) -> str:
    """Return the name of the subsystem with the highest activity magnitude."""
    active = [s for s in subsystems if s.active]
    if not active:
        return "none"
    return max(active, key=lambda s: np.linalg.norm(s.activity)).name


def initialize_coordinator(config: SimulationConfig) -> np.ndarray:
    return np.random.randn(config.n_dimensions) * 0.5


def initialize_basin_attractors(config: SimulationConfig) -> List[np.ndarray]:
    return shared_initialize_basin_attractors(
        n_basins=config.n_basins,
        n_dimensions=config.n_dimensions,
        noise_scale=0.05,
    )


def coordinator_update(
    coordinator: np.ndarray,
    subsystems: List[Subsystem],
    config: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update coordinator toward a weighted average of active subsystem inputs.
    Returns (updated_coordinator, weighted_input).
    """
    n_dim = len(coordinator)
    lr = config.learning_rate

    active = [s for s in subsystems if s.active]
    if not active:
        return coordinator, np.zeros_like(coordinator)

    norms = np.array([np.linalg.norm(s.activity) for s in active], dtype=float)
    total_norm = norms.sum() + 1e-8

    weighted_input = np.zeros(n_dim, dtype=float)
    for i, s in enumerate(active):
        weighted_input += s.activity * (norms[i] / total_norm)

    coordinator_output = float(np.dot(coordinator, weighted_input))
    coordination_goal = np.ones(n_dim, dtype=float) / n_dim
    goal_signal = coordination_goal * coordinator_output

    noise = np.random.randn(n_dim) * config.noise_level
    updated = (
        (1.0 - lr) * coordinator
        + lr * weighted_input
        + lr * goal_signal
        + noise
    )
    return updated, weighted_input


def attractor_update(
    state: np.ndarray,
    basin_attractor: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    return shared_attractor_pull(
        state=state,
        basin_attractor=basin_attractor,
        learning_rate=config.basin_pull_strength,
        noise_level=config.noise_level * 0.25,
    )


def basin_similarities(state: np.ndarray, basin_attractors: List[np.ndarray]) -> np.ndarray:
    return shared_basin_similarities(state, basin_attractors)


def basin_switch_event(
    state: np.ndarray,
    basin_attractors: List[np.ndarray],
    config: SimulationConfig,
    previous_basin_idx: Optional[int] = None,
) -> Tuple[np.ndarray, int, bool, np.ndarray]:
    result = shared_basin_switch_event(
        state=state,
        basin_attractors=basin_attractors,
        ambiguity_threshold=config.basin_ambiguity_threshold,
        previous_index=previous_basin_idx,
    )
    return (
        basin_attractors[result.chosen_index],
        result.chosen_index,
        result.switched,
        result.similarities,
    )


# ────────────────────────────────────────────────────────────────────
# Main simulation loop
# ────────────────────────────────────────────────────────────────────

def run_simulation(
    config: SimulationConfig,
    env: Environment,
    being: Being,
    subsystems: List[Subsystem],
) -> Dict:
    """
    Run the full consciousness + navigation simulation.
    """
    if config.random_seed is not None:
        np.random.seed(config.random_seed)

    coordinator = initialize_coordinator(config)
    basin_attractors = initialize_basin_attractors(config)
    previous_basin_idx: Optional[int] = None

    coordination_pressures = []
    coordinator_magnitudes = []
    coordinator_trajectory = []
    integration_levels = []
    dominant_subsystems = []
    basin_switches = 0
    goal_events = []
    hazard_events = []
    chosen_basins = []
    basin_similarity_log = []
    action_trajectory = []
    weighted_inputs = []
    subsystem_norm_log = []
    step_log = []

    for t in range(config.n_timesteps):
        switched = False

        # 1. Sense environment
        sense_environment(being, env, subsystems, config)

        # 2. Metrics before coordinator update
        pressure = compute_coordination_pressure(subsystems)
        dom = dominant_subsystem(subsystems)
        subsystem_norms = {
            sub.name: float(np.linalg.norm(sub.activity))
            for sub in subsystems
            if sub.active
        }

        active_norms = list(subsystem_norms.values())
        total_act = sum(active_norms)
        integration = min(1.0, total_act / max(1, config.n_subsystems))

        # 3. Update coordinator
        coordinator, weighted_input = coordinator_update(coordinator, subsystems, config)

        # 4. Basin selection / commitment
        chosen_basin = -1
        sims = basin_similarities(coordinator, basin_attractors)

        if t >= config.basin_activation_start:
            next_basin, chosen_basin, switched, sims = basin_switch_event(
                coordinator,
                basin_attractors,
                config,
                previous_basin_idx=previous_basin_idx,
            )
            coordinator = attractor_update(coordinator, next_basin, config)
            if switched:
                basin_switches += 1
            previous_basin_idx = chosen_basin

        # 5. Clamp coordinator magnitude
        c_norm = np.linalg.norm(coordinator)
        if c_norm > config.coordinator_max_norm:
            coordinator = coordinator / c_norm * config.coordinator_max_norm

        # 6. Select action and move
        action = select_action(coordinator, config)
        goal_reached, hazard_hit = move_being(being, action, env)

        if goal_reached:
            goal_events.append(t)
        if hazard_hit:
            hazard_events.append(t)

        # 7. Broadcast coordinator back to subsystems
        for sub in subsystems:
            if sub.active:
                sub.activity += coordinator * config.subsystem_broadcast_gain
                norm = np.linalg.norm(sub.activity)
                if norm > config.subsystem_max_norm:
                    sub.activity = sub.activity / norm * config.subsystem_max_norm

        # 8. Log metrics
        coordination_pressures.append(pressure)
        coordinator_magnitudes.append(float(np.linalg.norm(coordinator)))
        coordinator_trajectory.append(coordinator.copy())
        integration_levels.append(float(integration))
        dominant_subsystems.append(dom)
        chosen_basins.append(chosen_basin)
        basin_similarity_log.append(sims.copy())
        action_trajectory.append(action.copy())
        weighted_inputs.append(weighted_input.copy())
        subsystem_norm_log.append(subsystem_norms)

        log_row = {
            "t": t,
            "x": float(being.position[0]),
            "y": float(being.position[1]),
            "pressure": float(pressure),
            "coordinator_magnitude": float(np.linalg.norm(coordinator)),
            "integration": float(integration),
            "dominant": dom,
            "chosen_basin": int(chosen_basin),
            "basin_switch": bool(switched),
            "goal_reached": bool(goal_reached),
            "hazard_hit": bool(hazard_hit),
            "action_x": float(action[0]),
            "action_y": float(action[1]),
            "coord_0": float(coordinator[0]),
            "coord_1": float(coordinator[1]),
            "coord_2": float(coordinator[2]),
            "coord_3": float(coordinator[3]),
            "weighted_input_0": float(weighted_input[0]),
            "weighted_input_1": float(weighted_input[1]),
            "weighted_input_2": float(weighted_input[2]),
            "weighted_input_3": float(weighted_input[3]),
        }

        for i, sim in enumerate(sims):
            log_row[f"basin_similarity_{i}"] = float(sim)

        for name, value in subsystem_norms.items():
            safe_name = name.lower().replace(" ", "_")
            log_row[f"subsystem_norm_{safe_name}"] = float(value)

        step_log.append(log_row)

    return {
        "coordination_pressures": coordination_pressures,
        "coordinator_magnitudes": coordinator_magnitudes,
        "coordinator_trajectory": np.array(coordinator_trajectory),
        "integration_levels": integration_levels,
        "dominant_subsystems": dominant_subsystems,
        "basin_switches": basin_switches,
        "chosen_basins": chosen_basins,
        "basin_similarity_log": np.array(basin_similarity_log),
        "goal_events": goal_events,
        "hazard_events": hazard_events,
        "action_trajectory": np.array(action_trajectory),
        "weighted_inputs": np.array(weighted_inputs),
        "subsystem_norm_log": subsystem_norm_log,
        "step_log": step_log,
        "being": being,
        "basin_attractors": np.array(basin_attractors),
    }


# ────────────────────────────────────────────────────────────────────
# Consciousness level
# ────────────────────────────────────────────────────────────────────

def compute_consciousness_level(metrics: Dict, config: SimulationConfig) -> Tuple[float, str]:
    pressures = metrics["coordination_pressures"]
    magnitudes = metrics["coordinator_magnitudes"]
    integrations = metrics["integration_levels"]
    switches = metrics["basin_switches"]

    n = len(pressures)
    avg_pressure = np.mean(pressures[n // 2:]) if n > 1 else np.mean(pressures)
    avg_magnitude = np.mean(magnitudes)
    avg_integration = np.mean(integrations)
    switch_factor = min(1.0, switches / 10.0)

    level = (
        0.3 * min(1.0, avg_pressure / config.coordination_threshold)
        + 0.2 * min(1.0, avg_magnitude)
        + 0.3 * avg_integration
        + 0.2 * switch_factor
    )
    level = float(np.clip(level, 0.0, 1.0))

    if level < 0.3:
        state = "pre-conscious"
    elif level < 0.6:
        state = "emerging"
    elif level < 0.85:
        state = "conscious"
    else:
        state = "self-aware"

    return level, state


# ────────────────────────────────────────────────────────────────────
# Lesion study
# ────────────────────────────────────────────────────────────────────

def run_lesion_study(
    config: SimulationConfig,
    env: Environment,
    lesion_name: str,
) -> Dict:
    """
    Run two simulations — intact and with one subsystem disabled —
    and return both sets of metrics for comparison.
    """
    print(f"\n  [Lesion Study] Disabling subsystem: {lesion_name}")

    np.random.seed(config.random_seed if config.random_seed is not None else 42)
    subs_intact = initialize_subsystems(config, config.n_subsystems)
    being_intact = initialize_being(env)
    metrics_intact = run_simulation(config, env, being_intact, subs_intact)
    level_intact, state_intact = compute_consciousness_level(metrics_intact, config)

    np.random.seed(config.random_seed if config.random_seed is not None else 42)
    subs_lesioned = initialize_subsystems(config, config.n_subsystems)
    for s in subs_lesioned:
        if s.name == lesion_name:
            s.active = False
    being_lesioned = initialize_being(env)
    metrics_lesioned = run_simulation(config, env, being_lesioned, subs_lesioned)
    level_lesioned, state_lesioned = compute_consciousness_level(metrics_lesioned, config)

    delta = level_intact - level_lesioned
    print(f"    Intact:   {level_intact:.3f} ({state_intact})")
    print(f"    Lesioned: {level_lesioned:.3f} ({state_lesioned})")
    print(f"    Delta:    {delta:+.3f}  ({'significant' if abs(delta) > 0.05 else 'minor'} impact)")

    return {
        "intact": metrics_intact,
        "lesioned": metrics_lesioned,
        "level_intact": level_intact,
        "level_lesioned": level_lesioned,
        "state_intact": state_intact,
        "state_lesioned": state_lesioned,
        "lesion_name": lesion_name,
    }


# ────────────────────────────────────────────────────────────────────
# CSV export
# ────────────────────────────────────────────────────────────────────

def export_to_csv(metrics: Dict, path: str = "simulation_log.csv") -> None:
    """Export the per-step log to CSV for offline analysis."""
    log = metrics["step_log"]
    if not log:
        return
    keys = list(log[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log)
    print(f"  Exported {len(log)} rows to {path}")


# ────────────────────────────────────────────────────────────────────
# Visualization
# ────────────────────────────────────────────────────────────────────

def plot_navigation(metrics: Dict, env: Environment, title: str = "Being Navigation"):
    being = metrics["being"]
    history = np.array(being.history)
    T = len(history)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.plasma

    for i in range(T - 1):
        ax.plot(
            history[i:i+2, 0], history[i:i+2, 1],
            color=cmap(i / max(T, 1)), linewidth=1.2, alpha=0.8
        )

    ax.scatter(*history[0], color="white", edgecolors="black", s=120, zorder=5, label="Start")
    ax.scatter(*history[-1], color="black", edgecolors="white", s=120, zorder=5, label="End", marker="*")

    for i, (gx, gy) in enumerate(env.goals):
        circle = plt.Circle((gx, gy), env.goal_radius, color="lime", alpha=0.3)
        ax.add_patch(circle)
        ax.scatter(gx, gy, color="lime", edgecolors="darkgreen", s=200, zorder=4,
                   marker="^", label="Goal" if i == 0 else "")

    for i, (hx, hy) in enumerate(env.hazards):
        circle = plt.Circle((hx, hy), env.hazard_radius, color="red", alpha=0.2)
        ax.add_patch(circle)
        ax.scatter(hx, hy, color="darkred", s=200, zorder=4,
                   marker="x", label="Hazard" if i == 0 else "")

    for t in metrics["goal_events"]:
        if t < len(history):
            ax.scatter(*history[t], color="lime", s=80, zorder=6, marker="o", alpha=0.9)
    for t in metrics["hazard_events"]:
        if t < len(history):
            ax.scatter(*history[t], color="red", s=80, zorder=6, marker="x", alpha=0.9)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, T))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Timestep")

    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"{title}\nGoals reached: {being.goals_reached}  |  Hazards hit: {being.hazards_hit}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    plt.tight_layout()


def plot_phase_portrait(metrics: Dict):
    traj = metrics["coordinator_trajectory"]
    T = len(traj)
    cmap = plt.cm.viridis

    pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
    labels = ["Dim 0 vs 1", "Dim 0 vs 2", "Dim 1 vs 2", "Dim 2 vs 3"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (d1, d2), label in zip(axes, pairs, labels):
        for i in range(T - 1):
            ax.plot(
                traj[i:i+2, d1], traj[i:i+2, d2],
                color=cmap(i / max(T, 1)), linewidth=1.0, alpha=0.7
            )
        ax.scatter(traj[0, d1], traj[0, d2], color="white", edgecolors="black", s=80, zorder=5)
        ax.scatter(traj[-1, d1], traj[-1, d2], color="black", edgecolors="white", s=80, zorder=5, marker="*")
        ax.set_xlabel(f"Dim {d1}")
        ax.set_ylabel(f"Dim {d2}")
        ax.set_title(label)
        ax.grid(alpha=0.3)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, T))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Timestep", shrink=0.6)
    plt.suptitle("4D Phase Portrait (Coordinator Trajectory)", fontsize=13)
    plt.tight_layout()


def plot_dominance(metrics: Dict, subsystems: List[Subsystem]):
    names = [s.name for s in subsystems]
    dom_log = metrics["dominant_subsystems"]
    T = len(dom_log)

    dom_idx = [names.index(d) if d in names else -1 for d in dom_log]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    t = np.arange(T)

    axes[0].scatter(t, dom_idx, c=dom_idx, cmap="tab10", s=8, alpha=0.7)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].set_ylabel("Dominant Subsystem")
    axes[0].set_title("Subsystem Dominance Over Time")
    axes[0].grid(alpha=0.2)

    axes[1].plot(t, metrics["coordination_pressures"], color="blue", linewidth=1.5)
    axes[1].set_ylabel("Coordination Pressure")
    axes[1].set_title("Coordination Pressure (conflict between subsystems)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, metrics["integration_levels"], color="orange", linewidth=1.5)
    axes[2].set_ylabel("Integration Level")
    axes[2].set_xlabel("Timestep")
    axes[2].set_title("Information Integration (IIT-style)")
    axes[2].grid(alpha=0.3)

    for ax in axes:
        for t_g in metrics["goal_events"]:
            ax.axvline(t_g, color="lime", alpha=0.4, linewidth=1.5)
        for t_h in metrics["hazard_events"]:
            ax.axvline(t_h, color="red", alpha=0.3, linewidth=1.0)

    plt.tight_layout()


def plot_lesion_comparison(lesion_results: Dict):
    intact = lesion_results["intact"]
    lesioned = lesion_results["lesioned"]
    name = lesion_results["lesion_name"]
    T = len(intact["coordination_pressures"])
    t = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, key, ylabel, title in [
        (axes[0, 0], "coordination_pressures", "Pressure", "Coordination Pressure"),
        (axes[0, 1], "coordinator_magnitudes", "Magnitude", "Coordinator Magnitude"),
        (axes[1, 0], "integration_levels", "Integration", "Information Integration"),
    ]:
        ax.plot(t, intact[key], color="steelblue", linewidth=1.5, label="Intact")
        ax.plot(t, lesioned[key], color="salmon", linewidth=1.5, linestyle="--", label=f"Lesion: {name}")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    ax = axes[1, 1]
    h_intact = np.array(lesion_results["intact"]["being"].history)
    h_lesioned = np.array(lesion_results["lesioned"]["being"].history)
    ax.plot(h_intact[:, 0], h_intact[:, 1], color="steelblue", linewidth=1.2, label="Intact", alpha=0.8)
    ax.plot(h_lesioned[:, 0], h_lesioned[:, 1], color="salmon", linewidth=1.2,
            linestyle="--", label=f"Lesion: {name}", alpha=0.8)
    ax.set_title("Navigation Path Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(alpha=0.2)

    il = lesion_results["level_intact"]
    ll = lesion_results["level_lesioned"]
    si = lesion_results["state_intact"]
    sl = lesion_results["state_lesioned"]
    plt.suptitle(
        f"Lesion Study: {name} removed\n"
        f"Intact: {il:.3f} ({si})  →  Lesioned: {ll:.3f} ({sl})  |  Δ = {il - ll:+.3f}",
        fontsize=12
    )
    plt.tight_layout()


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    config = SimulationConfig()

    if config.random_seed is not None:
        np.random.seed(config.random_seed)

    env = Environment()
    subsystems = initialize_subsystems(config, config.n_subsystems)
    being = initialize_being(env)

    print(f"\n{'=' * 60}")
    print("  CONSCIOUSNESS EMERGENCE + NAVIGATION SIMULATION")
    print(f"{'=' * 60}")
    print(f"\n  Config: {config.n_subsystems} subsystems | {config.n_dimensions}D manifold | {config.n_timesteps} steps")
    print(f"  Environment: {config.env_size}x{config.env_size} | {len(env.goals)} goals | {len(env.hazards)} hazards\n")

    metrics = run_simulation(config, env, being, subsystems)
    level, state = compute_consciousness_level(metrics, config)

    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  Consciousness Level:  {level:.3f} ({state})")
    print(f"  Basin-switch events: {metrics['basin_switches']}")
    print(f"  Goals reached:       {being.goals_reached}")
    print(f"  Hazards hit:         {being.hazards_hit}")
    print(f"  Avg pressure:        {np.mean(metrics['coordination_pressures']):.3f}")
    print(f"  Avg integration:     {np.mean(metrics['integration_levels']):.3f}")

    dom_counts = Counter(metrics["dominant_subsystems"])
    print("\n  Subsystem dominance (top 3):")
    for name, count in dom_counts.most_common(3):
        print(f"    {name:<20} {count} steps ({100 * count // config.n_timesteps}%)")

    print(f"\n{'=' * 60}")
    print("  EXPORT")
    print(f"{'=' * 60}")
    export_to_csv(metrics, "simulation_log.csv")

    print(f"\n{'=' * 60}")
    print("  LESION STUDY")
    print(f"{'=' * 60}")
    lesion_results = run_lesion_study(config, env, lesion_name="Planning")

    print(f"\n{'=' * 60}")
    print("  VISUALIZATIONS  (close each window to see the next)")
    print(f"{'=' * 60}\n")

    plot_navigation(metrics, env, title="Being Navigation (4D Slice → 2D Action)")
    plot_phase_portrait(metrics)
    plot_dominance(metrics, subsystems)
    plot_lesion_comparison(lesion_results)

    plt.show()

    return metrics, level, state


if __name__ == "__main__":
    main()