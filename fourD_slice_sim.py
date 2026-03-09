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

import csv
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """Configuration for the consciousness simulation."""
    n_subsystems: int = 8
    n_dimensions: int = 4          # Dimensionality of the coordination manifold
    n_timesteps: int = 300
    coordination_threshold: float = 0.85
    noise_level: float = 0.1
    learning_rate: float = 0.005
    step_size: float = 0.4         # How far the being moves per timestep
    env_size: int = 20             # 2D environment side length

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
    goal_radius: float = 1.5   # Distance at which a goal is "reached"
    hazard_radius: float = 2.0 # Distance at which a hazard is "felt"

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
    pos = np.array([env.size / 2.0, env.size / 2.0])
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
    active: bool = True  # Set False for lesion studies

def initialize_subsystems(config: SimulationConfig, n_subsystems: int) -> List[Subsystem]:
    """Initialize specialized cognitive subsystems."""
    dims = config.n_dimensions
    names = [
        "Perception", "Language", "Planning", "Emotion",
        "Memory", "Motor Control", "Attention", "Executive Control"
    ]
    return [
        Subsystem(
            name=names[i],
            activity=np.random.randn(dims) * 0.1,
            weights=np.random.randn(dims) * 0.05,
            preferred_direction=np.random.randn(dims) * 0.1,
        )
        for i in range(min(n_subsystems, len(names)))
    ]

# ────────────────────────────────────────────────────────────────────
# Sensing: subsystems respond to the environment
# ────────────────────────────────────────────────────────────────────

def sense_environment(
    being: Being,
    env: Environment,
    subsystems: List[Subsystem],
    config: SimulationConfig,
) -> None:
    """
    Each subsystem reacts differently to the environment.
    This drives the coordinator toward navigation-relevant attractors.

    - Perception   : responds to proximity of any stimulus
    - Planning     : biases activity toward nearest goal direction
    - Emotion      : activates strongly near hazards (fear/avoidance)
    - Memory       : encodes displacement from recent average position
    - Attention    : amplifies the most salient stimulus
    - Motor Control: receives the committed action signal
    - Language/Exec: internal dynamics, weakly modulated by environment
    """
    pos = being.position
    n_dim = config.n_dimensions

    goal_vecs = [np.array(g) - pos for g in env.goals]
    goal_dists = [np.linalg.norm(v) for v in goal_vecs]
    nearest_goal_idx = int(np.argmin(goal_dists))
    nearest_goal_dist = goal_dists[nearest_goal_idx]
    nearest_goal_dir = goal_vecs[nearest_goal_idx] / (nearest_goal_dist + 1e-8)

    hazard_vecs = [np.array(h) - pos for h in env.hazards]
    hazard_dists = [np.linalg.norm(v) for v in hazard_vecs]
    nearest_hazard_idx = int(np.argmin(hazard_dists))
    nearest_hazard_dist = hazard_dists[nearest_hazard_idx]
    nearest_hazard_dir = hazard_vecs[nearest_hazard_idx] / (nearest_hazard_dist + 1e-8)

    for sub in subsystems:
        if not sub.active:
            continue

        if sub.name == "Perception":
            # Responds to overall stimulus density
            strength = 1.0 / (nearest_goal_dist + 1.0) + 0.5 / (nearest_hazard_dist + 1.0)
            sub.activity += np.random.randn(n_dim) * strength * 0.08

        elif sub.name == "Planning":
            # Encodes goal direction into first 2 coordinator dims
            signal = np.zeros(n_dim)
            signal[:2] = nearest_goal_dir * 0.2
            sub.activity += signal

        elif sub.name == "Emotion":
            # Fear: strong push away from hazard, encoded as negative hazard direction
            fear_strength = max(0.0, 1.0 - nearest_hazard_dist / env.hazard_radius)
            signal = np.zeros(n_dim)
            signal[:2] = -nearest_hazard_dir * fear_strength * 0.3
            sub.activity += signal + np.random.randn(n_dim) * fear_strength * 0.05

        elif sub.name == "Memory":
            # Integrates displacement from recent average (path memory)
            if len(being.history) >= 5:
                recent = np.mean(being.history[-5:], axis=0)
                displacement = pos - recent
                signal = np.zeros(n_dim)
                signal[:2] = displacement * 0.06
                sub.activity += signal

        elif sub.name == "Attention":
            # Boosts the most salient signal (goal or hazard, whichever closer)
            goal_sal = 1.0 / (nearest_goal_dist + 1.0)
            hazard_sal = 2.0 / (nearest_hazard_dist + 0.5)  # hazards more salient
            salience = max(goal_sal, hazard_sal)
            sub.activity += np.random.randn(n_dim) * salience * 0.06

        elif sub.name == "Motor Control":
            # Receives a copy of the current intended action (efference copy)
            signal = np.zeros(n_dim)
            signal[:2] = nearest_goal_dir * 0.1
            sub.activity += signal

        else:
            # Language, Executive Control: internal noise + weak env coupling
            sub.activity += np.random.randn(n_dim) * 0.03

# ────────────────────────────────────────────────────────────────────
# Navigation: the 4D slice concept
# ────────────────────────────────────────────────────────────────────

def select_action(coordinator: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """
    Select a navigation action by slicing the 4D coordinator state.

    The '4D slice' concept:
      The coordinator exists in 4D phase space. Consciousness selects a
      2D cross-section of this manifold to commit to as the next action.
      Dimensions 0 and 1 are projected onto the XY movement plane.
      The remaining dimensions (2, 3) encode internal state not yet acted on.

    This is the 'collapse as commitment' — the being reduces its high-dimensional
    state to a single directed movement.
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
        np.linalg.norm(being.position - np.array(g)) < env.goal_radius
        for g in env.goals
    )
    hazard_hit = any(
        np.linalg.norm(being.position - np.array(h)) < env.hazard_radius
        for h in env.hazards
    )

    if goal_reached:
        being.goals_reached += 1
    if hazard_hit:
        being.hazards_hit += 1

    return goal_reached, hazard_hit

# ────────────────────────────────────────────────────────────────────
# Core coordination functions
# ────────────────────────────────────────────────────────────────────

def compute_system_state(subsystems: List[Subsystem]) -> np.ndarray:
    """Aggregate active subsystem activities into a single 1D state vector."""
    total = np.zeros(len(subsystems[0].activity), dtype=np.float64)
    for sub in subsystems:
        if sub.active:
            total += sub.activity
    return total

def compute_coordination_pressure(subsystems: List[Subsystem]) -> float:
    """
    Coordination pressure: how much active subsystems disagree.
    Range [0, 1] — higher means more conflict, more need for a coordinator.
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
    # Start with stronger signal so action selection works from step 1
    return np.random.randn(config.n_dimensions) * 0.5

def coordinator_update(
    coordinator: np.ndarray,
    subsystems: List[Subsystem],
    config: SimulationConfig,
) -> np.ndarray:
    """
    Update coordinator toward a weighted average of active subsystem inputs.
    Acts as a global workspace: integrates, then broadcasts.
    """
    n_dim = len(coordinator)
    lr = config.learning_rate

    active = [s for s in subsystems if s.active]
    if not active:
        return coordinator

    norms = np.array([np.linalg.norm(s.activity) for s in active])
    total_norm = norms.sum() + 1e-8
    weighted_input = sum(
        s.activity * (norms[i] / total_norm) for i, s in enumerate(active)
    )

    coordinator_output = np.dot(coordinator, weighted_input)  # scalar alignment
    coordination_goal = np.ones(n_dim) / n_dim
    goal_signal = coordination_goal * coordinator_output

    noise = np.random.randn(n_dim) * config.noise_level

    return (
        (1 - lr) * coordinator
        + lr * weighted_input
        + lr * goal_signal
        + noise
    )

def attractor_update(
    state: np.ndarray,
    basin_attractor: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """Pull state toward a basin of attraction (commitment)."""
    delta = config.learning_rate * (basin_attractor - state)
    noise = np.random.randn(len(state)) * config.noise_level
    state = state + delta + noise
    norm = np.linalg.norm(state)
    if norm > 1e-8:
        state = state / norm
    return state

def basin_switch_event(
    state: np.ndarray,
    basin_attractors: List[np.ndarray],
    config: SimulationConfig,
) -> Tuple[np.ndarray, bool]:
    """
    Check for basin-switch (decision) events.
    When equidistant from competing basins, noise breaks symmetry.
    """
    distances = []
    for att in basin_attractors:
        ns = np.linalg.norm(state)
        na = np.linalg.norm(att)
        if ns < 1e-8 or na < 1e-8:
            distances.append(0.0)
            continue
        cos_sim = np.clip(np.dot(state, att) / (ns * na), -1.0, 1.0)
        distances.append(-np.arccos(cos_sim))
    distances = np.array(distances)

    spread = np.max(distances) - np.min(distances)
    next_idx = int(np.argmin(distances))

    if spread < 0.3:  # near-equidistant → decision point
        if np.random.rand() > 0.5:
            next_idx = (next_idx + 1) % len(basin_attractors)
            return basin_attractors[next_idx], True

    return basin_attractors[next_idx], False

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

    Each timestep:
      1. Subsystems sense the environment
      2. Coordinator integrates subsystem state
      3. Coordinator selects an action (4D slice → 2D movement)
      4. Being moves; goals/hazards are recorded
      5. Metrics are logged
    """
    coordinator = initialize_coordinator(config)

    n_basins = 5
    basin_attractors = []
    for _ in range(n_basins):
        v = np.random.randn(config.n_dimensions)
        v = v / (np.linalg.norm(v) + 1e-8)
        v += np.random.randn(config.n_dimensions) * 0.05
        basin_attractors.append(v)

    # Metrics
    coordination_pressures = []
    coordinator_magnitudes = []
    coordinator_trajectory = []  # full 4D state per step
    integration_levels = []
    dominant_subsystems = []
    basin_switches = 0
    goal_events = []
    hazard_events = []
    step_log = []

    for t in range(config.n_timesteps):
        # 1. Sense environment
        sense_environment(being, env, subsystems, config)

        # 2. Compute state metrics
        subsystem_state = compute_system_state(subsystems)
        pressure = compute_coordination_pressure(subsystems)
        dom = dominant_subsystem(subsystems)

        coordination_pressures.append(pressure)
        coordinator_magnitudes.append(np.linalg.norm(coordinator))
        coordinator_trajectory.append(coordinator.copy())
        dominant_subsystems.append(dom)

        active_norms = [np.linalg.norm(s.activity) for s in subsystems if s.active]
        total_act = sum(active_norms)
        integration = min(1.0, total_act / config.n_subsystems)
        integration_levels.append(integration)

        # 3. Update coordinator
        coordinator = coordinator_update(coordinator, subsystems, config)

        # 4. Basin switch (decision)
        if t > 10:
            next_basin, switched = basin_switch_event(coordinator, basin_attractors, config)
            if switched:
                basin_switches += 1
                coordinator = attractor_update(coordinator, next_basin, config)

        # 5. Select action and move
        action = select_action(coordinator, config)
        goal_reached, hazard_hit = move_being(being, action, env)

        if goal_reached:
            goal_events.append(t)
        if hazard_hit:
            hazard_events.append(t)

        # 6. Broadcast coordinator back to subsystems
        for sub in subsystems:
            if sub.active:
                sub.activity += coordinator * 0.05
                # Damping: prevent activity explosion
                norm = np.linalg.norm(sub.activity)
                if norm > 2.0:
                    sub.activity = sub.activity / norm * 2.0

        # Clamp coordinator magnitude to prevent overflow
        c_norm = np.linalg.norm(coordinator)
        if c_norm > 5.0:
            coordinator = coordinator / c_norm * 5.0

        # 7. Per-step log entry
        step_log.append({
            "t": t,
            "x": float(being.position[0]),
            "y": float(being.position[1]),
            "pressure": float(pressure),
            "coordinator_magnitude": float(np.linalg.norm(coordinator)),
            "integration": float(integration),
            "dominant": dom,
            "basin_switch": switched if t > 10 else False,
            "goal_reached": goal_reached,
            "hazard_hit": hazard_hit,
        })

    return {
        "coordination_pressures": coordination_pressures,
        "coordinator_magnitudes": coordinator_magnitudes,
        "coordinator_trajectory": np.array(coordinator_trajectory),
        "integration_levels": integration_levels,
        "dominant_subsystems": dominant_subsystems,
        "basin_switches": basin_switches,
        "goal_events": goal_events,
        "hazard_events": hazard_events,
        "step_log": step_log,
        "being": being,
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
    avg_pressure = np.mean(pressures[n // 2:])
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
# Study tool: lesion study
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

    # Intact run
    np.random.seed(42)
    subs_intact = initialize_subsystems(config, config.n_subsystems)
    being_intact = initialize_being(env)
    metrics_intact = run_simulation(config, env, being_intact, subs_intact)
    level_intact, state_intact = compute_consciousness_level(metrics_intact, config)

    # Lesioned run
    np.random.seed(42)
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
# Study tool: CSV export
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
    """
    Plot the being's path through the 2D environment.
    Color encodes time (early=dark, late=bright). Goals and hazards are marked.
    """
    being = metrics["being"]
    history = np.array(being.history)
    T = len(history)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Path colored by time
    cmap = plt.cm.plasma
    for i in range(T - 1):
        ax.plot(
            history[i:i+2, 0], history[i:i+2, 1],
            color=cmap(i / T), linewidth=1.2, alpha=0.8
        )

    # Start and end
    ax.scatter(*history[0], color='white', edgecolors='black', s=120, zorder=5, label='Start')
    ax.scatter(*history[-1], color='black', edgecolors='white', s=120, zorder=5, label='End', marker='*')

    # Goals
    for i, (gx, gy) in enumerate(env.goals):
        circle = plt.Circle((gx, gy), env.goal_radius, color='lime', alpha=0.3)
        ax.add_patch(circle)
        ax.scatter(gx, gy, color='lime', edgecolors='darkgreen', s=200, zorder=4,
                   marker='^', label='Goal' if i == 0 else '')

    # Hazards
    for i, (hx, hy) in enumerate(env.hazards):
        circle = plt.Circle((hx, hy), env.hazard_radius, color='red', alpha=0.2)
        ax.add_patch(circle)
        ax.scatter(hx, hy, color='red', edgecolors='darkred', s=200, zorder=4,
                   marker='x', label='Hazard' if i == 0 else '')

    # Goal/hazard events
    goal_ts = metrics["goal_events"]
    hazard_ts = metrics["hazard_events"]
    for t in goal_ts:
        if t < len(history):
            ax.scatter(*history[t], color='lime', s=80, zorder=6, marker='o', alpha=0.9)
    for t in hazard_ts:
        if t < len(history):
            ax.scatter(*history[t], color='red', s=80, zorder=6, marker='x', alpha=0.9)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, T))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Timestep')

    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title}\nGoals reached: {being.goals_reached}  |  Hazards hit: {being.hazards_hit}')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)

    plt.tight_layout()

def plot_phase_portrait(metrics: Dict):
    """
    Phase portrait: coordinator trajectory projected onto 2D dimension pairs.
    Shows identity orbits and basin-switch events in phase space.
    """
    traj = metrics["coordinator_trajectory"]  # shape (T, 4)
    T = len(traj)
    cmap = plt.cm.viridis

    pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
    labels = ['Dim 0 vs 1', 'Dim 0 vs 2', 'Dim 1 vs 2', 'Dim 2 vs 3']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (d1, d2), label in zip(axes, pairs, labels):
        for i in range(T - 1):
            ax.plot(
                traj[i:i+2, d1], traj[i:i+2, d2],
                color=cmap(i / T), linewidth=1.0, alpha=0.7
            )
        ax.scatter(traj[0, d1], traj[0, d2], color='white', edgecolors='black', s=80, zorder=5)
        ax.scatter(traj[-1, d1], traj[-1, d2], color='black', edgecolors='white', s=80, zorder=5, marker='*')
        ax.set_xlabel(f'Dim {d1}')
        ax.set_ylabel(f'Dim {d2}')
        ax.set_title(label)
        ax.grid(alpha=0.3)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, T))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label='Timestep', shrink=0.6)
    plt.suptitle('4D Phase Portrait (Coordinator Trajectory)', fontsize=13)
    plt.tight_layout()

def plot_dominance(metrics: Dict, subsystems: List[Subsystem]):
    """
    Subsystem dominance over time: which subsystem is steering at each step.
    Displayed as a categorical timeline.
    """
    names = [s.name for s in subsystems]
    dom_log = metrics["dominant_subsystems"]
    T = len(dom_log)

    dom_idx = [names.index(d) if d in names else -1 for d in dom_log]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    t = np.arange(T)

    # Dominance timeline
    axes[0].scatter(t, dom_idx, c=dom_idx, cmap='tab10', s=8, alpha=0.7)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].set_ylabel('Dominant Subsystem')
    axes[0].set_title('Subsystem Dominance Over Time')
    axes[0].grid(alpha=0.2)

    # Coordination pressure
    axes[1].plot(t, metrics["coordination_pressures"], color='blue', linewidth=1.5)
    axes[1].set_ylabel('Coordination Pressure')
    axes[1].set_title('Coordination Pressure (conflict between subsystems)')
    axes[1].grid(alpha=0.3)

    # Integration level
    axes[2].plot(t, metrics["integration_levels"], color='orange', linewidth=1.5)
    axes[2].set_ylabel('Integration Level')
    axes[2].set_xlabel('Timestep')
    axes[2].set_title('Information Integration (IIT-style)')
    axes[2].grid(alpha=0.3)

    # Mark goal and hazard events
    for ax in axes:
        for t_g in metrics["goal_events"]:
            ax.axvline(t_g, color='lime', alpha=0.4, linewidth=1.5)
        for t_h in metrics["hazard_events"]:
            ax.axvline(t_h, color='red', alpha=0.3, linewidth=1.0)

    plt.tight_layout()

def plot_lesion_comparison(lesion_results: Dict):
    """Compare intact vs lesioned run: pressure, magnitude, integration, position."""
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
        ax.plot(t, intact[key], color='steelblue', linewidth=1.5, label='Intact')
        ax.plot(t, lesioned[key], color='salmon', linewidth=1.5, linestyle='--', label=f'Lesion: {name}')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    # Navigation comparison
    ax = axes[1, 1]
    h_intact = np.array(lesion_results["intact"]["being"].history)
    h_lesioned = np.array(lesion_results["lesioned"]["being"].history)
    ax.plot(h_intact[:, 0], h_intact[:, 1], color='steelblue', linewidth=1.2, label='Intact', alpha=0.8)
    ax.plot(h_lesioned[:, 0], h_lesioned[:, 1], color='salmon', linewidth=1.2,
            linestyle='--', label=f'Lesion: {name}', alpha=0.8)
    ax.set_title('Navigation Path Comparison')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(alpha=0.2)

    il = lesion_results["level_intact"]
    ll = lesion_results["level_lesioned"]
    si = lesion_results["state_intact"]
    sl = lesion_results["state_lesioned"]
    plt.suptitle(
        f'Lesion Study: {name} removed\n'
        f'Intact: {il:.3f} ({si})  →  Lesioned: {ll:.3f} ({sl})  |  Δ = {il - ll:+.3f}',
        fontsize=12
    )
    plt.tight_layout()

# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    config = SimulationConfig(
        n_subsystems=8,
        n_dimensions=4,
        n_timesteps=300,
        coordination_threshold=0.85,
        noise_level=0.1,
        learning_rate=0.005,
        step_size=0.4,
        env_size=20,
    )

    env = Environment()
    subsystems = initialize_subsystems(config, config.n_subsystems)
    being = initialize_being(env)

    print(f"\n{'=' * 60}")
    print(f"  CONSCIOUSNESS EMERGENCE + NAVIGATION SIMULATION")
    print(f"{'=' * 60}")
    print(f"\n  Config: {config.n_subsystems} subsystems | {config.n_dimensions}D manifold | {config.n_timesteps} steps")
    print(f"  Environment: {config.env_size}x{config.env_size} | {len(env.goals)} goals | {len(env.hazards)} hazards\n")

    metrics = run_simulation(config, env, being, subsystems)
    level, state = compute_consciousness_level(metrics, config)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  Consciousness Level:  {level:.3f} ({state})")
    print(f"  Basin-switch events: {metrics['basin_switches']}")
    print(f"  Goals reached:       {being.goals_reached}")
    print(f"  Hazards hit:         {being.hazards_hit}")
    print(f"  Avg pressure:        {np.mean(metrics['coordination_pressures']):.3f}")
    print(f"  Avg integration:     {np.mean(metrics['integration_levels']):.3f}")

    # Dominance summary
    from collections import Counter
    dom_counts = Counter(metrics["dominant_subsystems"])
    print(f"\n  Subsystem dominance (top 3):")
    for name, count in dom_counts.most_common(3):
        print(f"    {name:<20} {count} steps ({100*count//config.n_timesteps}%)")

    # Export CSV
    print(f"\n{'=' * 60}")
    print(f"  EXPORT")
    print(f"{'=' * 60}")
    export_to_csv(metrics, "simulation_log.csv")

    # Lesion study
    print(f"\n{'=' * 60}")
    print(f"  LESION STUDY")
    print(f"{'=' * 60}")
    lesion_results = run_lesion_study(config, env, lesion_name="Planning")

    # Plots
    print(f"\n{'=' * 60}")
    print(f"  VISUALIZATIONS  (close each window to see the next)")
    print(f"{'=' * 60}\n")

    plot_navigation(metrics, env, title="Being Navigation (4D Slice → 2D Action)")
    plot_phase_portrait(metrics)
    plot_dominance(metrics, subsystems)
    plot_lesion_comparison(lesion_results)

    plt.show()

    return metrics, level, state

if __name__ == "__main__":
    main()
