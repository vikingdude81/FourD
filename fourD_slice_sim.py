#!/usr/bin/env python3
"""
Consciousness Simulation: Dual-Geometry 4D Manifold + Navigation

This simulation implements a dual-layer geometric structure for consciousness modeling:
- 600-cell layer (microstate/local): fine-grained transition mesh, attentional shifts
- 120-cell layer (macrostate/global): stable modes of consciousness, worldview enclosure

Framework: Consciousness as dual organization where local micro-dynamics remain richly
differentiated while global closure remains stable enough to bind them into one integrated mode.

Run: python fourD_slice_sim.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
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
    n_dimensions: int = 4          # Dimensionality of coordination manifold
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
            strength = 1.0 / (nearest_goal_dist + 1.0) + 0.5 / (nearest_hazard_dist + 1.0)
            sub.activity += np.random.randn(n_dim) * strength * 0.08

        elif sub.name == "Planning":
            signal = np.zeros(n_dim)
            signal[:2] = nearest_goal_dir * 0.2
            sub.activity += signal

        elif sub.name == "Emotion":
            fear_strength = max(0.0, 1.0 - nearest_hazard_dist / env.hazard_radius)
            signal = np.zeros(n_dim)
            signal[:2] = -nearest_hazard_dir * fear_strength * 0.3
            sub.activity += signal + np.random.randn(n_dim) * fear_strength * 0.05

        elif sub.name == "Memory":
            if len(being.history) >= 5:
                recent = np.mean(being.history[-5:], axis=0)
                displacement = pos - recent
                signal = np.zeros(n_dim)
                signal[:2] = displacement * 0.06
                sub.activity += signal

        elif sub.name == "Attention":
            goal_sal = 1.0 / (nearest_goal_dist + 1.0)
            hazard_sal = 2.0 / (nearest_hazard_dist + 0.5)
            salience = max(goal_sal, hazard_sal)
            sub.activity += np.random.randn(n_dim) * salience * 0.06

        elif sub.name == "Motor Control":
            signal = np.zeros(n_dim)
            signal[:2] = nearest_goal_dir * 0.1
            sub.activity += signal

        else:
            sub.activity += np.random.randn(n_dim) * 0.03

# ────────────────────────────────────────────────────────────────────
# Navigation: the 4D slice concept
# ────────────────────────────────────────────────────────────────────

def select_action(coordinator: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """
    Select a navigation action by slicing the 4D coordinator state.
    
    The '4D slice' concept: Dimensions 0 and 1 are projected onto the 
    XY movement plane. Remaining dimensions encode internal state.
    """
    action = coordinator[:2].copy()
    norm = np.linalg.norm(action)
    if norm > 1e-8:
        action = action / norm
    return action * config.step_size

def move_being(being: Being, action: np.ndarray, env: Environment) -> Tuple[bool, bool]:
    """Move the being and check for goal/hazard events."""
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
    """Aggregate active subsystem activities into a single state vector."""
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
            act_i = active[i].activity
            act_j = active[j].activity
            
            ni = np.linalg.norm(act_i)
            nj = np.linalg.norm(act_j)
            
            if ni < 1e-8 or nj < 1e-8:
                continue
                
            cos_sim = np.clip(
                np.dot(act_i, act_j) / (ni * nj),
                -1.0, 1.0
            )
            angle_diff = np.arccos(np.clip(cos_sim, -1.0, 1.0))
            
            avg_norm = (ni + nj) / 2
            conflict_factor = (angle_diff / np.pi) * avg_norm
            
            total_conflict += conflict_factor
            n_pairs += 1

    if n_pairs == 0:
        return 0.0
    
    max_possible = np.pi / 2.0
    pressure = total_conflict / (n_pairs * max_possible)
    pressure = min(1.5, pressure * 2.0)
    
    return float(np.clip(pressure, 0.0, 1.5))

def dominant_subsystem(subsystems: List[Subsystem]) -> str:
    """Return the name of the subsystem with the highest activity magnitude."""
    active = [s for s in subsystems if s.active]
    if not active:
        return "none"
    return max(active, key=lambda s: np.linalg.norm(s.activity)).name

def initialize_coordinator(config: SimulationConfig) -> np.ndarray:
    """Initialize coordinator with stronger signal."""
    return np.random.randn(config.n_dimensions) * 0.5

def normalize_to_hypersphere(state: np.ndarray) -> np.ndarray:
    """Normalize state to live on a bounded hyperspherical manifold in 4D."""
    norm = np.linalg.norm(state)
    if norm > 1e-8:
        return state / norm
    return state

def coordinator_update(
    coordinator: np.ndarray,
    subsystems: List[Subsystem],
    config: SimulationConfig,
) -> np.ndarray:
    """Update coordinator toward a weighted average of active subsystem inputs."""
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

    coordinator_output = np.dot(coordinator, weighted_input)
    coordination_goal = np.ones(n_dim) / n_dim
    goal_signal = coordination_goal * coordinator_output

    noise = np.random.randn(n_dim) * config.noise_level

    new_coordinator = (
        (1 - lr) * coordinator
        + lr * weighted_input
        + lr * goal_signal
        + noise
    )
    
    # Normalize to hyperspherical manifold
    return normalize_to_hypersphere(new_coordinator)

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
    """Check for basin-switch (decision) events with enhanced dynamics."""
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

    if spread < 0.5:
        c_norm = np.linalg.norm(state)
        base_probability = 0.3 + (0.5 - spread) * 2.0
        strength_bonus = min(0.4, (c_norm - 1.0) * 0.1) if c_norm > 1.0 else 0.0
        decision_threshold = max(0.3, min(0.8, base_probability + strength_bonus))
        
        if np.random.rand() < decision_threshold:
            next_idx = (next_idx + 1) % len(basin_attractors)
            return basin_attractors[next_idx], True

    return basin_attractors[next_idx], False

# ────────────────────────────────────────────────────────────────────
# Dual-Geometry Layer: 600-cell and 120-cell inspired structures
# ────────────────────────────────────────────────────────────────────

@dataclass
class MicroTransition:
    """A microstate transition in the 600-cell-like mesh."""
    from_node: int
    to_node: int
    simplex_id: int  # Which 4-simplex this belongs to
    weight: float = 1.0

@dataclass
class MacroBasin:
    """A macrostate basin in the 120-cell-like closure layer."""
    basin_id: int
    name: str
    center: np.ndarray
    radius: float = 0.3
    residence_time: int = 0
    
    # Mode associations
    is_exploratory: bool = False
    is_threat_locked: bool = False
    is_reflective: bool = False
    is_flow: bool = False
    is_fragmented: bool = False

class DualGeometryLayer:
    """
    Implements the 120-cell/600-cell duality for consciousness modeling.
    
    - Micro layer (600-cell-like): fine-grained transition mesh
    - Macro layer (120-cell-like): stable modes of consciousness
    
    The dual mapping ensures every local micro-transition implies a 
    global macrostate, and every global state constrains transitions.
    """
    
    def __init__(self, n_dimensions: int = 4, n_macro_basins: int = 5):
        self.n_dimensions = n_dimensions
        self.n_macro_basins = n_macro_basins
        
        # Generate macro basin centers (120-cell inspired)
        self.basins = []
        
        # Generate micro transition nodes and edges (600-cell inspired)
        self.micro_nodes = []
        self.transition_graph = {}
        
        # Dual mapping: which macro basin each micro node belongs to
        self.node_to_basin = {}
        
        # Build the geometry layers
        self.basins = self._generate_macro_basins()
        self.micro_nodes, self.transition_graph = self._generate_micro_structure()
        self.node_to_basin = self._build_dual_mapping()
    
    def _generate_macro_basins(self) -> List[MacroBasin]:
        """Generate macro basins representing stable modes of consciousness."""
        centers = []
        
        # Generate diverse attractor points in 4D hypersphere
        for i in range(self.n_macro_basins):
            v = np.random.randn(self.n_dimensions)
            v = normalize_to_hypersphere(v)
            centers.append(MacroBasin(
                basin_id=i,
                name=f"Mode_{i}",
                center=v,
                radius=0.25 + 0.1 * (i % 3),
            ))
        
        # Assign semantic meanings to basins
        if len(centers) > 0:
            centers[0].is_exploratory = True
            centers[1].is_threat_locked = True
            if len(centers) > 2:
                centers[2].is_reflective = True
            if len(centers) > 3:
                centers[3].is_flow = True
            if len(centers) > 4:
                centers[4].is_fragmented = True
        
        return centers
    
    def _generate_micro_structure(self) -> Tuple[List[np.ndarray], Dict[int, List[int]]]:
        """Generate micro transition nodes and adjacency graph (600-cell-like)."""
        # Generate more nodes for fine-grained transitions
        n_nodes = 24  # More nodes for richer local dynamics
        
        nodes = []
        for i in range(n_nodes):
            v = np.random.randn(self.n_dimensions)
            v = normalize_to_hypersphere(v)
            nodes.append(v)
        
        # Build adjacency: each node connected to k nearest neighbors
        k_neighbors = 6
        graph = {}
        
        for i, node in enumerate(nodes):
            distances = [(j, np.linalg.norm(node - nodes[j])) 
                        for j in range(len(nodes)) if j != i]
            distances.sort(key=lambda x: x[1])
            neighbors = [idx for idx, _ in distances[:k_neighbors]]
            graph[i] = neighbors
        
        return nodes, graph
    
    def _build_dual_mapping(self) -> Dict[int, int]:
        """Build mapping from micro nodes to macro basins."""
        mapping = {}
        
        for node_idx, node_pos in enumerate(self.micro_nodes):
            # Find nearest basin center
            distances = [(b.basin_id, np.linalg.norm(node_pos - b.center)) 
                        for b in self.basins]
            nearest_basin = min(distances, key=lambda x: x[1])[0]
            mapping[node_idx] = nearest_basin
        
        return mapping
    
    def get_current_macro_basin(self, state: np.ndarray) -> Tuple[int, float]:
        """Find which macro basin the current state occupies."""
        distances = [(b.basin_id, np.linalg.norm(state - b.center)) 
                    for b in self.basins]
        
        nearest_basin_id, distance = min(distances, key=lambda x: x[1])
        is_within = distance < self.basins[nearest_basin_id].radius
        
        return nearest_basin_id, float(distance)
    
    def get_current_micro_node(self, state: np.ndarray) -> int:
        """Find which micro node the current state is closest to."""
        distances = [(i, np.linalg.norm(state - node)) 
                    for i, node in enumerate(self.micro_nodes)]
        
        nearest_idx, _ = min(distances, key=lambda x: x[1])
        return nearest_idx
    
    def get_allowed_transitions(self, current_basin_id: int) -> List[int]:
        """Get micro transitions allowed by the current macro basin."""
        # Basin constrains which local transitions are possible
        allowed = []
        
        for node_idx, basin_id in self.node_to_basin.items():
            if basin_id == current_basin_id and node_idx in self.transition_graph:
                allowed.extend(self.transition_graph[node_idx])
        
        return list(set(allowed))  # Unique transitions
    
    def compute_closure_coherence(self, state: np.ndarray) -> float:
        """
        Compute how well local dynamics fit the current global state.
        
        Higher coherence = more integrated consciousness-like state.
        """
        basin_id, distance = self.get_current_macro_basin(state)
        basin = self.basins[basin_id]
        
        # Base coherence from proximity to basin center
        base_coherence = max(0, 1 - distance / (2 * basin.radius))
        
        return float(base_coherence)

# ────────────────────────────────────────────────────────────────────
# Main simulation loop with dual-geometry dynamics
# ────────────────────────────────────────────────────────────────────

def run_simulation(
    config: SimulationConfig,
    env: Environment,
    being: Being,
    subsystems: List[Subsystem],
) -> Dict:
    """
    Run the full consciousness + navigation simulation with dual-geometry dynamics.
    
    Each timestep alternates between:
    1. Local micro-transition (600-cell layer): differentiate
    2. Global macrostate reconciliation (120-cell layer): integrate
    """
    coordinator = initialize_coordinator(config)
    geometry_layer = DualGeometryLayer(n_dimensions=config.n_dimensions, n_macro_basins=5)

    # Metrics tracking
    coordination_pressures = []
    coordinator_magnitudes = []
    coordinator_trajectory = []
    integration_levels = []
    dominant_subsystems = []
    
    micro_transition_count = 0
    macro_state_changes = 0
    
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

        # 3. Update coordinator with dual-geometry dynamics
        
        # Local micro-transition step (600-cell layer): differentiate
        current_micro_node = geometry_layer.get_current_micro_node(coordinator)
        
        # Subsystem activity drives micro transitions within allowed set
        basin_id, _ = geometry_layer.get_current_macro_basin(coordinator)
        allowed_transitions = geometry_layer.get_allowed_transitions(basin_id)
        
        if allowed_transitions and np.random.rand() < 0.3:
            # Micro transition driven by dominant subsystem pressure
            next_micro_node = np.random.choice(allowed_transitions)
            micro_transition_count += 1
        
        coordinator = coordinator_update(coordinator, subsystems, config)

        # Global macrostate reconciliation (120-cell layer): integrate
        new_basin_id, _ = geometry_layer.get_current_macro_basin(coordinator)
        if new_basin_id != basin_id:
            macro_state_changes += 1
        
        # 4. Select action and move
        action = select_action(coordinator, config)
        goal_reached, hazard_hit = move_being(being, action, env)

        if goal_reached:
            goal_events.append(t)
        if hazard_hit:
            hazard_events.append(t)

        # 5. Broadcast coordinator back to subsystems
        for sub in subsystems:
            if sub.active:
                sub.activity += coordinator * 0.05
                norm = np.linalg.norm(sub.activity)
                if norm > 2.0:
                    sub.activity = sub.activity / norm * 2.0

        # Clamp coordinator magnitude to prevent overflow
        c_norm = np.linalg.norm(coordinator)
        if c_norm > 5.0:
            coordinator = coordinator / c_norm * 5.0

        # Per-step log entry with dual-geometry info
        step_log.append({
            "t": t,
            "x": float(being.position[0]),
            "y": float(being.position[1]),
            "pressure": float(pressure),
            "coordinator_magnitude": float(np.linalg.norm(coordinator)),
            "integration": float(integration),
            "dominant": dom,
            "micro_node": current_micro_node,
            "macro_basin_id": new_basin_id,
            "closure_coherence": float(geometry_layer.compute_closure_coherence(coordinator)),
            "goal_reached": goal_reached,
            "hazard_hit": hazard_hit,
        })

    return {
        "coordination_pressures": coordination_pressures,
        "coordinator_magnitudes": coordinator_magnitudes,
        "coordinator_trajectory": np.array(coordinator_trajectory),
        "integration_levels": integration_levels,
        "dominant_subsystems": dominant_subsystems,
        "basin_switches": macro_state_changes,
        "micro_transitions": micro_transition_count,
        "goal_events": goal_events,
        "hazard_events": hazard_events,
        "step_log": step_log,
        "being": being,
        "geometry_layer": geometry_layer,
    }

# ────────────────────────────────────────────────────────────────────
# Consciousness level with dual-geometry awareness
# ────────────────────────────────────────────────────────────────────

def compute_consciousness_level(metrics: Dict, config: SimulationConfig) -> Tuple[float, str]:
    pressures = metrics["coordination_pressures"]
    magnitudes = metrics["coordinator_magnitudes"]
    integrations = metrics["integration_levels"]
    switches = metrics["basin_switches"]
    
    # Include dual-geometry metrics if available
    micro_transitions = metrics.get("micro_transitions", 0)
    
    n = len(pressures)
    avg_pressure = np.mean(pressures[n // 2:])
    avg_magnitude = np.mean(magnitudes)
    avg_integration = np.mean(integrations)
    switch_factor = min(1.0, switches / 10.0)
    
    # Micro-activity factor: rich local dynamics
    micro_factor = min(1.0, micro_transitions / 30.0)

    level = (
        0.25 * min(1.0, avg_pressure / config.coordination_threshold)
        + 0.15 * min(1.0, avg_magnitude)
        + 0.25 * avg_integration
        + 0.15 * switch_factor
        + 0.20 * micro_factor
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
# Study tool: lesion study with dual-geometry analysis
# ────────────────────────────────────────────────────────────────────

def run_lesion_study(
    config: SimulationConfig,
    env: Environment,
    lesion_name: str,
) -> Dict:
    """Run intact vs lesioned simulation comparison."""
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
# Study tool: CSV export with dual-geometry columns
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
# Visualization with dual-geometry views
# ────────────────────────────────────────────────────────────────────

def plot_navigation(metrics: Dict, env: Environment, title: str = "Being Navigation"):
    """Plot the being's path through the 2D environment."""
    being = metrics["being"]
    history = np.array(being.history)
    T = len(history)

    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = plt.cm.plasma
    for i in range(T - 1):
        ax.plot(
            history[i:i+2, 0], history[i:i+2, 1],
            color=cmap(i / T), linewidth=1.2, alpha=0.8
        )

    ax.scatter(*history[0], color='white', edgecolors='black', s=120, zorder=5, label='Start')
    ax.scatter(*history[-1], color='black', edgecolors='white', s=120, zorder=5, label='End', marker='*')

    for i, (gx, gy) in enumerate(env.goals):
        circle = plt.Circle((gx, gy), env.goal_radius, color='lime', alpha=0.3)
        ax.add_patch(circle)
        ax.scatter(gx, gy, color='lime', edgecolors='darkgreen', s=200, zorder=4,
                   marker='^', label='Goal' if i == 0 else '')

    for i, (hx, hy) in enumerate(env.hazards):
        circle = plt.Circle((hx, hy), env.hazard_radius, color='red', alpha=0.2)
        ax.add_patch(circle)
        ax.scatter(hx, hy, color='red', edgecolors='darkred', s=200, zorder=4,
                   marker='x', label='Hazard' if i == 0 else '')

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

def plot_micro_macro_layers(metrics: Dict):
    """
    Plot dual-geometry layers: micro transition activity and macro basin occupancy.
    
    View 2: Micro transition activity (600-cell layer)
    View 3: Macro closure occupancy (120-cell layer)
    """
    geometry = metrics["geometry_layer"]
    log = metrics["step_log"]
    
    T = len(log)
    t = np.arange(T)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Micro transition activity (600-cell layer)
    ax = axes[0]
    micro_nodes = [log[i]["micro_node"] for i in range(T)]
    ax.scatter(t, micro_nodes, c=micro_nodes, cmap='tab20', s=15, alpha=0.7)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Micro Node (600-cell layer)')
    ax.set_title('Micro Transition Activity\nFine-grained local dynamics')
    ax.grid(alpha=0.3)
    
    # Macro basin occupancy (120-cell layer)
    ax = axes[1]
    macro_basins = [log[i]["macro_basin_id"] for i in range(T)]
    ax.scatter(t, macro_basins, c=macro_basins, cmap='tab10', s=15, alpha=0.7)
    
    # Mark basin change points
    changes = []
    for i in range(1, T):
        if macro_basins[i] != macro_basins[i-1]:
            changes.append(i)
    for c in changes:
        ax.axvline(c, color='black', alpha=0.5, linewidth=0.5)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Macro Basin (120-cell layer)')
    ax.set_title('Macro Closure Occupancy\nStable modes of consciousness')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()

def plot_closure_coherence(metrics: Dict):
    """Plot closure coherence over time."""
    log = metrics["step_log"]
    t = np.arange(len(log))
    coherences = [log[i]["closure_coherence"] for i in range(len(log))]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, coherences, color='purple', linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Closure Coherence')
    ax.set_title('Closure Coherence\nHow well local dynamics fit global state')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()

def plot_phase_portrait(metrics: Dict):
    """Phase portrait of coordinator trajectory in 4D."""
    traj = metrics["coordinator_trajectory"]
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
    """Plot subsystem dominance and coordination metrics."""
    names = [s.name for s in subsystems]
    dom_log = metrics["dominant_subsystems"]
    T = len(dom_log)

    dom_idx = [names.index(d) if d in names else -1 for d in dom_log]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    t = np.arange(T)

    axes[0].scatter(t, dom_idx, c=dom_idx, cmap='tab10', s=8, alpha=0.7)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].set_ylabel('Dominant Subsystem')
    axes[0].set_title('Subsystem Dominance Over Time')
    axes[0].grid(alpha=0.2)

    axes[1].plot(t, metrics["coordination_pressures"], color='blue', linewidth=1.5)
    axes[1].set_ylabel('Coordination Pressure')
    axes[1].set_title('Coordination Pressure (conflict between subsystems)')
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, metrics["integration_levels"], color='orange', linewidth=1.5)
    axes[2].set_ylabel('Integration Level')
    axes[2].set_xlabel('Timestep')
    axes[2].set_title('Information Integration (IIT-style)')
    axes[2].grid(alpha=0.3)

    for ax in axes:
        for t_g in metrics["goal_events"]:
            ax.axvline(t_g, color='lime', alpha=0.4, linewidth=1.5)
        for t_h in metrics["hazard_events"]:
            ax.axvline(t_h, color='red', alpha=0.3, linewidth=1.0)

    plt.tight_layout()

def plot_lesion_comparison(lesion_results: Dict):
    """Compare intact vs lesioned run."""
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
# CLI argument parsing
# ────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 fourD_slice_sim.py",
        description="Consciousness Emergence + Navigation Simulation"
    )
    p.add_argument("--n-subsystems", type=int, default=None, help="Number of subsystems")
    p.add_argument("--n-dims", type=int, default=None, help="Coordinator dimensionality")
    p.add_argument("--n-timesteps", type=int, default=None, help="Number of simulation steps")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--perception-gain", type=float, default=None, help="Perception subsystem gain")
    p.add_argument("--language-gain", type=float, default=None, help="Language subsystem gain")
    p.add_argument("--planning-gain", type=float, default=None, help="Planning subsystem gain")
    p.add_argument("--emotion-gain", type=float, default=None, help="Emotion subsystem gain")
    p.add_argument("--memory-gain", type=float, default=None, help="Memory subsystem gain")
    p.add_argument("--motor-gain", type=float, default=None, help="Motor Control subsystem gain")
    p.add_argument("--attention-gain", type=float, default=None, help="Attention subsystem gain")
    p.add_argument("--executive-gain", type=float, default=None, help="Executive Control subsystem gain")
    p.add_argument("--basin-ambiguity-threshold", type=float, default=None, help="Basin switching ambiguity threshold")
    p.add_argument("--basin-pull-strength", type=float, default=None, help="Basin attractor pull strength")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory (default: outputs/)")
    return p


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None):
    import os
    from datetime import datetime
    
    parser = _build_parser()
    args = parser.parse_args(argv)
    
    # Start with default config, then override with CLI arguments
    config = SimulationConfig()
    
    # Override config values with CLI arguments if provided
    if args.n_subsystems is not None:
        config.n_subsystems = args.n_subsystems
    if args.n_dims is not None:
        config.n_dimensions = args.n_dims
    if args.n_timesteps is not None:
        config.n_timesteps = args.n_timesteps
    if args.seed is not None:
        config.random_seed = args.seed
    if args.perception_gain is not None:
        config.perception_gain = args.perception_gain
    if args.language_gain is not None:
        config.language_gain = args.language_gain
    if args.planning_gain is not None:
        config.planning_gain = args.planning_gain
    if args.emotion_gain is not None:
        config.emotion_gain = args.emotion_gain
    if args.memory_gain is not None:
        config.memory_gain = args.memory_gain
    if args.motor_gain is not None:
        config.motor_gain = args.motor_gain
    if args.attention_gain is not None:
        config.attention_gain = args.attention_gain
    if args.executive_gain is not None:
        config.executive_gain = args.executive_gain
    if args.basin_ambiguity_threshold is not None:
        config.basin_ambiguity_threshold = args.basin_ambiguity_threshold
    if args.basin_pull_strength is not None:
        config.basin_pull_strength = args.basin_pull_strength
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs/simulation_run_{}".format(timestamp))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config.random_seed is not None:
        np.random.seed(config.random_seed)

    env = Environment()
    subsystems = initialize_subsystems(config, config.n_subsystems)
    being = initialize_being(env)

    print(f"\n{'=' * 70}")
    print(f"  CONSCIOUSNESS EMERGENCE + NAVIGATION SIMULATION")
    print(f"{'=' * 70}")
    print(f"\n  Config: {config.n_subsystems} subsystems | {config.n_dimensions}D manifold | {config.n_timesteps} steps")
    print(f"  Environment: {config.env_size}x{config.env_size} | {len(env.goals)} goals | {len(env.hazards)} hazards\n")

    metrics = run_simulation(config, env, being, subsystems)
    level, state = compute_consciousness_level(metrics, config)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  Consciousness Level:  {level:.3f} ({state})")
    print(f"  Basin-switch events: {metrics['basin_switches']}")
    print(f"  Micro-transitions:   {metrics['micro_transitions']} (600-cell layer)")
    print(f"  Goals reached:       {being.goals_reached}")
    print(f"  Hazards hit:         {being.hazards_hit}")
    print(f"  Avg pressure:        {np.mean(metrics['coordination_pressures']):.3f}")
    print(f"  Avg integration:     {np.mean(metrics['integration_levels']):.3f}")

    dom_counts = Counter(metrics["dominant_subsystems"])
    print(f"\n  Subsystem dominance (top 3):")
    for name, count in dom_counts.most_common(3):
        print(f"    {name:<20} {count} steps ({100*count//config.n_timesteps}%)")

    print(f"\n{'=' * 70}")
    print(f"  EXPORT")
    print(f"{'=' * 70}")

    # Export CSV to output directory
    csv_path = output_dir / "simulation_log.csv"
    export_to_csv(metrics, str(csv_path))

    print(f"\n{'=' * 70}")
    print(f"  LESION STUDY")
    print(f"{'=' * 70}")
    lesion_results = run_lesion_study(config, env, lesion_name="Planning")

    # Save all figures to output directory
    print(f"\n{'=' * 70}")
    print(f"  SAVING VISUALIZATIONS")
    print(f"{'=' * 70}\n")

    fig_nav = plt.figure(figsize=(8, 8))
    plot_navigation(metrics, env, title="Being Navigation (4D Slice → 2D Action)")
    nav_path = output_dir / "navigation.png"
    fig_nav.savefig(nav_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {nav_path}")

    fig_phase = plot_phase_portrait(metrics)
    phase_path = output_dir / "phase_portrait.png"
    plt.gcf().savefig(phase_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {phase_path}")

    fig_dom = plot_dominance(metrics, subsystems)
    dom_path = output_dir / "dominance.png"
    plt.gcf().savefig(dom_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {dom_path}")

    fig_lesion = plot_lesion_comparison(lesion_results)
    lesion_path = output_dir / "lesion_study.png"
    plt.gcf().savefig(lesion_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {lesion_path}")

    # Save summary JSON
    summary = {
        "timestamp": timestamp,
        "config": {
            "n_subsystems": config.n_subsystems,
            "n_dimensions": config.n_dimensions,
            "n_timesteps": config.n_timesteps,
            "coordination_threshold": config.coordination_threshold,
            "noise_level": config.noise_level,
            "random_seed": config.random_seed,
        },
        "results": {
            "consciousness_level": level,
            "consciousness_state": state,
            "basin_switches": metrics["basin_switches"],
            "goals_reached": being.goals_reached,
            "hazards_hit": being.hazards_hit,
            "avg_pressure": float(np.mean(metrics['coordination_pressures'])),
            "avg_integration": float(np.mean(metrics['integration_levels'])),
        },
        "subsystem_dominance": {name: count for name, count in dom_counts.most_common()},
        "lesion_study": {
            "lesioned_subsystem": "Planning",
            "intact_level": lesion_results["level_intact"],
            "lesioned_level": lesion_results["level_lesioned"],
            "delta": lesion_results["level_intact"] - lesion_results["level_lesioned"],
        },
        "output_files": {
            "simulation_log_csv": str(csv_path),
            "navigation_plot": str(nav_path),
            "phase_portrait_plot": str(phase_path),
            "dominance_plot": str(dom_path),
            "lesion_study_plot": str(lesion_path),
        }
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    plt.close('all')

    return metrics, level, state

if __name__ == "__main__":
    main()