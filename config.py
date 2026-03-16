#!/usr/bin/env python3
"""
Configuration module for the Consciousness Simulation.

Provides predefined simulation configurations and utilities for loading/saving settings.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional


@dataclass
class SimulationConfig:
    """Configuration for the consciousness simulation."""
    
    # Core simulation parameters
    n_subsystems: int = 8          # Number of cognitive subsystems
    n_dimensions: int = 4          # Dimensionality of coordination manifold
    n_timesteps: int = 300         # Simulation duration
    
    # Coordination dynamics
    coordination_threshold: float = 0.85  # Threshold for high-pressure states
    noise_level: float = 0.1           # Random perturbation magnitude
    learning_rate: float = 0.005       # Coordinator update speed
    
    # Navigation parameters
    step_size: float = 0.4             # Movement distance per timestep
    
    # Environment settings
    env_size: int = 20                 # Grid world side length
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SimulationConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# ────────────────────────────────────────────────────────────────────
# Predefined configurations
# ────────────────────────────────────────────────────────────────────

def get_config(name: str) -> SimulationConfig:
    """Load a predefined simulation configuration by name."""
    
    configs = {
        "default": SimulationConfig(
            n_subsystems=8,
            n_dimensions=4,
            n_timesteps=300,
            coordination_threshold=0.85,
            noise_level=0.1,
            learning_rate=0.005,
            step_size=0.4,
            env_size=20,
        ),
        
        "quick": SimulationConfig(
            n_subsystems=6,
            n_dimensions=4,
            n_timesteps=100,
            coordination_threshold=0.85,
            noise_level=0.1,
            learning_rate=0.005,
            step_size=0.4,
            env_size=20,
        ),
        
        "extended": SimulationConfig(
            n_subsystems=8,
            n_dimensions=4,
            n_timesteps=500,
            coordination_threshold=0.85,
            noise_level=0.1,
            learning_rate=0.005,
            step_size=0.4,
            env_size=20,
        ),
        
        "high_conflict": SimulationConfig(
            n_subsystems=8,
            n_dimensions=4,
            n_timesteps=300,
            coordination_threshold=0.7,   # Lower threshold for more decision points
            noise_level=0.15,             # Higher noise for more variability
            learning_rate=0.008,          # Faster coordinator updates
            step_size=0.4,
            env_size=20,
        ),
        
        "minimal": SimulationConfig(
            n_subsystems=3,
            n_dimensions=2,
            n_timesteps=150,
            coordination_threshold=0.85,
            noise_level=0.05,
            learning_rate=0.003,
            step_size=0.3,
            env_size=15,
        ),
    }
    
    return configs.get(name, configs["default"])


# ────────────────────────────────────────────────────────────────────
# Environment configurations
# ────────────────────────────────────────────────────────────────────

@dataclass
class EnvironmentConfig:
    """Configuration for the simulation environment."""
    
    size: int = 20
    goal_radius: float = 1.5
    hazard_radius: float = 2.0
    
    # Goal positions (x, y)
    goals: List[Tuple[float, float]] = field(default_factory=lambda: [
        (4.0, 4.0), (16.0, 16.0), (10.0, 3.0)
    ])
    
    # Hazard positions (x, y)  
    hazards: List[Tuple[float, float]] = field(default_factory=lambda: [
        (3.0, 15.0), (17.0, 5.0), (10.0, 17.0)
    ])


def get_environment_config(name: str = "default") -> EnvironmentConfig:
    """Load a predefined environment configuration by name."""
    
    env_configs = {
        "default": EnvironmentConfig(),
        
        "sparse_goals": EnvironmentConfig(
            size=25,
            goals=[(5.0, 5.0), (20.0, 20.0)],
            hazards=[],
        ),
        
        "challenging": EnvironmentConfig(
            size=30,
            goal_radius=1.0,
            hazard_radius=2.5,
            goals=[(3.0, 3.0), (27.0, 27.0)],
            hazards=[(15.0, 5.0), (25.0, 8.0), (5.0, 25.0), (15.0, 25.0)],
        ),
    }
    
    return env_configs.get(name, env_configs["default"])


# ────────────────────────────────────────────────────────────────────
# Configuration utilities
# ────────────────────────────────────────────────────────────────────

def save_config(config: SimulationConfig, path: str) -> None:
    """Save configuration to JSON file."""
    import json
    
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def load_config(path: str) -> SimulationConfig:
    """Load configuration from JSON file."""
    import json
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return SimulationConfig.from_dict(data)


if __name__ == "__main__":
    # Demo: print all available configurations
    print("Available simulation configurations:")
    for name in get_config.__code__.co_consts:
        if isinstance(name, str) and not name.startswith('_'):
            try:
                config = get_config(name)
                print(f"\n  {name}:")
                print(f"    Subsystems: {config.n_subsystems}")
                print(f"    Dimensions: {config.n_dimensions}")
                print(f"    Timesteps:  {config.n_timesteps}")
            except:
                pass
    
    print("\n\nAvailable environment configurations:")
    for name in get_environment_config.__code__.co_consts:
        if isinstance(name, str) and not name.startswith('_'):
            try:
                env = get_environment_config(name)
                print(f"\n  {name}:")
                print(f"    Size:       {env.size}x{env.size}")
                print(f"    Goals:      {len(env.goals)}")
                print(f"    Hazards:    {len(env.hazards)}")
            except:
                pass