#!/usr/bin/env python3
"""Unit tests for the Consciousness Simulation."""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fourD_slice_sim import (
    SimulationConfig,
    Environment,
    Being,
    Subsystem,
    initialize_being,
    initialize_subsystems,
    compute_coordination_pressure,
    compute_system_state,
    dominant_subsystem,
    select_action,
    move_being,
    basin_switch_event,
    attractor_update,
)


class TestSimulation(unittest.TestCase):
    """Test cases for simulation components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SimulationConfig(
            n_subsystems=4,
            n_dimensions=4,
            coordination_threshold=0.85,
            noise_level=0.1,
            learning_rate=0.005,
            step_size=0.4,
            env_size=20,
        )
        self.env = Environment()

    def test_initialization(self):
        """Test that initialization creates valid objects."""
        being = initialize_being(self.env)
        self.assertEqual(len(being.position), 2)
        # Position should be near center (10, 10) for 20x20 environment
        self.assertAlmostEqual(being.position[0], 10.0, places=1)
        self.assertAlmostEqual(being.position[1], 10.0, places=1)

    def test_subsystem_initialization(self):
        """Test subsystem initialization creates correct number and structure."""
        subs = initialize_subsystems(self.config, 4)
        self.assertEqual(len(subs), 4)
        
        for sub in subs:
            self.assertIsNotNone(sub.name)
            self.assertEqual(len(sub.activity), self.config.n_dimensions)
            self.assertEqual(len(sub.weights), self.config.n_dimensions)

    def test_computed_coordination_pressure_single_subsystem(self):
        """Test coordination pressure with single subsystem is zero."""
        subs = initialize_subsystems(self.config, 1)
        pressure = compute_coordination_pressure(subs)
        self.assertEqual(pressure, 0.0)

    def test_coordinated_pressure_no_conflict(self):
        """Test that identical activities result in zero conflict."""
        activity = np.array([1.0, 0.0, 0.0, 0.0])
        subs = [
            Subsystem(name="A", activity=activity.copy(), weights=np.zeros(4), preferred_direction=np.zeros(4)),
            Subsystem(name="B", activity=activity.copy(), weights=np.zeros(4), preferred_direction=np.zeros(4)),
        ]
        pressure = compute_coordination_pressure(subs)
        self.assertEqual(pressure, 0.0)

    def test_coordinated_pressure_max_conflict(self):
        """Test that opposite activities result in high conflict."""
        activity1 = np.array([1.0, 0.0, 0.0, 0.0])
        activity2 = np.array([-1.0, 0.0, 0.0, 0.0])
        subs = [
            Subsystem(name="A", activity=activity1.copy(), weights=np.zeros(4), preferred_direction=np.zeros(4)),
            Subsystem(name="B", activity=activity2.copy(), weights=np.zeros(4), preferred_direction=np.zeros(4)),
        ]
        pressure = compute_coordination_pressure(subs)
        self.assertGreater(pressure, 0.5)

    def test_dominant_subsystem(self):
        """Test that dominant subsystem is correctly identified."""
        subs = initialize_subsystems(self.config, 3)
        
        # Boost one subsystem's activity
        subs[1].activity = np.ones(4) * 10.0
        
        dom = dominant_subsystem(subs)
        self.assertEqual(dom, subs[1].name)

    def test_dominant_subsystem_no_active(self):
        """Test that 'none' is returned when no subsystems are active."""
        subs = initialize_subsystems(self.config, 3)
        for sub in subs:
            sub.active = False
        
        dom = dominant_subsystem(subs)
        self.assertEqual(dom, "none")

    def test_select_action_normalization(self):
        """Test that select action returns normalized vector."""
        coordinator = np.array([3.0, 4.0, 0.0, 0.0])  # Magnitude = 5
        action = select_action(coordinator, self.config)
        
        # Action should be normalized to unit length (before step_size scaling)
        norm = np.linalg.norm(action[:2])
        self.assertAlmostEqual(norm, self.config.step_size, places=6)

    def test_move_being_boundary(self):
        """Test that being stays within environment bounds."""
        being = initialize_being(self.env)
        
        # Move toward boundary (starting at 10, 10 in 20x20 env)
        action = np.array([15.0, 0.0])  # Large movement toward edge
        
        goal_reached, hazard_hit = move_being(being, action, self.env)
        
        # Position should be clamped to boundary (0 or 20)
        self.assertGreaterEqual(being.position[0], 0.0)
        self.assertLessEqual(being.position[0], self.env.size)
        self.assertGreaterEqual(being.position[1], 0.0)
        self.assertLessEqual(being.position[1], self.env.size)

    def test_basin_switch_near_equidistant(self):
        """Test basin switching when equidistant from attractors."""
        coordinator = np.array([0.5, 0.5, 0.5, 0.5])
        
        # Create two nearly equidistant attractors
        attractor1 = np.array([1.0, 0.0, 0.0, 0.0])
        attractor2 = np.array([-1.0, 0.0, 0.0, 0.0])
        
        basin_attractors = [attractor1, attractor2]
        next_basin, switched = basin_switch_event(coordinator, basin_attractors, self.config)
        
        # Should return one of the basins (compare arrays properly)
        is_attractor1 = np.allclose(next_basin, attractor1)
        is_attractor2 = np.allclose(next_basin, attractor2)
        self.assertTrue(is_attractor1 or is_attractor2, "next_basin should be one of the basin attractors")

    def test_attractor_update_normalization(self):
        """Test that attractor update maintains normalized state."""
        state = np.array([3.0, 4.0, 0.0, 0.0])  # Magnitude = 5
        
        attractor = np.array([1.0, 0.0, 0.0, 0.0])
        
        updated = attractor_update(state, attractor, self.config)
        
        # State should be normalized (magnitude ≈ 1)
        norm = np.linalg.norm(updated)
        self.assertGreater(norm, 0.9)
        self.assertLess(norm, 1.1)


class TestEnvironment(unittest.TestCase):
    """Test environment-related functionality."""

    def test_goal_reached(self):
        """Test that goals are correctly detected when reached."""
        env = Environment()
        being = Being(position=np.array([4.0, 4.0]))  # At first goal
        
        goal_reached, _ = move_being(being, np.zeros(2), env)
        
        self.assertTrue(goal_reached)

    def test_hazard_hit(self):
        """Test that hazards are correctly detected when hit."""
        env = Environment()
        being = Being(position=np.array([3.0, 15.0]))  # At first hazard
        
        goal_reached, hazard_hit = move_being(being, np.zeros(2), env)
        
        self.assertTrue(hazard_hit)

    def test_no_goal_or_hazard(self):
        """Test neutral state when not near goals or hazards."""
        env = Environment()
        being = Being(position=np.array([10.0, 10.0]))  # Center of map
        
        goal_reached, hazard_hit = move_being(being, np.zeros(2), env)
        
        self.assertFalse(goal_reached)
        self.assertFalse(hazard_hit)


if __name__ == "__main__":
    unittest.main()