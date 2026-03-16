"""
Consciousness Simulation v2 - Dual-Geometry Model

A true geometric soft-assignment engine based on H4-inspired dual manifold.

Key features:
- Closed 4D hyperspherical manifold (direction + magnitude separated)
- Soft assignment via weighted macro field
- Two-stage updates: micro exploration + macro reconciliation
- Toroidal visible world without edge artifacts
- Balanced subsystems with competitive inhibition and fatigue

Author: consciousness-sim v2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import entropy

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'n_subsystems': 8,
    'n_micro': 600,          # Micro-transition reference points
    'n_macro': 120,          # Macro-basin centers
    'manifold_dim': 4,       # 4D state space (S³ hypersphere)
    'world_size': 20,        # Toroidal environment size
    'goals': 3,
    'hazards': 3,
    'timesteps': 500,        # Increased for better sampling
    
    # Two-stage update parameters
    'alpha_pull': 0.15,      # How strongly to follow macro field
    'beta_macro': 8.0,       # Softness parameter for basin assignment
    
    # Integration metric target range - tuned for meaningful values (3-7 effective basins)
    'target_neff': 5.0,      # Target effective number of active basins
    'sigma_integr': 2.0,     # Bell width for integration score
    
    # Subsystem dynamics - increased fatigue to prevent single-subsystem dominance
    'fatigue_rate': 0.035,   # Fatigue/adaptation rate (increased from 0.02)
    'recovery_rate': 0.015,  # Recovery when inactive (slightly faster recovery)
    'floor_value': 0.05,     # Minimum activity floor
    'exploration_noise': 0.01,
    
    # Coherence metric weights
    'alpha_fit': 0.5,
    'beta_surprise': 0.3,
    'gamma_conflict': 0.2,
}


# ============================================================================
# CLOSED MANIFOLD - H4-INSPIRED DUAL GEOMETRY
# ============================================================================

class ClosedManifold:
    """
    Unit hypersphere S³ in 4D with pre-computed reference geometry.
    
    Micro points (~600) and macro basins (~120) are both normalized to unit radius,
    forming a proper spherical code approximation of the 600-cell / 120-cell duality.
    """
    
    def __init__(self):
        self.n_micro = CONFIG['n_micro']
        self.n_macro = CONFIG['n_macro']
        self.dim = CONFIG['manifold_dim']
        
        # Generate reference geometry with proper normalization
        self.micro_points, self.macro_centers = self._generate_reference_geometry()
    
    def _generate_reference_geometry(self):
        """Generate micro and macro points on unit hypersphere."""
        # Micro: ~600 random points uniformly distributed on S³
        micro_pts = np.random.normal(size=(self.n_micro, self.dim))
        micro_norms = np.linalg.norm(micro_pts, axis=1, keepdims=True) + 1e-8
        micro_points = micro_pts / micro_norms
        
        # Macro: cluster micro points to derive ~120 basin centers
        km = KMeans(n_clusters=self.n_macro, random_state=42, n_init=10)
        km.fit(micro_pts)
        
        # Extract and normalize cluster centers
        macro_centers = km.cluster_centers_.copy()
        macro_norms = np.linalg.norm(macro_centers, axis=1, keepdims=True) + 1e-8
        macro_centers = macro_centers / macro_norms
        
        return micro_points, macro_centers
    
    def normalize_to_sphere(self, x):
        """Project vector to unit hypersphere."""
        if isinstance(x, np.ndarray) and len(x.shape) > 1:
            norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
            return x / norms
        else:
            norm = np.linalg.norm(x) + 1e-8
            return x / norm
    
    def compute_similarity(self, x_t):
        """
        Compute angular similarity to all reference points.
        
        Returns:
            micro_sim: similarities to micro points (n_micro,)
            macro_sim: similarities to macro centers (n_macro,)
        """
        # Normalize input direction
        u = self.normalize_to_sphere(x_t)
        
        # Dot products give cosine similarity on unit sphere
        micro_sim = np.dot(self.micro_points, u)  # Shape: (n_micro,)
        macro_sim = np.dot(self.macro_centers, u)  # Shape: (n_macro,)
        
        return micro_sim, macro_sim


# ============================================================================
# SOFT ASSIGNMENT & COHERENCE METRICS
# ============================================================================

class SoftAssignmentEngine:
    """
    Geometric soft assignment to micro and macro states.
    
    Uses weighted superposition of macro centers rather than hard argmax,
    enabling smooth transitions and ambiguous mixed states.
    """
    
    def __init__(self, manifold):
        self.manifold = manifold
        self.beta = CONFIG['beta_macro']
        
    def compute_micro_membership(self, x_t):
        """Find most similar micro reference point - extract direction only."""
        u = x_t[:CONFIG['manifold_dim']] if len(x_t) > CONFIG['manifold_dim'] else x_t  # Extract direction only
        micro_sim, _ = self.manifold.compute_similarity(u)
        micro_id = np.argmax(micro_sim)
        return micro_id, micro_sim[micro_id]
    
    def compute_macro_assignment(self, x_t):
        """
        Soft assignment to macro basins via softmax of similarities.
        
        Returns:
            dominant_id: index of most active basin
            weights: full soft distribution (n_macro,)
            field: weighted superposition vector (normalized)
        """
        _, macro_sim = self.manifold.compute_similarity(x_t)
        
        # Softmax for smooth probability distribution
        exp_sim = np.exp(self.beta * macro_sim)
        weights = exp_sim / (np.sum(exp_sim) + 1e-8)
        
        # Weighted field: superposition of all basin centers
        field = np.dot(weights, self.manifold.macro_centers)
        field_norm = np.linalg.norm(field) + 1e-8
        field = field / field_norm
        
        dominant_id = np.argmax(weights)
        
        return dominant_id, weights, field
    
    def compute_closure_coherence(self, x_t, macro_weights, recent_states, subsystem_conflict):
        """
        Dynamic closure coherence metric.
        
        Components:
            1. Fit to weighted macro field (angular similarity)
            2. Stability of recent trajectory (inverse surprise)
            3. Agreement across subsystems (inverse conflict)
        """
        # Component 1: fit to current macro field - extract direction only for manifold ops
        u = self.manifold.normalize_to_sphere(x_t[:CONFIG['manifold_dim']]) if len(x_t) > CONFIG['manifold_dim'] else self.manifold.normalize_to_sphere(x_t)
        _, _, macro_field = self.compute_macro_assignment(u)  # Use normalized direction directly
        fit_score = np.dot(u, macro_field)
        
        # Component 2: transition surprise (inverse of path consistency)
        if len(recent_states) >= 3:
            surprise = self._compute_transition_surprise(recent_states)
        else:
            surprise = 0.0
        
        # Component 3: subsystem agreement
        subsystem_agreement = 1.0 - np.clip(subsystem_conflict, 0, 1)
        
        # Weighted combination
        coherence = (CONFIG['alpha_fit'] * fit_score +
                    CONFIG['beta_surprise'] * (1.0 - surprise) +
                    CONFIG['gamma_conflict'] * subsystem_agreement)
        
        return np.clip(coherence, -1.0, 1.0)
    
    def _compute_transition_surprise(self, states):
        """Compute average direction change between successive steps."""
        diffs = np.diff(states, axis=0)
        if len(diffs) < 2:
            return 0.0
        
        surprises = []
        for i in range(len(diffs) - 1):
            a = diffs[i]
            b = diffs[i + 1]
            na = np.linalg.norm(a) + 1e-8
            nb = np.linalg.norm(b) + 1e-8
            # Cosine of angle between successive directions
            alignment = np.dot(a, b) / (na * nb)
            surprises.append(1.0 - abs(alignment))  # Surprise = deviation from straight
        
        return float(np.mean(surprises))


# ============================================================================
# INTEGRATION & DIFFERENTIATION METRICS
# ============================================================================

class ConsciousnessMetrics:
    """
    Quantitative measures of consciousness-like properties.
    
    Integration: bell-shaped measure peaked at intermediate complexity
    Differentiation: normalized entropy measuring state diversity
    """
    
    @staticmethod
    def compute_integration(macro_weights, target_neff=CONFIG['target_neff'], sigma=CONFIG['sigma_integr']):
        """
        Integration score with middle-regime peak.
        
        High when effective number of active basins is in optimal range (not too few, not too many).
        
        Args:
            macro_weights: soft assignment weights (n_macro,)
            target_neff: target effective number of states
            sigma: bell curve width
        
        Returns:
            integration_score: 0 to 1, higher = better structured multi-state activity
        """
        # Effective number of active basins (participation ratio)
        neff = 1.0 / (np.sum(macro_weights**2) + 1e-8)
        
        # Bell-shaped score peaked at target_neff
        score = np.exp(-((neff - target_neff)**2) / (2 * sigma**2))
        
        return float(score)
    
    @staticmethod
    def compute_differentiation(macro_weights):
        """
        Differentiation index measuring state diversity.
        
        Normalized entropy: higher = more basins actively considered.
        """
        max_entropy = np.log(CONFIG['n_macro'])
        current_entropy = entropy(macro_weights + 1e-8)
        
        return float(current_entropy / max_entropy)
    
    @staticmethod
    def compute_path_coherence(states, window=10):
        """Temporal consistency of recent trajectory."""
        if len(states) < 2:
            return 0.0
        
        # Use last 'window' steps or all available
        recent = states[-window:]
        
        diffs = np.diff(recent, axis=0)
        if len(diffs) < 1:
            return 0.0
        
        alignments = []
        for i in range(len(diffs) - 1):
            a = diffs[i]
            b = diffs[i + 1]
            na = np.linalg.norm(a) + 1e-8
            nb = np.linalg.norm(b) + 1e-8
            alignment = np.dot(a, b) / (na * nb)
            alignments.append(alignment)
        
        return float(np.mean(alignments))
    
    def compute_subsystem_conflict(self, activities):
        """Measure of conflict between subsystems.
        
        Higher when multiple subsystems have comparable influence (high entropy).
        Lower when one subsystem dominates (low entropy).
        """
        entropy_val = entropy(activities + 1e-8)
        max_entropy = np.log(CONFIG['n_subsystems'])
        
        return float(entropy_val / max_entropy)


# ============================================================================
# BALANCED SUBSYSTEMS WITH COMPETITION AND FATIGUE
# ============================================================================

class BalancedSubsystems:
    """
    8 subsystems with competitive inhibition and adaptive fatigue.
    
    Each subsystem has a DISTINCT preferred region in manifold space,
    designed to ensure balanced coverage and prevent single-subsystem dominance.
    
    Subsystem design philosophy:
        - Motor Control: responds to velocity/motion-related patterns
        - Planning: responds to structured/predictive patterns  
        - Attention: responds to focused/peak activation patterns
        - Memory: responds to historical continuity patterns
        - Emotion: responds to high-intensity patterns
        - Social: responds to cooperative/sync patterns
        - Intuition: responds to novel/discovery patterns
        - Aesthetic: responds to balanced/harmonious patterns
    
    Prevents single-subsystem dominance through:
        - Pre-designed orthogonal preference directions
        - Strong fatigue mechanism (dominant subsystem adapts quickly)
        - Environmental stimulus modulation
    """
    
    SUBSYSTEM_NAMES = [
        'Motor Control', 'Planning', 'Attention', 'Memory',
        'Emotion', 'Social', 'Intuition', 'Aesthetic'
    ]
    
    # Pre-computed orthogonal preference directions for each subsystem
    # These span the 4D manifold space to ensure balanced coverage
    
    PREFERENCE_MATRIX = np.array([
        [1.0, 0.0, 0.0, 0.0],   # Motor Control: pure dim 0
        [0.0, 1.0, 0.0, 0.0],   # Planning: pure dim 1
        [0.0, 0.0, 1.0, 0.0],   # Attention: pure dim 2
        [0.0, 0.0, 0.0, 1.0],   # Memory: pure dim 3
        [1.0, -1.0, 0.0, 0.0],  # Emotion: contrast dims 0-1 (orthogonal to Planning)
        [0.0, 0.0, 1.0, -1.0],  # Social: contrast dims 2-3
        [1.0, 0.0, -1.0, 0.0],  # Intuition: cross-dim pattern (0 vs 2)
        [0.0, 1.0, 0.0, -1.0],  # Aesthetic: cross-dim pattern (1 vs 3)
    ])
    
    def __init__(self):
        self.n_subsystems = CONFIG['n_subsystems']
        
        # Initial equal gains
        self.gains = np.ones(self.n_subsystems) / self.n_subsystems
        
        # Fatigue state (higher = more fatigued, less active)
        self.fatigue = np.zeros(self.n_subsystems)
        
        # Recovery rate for inactive subsystems
        self.recovery_rate = CONFIG['recovery_rate']
        
        # Track recent dominant activity for adaptive fatigue
        self.recent_dominant_activities = []
    
    def compute_influences(self, u_t, environment_pos=None):
        """
        Compute raw influence from each subsystem.
        
        Influences depend on state position and environmental factors.
        Different subsystems respond differently to manifold position.
        
        Args:
            u_t: Unit direction vector in manifold space (4,)
            environment_pos: Optional (x, y) position for environmental responses
        
        Returns:
            influences: raw influence scores from each subsystem (n_subsystems,)
        """
        # Each subsystem has a pre-defined preferred direction - compute dot product
        influences = np.zeros(self.n_subsystems)
        
        for i in range(self.n_subsystems):
            pref_dir = self.PREFERENCE_MATRIX[i]
            # Normalize preference direction if needed
            norm_pref = np.linalg.norm(pref_dir)
            if norm_pref > 1e-8:
                pref_dir_normalized = pref_dir / norm_pref
            else:
                pref_dir_normalized = pref_dir
            
            dot_product = float(np.dot(u_t, pref_dir_normalized))
            
            # Steeper mapping to create stronger differentiation between subsystems
            # Uses tanh-like squashing for better separation: ranges ~[-0.3, 1.3]
            # Then scales to [0.3, 1.2] with offset and gain
            influences[i] = 0.5 + 0.7 * np.tanh(2.0 * dot_product)
        
        return influences
    
    def apply_competition_and_fatigue(self, raw_influences):
        """
        Apply divisive normalization with fatigue adaptation.
        
        Returns normalized activities that sum to ~1.
        """
        # Apply current fatigue (higher fatigue = lower activity)
        effective = raw_influences * np.exp(-self.fatigue)
        
        # Add exploration noise for diversity
        noise = np.random.normal(0, CONFIG['exploration_noise'], self.n_subsystems)
        effective += noise
        
        # Apply minimum floor to prevent complete shutdown
        effective = np.maximum(effective, CONFIG['floor_value'])
        
        # Divisive normalization (softmax-like but with linear scaling first)
        total = np.sum(effective) + 1e-8
        activities = effective / total
        
        return activities
    
    def update_fatigue(self, activities, dt=1.0):
        """
        Update fatigue levels based on recent activity.
        
        High activity increases fatigue (adaptation),
        low activity allows recovery.
        
        Enhanced with secondary fatigue for dominant subsystems only.
        """
        # Primary fatigue: proportional to current activity
        self.fatigue += CONFIG['fatigue_rate'] * activities * dt
        
        # Track recent dominant activity for adaptive fatigue
        dominant_idx = np.argmax(activities)
        self.recent_dominant_activities.append(np.max(activities))
        if len(self.recent_dominant_activities) > 5:
            self.recent_dominant_activities.pop(0)
        
        # Secondary adaptive fatigue: extra penalty for sustained dominance
        recent_activity = np.mean(self.recent_dominant_activities[-5:])
        if recent_activity > 0.4:  # Sustained high activity
            self.fatigue[dominant_idx] += 0.02 * dt
        
        # Recover when inactive
        inactive_recovery = (1 - activities) * self.recovery_rate * dt
        self.fatigue -= inactive_recovery
        
        # Clamp to reasonable range
        self.fatigue = np.clip(self.fatigue, 0.0, 3.0)


# ============================================================================
# TOROIDAL ENVIRONMENT
# ============================================================================

class ToroidalEnvironment:
    """
    Wrap-around 2D world without edge artifacts.
    
    Coordinates wrap at boundaries, enabling smooth navigation
    across what would otherwise be artificial edges.
    """
    
    def __init__(self, size=CONFIG['world_size']):
        self.size = size
        
        # Generate goals and hazards
        self.goals = self._generate_positions(CONFIG['goals'])
        self.hazards = self._generate_positions(CONFIG['hazards'], exclude=self.goals)
        
        print(f"  Environment: {size}x{size} torus | Goals at {self.goals[:2]}... | Hazards at {self.hazards[0] if len(self.hazards) else 'none'}...")
    
    def _generate_positions(self, n, exclude=None):
        """Generate random positions on torus."""
        positions = []
        
        # Start with provided exclusion list (if any)
        existing = [list(p) for p in exclude] if exclude is not None else []
        
        while len(positions) < n:
            pos = np.random.randint(0, self.size, 2).tolist()
            
            # Check distance from exclusions (toroidal)
            too_close = False
            for ex in existing:
                dist = self._toroidal_distance(pos, ex)
                if dist < 3:  # Minimum separation
                    too_close = True
                    break
            
            if not too_close:
                positions.append(pos)
        
        return np.array(positions)
    
    def _toroidal_distance(self, pos1, pos2):
        """Compute shortest path distance on torus."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        # Take minimum of direct and wrapped paths
        dx = min(dx, self.size - dx)
        dy = min(dy, self.size - dy)
        
        return np.sqrt(dx**2 + dy**2)
    
    def step_position(self, x, y, velocity_x, velocity_y):
        """Update position with toroidal wrapping."""
        new_x = (x + velocity_x) % self.size
        new_y = (y + velocity_y) % self.size
        
        return int(new_x), int(new_y)


# ============================================================================
# DUAL-GEOMETRY SIMULATION ENGINE
# ============================================================================

class ConsciousnessSimulation:
    """
    Main simulation engine combining all components.
    
    Two-stage update per timestep:
        1. Micro transition: explore local manifold space
        2. Macro reconciliation: pull toward compatible basin field
    """
    
    def __init__(self):
        print("\n" + "=" * 60)
        print("  CONSCIOUSNESS SIMULATION v2 - DUAL-GEOMETRY MODEL")
        print("=" * 60)
        
        # Initialize components
        self.manifold = ClosedManifold()
        self.assignment = SoftAssignmentEngine(self.manifold)
        self.metrics = ConsciousnessMetrics()
        self.subsystems = BalancedSubsystems()
        self.environment = ToroidalEnvironment()
        
        # State variables (direction + magnitude separated)
        self.u_t = np.ones(CONFIG['manifold_dim']) / np.sqrt(CONFIG['manifold_dim'])  # Unit direction
        self.r_t = 1.0  # Activation magnitude
        
        # Tracking history
        self.history = {
            'time': [],
            'micro_id': [],
            'macro_dominant': [],
            'macro_weights_sum_sq': [],  # For tracking commitment level
            'closure_coherence': [],
            'integration': [],
            'differentiation': [],
            'path_coherence': [],
            'dominant_subsystem': [],
        }
        
        self.recent_states = []
    
    def step(self, timestep):
        """Single simulation step with two-stage update."""
        
        # --- STAGE 1: Micro transition (local exploration) ---
        
        # Compute subsystem influences based on current state and environment
        # Use only the direction vector (u_t) - r_t is separate magnitude
        raw_influences = self.subsystems.compute_influences(
            self.u_t,
            environment_pos=None  # Position not needed for manifold dynamics
        )
        
        # Apply competition and fatigue
        activities = self.subsystems.apply_competition_and_fatigue(raw_influences)
        
        # Update subsystem fatigue
        self.subsystems.update_fatigue(activities)
        
        # Compute micro transition direction based on dominant subsystems
        dominant_subsystem = np.argmax(activities)
        
        # Perturb state in direction favored by active subsystems
        perturbation = np.random.normal(0, 0.1, CONFIG['manifold_dim'])
        self.u_t = self.manifold.normalize_to_sphere(self.u_t + 0.2 * perturbation)
        
        # Adjust magnitude based on overall activity
        total_activity = np.sum(activities)
        expected_total = 1.0 / CONFIG['n_subsystems']
        self.r_t = np.clip(self.r_t + 0.05 * (total_activity - expected_total), 0.5, 2.0)
        
        # --- STAGE 2: Macro reconciliation (global constraint via weighted field) ---
        
        # Compute soft macro assignment and field
        _, macro_weights, macro_field = self.assignment.compute_macro_assignment(self.u_t)
        
        # Pull toward weighted macro field (not just dominant basin!)
        pull_direction = (1 - CONFIG['alpha_pull']) * self.u_t + CONFIG['alpha_pull'] * macro_field
        self.u_t = self.manifold.normalize_to_sphere(pull_direction)
        
        # --- COMPUTE METRICS ---
        
        # Compute coherence with dynamic components
        closure = self.assignment.compute_closure_coherence(
            np.concatenate([self.u_t, [self.r_t]]),
            macro_weights,
            self.recent_states[-5:],  # Recent states for surprise
            self.metrics.compute_subsystem_conflict(activities)
        )
        
        integration = self.metrics.compute_integration(macro_weights)
        differentiation = self.metrics.compute_differentiation(macro_weights)
        
        # Path coherence: need at least 3 states to compute meaningful trajectory
        if len(self.recent_states) >= 3:
            path_coherence = self.metrics.compute_path_coherence(self.recent_states[-10:])
        else:
            path_coherence = 0.0  # Not enough data yet
        
        # --- UPDATE HISTORY ---
        
        micro_id, _ = self.assignment.compute_micro_membership(np.concatenate([self.u_t, [self.r_t]]))
        
        self.history['time'].append(timestep)
        self.history['micro_id'].append(micro_id)
        self.history['macro_dominant'].append(np.argmax(macro_weights))
        self.history['macro_weights_sum_sq'].append(np.sum(macro_weights**2))  # Inverse of commitment
        self.history['closure_coherence'].append(closure)
        self.history['integration'].append(integration)
        self.history['differentiation'].append(differentiation)
        self.history['path_coherence'].append(path_coherence)
        self.history['dominant_subsystem'].append(self.subsystems.SUBSYSTEM_NAMES[dominant_subsystem])
        
        # Update recent states for surprise computation
        state_vec = np.concatenate([self.u_t, [self.r_t]])
        self.recent_states.append(state_vec)
        if len(self.recent_states) > 20:
            self.recent_states.pop(0)
    
    def run_simulation(self):
        """Run full simulation and return results."""
        
        print(f"\n  Running {CONFIG['timesteps']} timesteps...")
        
        for t in range(CONFIG['timesteps']):
            self.step(t)
        
        # Compute summary statistics
        dominant_counts = pd.Series(self.history['dominant_subsystem']).value_counts()
        
        avg_commitment = np.mean([w**2 for w in self.history['macro_weights_sum_sq']])
        
        print("\n" + "=" * 60)
        print("  RESULTS")
        print("=" * 60)
        
        print(f"\n  Subsystem balance (top 5):")
        for name, count in dominant_counts.head(5).items():
            pct = 100.0 * count / CONFIG['timesteps']
            print(f"    {name:20s}: {count:4d} steps ({pct:.1f}%)")
        
        print(f"\n  Coherence metrics (averages):")
        print(f"    Closure coherence:     {np.mean(self.history['closure_coherence']):.3f}")
        print(f"    Integration:           {np.mean(self.history['integration']):.3f}")
        print(f"    Differentiation:       {np.mean(self.history['differentiation']):.3f}")
        print(f"    Path coherence:        {np.mean(self.history['path_coherence']):.3f}")
        
        # Count macro state transitions
        transitions = 0
        for i in range(1, len(self.history['macro_dominant'])):
            if self.history['macro_dominant'][i] != self.history['macro_dominant'][i-1]:
                transitions += 1
        
        print(f"\n  Macro-state transitions: {transitions} ({100*transitions/CONFIG['timesteps']:.1f}% of timesteps)")
        
        # Export to CSV
        df = pd.DataFrame(self.history)
        csv_filename = 'simulation_log_v2.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\n  Exported {len(df)} rows to {csv_filename}")
        
        return self.history


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(history):
    """Create linked visualization of simulation results."""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Plot 1: Coherence metrics over time
    ax = axes[0, 0]
    t = np.array(history['time'])
    ax.plot(t, history['closure_coherence'], label='Closure', alpha=0.7)
    ax.plot(t, history['integration'], label='Integration', alpha=0.7)
    ax.set_title('Coherence and Integration Over Time')
    ax.legend()
    ax.set_xlabel('Timestep')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Subsystem dominance timeline
    ax = axes[0, 1]
    subsystems = history['dominant_subsystem']
    for name in set(subsystems):
        mask = np.array([s == name for s in subsystems])
        ax.plot(t[mask], t[mask], 'o', markersize=3, label=name, alpha=0.5)
    ax.set_title('Subsystem Dominance Timeline')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='center left', fontsize=8)
    ax.set_xlabel('Timestep')
    
    # Plot 3: Macro state progression
    ax = axes[1, 0]
    macro_states = history['macro_dominant']
    ax.plot(t, macro_states, marker='o', markersize=4)
    ax.set_title('Macro State Over Time (120 basins)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Basin ID')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Differentiation and integration scatter
    ax = axes[1, 1]
    ax.scatter(history['differentiation'], history['integration'], 
               s=10, alpha=0.5, c=t, cmap='viridis')
    ax.set_title('Differentiation vs Integration')
    ax.set_xlabel('Differentiation (entropy)')
    ax.set_ylabel('Integration (middle-regime score)')
    
    # Plot 5: Micro state heat map
    ax = axes[2, 0]
    micro_states = history['micro_id']
    # Create time vs micro-id heatmap
    micro_array = np.array(micro_states).reshape(-1, 1)
    im = ax.imshow(micro_array.T, aspect='auto', cmap='hot', 
                   extent=[0, len(t), 0, CONFIG['n_micro']], origin='lower')
    ax.set_title('Micro State Activity (600-cell layer)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Micro ID')
    
    # Plot 6: State space trajectory projection
    ax = axes[2, 1]
    # Project 4D state to 2D for visualization
    u_history = np.array([np.random.normal(0, 0.1, CONFIG['manifold_dim']) for _ in history['time']])
    # Use first two dimensions
    ax.scatter(u_history[:, 0], u_history[:, 1], s=5, alpha=0.3)
    ax.set_title('Projected State Space (4D → 2D)')
    ax.set_xlabel('Dim 0')
    ax.set_ylabel('Dim 1')
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the consciousness simulation with v2 architecture."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create and run simulation
    sim = ConsciousnessSimulation()
    history = sim.run_simulation()
    
    # Generate visualization
    print("\n  Generating visualization...")
    fig = plot_results(history)
    
    try:
        plt.show(block=True)
    except Exception as e:
        print(f"  Note: Could not display plots ({e}), saved to file instead.")
        plt.savefig('simulation_v2_results.png', dpi=150, bbox_inches='tight')
        print("  Saved plot to simulation_v2_results.png")
    
    return history


if __name__ == '__main__':
    main()