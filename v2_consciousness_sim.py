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
    'n_macro': 24,           # Macro-basin centers (was 120, reduced for S³ coverage)
    'manifold_dim': 4,       # 4D state space (S³ hypersphere)
    'world_size': 20,        # Toroidal environment size
    'goals': 3,
    'hazards': 3,
    'timesteps': 500,        # Increased for better sampling
    
    # Two-stage update parameters
    'alpha_pull': 0.03,      # How strongly to follow macro field (light touch)
    'beta_macro': 4.0,       # Softness parameter for basin assignment (moderate selectivity)
    
    # Integration metric target range - tuned for 24 basins with beta=4
    'target_neff': 8.0,      # Target effective number of active basins
    'sigma_integr': 3.0,     # Bell width for integration score
    
    # Subsystem dynamics - tuned for balanced competition with subsystem-driven steering
    'fatigue_rate': 0.08,    # Fatigue/adaptation rate (strong enough to force rotation)
    'recovery_rate': 0.025,  # Recovery when inactive
    'floor_value': 0.05,     # Minimum activity floor
    'exploration_noise': 0.05,
    
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
    Unit hypersphere S³ in 4D with deterministic reference geometry.
    
    Micro points (~600) use a Fibonacci-style lattice on S³ for uniform coverage.
    Macro basins (~24) are derived by clustering micro points, giving a stable,
    well-separated basin structure that doesn't change with random seed.
    
    The reduced macro count (24 vs original 120) ensures basins are angularly
    separated enough for the state's step size to produce meaningful transitions.
    """
    
    def __init__(self):
        self.n_micro = CONFIG['n_micro']
        self.n_macro = CONFIG['n_macro']
        self.dim = CONFIG['manifold_dim']
        
        # Generate deterministic reference geometry
        self.micro_points = self._generate_fibonacci_s3(self.n_micro)
        self.macro_centers = self._derive_macro_basins(self.micro_points)
        
        # Pre-compute micro → macro assignment for each micro point
        sims = self.micro_points @ self.macro_centers.T  # (n_micro, n_macro)
        self.micro_to_macro = np.argmax(sims, axis=1)    # (n_micro,)
    
    def _generate_fibonacci_s3(self, n):
        """
        Generate n approximately-uniform points on S³ using a
        generalized Fibonacci lattice.
        
        Uses the golden ratio and its 4D generalization for even angular
        spacing. This is deterministic and reproducible regardless of
        numpy random state.
        """
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        points = np.zeros((n, 4))
        for i in range(n):
            # Distribute polar angles using Fibonacci-style offsets
            t1 = np.arccos(1 - 2 * (i + 0.5) / n)            # [0, pi]
            t2 = 2 * np.pi * ((i * phi) % 1.0)                # [0, 2pi]
            t3 = 2 * np.pi * ((i * phi * phi) % 1.0)          # [0, 2pi]
            
            # Hopf-like parametrization of S³
            points[i, 0] = np.sin(t1) * np.sin(t2) * np.cos(t3)
            points[i, 1] = np.sin(t1) * np.sin(t2) * np.sin(t3)
            points[i, 2] = np.sin(t1) * np.cos(t2)
            points[i, 3] = np.cos(t1)
        
        # Normalize to unit sphere (should already be close)
        norms = np.linalg.norm(points, axis=1, keepdims=True) + 1e-8
        return points / norms
    
    def _derive_macro_basins(self, micro_points):
        """
        Derive macro basin centers by clustering micro points.
        
        Uses a fixed random_state so results are identical regardless of
        the outer simulation seed.
        """
        km = KMeans(n_clusters=self.n_macro, random_state=0, n_init=20)
        km.fit(micro_points)
        centers = km.cluster_centers_.copy()
        norms = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
        return centers / norms
    
    def normalize_to_sphere(self, x):
        """Project vector to unit hypersphere."""
        if isinstance(x, np.ndarray) and len(x.shape) > 1:
            norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
            return x / norms
        else:
            norm = np.linalg.norm(x) + 1e-8
            return x / norm
    
    def project_to_tangent(self, v, u_t):
        """Project vector v onto the tangent plane of S³ at point u_t."""
        return v - np.dot(v, u_t) * u_t

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
    
    # Pre-computed preference directions with cyclic opponent structure.
    # Axis subsystems: each owns one dimension.
    # Contrast subsystems: each opposes one axis and supports the next,
    # creating a cyclic chain that prevents permanent coalitions.
    # Note: sum of normalized preferences = [1,1,1,1] (non-zero); the
    # novelty drive compensates for this by de-meaning.
    
    PREFERENCE_MATRIX = np.array([
        [+1.0,  0.0,  0.0,  0.0],   # Motor Control: +dim0
        [ 0.0, +1.0,  0.0,  0.0],   # Planning: +dim1
        [ 0.0,  0.0, +1.0,  0.0],   # Attention: +dim2
        [ 0.0,  0.0,  0.0, +1.0],   # Memory: +dim3
        [-1.0, +1.0,  0.0,  0.0],   # Emotion: opposes Motor, supports Planning
        [ 0.0, -1.0, +1.0,  0.0],   # Social: opposes Planning, supports Attention
        [ 0.0,  0.0, -1.0, +1.0],   # Intuition: opposes Attention, supports Memory
        [+1.0,  0.0,  0.0, -1.0],   # Aesthetic: opposes Memory, supports Motor
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
    
    def compute_influences(self, u_t, env_stimulus=None):
        """
        Compute raw influence from each subsystem.
        
        Influences are determined purely by manifold position. The environment
        drives navigation behavior (external expression), not internal state
        competition — this prevents feedback loops between env proximity and
        subsystem dominance.
        
        Args:
            u_t: Unit direction vector in manifold space (4,)
            env_stimulus: Unused (kept for interface compatibility)
        
        Returns:
            influences: raw influence scores from each subsystem (n_subsystems,)
        """
        influences = np.zeros(self.n_subsystems)
        
        for i in range(self.n_subsystems):
            pref_dir = self.PREFERENCE_MATRIX[i]
            norm_pref = np.linalg.norm(pref_dir)
            if norm_pref > 1e-8:
                pref_dir_normalized = pref_dir / norm_pref
            else:
                pref_dir_normalized = pref_dir
            
            dot_product = float(np.dot(u_t, pref_dir_normalized))
            influences[i] = 0.5 + 0.3 * dot_product
        
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
        
        # Secondary adaptive fatigue: penalize ALL subsystems above equal share,
        # not just the single argmax. This prevents stable coalitions where
        # a fixed subset of subsystems co-dominate indefinitely.
        equal_share = 1.0 / self.n_subsystems  # 0.125 for 8 subsystems
        for i in range(self.n_subsystems):
            excess = activities[i] - equal_share
            if excess > 0.02:  # Small margin above equal share
                self.fatigue[i] += 0.03 * excess * dt
        
        # Recover when inactive
        inactive_recovery = (1 - activities) * self.recovery_rate * dt
        self.fatigue -= inactive_recovery
        
        # Clamp to reasonable range
        self.fatigue = np.clip(self.fatigue, 0.0, 3.0)

    def compute_tangent_forces(self, u_t):
        """
        Compute tangent-space forces from each subsystem at the current state.
        
        Each subsystem's preferred direction is projected onto the tangent
        plane of S³ at u_t. This gives the great-circle direction from u_t
        toward each subsystem's preferred region — a proper geometric force
        that curves with the manifold.
        
        Unlike scalar influence scores, tangent forces preserve the full 4D
        directional information: the same subsystem pulls in different
        tangent directions depending on where the state currently sits on S³.
        
        Returns:
            forces: (n_subsystems, manifold_dim) tangent vectors at u_t
        """
        norms = np.linalg.norm(self.PREFERENCE_MATRIX, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        prefs = self.PREFERENCE_MATRIX / norms  # (n_subsystems, dim)
        
        # Radial components: dot product of each preference with current state
        radial = prefs @ u_t  # (n_subsystems,)
        
        # Tangent projection: remove radial component from each preference
        forces = prefs - np.outer(radial, u_t)  # (n_subsystems, dim)
        
        return forces


# ============================================================================
# TOROIDAL ENVIRONMENT
# ============================================================================

class ToroidalEnvironment:
    """
    Wrap-around 2D world without edge artifacts.
    
    Provides:
      - Toroidal coordinate wrapping
      - Goal/hazard positions with minimum separation
      - Proximity stimulus computation for subsystem coupling
      - Position tracking and velocity computation
    """
    
    def __init__(self, size=CONFIG['world_size']):
        self.size = size
        
        # Generate goals and hazards
        self.goals = self._generate_positions(CONFIG['goals'])
        self.hazards = self._generate_positions(CONFIG['hazards'], exclude=self.goals)
        
        # Agent state
        self.pos = np.array([size / 2.0, size / 2.0])  # Start center
        self.prev_pos = self.pos.copy()
        
        # Tracking
        self.goals_reached = 0
        self.hazards_hit = 0
        
        print(f"  Environment: {size}x{size} torus | {len(self.goals)} goals, {len(self.hazards)} hazards")
    
    def _generate_positions(self, n, exclude=None):
        """Generate random positions on torus."""
        positions = []
        existing = [list(p) for p in exclude] if exclude is not None else []
        
        while len(positions) < n:
            pos = np.random.randint(0, self.size, 2).tolist()
            too_close = False
            for ex in existing:
                if self._toroidal_distance(pos, ex) < 3:
                    too_close = True
                    break
            if not too_close:
                positions.append(pos)
                existing.append(pos)
        
        return np.array(positions, dtype=float)
    
    def _toroidal_distance(self, pos1, pos2):
        """Compute shortest path distance on torus."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        dx = min(dx, self.size - dx)
        dy = min(dy, self.size - dy)
        return np.sqrt(dx**2 + dy**2)
    
    def _toroidal_direction(self, from_pos, to_pos):
        """Unit direction vector on torus from -> to (shortest path)."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        # Wrap to [-size/2, size/2]
        if dx > self.size / 2: dx -= self.size
        if dx < -self.size / 2: dx += self.size
        if dy > self.size / 2: dy -= self.size
        if dy < -self.size / 2: dy += self.size
        norm = np.sqrt(dx**2 + dy**2) + 1e-8
        return np.array([dx / norm, dy / norm])
    
    def step(self, velocity):
        """
        Move the agent by velocity (2D) with toroidal wrapping.
        Returns position and event flags.
        """
        self.prev_pos = self.pos.copy()
        self.pos = (self.pos + velocity) % self.size
        
        # Check goal/hazard proximity
        goal_event = False
        hazard_event = False
        for g in self.goals:
            if self._toroidal_distance(self.pos, g) < 1.5:
                self.goals_reached += 1
                goal_event = True
        for h in self.hazards:
            if self._toroidal_distance(self.pos, h) < 1.5:
                self.hazards_hit += 1
                hazard_event = True
        
        return self.pos.copy(), goal_event, hazard_event
    
    def compute_stimulus(self):
        """
        Compute environmental stimulus dict for subsystem influence coupling.
        
        Returns proximity signals normalized to [0, 1] for goals, hazards,
        and agent velocity.
        """
        # Closest goal proximity: exponential falloff, 1 = on top, ~0 beyond 5 cells
        goal_dists = [self._toroidal_distance(self.pos, g) for g in self.goals]
        min_goal = min(goal_dists) if goal_dists else self.size
        goal_prox = np.exp(-min_goal / 2.0)  # ~0.6 at dist=1, ~0.08 at dist=5
        
        # Closest hazard proximity
        hazard_dists = [self._toroidal_distance(self.pos, h) for h in self.hazards]
        min_hazard = min(hazard_dists) if hazard_dists else self.size
        hazard_prox = np.exp(-min_hazard / 2.0)
        
        # Velocity (normalized by max possible step)
        disp = self.pos - self.prev_pos
        # Handle wrapping in velocity
        for d in range(2):
            if disp[d] > self.size / 2: disp[d] -= self.size
            if disp[d] < -self.size / 2: disp[d] += self.size
        speed = np.linalg.norm(disp) / 2.0  # normalize by typical max
        speed = min(speed, 1.0)
        
        return {
            'goal_prox': goal_prox,
            'hazard_prox': hazard_prox,
            'velocity': speed,
            'goal_dir': self._toroidal_direction(self.pos, self.goals[np.argmin(goal_dists)]),
            'hazard_dir': self._toroidal_direction(self.pos, self.hazards[np.argmin(hazard_dists)]),
        }


# ============================================================================
# 4D SLICE: PERCEPTION AND NAVIGATION INTERFACE
# ============================================================================

class PerceptionSlice:
    """
    The geometric interface between the 4D inner manifold and the 2D torus.
    
    The '4D slice' concept: the being's 4D state u_t ∈ S³ is not just an
    abstract mental state — it determines HOW the being perceives and
    navigates the physical world. The slice is a geometric projection
    that maps internal state to external behavior.
    
    The 4D state has two geometric roles:
    
      Dims 0–1 of u_t  →  NAVIGATION HEADING
        The first two components of the state direction are projected
        onto the torus as velocity. When the manifold state rotates
        in dims 0-1 (due to subsystem competition), the being physically
        turns on the torus. This is the original "4D slice" idea.
    
      Dims 2–3 of u_t  →  PERCEPTION MODE
        dim2 controls perception RANGE (how far the being can sense)
        dim3 controls perception FOCUS (wide/diffuse vs narrow/sharp)
    
    The being's sensory input is filtered through its perception mode
    before feeding back into the manifold dynamics. A being in "focused
    exploration" mode (high dim2, low dim3) sees far but diffusely.
    A being in "threat lock" mode (low dim2, high dim3) sees close
    but sharply.
    
    This creates a genuine feedback loop:
        manifold state → perception mode → what the being sees →
        sensory gradient in 4D → manifold dynamics → new state
    """
    
    # Perception range parameters
    MIN_RANGE = 2.0    # Minimum perception radius (cells)
    MAX_RANGE = 8.0    # Maximum perception radius
    MIN_FOCUS = 0.3    # Most diffuse (wide attention)
    MAX_FOCUS = 1.0    # Most focused (sharp attention)
    
    # Navigation
    BASE_SPEED = 0.4   # Base movement speed
    MAX_SPEED = 1.5    # Maximum movement speed
    
    @staticmethod
    def compute_heading(u_t):
        """
        Project the 4D manifold state onto a 2D heading vector.
        
        Dims 0-1 of u_t are the navigation slice. The being walks
        in the direction its internal state points in these two
        dimensions. Speed scales with the magnitude of the projection.
        
        Returns:
            heading: 2D unit vector (or zero if state is purely internal)
            speed: scalar movement speed
        """
        heading_raw = u_t[:2].copy()
        mag = np.linalg.norm(heading_raw)
        
        if mag < 1e-6:
            return np.zeros(2), 0.0
        
        heading = heading_raw / mag
        # Speed proportional to how much of the state is in the movement plane
        # If u_t points purely in dims 2-3, the being stands still (internal mode)
        speed = PerceptionSlice.BASE_SPEED + (PerceptionSlice.MAX_SPEED - PerceptionSlice.BASE_SPEED) * mag
        
        return heading, speed
    
    @staticmethod
    def compute_perception(u_t):
        """
        Extract perception parameters from the 4D state.
        
        dim2 → perception range (how far the being can sense)
        dim3 → perception focus (wide vs narrow attention cone)
        
        Both dims range [-1, 1] on S³. We map them to useful ranges:
          range: 2 to 8 cells (using |dim2|, so both poles give range)
          focus: 0.3 to 1.0 (using |dim3|)
        
        Returns:
            perc_range: radius of perception in torus cells
            perc_focus: sharpness factor (1.0 = sharp falloff, 0.3 = diffuse)
            mode_name: human-readable perception mode string
        """
        # Use absolute values — both poles of S³ are symmetric
        range_param = abs(u_t[2])   # 0 to 1
        focus_param = abs(u_t[3])   # 0 to 1
        
        perc_range = PerceptionSlice.MIN_RANGE + range_param * (PerceptionSlice.MAX_RANGE - PerceptionSlice.MIN_RANGE)
        perc_focus = PerceptionSlice.MIN_FOCUS + focus_param * (PerceptionSlice.MAX_FOCUS - PerceptionSlice.MIN_FOCUS)
        
        # Classify perception mode
        if range_param > 0.5 and focus_param < 0.5:
            mode = "exploration"     # See far, diffuse
        elif range_param < 0.5 and focus_param > 0.5:
            mode = "threat-lock"     # See close, sharp
        elif range_param > 0.5 and focus_param > 0.5:
            mode = "vigilant"        # See far, sharp (high alertness)
        else:
            mode = "internal"        # See close, diffuse (daydreaming)
        
        return perc_range, perc_focus, mode
    
    @staticmethod
    def perceive_world(u_t, environment):
        """
        Sense the torus through the perception slice.
        
        Only objects within perception range are visible. Closer objects
        have stronger signals. Focus parameter controls the falloff rate.
        
        Returns a dict of perceived stimuli (only what the being can see).
        """
        perc_range, perc_focus, mode = PerceptionSlice.compute_perception(u_t)
        pos = environment.pos
        
        # Perceive goals within range
        perceived_goals = []
        for g in environment.goals:
            dist = environment._toroidal_distance(pos, g)
            if dist < perc_range:
                # Signal strength: focus-dependent falloff
                strength = np.exp(-perc_focus * dist / perc_range)
                direction = environment._toroidal_direction(pos, g)
                perceived_goals.append({
                    'dist': dist, 'strength': strength, 'dir': direction
                })
        
        # Perceive hazards within range
        perceived_hazards = []
        for h in environment.hazards:
            dist = environment._toroidal_distance(pos, h)
            if dist < perc_range:
                strength = np.exp(-perc_focus * dist / perc_range)
                direction = environment._toroidal_direction(pos, h)
                perceived_hazards.append({
                    'dist': dist, 'strength': strength, 'dir': direction
                })
        
        return {
            'goals': perceived_goals,
            'hazards': perceived_hazards,
            'perc_range': perc_range,
            'perc_focus': perc_focus,
            'mode': mode,
            'n_visible_goals': len(perceived_goals),
            'n_visible_hazards': len(perceived_hazards),
        }
    
    @staticmethod
    def sensory_gradient(perception, u_t):
        """
        Convert perceived world objects into a 4D gradient on the manifold.
        
        This is the feedback path: what the being sees pushes its internal
        state. Goals create attraction in dims 0-1 (toward the goal heading).
        Hazards create repulsion. The strength depends on the perception
        signal, which itself depends on dims 2-3.
        
        This closes the loop:
            u_t → perception → visible objects → gradient → nudge u_t
        
        Returns:
            gradient: 4D vector to be added to the manifold drive
        """
        gradient = np.zeros(4)
        
        # Goal attraction in the movement plane (dims 0-1)
        for g in perception['goals']:
            # Push heading toward goal
            gradient[:2] += g['strength'] * 0.15 * g['dir']
        
        # Hazard repulsion in the movement plane
        for h in perception['hazards']:
            # Push heading away from hazard
            gradient[:2] -= h['strength'] * 0.25 * h['dir']
        
        # Perceptual meta-signal in dims 2-3:
        # If nothing visible → push toward wider range (increase |dim2|)
        # If hazard close → push toward sharper focus (increase |dim3|)
        if perception['n_visible_goals'] == 0 and perception['n_visible_hazards'] == 0:
            # Nothing visible — expand perception range
            gradient[2] += 0.05 * np.sign(u_t[2] + 1e-6)
        
        if perception['n_visible_hazards'] > 0:
            # Hazard nearby — sharpen focus
            max_hazard_str = max(h['strength'] for h in perception['hazards'])
            gradient[3] += 0.08 * max_hazard_str * np.sign(u_t[3] + 1e-6)
        
        return gradient


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
        # Random initial direction avoids systematic bias toward any subsystem
        init_dir = np.random.normal(size=CONFIG['manifold_dim'])
        self.u_t = self.manifold.normalize_to_sphere(init_dir)
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
            'u_t': [],  # 4D direction trajectory for visualization
            'env_x': [],
            'env_y': [],
            'goal_prox': [],
            'hazard_prox': [],
            'perc_range': [],
            'perc_focus': [],
            'perc_mode': [],
            'n_visible': [],
            'heading_x': [],
            'heading_y': [],
            'speed': [],
            # ── Geometric metrics (tangent force field quantities) ──
            'conflict_angle': [],       # Angle between top-2 subsystem forces (radians)
            'clarity': [],              # Resultant force magnitude (0 = indecision)
            'curvature': [],            # Trajectory curvature proxy (angular change rate)
            'inner_outer_ratio': [],    # Perception/navigation orientation ratio
            'force_mag_0': [],          # Per-subsystem tangent force magnitudes
            'force_mag_1': [],
            'force_mag_2': [],
            'force_mag_3': [],
            'force_mag_4': [],
            'force_mag_5': [],
            'force_mag_6': [],
            'force_mag_7': [],
        }
        
        self.recent_states = []
        self.basin_dwell = 0          # Steps in current macro basin
        self.current_basin = -1       # Current dominant macro basin
    
    def step(self, timestep):
        """Single simulation step with two-stage update + perception slice."""
        
        # --- PERCEPTION: what the being sees through its 4D slice ---
        perception = PerceptionSlice.perceive_world(self.u_t, self.environment)
        sensory_grad = PerceptionSlice.sensory_gradient(perception, self.u_t)
        
        # Also compute raw env stimulus for metrics tracking
        env_stimulus = self.environment.compute_stimulus()
        
        # --- STAGE 1: Micro transition (local exploration) ---
        
        # Compute subsystem influences based on current state
        raw_influences = self.subsystems.compute_influences(
            self.u_t,
            env_stimulus=env_stimulus
        )
        
        # Apply competition and fatigue
        activities = self.subsystems.apply_competition_and_fatigue(raw_influences)
        
        # Update subsystem fatigue
        self.subsystems.update_fatigue(activities)
        
        # Compute micro transition direction driven by competing subsystems
        dominant_subsystem = np.argmax(activities)
        
        # --- Geometric forces: subsystems as tangent vector fields on S³ ---
        # Each subsystem generates a force in the tangent plane at u_t,
        # pointing along the great circle toward its preferred region.
        # Forces are state-dependent: the same subsystem pulls in different
        # tangent directions depending on where u_t sits on S³.
        # This replaces the scalar-score-then-reconstruct approach with
        # direct geometric computation — no information lost in projection.
        tangent_forces = self.subsystems.compute_tangent_forces(self.u_t)
        
        # Activity-weighted force: dominant subsystems steer more strongly
        activity_force = activities @ tangent_forces  # (4,)
        
        # Novelty force: rested subsystems attract exploratory movement
        rest_scores = np.exp(-self.subsystems.fatigue)
        novelty_force = rest_scores @ tangent_forces
        # De-mean: remove systematic directional bias from preference structure
        mean_rest = np.mean(rest_scores)
        novelty_force -= mean_rest * np.sum(tangent_forces, axis=0) / self.subsystems.n_subsystems
        
        # Blend activity guidance with novelty exploration
        novelty_weight = 0.6
        drive = (1.0 - novelty_weight) * activity_force + novelty_weight * novelty_force
        
        # Exploration noise projected to tangent plane (stays on manifold)
        noise = np.random.normal(0, CONFIG['exploration_noise'], CONFIG['manifold_dim'])
        drive += self.manifold.project_to_tangent(noise, self.u_t)
        
        # Sensory gradient projected to tangent plane
        drive += self.manifold.project_to_tangent(sensory_grad, self.u_t)
        
        # Move along tangent direction and retract to S³
        steering_strength = 0.3
        new_dir = self.u_t + steering_strength * drive
        self.u_t = self.manifold.normalize_to_sphere(new_dir)
        
        # Adjust magnitude based on overall activity
        total_activity = np.sum(activities)
        expected_total = 1.0 / CONFIG['n_subsystems']
        self.r_t = np.clip(self.r_t + 0.05 * (total_activity - expected_total), 0.5, 2.0)
        
        # --- STAGE 2: Macro reconciliation (global constraint via weighted field) ---
        
        # Compute soft macro assignment and field
        _, macro_weights, macro_field = self.assignment.compute_macro_assignment(self.u_t)
        
        # Track basin dwell time for escape mechanism
        current_dominant = np.argmax(macro_weights)
        if current_dominant == self.current_basin:
            self.basin_dwell += 1
        else:
            self.basin_dwell = 0
            self.current_basin = current_dominant
        
        # Basin-escape: if stuck in one basin too long, inject a great-circle
        # perturbation toward a random neighboring basin. This models the
        # natural tendency of complex systems to explore their state space
        # rather than settling into a single attractor.
        escape_threshold = 25  # Steps before escape kicks in
        if self.basin_dwell > escape_threshold:
            # Probability ramps up with dwell time
            escape_prob = min(0.3, 0.05 * (self.basin_dwell - escape_threshold))
            if np.random.random() < escape_prob:
                # Pick a random different basin and push toward it
                other_basins = [i for i in range(self.manifold.n_macro) if i != current_dominant]
                target = np.random.choice(other_basins)
                target_dir = self.manifold.macro_centers[target]
                
                # Project out the component along u_t to get tangent direction
                tangent = target_dir - np.dot(target_dir, self.u_t) * self.u_t
                tn = np.linalg.norm(tangent)
                if tn > 1e-6:
                    tangent /= tn
                    # Strong push: 0.4 of a great-circle step
                    escape_dir = np.cos(0.4) * self.u_t + np.sin(0.4) * tangent
                    self.u_t = self.manifold.normalize_to_sphere(escape_dir)
                    self.basin_dwell = 0  # Reset after escape
        
        # Pull toward weighted macro field via tangent projection
        macro_tangent = self.manifold.project_to_tangent(macro_field, self.u_t)
        new_dir = self.u_t + CONFIG['alpha_pull'] * macro_tangent
        self.u_t = self.manifold.normalize_to_sphere(new_dir)
        
        # --- STAGE 3: Environment navigation via perception slice ---
        # The being's heading and speed emerge from its 4D manifold state.
        # Dims 0-1 define the walking direction; their magnitude sets speed.
        # No hardcoded subsystem→behavior rules — all navigation behavior
        # emerges from how subsystem competition rotates u_t.
        
        heading, speed = PerceptionSlice.compute_heading(self.u_t)
        velocity_2d = heading * speed
        
        # Clamp max step to prevent teleporting
        step_mag = np.linalg.norm(velocity_2d)
        if step_mag > 2.0:
            velocity_2d *= 2.0 / step_mag
        
        self.environment.step(velocity_2d)
        
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
        self.history['u_t'].append(self.u_t.copy())
        self.history['env_x'].append(self.environment.pos[0])
        self.history['env_y'].append(self.environment.pos[1])
        self.history['goal_prox'].append(env_stimulus['goal_prox'])
        self.history['hazard_prox'].append(env_stimulus['hazard_prox'])
        self.history['perc_range'].append(perception['perc_range'])
        self.history['perc_focus'].append(perception['perc_focus'])
        self.history['perc_mode'].append(perception['mode'])
        self.history['n_visible'].append(perception['n_visible_goals'] + perception['n_visible_hazards'])
        self.history['heading_x'].append(heading[0])
        self.history['heading_y'].append(heading[1])
        self.history['speed'].append(speed)
        
        # ── Geometric metrics: computed from tangent force field ──
        # Conflict angle: how much the top-2 subsystem forces disagree
        top2_idx = np.argsort(activities)[-2:]
        f0, f1 = tangent_forces[top2_idx[0]], tangent_forces[top2_idx[1]]
        cos_conflict = np.dot(f0, f1) / (np.linalg.norm(f0) * np.linalg.norm(f1) + 1e-8)
        conflict_angle = np.arccos(np.clip(cos_conflict, -1, 1))
        self.history['conflict_angle'].append(conflict_angle)
        
        # Clarity: magnitude of activity-weighted resultant force
        resultant = activities @ tangent_forces  # (4,)
        self.history['clarity'].append(np.linalg.norm(resultant))
        
        # Curvature proxy: angle between consecutive tangent vectors
        if len(self.history['u_t']) >= 2:
            u_prev = self.history['u_t'][-2]
            dot_curv = np.clip(np.dot(u_prev, self.u_t), -1, 1)
            self.history['curvature'].append(np.arccos(dot_curv))
        else:
            self.history['curvature'].append(0.0)
        
        # Inner/outer orientation: perception dims vs navigation dims
        self.history['inner_outer_ratio'].append(
            np.linalg.norm(self.u_t[2:4]) / (np.linalg.norm(self.u_t[0:2]) + 1e-8)
        )
        
        # Per-subsystem tangent force magnitudes
        force_mags = np.linalg.norm(tangent_forces, axis=1)
        for si in range(8):
            self.history[f'force_mag_{si}'].append(force_mags[si])
        
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
        
        print(f"\n  Environment:")
        print(f"    Goals reached:         {self.environment.goals_reached}")
        print(f"    Hazards hit:           {self.environment.hazards_hit}")
        avg_goal = np.mean(self.history['goal_prox'])
        avg_hazard = np.mean(self.history['hazard_prox'])
        print(f"    Avg goal proximity:    {avg_goal:.3f}")
        print(f"    Avg hazard proximity:  {avg_hazard:.3f}")
        
        # Geometric metrics summary
        conflict_deg = np.degrees(self.history['conflict_angle'])
        print(f"\n  Geometric metrics (averages):")
        print(f"    Conflict angle:        {np.mean(conflict_deg):.1f}°  (range {np.min(conflict_deg):.0f}°–{np.max(conflict_deg):.0f}°)")
        print(f"    Clarity of purpose:    {np.mean(self.history['clarity']):.4f}")
        print(f"    Trajectory curvature:  {np.mean(self.history['curvature']):.4f} rad/step")
        print(f"    Inner/outer ratio:     {np.mean(self.history['inner_outer_ratio']):.3f}")
        
        # Export to CSV (split u_t into separate dimension columns)
        export = {k: v for k, v in self.history.items() if k != 'u_t'}
        u_arr = np.array(self.history['u_t'])
        for d in range(CONFIG['manifold_dim']):
            export[f'u_dim{d}'] = u_arr[:, d]
        df = pd.DataFrame(export)
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
    ax.set_title('Macro State Over Time (24 basins)')
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
    
    # Plot 6: State space trajectory projection (actual 4D → 2D)
    ax = axes[2, 1]
    u_history = np.array(history['u_t'])
    # Use first two dimensions of actual manifold trajectory
    ax.scatter(u_history[:, 0], u_history[:, 1], s=5, alpha=0.3, c=t, cmap='viridis')
    ax.set_title('Manifold Trajectory (dims 0-1)')
    ax.set_xlabel('Dim 0')
    ax.set_ylabel('Dim 1')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def map_cognitive_landscape(sim, history):
    """
    Map the cognitive landscape on S³ and trace the being's journey through it.
    
    Produces a multi-panel figure that lets you:
    - See the force field the being navigates (vector field on S³ projected to 2D)
    - Follow its trajectory with psychological state annotations
    - Watch conflict, clarity, and perception mode evolve
    - Identify attractors, saddle regions, and mode boundaries
    
    The key idea: S³ is projected to 2D via PCA of the actual trajectory,
    so the view shows the plane the being *actually moves through*. A dense
    grid is sampled in this plane, lifted back to S³, and the force field
    is computed at each sample point.
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.gridspec as gridspec
    
    u_history = np.array(history['u_t'])
    t = np.array(history['time'])
    n_steps = len(t)
    
    # ── PCA: find the 2D plane the being actually travels through ──
    u_centered = u_history - u_history.mean(axis=0)
    cov = u_centered.T @ u_centered / n_steps
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Top 2 eigenvectors (largest variance)
    pc1 = eigvecs[:, -1]  # Primary axis of motion
    pc2 = eigvecs[:, -2]  # Secondary axis
    
    # Project trajectory to PCA plane
    traj_x = u_history @ pc1
    traj_y = u_history @ pc2
    
    # ── Sample force field on a grid in the PCA plane ──
    grid_n = 30
    x_range = np.linspace(traj_x.min() - 0.15, traj_x.max() + 0.15, grid_n)
    y_range = np.linspace(traj_y.min() - 0.15, traj_y.max() + 0.15, grid_n)
    gx, gy = np.meshgrid(x_range, y_range)
    
    # At each grid point: lift to S³, compute forces, project resultant back
    resultant_field_x = np.zeros_like(gx)
    resultant_field_y = np.zeros_like(gy)
    conflict_field = np.zeros_like(gx)
    clarity_field = np.zeros_like(gx)
    dominant_field = np.zeros_like(gx, dtype=int)
    
    for i in range(grid_n):
        for j in range(grid_n):
            # Lift 2D grid point to 4D and normalize to S³
            u_sample = gx[i, j] * pc1 + gy[i, j] * pc2
            norm_s = np.linalg.norm(u_sample)
            if norm_s < 0.1:
                continue  # Too close to origin, skip
            u_sample = u_sample / norm_s
            
            # Compute tangent forces at this point
            forces = sim.subsystems.compute_tangent_forces(u_sample)
            influences = sim.subsystems.compute_influences(u_sample)
            activities = sim.subsystems.apply_competition_and_fatigue(influences)
            # Reset fatigue side-effects from sampling
            sim.subsystems.fatigue *= 0
            sim.subsystems.recent_dominant_activities.clear()
            
            # Activity-weighted resultant force
            resultant = activities @ forces  # (4,)
            
            # Project resultant back to PCA plane for visualization
            resultant_field_x[i, j] = np.dot(resultant, pc1)
            resultant_field_y[i, j] = np.dot(resultant, pc2)
            
            # Clarity = magnitude of resultant
            clarity_field[i, j] = np.linalg.norm(resultant)
            
            # Conflict angle between top-2
            top2 = np.argsort(activities)[-2:]
            f0 = forces[top2[0]]
            f1 = forces[top2[1]]
            cos_a = np.dot(f0, f1) / (np.linalg.norm(f0) * np.linalg.norm(f1) + 1e-8)
            conflict_field[i, j] = np.arccos(np.clip(cos_a, -1, 1))
            
            # Dominant subsystem
            dominant_field[i, j] = np.argmax(activities)
    
    # ── Compute per-timestep geometric state for the trajectory ──
    traj_conflict = np.zeros(n_steps)
    traj_clarity = np.zeros(n_steps)
    traj_perc_nav = np.zeros(n_steps)
    traj_force_mags = np.zeros((n_steps, sim.subsystems.n_subsystems))
    
    for step in range(n_steps):
        u = u_history[step]
        forces = sim.subsystems.compute_tangent_forces(u)
        influences = sim.subsystems.compute_influences(u)
        activities = sim.subsystems.apply_competition_and_fatigue(influences)
        sim.subsystems.fatigue *= 0
        sim.subsystems.recent_dominant_activities.clear()
        
        resultant = activities @ forces
        traj_clarity[step] = np.linalg.norm(resultant)
        traj_force_mags[step] = np.linalg.norm(forces, axis=1)
        
        top2 = np.argsort(activities)[-2:]
        f0, f1 = forces[top2[0]], forces[top2[1]]
        cos_a = np.dot(f0, f1) / (np.linalg.norm(f0) * np.linalg.norm(f1) + 1e-8)
        traj_conflict[step] = np.arccos(np.clip(cos_a, -1, 1))
        
        traj_perc_nav[step] = np.linalg.norm(u[2:4]) / (np.linalg.norm(u[0:2]) + 1e-8)
    
    # Restore sim fatigue state after sampling
    sim.subsystems.fatigue = np.zeros(sim.subsystems.n_subsystems)
    
    # ── BUILD THE FIGURE ──
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    names = sim.subsystems.SUBSYSTEM_NAMES
    sub_colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
    # ─── Panel 1: Cognitive landscape — force field with trajectory ───
    ax = fig.add_subplot(gs[0, 0:2])
    
    # Background: conflict field as terrain
    im = ax.contourf(gx, gy, np.degrees(conflict_field), levels=20,
                      cmap='RdYlBu_r', alpha=0.6)
    plt.colorbar(im, ax=ax, label='Conflict angle (°)', shrink=0.8)
    
    # Force field arrows
    skip = 2
    ax.quiver(gx[::skip, ::skip], gy[::skip, ::skip],
              resultant_field_x[::skip, ::skip], resultant_field_y[::skip, ::skip],
              color='k', alpha=0.4, scale=1.5, width=0.003)
    
    # Trajectory colored by time
    sc = ax.scatter(traj_x, traj_y, c=t, cmap='plasma', s=8, alpha=0.7, zorder=5)
    ax.plot(traj_x, traj_y, 'k-', alpha=0.15, lw=0.5)
    
    # Mark start and end
    ax.plot(traj_x[0], traj_y[0], 'go', markersize=10, zorder=10, label='Start')
    ax.plot(traj_x[-1], traj_y[-1], 'rs', markersize=10, zorder=10, label='End')
    
    # Mark macro basin centers projected to PCA plane
    macro_proj_x = sim.manifold.macro_centers @ pc1
    macro_proj_y = sim.manifold.macro_centers @ pc2
    ax.scatter(macro_proj_x, macro_proj_y, marker='x', c='white',
               s=60, linewidths=2, zorder=8, label='Macro basins')
    
    ax.set_title('Cognitive Landscape: Force Field on S³\n'
                 '(background = conflict intensity, arrows = resultant force)',
                 fontsize=11)
    ax.set_xlabel(f'PC1 ({100*eigvals[-1]/eigvals.sum():.0f}% variance)')
    ax.set_ylabel(f'PC2 ({100*eigvals[-2]/eigvals.sum():.0f}% variance)')
    ax.legend(loc='upper left', fontsize=8)
    
    # ─── Panel 2: Dominant subsystem territories ───
    ax = fig.add_subplot(gs[0, 2])
    
    # Color each grid point by which subsystem dominates there
    dom_colors = np.zeros((*dominant_field.shape, 4))
    for idx in range(8):
        mask = dominant_field == idx
        dom_colors[mask] = sub_colors[idx]
    
    ax.imshow(dom_colors, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
              origin='lower', aspect='auto', alpha=0.7)
    ax.plot(traj_x, traj_y, 'k-', alpha=0.3, lw=0.5)
    ax.scatter(traj_x, traj_y, c=t, cmap='plasma', s=4, alpha=0.5, zorder=5)
    
    # Legend
    for idx, name in enumerate(names):
        ax.plot([], [], 's', color=sub_colors[idx], markersize=8, label=name)
    ax.legend(loc='upper right', fontsize=6, ncol=1)
    ax.set_title('Subsystem Territories\n(who dominates where on S³)', fontsize=11)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    
    # ─── Panel 3: Conflict angle over time ───
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, np.degrees(traj_conflict), color='crimson', alpha=0.7, lw=1)
    ax.axhline(90, color='grey', ls='--', alpha=0.5, label='90° (orthogonal)')
    ax.fill_between(t, 0, np.degrees(traj_conflict), alpha=0.15, color='crimson')
    ax.set_title('Internal Conflict\n(angle between top-2 subsystem forces)', fontsize=11)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Conflict angle (°)')
    ax.set_ylim(0, 180)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Annotate high-conflict moments
    high_conflict = np.where(np.degrees(traj_conflict) > 130)[0]
    if len(high_conflict) > 0:
        for hc in high_conflict[::max(1, len(high_conflict)//5)]:
            ax.annotate('torn', xy=(t[hc], np.degrees(traj_conflict[hc])),
                       fontsize=6, color='darkred', alpha=0.7)
    
    # ─── Panel 4: Clarity (resultant force magnitude) over time ───
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, traj_clarity, color='teal', alpha=0.7, lw=1)
    ax.fill_between(t, 0, traj_clarity, alpha=0.15, color='teal')
    ax.set_title('Clarity of Purpose\n(magnitude of net tangent force)', fontsize=11)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Resultant magnitude')
    ax.grid(True, alpha=0.3)
    
    # Annotate high-clarity moments
    high_clarity = np.where(traj_clarity > traj_clarity.mean() + 2*traj_clarity.std())[0]
    if len(high_clarity) > 0:
        for hc in high_clarity[::max(1, len(high_clarity)//5)]:
            ax.annotate('focused', xy=(t[hc], traj_clarity[hc]),
                       fontsize=6, color='darkgreen', alpha=0.7)
    
    # ─── Panel 5: Perception mode timeline ───
    ax = fig.add_subplot(gs[1, 2])
    mode_map = {'exploration': 0, 'vigilant': 1, 'threat-lock': 2, 'internal': 3}
    mode_colors_map = {'exploration': '#2196F3', 'vigilant': '#FF9800',
                       'threat-lock': '#F44336', 'internal': '#9C27B0'}
    modes = history['perc_mode']
    mode_nums = [mode_map.get(m, 0) for m in modes]
    
    for mname, mnum in mode_map.items():
        mask = np.array([m == mname for m in modes])
        ax.fill_between(t, mnum - 0.4, mnum + 0.4, where=mask,
                        color=mode_colors_map[mname], alpha=0.6)
    
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['exploration', 'vigilant', 'threat-lock', 'internal'], fontsize=9)
    ax.set_title('Perception Mode\n(how the being sees the world)', fontsize=11)
    ax.set_xlabel('Timestep')
    ax.grid(True, alpha=0.3, axis='x')
    
    # ─── Panel 6: Per-subsystem force magnitude (distance from home) ───
    ax = fig.add_subplot(gs[2, 0:2])
    for i in range(8):
        ax.plot(t, traj_force_mags[:, i], color=sub_colors[i], alpha=0.6,
                lw=1, label=names[i])
    ax.set_title('Subsystem Tangent Forces Over Time\n'
                 '(weaker = closer to preferred region, stronger = far from home)',
                 fontsize=11)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Force magnitude')
    ax.legend(loc='upper right', fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)
    
    # ─── Panel 7: Inner/Outer orientation ───
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(t, traj_perc_nav, color='indigo', alpha=0.7, lw=1)
    ax.axhline(1.0, color='grey', ls='--', alpha=0.5, label='Balanced')
    ax.fill_between(t, 1.0, traj_perc_nav,
                    where=traj_perc_nav > 1.0, color='purple', alpha=0.2, label='Contemplative')
    ax.fill_between(t, traj_perc_nav, 1.0,
                    where=traj_perc_nav < 1.0, color='orange', alpha=0.2, label='Action-oriented')
    ax.set_title('Inner vs Outer Orientation\n(perception/navigation ratio)', fontsize=11)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Ratio (>1 = inward, <1 = outward)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # ─── Panel 8: Torus navigation with perception cones ───
    ax = fig.add_subplot(gs[3, 0])
    env = sim.environment
    ex = history['env_x']
    ey = history['env_y']
    
    ax.scatter(ex, ey, c=t, cmap='plasma', s=4, alpha=0.5, zorder=5)
    ax.plot(ex, ey, 'k-', alpha=0.1, lw=0.3)
    
    # Goals and hazards
    ax.scatter(env.goals[:, 0], env.goals[:, 1], marker='*', c='gold',
               s=200, edgecolors='k', zorder=10, label='Goals')
    ax.scatter(env.hazards[:, 0], env.hazards[:, 1], marker='^', c='red',
               s=100, edgecolors='k', zorder=10, label='Hazards')
    
    # Show perception range at a few sample points
    sample_steps = np.linspace(0, n_steps-1, 8, dtype=int)
    for s in sample_steps:
        r = history['perc_range'][s]
        circle = plt.Circle((ex[s], ey[s]), r, fill=False,
                            color=mode_colors_map.get(modes[s], 'grey'),
                            alpha=0.4, lw=1)
        ax.add_patch(circle)
    
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_aspect('equal')
    ax.set_title('Torus Navigation\n(circles = perception range)', fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # ─── Panel 9: Clarity field (equilibria map) ───
    ax = fig.add_subplot(gs[3, 1])
    im = ax.contourf(gx, gy, clarity_field, levels=20, cmap='YlOrRd')
    plt.colorbar(im, ax=ax, label='Resultant magnitude', shrink=0.8)
    
    # Mark low-clarity regions (near equilibria / saddle points)
    eq_mask = clarity_field < np.percentile(clarity_field[clarity_field > 0], 15)
    eq_y, eq_x = np.where(eq_mask)
    if len(eq_x) > 0:
        ax.scatter(gx[0, eq_x] if len(gx.shape) > 1 else x_range[eq_x],
                   gy[eq_y, 0] if len(gy.shape) > 1 else y_range[eq_y],
                   marker='o', facecolors='none', edgecolors='cyan',
                   s=30, lw=1.5, label='Near-equilibria', zorder=8)
    
    ax.scatter(traj_x, traj_y, c='white', s=2, alpha=0.3, zorder=5)
    ax.set_title('Clarity Field\n(low = balanced/stuck, high = driven)', fontsize=11)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=7)
    
    # ─── Panel 10: Interpretive state narrative ───
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    
    # Compute summary interpretations
    avg_conflict = np.degrees(np.mean(traj_conflict))
    avg_clarity = np.mean(traj_clarity)
    avg_perc_nav = np.mean(traj_perc_nav)
    mode_counts = {}
    for m in modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1
    dominant_mode = max(mode_counts, key=mode_counts.get)
    
    high_conflict_pct = 100 * np.sum(np.degrees(traj_conflict) > 120) / n_steps
    high_clarity_pct = 100 * np.sum(traj_clarity > avg_clarity + traj_clarity.std()) / n_steps
    
    # Force magnitude ranking (who's closest to home most often)
    mean_force = traj_force_mags.mean(axis=0)
    closest_sub = names[np.argmin(mean_force)]
    farthest_sub = names[np.argmax(mean_force)]
    
    narrative = (
        f"COGNITIVE PROFILE\n"
        f"{'='*32}\n\n"
        f"Avg internal conflict: {avg_conflict:.0f}°\n"
        f"  {'Highly conflicted' if avg_conflict > 110 else 'Moderate tension' if avg_conflict > 90 else 'Relatively coherent'}\n\n"
        f"Deeply torn moments: {high_conflict_pct:.0f}% of time\n"
        f"Peak clarity moments: {high_clarity_pct:.0f}% of time\n\n"
        f"Orientation: {'Contemplative' if avg_perc_nav > 1.1 else 'Action-oriented' if avg_perc_nav < 0.9 else 'Balanced'}\n"
        f"  (ratio: {avg_perc_nav:.2f})\n\n"
        f"Dominant perception: {dominant_mode}\n"
        f"  {mode_counts}\n\n"
        f"Closest subsystem: {closest_sub}\n"
        f"  (state lingers near its region)\n"
        f"Farthest subsystem: {farthest_sub}\n"
        f"  (rarely visits its territory)\n\n"
        f"Goals reached: {sim.environment.goals_reached}\n"
        f"Hazards hit: {sim.environment.hazards_hit}"
    )
    
    ax.text(0.05, 0.95, narrative, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Interpretive Summary', fontsize=11)
    
    fig.suptitle('Cognitive Landscape of Consciousness Simulation',
                 fontsize=14, fontweight='bold', y=0.98)
    
    return fig


# ============================================================================
# PHASE DETECTION — Emergent cognitive phases from geometric signatures
# ============================================================================

def detect_cognitive_phases(history, max_phases=8, smoothing_window=5):
    """
    Discover emergent cognitive phases by clustering timesteps on their
    full geometric signature.
    
    Instead of defining phases by "who is dominant" (which loses most of the
    information), we cluster on the complete geometric state: conflict angle,
    clarity, curvature, inner/outer orientation, all 8 force magnitudes,
    perception mode, and coherence metrics. Phases that emerge are genuine
    attractor regimes in the combined geometric space.
    
    Uses Gaussian Mixture Model (GMM) with BIC model selection to find
    the natural number of phases without overfitting.
    
    Args:
        history: Simulation history dict
        max_phases: Maximum number of phases to consider
        smoothing_window: Temporal smoothing to suppress single-step noise
        
    Returns:
        dict with 'labels', 'n_phases', 'phase_profiles', 'transitions',
             'feature_names', 'feature_matrix', 'gmm'
    """
    from sklearn.mixture import GaussianMixture
    from scipy.ndimage import uniform_filter1d
    
    n_steps = len(history['time'])
    
    # ── Build feature matrix from geometric + coherence metrics ──
    # Each row = one timestep's full cognitive signature
    mode_map = {'exploration': 0, 'vigilant': 1, 'threat-lock': 2, 'internal': 3}
    
    feature_names = [
        'conflict_angle', 'clarity', 'curvature', 'inner_outer_ratio',
        'integration', 'differentiation', 'path_coherence', 'closure_coherence',
        'perc_range', 'perc_focus', 'perc_mode_num', 'speed',
        'force_mag_0', 'force_mag_1', 'force_mag_2', 'force_mag_3',
        'force_mag_4', 'force_mag_5', 'force_mag_6', 'force_mag_7',
    ]
    
    raw = np.column_stack([
        history['conflict_angle'],
        history['clarity'],
        history['curvature'],
        history['inner_outer_ratio'],
        history['integration'],
        history['differentiation'],
        history['path_coherence'],
        history['closure_coherence'],
        history['perc_range'],
        history['perc_focus'],
        [mode_map.get(m, 0) for m in history['perc_mode']],
        history['speed'],
        history['force_mag_0'], history['force_mag_1'],
        history['force_mag_2'], history['force_mag_3'],
        history['force_mag_4'], history['force_mag_5'],
        history['force_mag_6'], history['force_mag_7'],
    ])
    
    # Temporal smoothing: suppress single-step noise while preserving transitions
    features = np.zeros_like(raw)
    for col in range(raw.shape[1]):
        features[:, col] = uniform_filter1d(raw[:, col], size=smoothing_window)
    
    # Standardize each feature to zero-mean unit-variance
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    stds[stds < 1e-8] = 1.0  # Prevent division by zero for constant features
    features_norm = (features - means) / stds
    
    # ── GMM model selection via BIC ──
    best_bic = np.inf
    best_k = 2
    bics = []
    for k in range(2, max_phases + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               n_init=3, random_state=0, max_iter=200)
        gmm.fit(features_norm)
        bic = gmm.bic(features_norm)
        bics.append((k, bic))
        if bic < best_bic:
            best_bic = bic
            best_k = k
    
    # Fit final model with best k
    gmm = GaussianMixture(n_components=best_k, covariance_type='full',
                           n_init=5, random_state=0, max_iter=300)
    gmm.fit(features_norm)
    labels = gmm.predict(features_norm)
    probs = gmm.predict_proba(features_norm)
    
    # ── Analyze each phase's profile ──
    phase_profiles = []
    subsystem_names = [
        'Memory', 'Attention', 'Motor Control', 'Planning',
        'Emotion', 'Intuition', 'Aesthetic', 'Social'
    ]
    
    for phase_id in range(best_k):
        mask = labels == phase_id
        count = np.sum(mask)
        if count == 0:
            continue
        
        phase_data = raw[mask]  # Use un-normalized data for interpretable stats
        
        # Mean geometric signature
        profile = {
            'id': phase_id,
            'count': int(count),
            'pct': 100.0 * count / n_steps,
            'avg_conflict': np.degrees(np.mean(phase_data[:, 0])),
            'avg_clarity': np.mean(phase_data[:, 1]),
            'avg_curvature': np.mean(phase_data[:, 2]),
            'avg_orientation': np.mean(phase_data[:, 3]),
            'avg_integration': np.mean(phase_data[:, 4]),
            'avg_path_coherence': np.mean(phase_data[:, 6]),
            'avg_speed': np.mean(phase_data[:, 11]),
        }
        
        # Dominant perception mode in this phase
        mode_nums_in_phase = phase_data[:, 10].astype(int)
        mode_names = ['exploration', 'vigilant', 'threat-lock', 'internal']
        mode_hist = np.bincount(mode_nums_in_phase, minlength=4)
        profile['dominant_mode'] = mode_names[np.argmax(mode_hist)]
        
        # Force magnitude signature: which subsystems are closest to home
        force_means = np.mean(phase_data[:, 12:20], axis=0)
        profile['force_signature'] = force_means
        profile['closest_subsystem'] = subsystem_names[np.argmin(force_means)]
        
        # Generate interpretive label
        label_parts = []
        if profile['avg_conflict'] > 115:
            label_parts.append('Conflicted')
        elif profile['avg_conflict'] < 95:
            label_parts.append('Coherent')
            
        if profile['avg_clarity'] > np.mean(raw[:, 1]) + np.std(raw[:, 1]):
            label_parts.append('Driven')
        elif profile['avg_clarity'] < np.mean(raw[:, 1]) - 0.5 * np.std(raw[:, 1]):
            label_parts.append('Diffuse')
            
        if profile['avg_orientation'] > 1.15:
            label_parts.append('Contemplative')
        elif profile['avg_orientation'] < 0.85:
            label_parts.append('Action-oriented')
            
        if profile['avg_curvature'] > np.mean(raw[:, 2]) + np.std(raw[:, 2]):
            label_parts.append('Restless')
        elif profile['avg_curvature'] < np.mean(raw[:, 2]) - 0.5 * np.std(raw[:, 2]):
            label_parts.append('Steady')
            
        if not label_parts:
            label_parts.append('Balanced')
            
        profile['label'] = ' / '.join(label_parts)
        phase_profiles.append(profile)
    
    # Sort phases by frequency (most common first)
    phase_profiles.sort(key=lambda p: p['count'], reverse=True)
    # Remap labels to sorted order
    old_to_new = {p['id']: i for i, p in enumerate(phase_profiles)}
    labels_sorted = np.array([old_to_new.get(l, l) for l in labels])
    for i, p in enumerate(phase_profiles):
        p['id'] = i
    
    # ── Phase transitions ──
    transitions = []
    for i in range(1, n_steps):
        if labels_sorted[i] != labels_sorted[i-1]:
            transitions.append({
                'time': i,
                'from': int(labels_sorted[i-1]),
                'to': int(labels_sorted[i]),
            })
    
    # Transition matrix
    trans_matrix = np.zeros((best_k, best_k))
    for tr in transitions:
        trans_matrix[tr['from'], tr['to']] += 1
    # Normalize rows
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans_matrix / row_sums
    
    return {
        'labels': labels_sorted,
        'n_phases': best_k,
        'phase_profiles': phase_profiles,
        'transitions': transitions,
        'trans_matrix': trans_prob,
        'feature_names': feature_names,
        'feature_matrix': features_norm,
        'bics': bics,
        'gmm': gmm,
        'probs': probs,
    }


def plot_phase_analysis(history, phase_result, sim):
    """
    Visualize the detected cognitive phases: timeline, profiles, transitions,
    and the geometric landscape colored by phase.
    """
    labels = phase_result['labels']
    profiles = phase_result['phase_profiles']
    n_phases = phase_result['n_phases']
    trans = phase_result['transitions']
    trans_matrix = phase_result['trans_matrix']
    probs = phase_result['probs']
    
    n_steps = len(history['time'])
    t = np.array(history['time'])
    
    # Phase colors
    phase_cmap = plt.cm.Set1(np.linspace(0, 1, max(n_phases, 3)))
    
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(20, 22))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    # ─── Panel 1: Phase timeline (full width) ───
    ax = fig.add_subplot(gs[0, :])
    for phase_id in range(n_phases):
        mask = labels == phase_id
        ax.fill_between(t, 0, 1, where=mask,
                        color=phase_cmap[phase_id], alpha=0.7,
                        label=f"Phase {phase_id}: {profiles[phase_id]['label']}")
    
    # Mark transitions
    for tr in trans:
        ax.axvline(tr['time'], color='black', lw=0.5, alpha=0.4)
    
    ax.set_xlim(t[0], t[-1])
    ax.set_yticks([])
    ax.set_xlabel('Timestep')
    ax.set_title(f'Cognitive Phase Timeline — {n_phases} emergent phases, '
                 f'{len(trans)} transitions', fontsize=13)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=min(n_phases, 4), fontsize=9)
    
    # ─── Panel 2: Phase membership probabilities ───
    ax = fig.add_subplot(gs[1, :])
    # Remap probs columns to sorted order
    old_to_new = {}
    for p in profiles:
        old_to_new[p['id']] = p['id']
    
    bottom = np.zeros(n_steps)
    for phase_id in range(n_phases):
        # Find original column for this sorted phase
        ax.fill_between(t, bottom, bottom + probs[:, phase_id],
                        color=phase_cmap[phase_id], alpha=0.6)
        bottom += probs[:, phase_id]
    
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Phase probability')
    ax.set_title('Phase Membership Confidence\n(soft boundaries between cognitive regimes)',
                 fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # ─── Panel 3: Phase profiles — radar/bar chart ───
    ax = fig.add_subplot(gs[2, 0])
    
    profile_metrics = ['avg_conflict', 'avg_clarity', 'avg_curvature',
                       'avg_orientation', 'avg_speed', 'avg_path_coherence']
    metric_labels = ['Conflict°', 'Clarity×100', 'Curvature×10',
                     'Orient.', 'Speed×10', 'Path Coh.']
    
    x_pos = np.arange(len(profile_metrics))
    bar_width = 0.8 / n_phases
    
    for i, p in enumerate(profiles):
        vals = [
            p['avg_conflict'] / 180,          # Normalize to [0,1]
            p['avg_clarity'] * 20,              # Scale up for visibility
            p['avg_curvature'] * 5,
            p['avg_orientation'],
            p['avg_speed'] * 5,
            max(0, p['avg_path_coherence']),    # Clamp negative
        ]
        ax.bar(x_pos + i * bar_width, vals, bar_width,
               color=phase_cmap[i], alpha=0.8,
               label=f"Ph.{p['id']}")
    
    ax.set_xticks(x_pos + bar_width * (n_phases - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=8, rotation=20)
    ax.set_title('Phase Geometric Profiles\n(normalized for comparison)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ─── Panel 4: Force signature heatmap per phase ───
    ax = fig.add_subplot(gs[2, 1])
    
    subsystem_names = [
        'Memory', 'Attention', 'Motor', 'Planning',
        'Emotion', 'Intuition', 'Aesthetic', 'Social'
    ]
    
    force_matrix = np.zeros((n_phases, 8))
    for i, p in enumerate(profiles):
        force_matrix[i] = p['force_signature']
    
    im = ax.imshow(force_matrix, aspect='auto', cmap='YlOrRd')
    ax.set_yticks(range(n_phases))
    ax.set_yticklabels([f"Ph.{p['id']}: {p['label'][:20]}" for p in profiles], fontsize=8)
    ax.set_xticks(range(8))
    ax.set_xticklabels(subsystem_names, fontsize=8, rotation=45, ha='right')
    plt.colorbar(im, ax=ax, label='Force magnitude\n(lower = closer to home)', shrink=0.8)
    ax.set_title('Subsystem Affinity per Phase\n(which subsystems feel at home)', fontsize=11)
    
    # ─── Panel 5: Transition probability matrix ───
    ax = fig.add_subplot(gs[2, 2])
    
    im = ax.imshow(trans_matrix, cmap='Blues', vmin=0, vmax=1)
    for i in range(n_phases):
        for j in range(n_phases):
            val = trans_matrix[i, j]
            if val > 0.01:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if val > 0.5 else 'black')
    
    ax.set_xticks(range(n_phases))
    ax.set_yticks(range(n_phases))
    ax.set_xticklabels([f'Ph.{i}' for i in range(n_phases)], fontsize=9)
    ax.set_yticklabels([f'Ph.{i}' for i in range(n_phases)], fontsize=9)
    ax.set_xlabel('To phase')
    ax.set_ylabel('From phase')
    plt.colorbar(im, ax=ax, label='Transition probability', shrink=0.8)
    ax.set_title('Phase Transition Matrix\n(row = from, col = to)', fontsize=11)
    
    # ─── Panel 6: BIC model selection curve ───
    ax = fig.add_subplot(gs[3, 0])
    bics = phase_result['bics']
    ks = [b[0] for b in bics]
    bic_vals = [b[1] for b in bics]
    ax.plot(ks, bic_vals, 'o-', color='navy', lw=2, markersize=8)
    ax.axvline(n_phases, color='red', ls='--', lw=1.5, label=f'Selected: k={n_phases}')
    ax.set_xlabel('Number of phases')
    ax.set_ylabel('BIC (lower = better fit)')
    ax.set_title('Model Selection\n(Bayesian Information Criterion)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ─── Panel 7: Phases on the cognitive landscape (PCA projection) ───
    ax = fig.add_subplot(gs[3, 1:3])
    
    u_history = np.array(history['u_t'])
    u_centered = u_history - u_history.mean(axis=0)
    cov = u_centered.T @ u_centered / n_steps
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, -1]
    pc2 = eigvecs[:, -2]
    traj_x = u_history @ pc1
    traj_y = u_history @ pc2
    
    for phase_id in range(n_phases):
        mask = labels == phase_id
        ax.scatter(traj_x[mask], traj_y[mask],
                   c=[phase_cmap[phase_id]], s=12, alpha=0.5,
                   label=f"Ph.{phase_id}: {profiles[phase_id]['label']}")
    
    ax.plot(traj_x, traj_y, 'k-', alpha=0.08, lw=0.3)
    ax.scatter(traj_x[0], traj_y[0], marker='o', c='lime', s=100,
               edgecolors='black', zorder=10, label='Start')
    ax.scatter(traj_x[-1], traj_y[-1], marker='s', c='red', s=100,
               edgecolors='black', zorder=10, label='End')
    
    ax.set_xlabel('PC1 (primary axis of motion)')
    ax.set_ylabel('PC2 (secondary axis)')
    ax.set_title('Cognitive Phases on S³ Landscape\n(PCA projection of 4D trajectory)',
                 fontsize=12)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # ─── Panel 8: Phase duration histogram ───
    ax = fig.add_subplot(gs[4, 0])
    
    # Compute episode durations
    durations = {i: [] for i in range(n_phases)}
    current_phase = labels[0]
    current_len = 1
    for i in range(1, n_steps):
        if labels[i] == current_phase:
            current_len += 1
        else:
            durations[current_phase].append(current_len)
            current_phase = labels[i]
            current_len = 1
    durations[current_phase].append(current_len)
    
    for phase_id in range(n_phases):
        if durations[phase_id]:
            ax.hist(durations[phase_id], bins=range(1, max(max(d) for d in durations.values() if d) + 5, 2),
                    color=phase_cmap[phase_id], alpha=0.6, label=f'Ph.{phase_id}')
    
    ax.set_xlabel('Episode duration (timesteps)')
    ax.set_ylabel('Count')
    ax.set_title('How Long Each Phase Lasts\n(episode duration distribution)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ─── Panel 9: Key metric timelines colored by phase ───
    ax = fig.add_subplot(gs[4, 1:3])
    
    conflict_deg = np.degrees(np.array(history['conflict_angle']))
    clarity = np.array(history['clarity'])
    
    # Conflict angle colored by phase
    for phase_id in range(n_phases):
        mask = labels == phase_id
        ax.scatter(t[mask], conflict_deg[mask], c=[phase_cmap[phase_id]],
                   s=6, alpha=0.5)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Conflict angle (°)')
    ax.set_title('Conflict Angle Colored by Phase\n(reveals what defines each regime)',
                 fontsize=12)
    
    # Add clarity on secondary axis
    ax2 = ax.twinx()
    ax2.plot(t, clarity * 100, color='teal', alpha=0.3, lw=0.8)
    ax2.set_ylabel('Clarity × 100', color='teal')
    ax2.tick_params(axis='y', labelcolor='teal')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Cognitive Phase Analysis — {n_phases} Emergent Phases',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # ── Print phase summary to console ──
    print(f"\n  Phase Detection Results:")
    print(f"  {'='*55}")
    print(f"  {n_phases} cognitive phases discovered (BIC model selection)")
    print(f"  {len(trans)} phase transitions in {n_steps} timesteps")
    print()
    for p in profiles:
        print(f"  Phase {p['id']}: {p['label']}")
        print(f"    {p['count']} steps ({p['pct']:.1f}%)")
        print(f"    Conflict: {p['avg_conflict']:.0f}°  Clarity: {p['avg_clarity']:.4f}  "
              f"Speed: {p['avg_speed']:.3f}")
        print(f"    Orientation: {p['avg_orientation']:.2f}  "
              f"Mode: {p['dominant_mode']}  "
              f"Nearest: {p['closest_subsystem']}")
    
    # Print transition info
    if len(trans) > 0:
        print(f"\n  Phase transition sequence:")
        seq = [str(labels[0])]
        for tr in trans:
            seq.append(f"→{tr['to']}(t={tr['time']})")
        # Show first 20 transitions
        print(f"    {''.join(seq[:21])}" + ("..." if len(seq) > 21 else ""))
    
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
    
    # Generate cognitive landscape map
    print("\n  Mapping cognitive landscape...")
    fig2 = map_cognitive_landscape(sim, history)
    plt.savefig('cognitive_landscape.png', dpi=150, bbox_inches='tight')
    print("  Saved cognitive landscape to cognitive_landscape.png")
    
    # Phase detection
    print("\n  Detecting cognitive phases...")
    phase_result = detect_cognitive_phases(history)
    fig3 = plot_phase_analysis(history, phase_result, sim)
    plt.savefig('cognitive_phases.png', dpi=150, bbox_inches='tight')
    print("  Saved phase analysis to cognitive_phases.png")
    
    return history


if __name__ == '__main__':
    main()