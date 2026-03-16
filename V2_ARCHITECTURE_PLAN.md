# Consciousness Simulation v2 Architecture Plan

## Guiding Principles for v2

1. **True geometric duality** - not symbolic mapping
2. **Closed manifold dynamics** - no edge artifacts
3. **Soft assignment** - ambiguous, mixed states possible
4. **Two-stage updates** - micro transitions + macro reconciliation
5. **Balanced subsystems** - before adding learning

---

## 1. Manifold Geometry (Closed 4D)

### Current Issue
- Free-drifting 4D Euclidean space allows unbounded growth
- Magnitude can explode or collapse arbitrarily

### v2 Solution: Normalized Hyperspherical Manifold

```python
class ClosedManifold:
    """Unit 3-sphere in 4D with variable magnitude shell"""
    
    N_MICRO = 600      # Micro-transition reference points
    N_MACRO = 120      # Macro-basin centers
    
    def __init__(self):
        # Pre-compute reference geometry on unit hypersphere
        self.micro_points = self._generate_600_micro()      # Shape: (600, 4)
        self.macro_centers = self._generate_120_macro()     # Shape: (120, 4)
        
    def _generate_600_micro(self):
        """Generate ~600 points with good spherical coverage"""
        # Option A: Fibonacci lattice on S³
        # Option B: Random uniform on hypersphere
        # Option C: Optimized spherical code (maximal separation)
        return np.random.normal(size=(self.N_MICRO, 4))
    
    def _generate_120_macro(self):
        """Generate ~120 basin centers with good coverage"""
        # Use k-means clustering on micro points to derive basins
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.N_MACRO, random_state=42)
        return km.fit_predict(self.micro_points)  # Shape: (120, 4)
    
    def normalize_to_sphere(self, x):
        """Project to unit hypersphere"""
        norm = np.linalg.norm(x) + 1e-8
        return x / norm
    
    def similarity(self, x_t, reference_points):
        """Angular similarity (cosine distance on sphere)"""
        # x_t: (4,), reference_points: (N, 4)
        # Returns similarities: (N,)
        normalized_x = self.normalize_to_sphere(x_t)
        return np.dot(reference_points, normalized_x)
```

---

## 2. Soft Assignment Mechanism

### Current Issue
- Hard assignment via `node // BASINS_PER_NODE` is purely symbolic
- No geometric relationship between local and global states

### v2 Solution: Geometric Distance-Based Membership

```python
class DualGeometryEngine:
    def __init__(self, manifold):
        self.manifold = manifold
        self.beta = 10.0      # Softness parameter for macro assignment
        
    def compute_micro_membership(self, x_t):
        """Find most similar micro reference point"""
        similarities = self.manifold.similarity(x_t, self.manifold.micro_points)
        micro_id = np.argmax(similarities)
        return micro_id, similarities[micro_id]  # Return ID and confidence
    
    def compute_macro_membership(self, x_t):
        """Soft assignment to macro basins"""
        similarities = self.manifold.similarity(x_t, self.manifold.macro_centers)
        # Softmax for smooth distribution
        weights = softmax(self.beta * similarities)
        macro_id = np.argmax(weights)
        return macro_id, weights  # Return dominant ID and full distribution
    
    def compute_closure_coherence(self, x_t, macro_weights, subsystem_conflict):
        """Dynamic coherence metric"""
        # Component 1: fit to current macro basin (angular similarity)
        fit_score = np.dot(
            self.manifold.normalize_to_sphere(x_t),
            self.manifold.macro_centers[self.dominant_macro_id]
        )
        
        # Component 2: stability of recent path
        transition_surprise = compute_transition_surprise(recent_states)
        
        # Component 3: agreement across subsystems (inverse of conflict)
        subsystem_agreement = 1.0 - subsystem_conflict
        
        # Weighted combination
        alpha, beta, gamma = 0.5, 0.3, 0.2
        coherence = (alpha * fit_score + 
                    beta * (1.0 - transition_surprise) + 
                    gamma * subsystem_agreement)
        
        return np.clip(coherence, -1.0, 1.0)
```

---

## 3. Two-Stage State Update

### Current Issue
- Single update step conflates local dynamics with global organization
- Micro transitions can violate macro constraints

### v2 Solution: Separate Local and Global Updates

```python
class DualGeometryEngine:
    def step(self, subsystem_states, dt=1.0):
        """Two-stage update per timestep"""
        
        # --- STAGE 1: Micro transition (fast local dynamics) ---
        x_local = self._micro_transition_step(subsystem_states, dt)
        
        # --- STAGE 2: Macro reconciliation (global constraint) ---
        macro_id, macro_weights = self.compute_macro_membership(x_local)
        basin_center = self.manifold.macro_centers[macro_id]
        
        # Pull toward compatible basin center with relaxation factor
        alpha_pull = 0.1  # How strongly to follow global structure
        x_global = (1 - alpha_pull) * x_local + alpha_pull * basin_center
        
        # Normalize back to manifold
        x_t = self.manifold.normalize_to_sphere(x_global)
        
        return x_t, macro_id, macro_weights
```

---

## 4. Toroidal Visible World

### Current Issue
- Hard boundaries at x=20, y=20 cause artificial trapping
- Edge behavior distorts navigation dynamics
- Contradicts closure philosophy (incomplete manifold)

### v2 Solution: Wrap-around torus topology

```python
class ToroidalEnvironment:
    def __init__(self, size=20):
        self.size = size
        
    def apply_boundary(self, x, y):
        """Wrap coordinates on torus"""
        return (x % self.size, y % self.size)
    
    def compute_distance_toroidal(self, pos1, pos2):
        """Shortest path distance on torus"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        # Take minimum of direct and wrapped paths
        dx = min(dx, self.size - dx)
        dy = min(dy, self.size - dy)
        return np.sqrt(dx**2 + dy**2)
    
    def step_position(self, x, y, velocity_x, velocity_y):
        """Update position with toroidal wrapping"""
        new_x = (x + velocity_x) % self.size
        new_y = (y + velocity_y) % self.size
        return new_x, new_y
```

---

## 5. Balanced Subsystem Dynamics

### Current Issue
- Planning dominates 98% of timesteps
- System lacks rich exploration due to single-channel dominance

### v2 Solution: Competitive inhibition with fatigue

```python
class BalancedSubsystems:
    def __init__(self, n_subsystems=8):
        self.n_subsystems = n_subsystems
        
        # Individual gains for each subsystem
        self.gains = np.ones(n_subsystems) * 0.125  # Equal initial influence
        
        # Fatigue/adaptation rates
        self.fatigue_rates = np.ones(n_subsystems) * 0.001
        
        # Current activity levels (for competition)
        self.activities = np.zeros(n_subsystems)
    
    def apply_competition(self, raw_influences):
        """Inhibitory competition between subsystems"""
        # Normalize to sum=1 before applying gains
        total_activity = np.sum(raw_influences * self.gains) + 1e-8
        normalized = (raw_influences * self.gains) / total_activity
        
        return normalized
    
    def apply_fatigue(self, dt):
        """Reduce activity of dominant subsystems"""
        # Slower for less active, faster for more active
        self.activities *= np.exp(-self.fatigue_rates * dt)
```

---

## 6. Revised Consciousness Metrics

### Current Issue
- Closure coherence only measures static fit to basin center
- Doesn't capture dynamic quality of state evolution

### v2 Solution: Multi-component Dynamic Coherence

```python
def compute_consciousness_metrics(state_history, macro_weights, subsystem_conflict):
    """Enhanced metrics for consciousness-like behavior"""
    
    # 1. Closure coherence (dynamic)
    closure = compute_closure_coherence(
        x_t=state_history[-1],
        macro_weights=macro_weights,
        recent_states=state_history[-5:],
        subsystem_conflict=subsystem_conflict
    )
    
    # 2. Integration level (IIT-style)
    integration = compute_integration(macro_weights)
    # Higher when distribution is broad but not uniform
    
    # 3. Differentiation index
    differentiation = entropy(macro_weights) / log(N_MACRO)
    # Measures how many basins are actively considered
    
    # 4. Path coherence (temporal stability)
    path_coherence = compute_temporal_consistency(state_history[-10:])
    
    # 5. Transition surprise rate
    surprise_rate = compute_surprise_rate(state_history)
    
    return {
        'closure': closure,
        'integration': integration,
        'differentiation': differentiation,
        'path_coherence': path_coherence,
        'surprise_rate': surprise_rate
    }

def compute_integration(macro_weights):
    """Integration = high when few basins dominate but not one"""
    # Use Gini coefficient or similar measure
    sorted_w = np.sort(macro_weights)[::-1]
    cumulative = np.cumsum(sorted_w)
    gini = (2 * np.sum((np.arange(1, len(cumulative)+1) * sorted_w))) / (len(cumulative)*np.sum(sorted_w)) - 1/(len(cumulative)-1)
    return 1.0 - gini

def compute_temporal_consistency(states):
    """How consistent is the recent trajectory?"""
    # Measure alignment of successive state differences
    diffs = np.diff(states, axis=0)
    alignments = np.sum([np.dot(diffs[i], diffs[i+1]) 
                        for i in range(len(diffs)-1)])
    return np.clip(alignments / (len(diffs)+1e-8), -1.0, 1.0)
```

---

## 7. Visualization Design (Linked Views)

### Required Views
```python
def create_linked_visualizations(time_series_data):
    """Three linked views with shared timeline"""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 8))
    
    # View 1: 2D projected path (visible world)
    ax_path = axes[0, 0]
    plot_toroidal_path(time_series_data['position_x'], 
                       time_series_data['position_y'])
    ax_path.set_title('Visible World Trajectory (Toroidal)')
    
    # View 2: Micro transition activity
    ax_micro = axes[0, 1]
    plot_activity_heatmap(time_series_data['micro_ids'])
    ax_micro.set_title('Micro Transition Activity (600-cell layer)')
    
    # View 3: Macro basin occupancy
    ax_macro = axes[1, 0]
    plot_basin_occupancy(time_series_data['macro_weights'])
    ax_macro.set_title('Macro Basin Occupancy (120-cell layer)')
    
    # View 4: Coherence timeline
    ax_coherence = axes[1, 1]
    plot_timeline(time_series_data['closure'], 
                  time_series_data['integration'])
    ax_coherence.set_title('Coherence Metrics Over Time')
    
    # View 5: Subsystem dominance
    ax_subsystems = axes[2, 0]
    plot_subsystem_dominance(time_series_data['dominant_subsystem'])
    ax_subsystems.set_title('Subsystem Dominance Timeline')
    
    # View 6: Basin-switch events
    ax_switches = axes[2, 1]
    plot_basin_switches(time_series_data['basin_changes'])
    ax_switches.set_title('Macro-state Transition Events')
```

---

## Implementation Order (v2)

### Phase 1: Manifold Foundation (1-2 hours)
1. Implement `ClosedManifold` class with pre-computed micro/macro points
2. Add normalization and similarity functions
3. Test spherical coverage of reference points

### Phase 2: Soft Assignment (1 hour)
4. Implement `compute_micro_membership()` and `compute_macro_membership()`
5. Add softmax weighting for macro assignment
6. Verify soft transitions between basins

### Phase 3: Two-Stage Updates (1-2 hours)
7. Implement `_micro_transition_step()` with subsystem influences
8. Implement `_macro_reconciliation_step()` with basin pull
9. Test that micro can explore while macro constrains

### Phase 4: Toroidal World (30 min)
10. Replace boundary clamping with toroidal wrapping
11. Update collision detection for wrapped distances
12. Verify no edge trapping occurs

### Phase 5: Subsystem Balance (1 hour)
13. Add competitive inhibition between subsystems
14. Implement fatigue/adaptation mechanism
15. Tune gains to achieve ~20-30% dominant subsystem (not 98%)

### Phase 6: Enhanced Metrics (30 min)
16. Implement dynamic closure coherence with all components
17. Add integration and differentiation measures
18. Verify metrics respond appropriately to state changes

### Phase 7: Visualization Updates (1 hour)
19. Create linked view layout
20. Test non-blocking display with `plt.pause()`
21. Verify all key behaviors visible across views

---

## Expected Behavioral Changes

| Before (v1) | After (v2) |
|-------------|------------|
| Planning dominates 98% time | Multiple subsystems compete (~30-40% each top 3) |
| Hard basin assignment | Soft, ambiguous basin membership |
| Edge trapping at boundaries | Smooth toroidal navigation |
| Static coherence metric | Dynamic coherence with surprise component |
| Symbolic micro→macro mapping | Geometric distance-based assignment |

---

## Success Criteria for v2

1. **No edge artifacts** - Being navigates freely on torus
2. **Balanced competition** - At least 3 subsystems each dominate >5% of time
3. **Soft transitions** - Basin switching shows gradual probability shifts
4. **Meaningful coherence** - Closure metric varies meaningfully (0.3-0.9 range)
5. **Rich exploration** - State space coverage improves significantly

---

## Next Steps After v2 Complete

Once the geometric foundation is solid:

1. Add Hebbian learning to subsystem weights
2. Implement reinforcement signal from goal/hazard interactions  
3. Study how coherence emerges under different parameter regimes
4. Run lesion studies on macro vs micro layer removal
5. Compare with baseline "consciousness" metrics from other models