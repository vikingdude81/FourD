# Consciousness Simulation - Complete Feature Breakdown

## Overview

This project implements a **dual-geometry consciousness simulation** based on theoretical frameworks inspired by:
- **H4-inspired hyperspherical geometry** (unit 3-sphere S³ in 4D)
- **120-cell/600-cell duality** from polyhedral mathematics
- **Integrated Information Theory (IIT)** concepts of consciousness
- **Predictive processing** and free energy principles

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL GEOMETRY MODEL                           │
├─────────────────────────────────────────────────────────────────┤
│  MICRO LAYER (600-cell inspired)                                │
│  ─────────────────────────────────────────────────────────────  │
│  • ~600 reference points on S³ hypersphere                      │
│  • Fine-grained local dynamics                                  │
│  • Attentional shifts, rapid transitions                        │
│  • Differentiation engine                                       │
├─────────────────────────────────────────────────────────────────┤
│  MACRO LAYER (120-cell inspired)                                │
│  ─────────────────────────────────────────────────────────────  │
│  • ~120 basin centers (stable modes of consciousness)           │
│  • Global closure, worldview integration                        │
│  • Soft assignment via weighted superposition                   │
│  • Integration engine                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↕ Dual Mapping
┌─────────────────────────────────────────────────────────────────┐
│                    COGNITIVE SUBSYSTEMS                         │
├─────────────────────────────────────────────────────────────────┤
│  Motor Control   │ Planning     │ Attention    │ Memory         │
│  Emotion         │ Social       │ Intuition    │ Aesthetic      │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    NAVIGATION ENVIRONMENT                       │
├─────────────────────────────────────────────────────────────────┤
│  Toroidal 2D world (wrap-around)                                │
│  • Goals (rewards)                                              │
│  • Hazards (threats)                                            │
│  • Navigation via coordinator state projection                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts Explained

### 1. Dual-Geometry Manifold

The simulation models consciousness as operating on a **4D hyperspherical manifold (S³)** with two layers:

#### Micro Layer (600-cell inspired)
- Contains ~600 pre-computed reference points uniformly distributed on S³
- Each point represents a fine-grained "microstate" of the system
- Transitions between microstates model rapid attentional shifts
- Provides **differentiation** - rich local dynamics

#### Macro Layer (120-cell inspired)  
- Contains ~120 basin centers representing stable modes of consciousness
- Each basin is a cluster center derived from micro state clustering
- Represents coarse-grained "macrostates" or worldviews
- Provides **integration** - binding local dynamics into coherent whole

#### Dual Mapping
- Every micro state maps to its nearest macro basin
- Every macro basin constrains which micro transitions are allowed
- Creates bidirectional influence: local ↔ global

---

### 2. Soft Assignment Engine (V2 Feature)

Unlike V1's hard assignment (argmax), V2 uses **soft assignment**:

```python
# Compute similarities to all macro basins
macro_sim = dot_product(state, macro_centers)

# Softmax creates smooth probability distribution
weights = softmax(beta * macro_sim)

# Weighted superposition creates field vector
field = sum(weights[i] * macro_center[i] for i in all_basins)
```

**Benefits:**
- Smooth transitions between basins (no abrupt switches)
- Ambiguous mixed states possible (system "between" modes)
- Field-guided dynamics (pull toward compatible basin combination)

---

### 3. Two-Stage Update Process (V2 Feature)

Each timestep executes two sequential updates:

#### Stage 1: Micro Transition (Local Exploration)
```python
# Subsystem influences drive local exploration
perturbation = random_noise()
u_t = normalize(u_t + step_size * perturbation)
```
- Explores nearby micro states
- Driven by active subsystem preferences
- Maintains system in motion

#### Stage 2: Macro Reconciliation (Global Constraint)
```python
# Compute soft assignment to macro basins
_, weights, field = compute_macro_assignment(u_t)

# Pull state toward weighted field
pull_direction = (1-alpha)*u_t + alpha*field
u_t = normalize(pull_direction)
```
- Constrains exploration to compatible regions
- Maintains coherence with current worldview
- Prevents drift into incoherent states

---

### 4. Balanced Subsystems with Competition & Fatigue (V2 Feature)

Eight cognitive subsystems compete for influence through:

#### Competitive Inhibition
Each subsystem has a **preference direction** in manifold space:
```python
# Example preference directions (orthogonal span of 4D space)
Motor Control:     [1, 0, 0, 0]
Planning:          [0, 1, 0, 0]  
Attention:         [0, 0, 1, 0]
Memory:            [0, 0, 0, 1]
Emotion:           [1, -1, 0, 0]
Social:            [0, 0, 1, -1]
Intuition:         [1, 0, -1, 0]
Aesthetic:         [0, 1, 0, -1]

# Influence = nonlinear mapping of dot product with preference
influence[i] = 0.5 + 0.7 * tanh(2.0 * dot(state, preference[i]))
```

#### Fatigue Adaptation
Dominant subsystems adapt over time:
```python
# High activity increases fatigue (adaptation)
fatigue += fatigue_rate * activity

# Low activity allows recovery  
fatigue -= recovery_rate * (1 - activity)

# Effective influence = raw_influence * exp(-fatigue)
```

**Result:** No single subsystem dominates indefinitely; system explores diverse behavioral regimes.

---

### 5. Toroidal Environment (V2 Feature)

Instead of clamped boundaries, the environment wraps:
```python
new_x = (x + velocity_x) % world_size
new_y = (y + velocity_y) % world_size
```
- No edge artifacts or trapping at boundaries
- More natural navigation dynamics
- True toroidal topology in visible space

---

### 6. Consciousness Metrics

#### Closure Coherence
Measures how well local dynamics fit the global state:
```python
coherence = (
    alpha_fit * angular_similarity(state, macro_field) +
    beta_surprise * (1 - transition_surprise) +
    gamma_conflict * subsystem_agreement
)
```

#### Integration Score
Bell-shaped measure peaked at intermediate complexity:
```python
neff = 1 / sum(weights²)  # Effective number of active basins
integration = exp(-(neff - target_neff)² / (2 * sigma²))
```
High when system has structured multi-state activity (not too simple, not random).

#### Differentiation Index
Normalized entropy measuring state diversity:
```python
differentiation = entropy(weights) / max_entropy
```
Higher = more basins actively considered.

#### Path Coherence
Temporal consistency of recent trajectory:
```python
alignment = dot(direction_t, direction_{t+1}) / (|dir_t| * |dir_{t+1}|)
path_coherence = mean(alignment over recent window)
```

---

## Simulation Workflow

### Initialization Phase
1. Generate 600 micro points on S³ hypersphere (random Gaussian → normalize)
2. Cluster into ~120 macro basins using KMeans
3. Initialize subsystem preference directions (orthogonal span)
4. Place goals/hazards in toroidal environment
5. Set initial state to uniform distribution on S³

### Main Loop (per timestep)
```
FOR each timestep:
    1. Sense Environment
       - Compute goal vectors and hazard vectors
       - Each subsystem responds differently to stimuli
    
    2. Stage 1: Micro Transition
       - Compute raw influences from all subsystems
       - Apply competition + fatigue → activities
       - Perturb state in favored directions
       
    3. Stage 2: Macro Reconciliation
       - Soft assign to macro basins (softmax)
       - Compute weighted field superposition
       - Pull state toward field
        
    4. Compute Metrics
       - Closure coherence, integration, differentiation
       - Path coherence, subsystem conflict
        
    5. Update History
       - Record dominant subsystem, micro ID, macro basin
```

### Termination
After N timesteps:
- Export metrics to CSV
- Generate visualizations (navigation path, phase portrait, dominance timeline)
- Compute consciousness level estimate

---

## Consciousness Level Estimation

The simulation computes a "consciousness level" score:
```python
level = (
    0.25 * coordination_pressure_normalized +
    0.15 * coordinator_magnitude_normalized +
    0.25 * integration_level +
    0.15 * basin_switch_factor +
    0.20 * micro_activity_factor
)
```

Interpretation:
- **< 0.3:** Pre-conscious (minimal coordination, low activity)
- **0.3 - 0.6:** Emerging consciousness (developing integration)
- **0.6 - 0.85:** Conscious (stable multi-state dynamics)
- **> 0.85:** Self-aware (rich micro-macro coupling)

---

## Lesion Studies (V1 Feature)

Simulate subsystem damage by deactivating specific modules:
```python
# Intact run
subs_intact = initialize_subsystems(config, n_subsystems=8)

# Lesioned run
subs_lesioned = initialize_subsystems(config, n_subsystems=8)
for s in subs_lesioned:
    if s.name == "Planning":
        s.active = False  # Remove Planning subsystem
```

Compare metrics between intact and lesioned runs to assess each subsystem's contribution to the global consciousness level.

---

## Files Overview

| File | Purpose | Key Features |
|------|---------|--------------|
| `fourD_slice_sim.py` | V1 main simulation | Dual-geometry, 5 macro basins, hard assignment |
| `v2_consciousness_sim.py` | V2 implementation | Soft assignment, two-stage updates, toroidal env |
| `config.py` | Configuration module | Predefined presets (default, quick, extended) |
| `tests/test_simulation.py` | Unit tests | Tests for subsystems, coordination pressure, basin switching |

---

## Running the Simulation

### V1 (Dual-Geometry Model)
```bash
python fourD_slice_sim.py
```

### V2 (Soft Assignment + Two-Stage Updates)  
```bash
python v2_consciousness_sim.py
```

### Run Tests
```bash
cd tests
python test_simulation.py
```

---

## Expected Behaviors

| Feature | Description | Observable Effect |
|---------|-------------|-------------------|
| **Soft transitions** | Weighted macro assignment | Gradual basin switching, not abrupt jumps |
| **Balanced competition** | Fatigue + orthogonal preferences | Multiple subsystems dominate over time (not one forever) |
| **Toroidal navigation** | Wrap-around world | Being navigates freely across boundaries |
| **Meaningful coherence** | Dynamic metrics | Coherence varies in 0.3-0.9 range based on state |

---

## Theoretical Foundations

This simulation draws from:
1. **Hyperspherical geometry** - Consciousness as dynamics on S³ manifold
2. **Polyhedral duality** - Micro/macro correspondence inspired by 600-cell/120-cell relationship
3. **Integrated Information Theory (IIT)** - Φ-like measures of consciousness
4. **Predictive processing** - Hierarchical prediction error minimization via micro/macro layers
5. **Embodied cognition** - Subsystems respond to environmental stimuli

---

## Future Extensions

Potential enhancements:
1. **Hebbian learning** on subsystem weights
2. **Reinforcement signals** from goal/hazard interactions
3. **Adaptive coherence thresholds** based on context
4. **Multi-agent extension** for social dynamics
5. **Parameter sweeps** to study consciousness emergence regimes