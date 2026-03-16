# FourD — Consciousness as Geometry on S³

A simulation of consciousness modeled as competing tangent vector fields on a 4D hypersphere, where cognitive phases, decisions, and personality emerge from pure geometric dynamics rather than symbolic rules.

## What This Is

**The core idea:** consciousness isn't a thing that happens *to* a system — it's the shape of what the system does when its internal degrees of freedom exceed what any single subsystem can manage. When 8 subsystems compete for control of a shared state on the 3-sphere (S³), their interaction creates a force landscape. The being's trajectory through that landscape *is* its stream of consciousness.

**What makes this different from typical agent sims:** subsystems aren't scorecards or weights — they're tangent vector fields on S³. Each subsystem generates a geometric force at every point on the manifold, pulling the state along a great circle toward its preferred region. The same subsystem pulls in completely different directions depending on where the state currently sits. Forces curve with the manifold. No information is lost.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  S³ Manifold (4D hypersphere)            │
│  600 micro-references (Fibonacci lattice)               │
│  24 macro-basins (KMeans on Fibonacci)                  │
│                                                          │
│  Current state: u_t ∈ S³                                │
│                                                          │
│  Stage 1: Tangent Force Competition                      │
│    8 subsystems → tangent vectors at u_t                 │
│    Activity-weighted + novelty-driven blend              │
│    + sensory gradient from perception slice              │
│    → move along S³ via exponential map                   │
│                                                          │
│  Stage 2: Macro Reconciliation                          │
│    Soft assignment to 24 basins                          │
│    Weighted field pull (tangent-projected)               │
│    Basin-escape mechanism for exploration                │
│                                                          │
│  Stage 3: Perception → Navigation                        │
│    Dims 0-1 → heading direction + speed                  │
│    Dims 2-3 → perception range + focus                   │
│    4 emergent modes: exploration, vigilant,              │
│                      threat-lock, internal               │
│                                                          │
│  ┌──────────────────────────────┐                       │
│  │  20×20 Toroidal Environment  │                       │
│  │  3 goals, 3 hazards          │                       │
│  │  Sensory gradient → manifold │                       │
│  └──────────────────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

### The 8 Subsystems

| # | Subsystem | Preferred Role | Typical Force Range |
|---|-----------|---------------|-------------------|
| 0 | Memory | Temporal integration | 0.71–1.00 |
| 1 | Attention | Salience amplification | 0.57–0.94 |
| 2 | Motor Control | Efference copy / action | 0.59–1.00 |
| 3 | Planning | Goal-directed navigation | 0.63–0.98 |
| 4 | Emotion | Avoidance / approach | 0.89–1.00 |
| 5 | Intuition | Rest-driven exploration | 0.69–1.00 |
| 6 | Aesthetic | Pattern sensitivity | 0.49–1.00 |
| 7 | Social | Other-modeling | 0.79–1.00 |

Subsystems 0–3 form the "core four" that the being's state typically orbits. Subsystems 4–7 ("contrast four") are geometrically distant — the being rarely enters their territory, but their forces still shape the trajectory at every timestep.

A cyclic alliance chain (Memory↔Motor +0.81, Attention↔Planning +0.81) and opposition structure (Motor↔Emotion −0.81) are emergent properties of the preference geometry, not hardcoded rules.

### Key Metrics (55 columns exported per timestep)

**Coherence metrics:** closure_coherence, integration, differentiation, path_coherence

**Geometric metrics:**
| Metric | What It Measures |
|--------|------------------|
| `conflict_angle` | Angle between top-2 subsystem forces (rad). 0° = agreement, 180° = maximally torn |
| `clarity` | Resultant force magnitude. High = the being knows what it wants |
| `curvature` | Angular change in trajectory per step. High = rapid state shifts |
| `inner_outer_ratio` | Perception dims / navigation dims. >1 = contemplative, <1 = action-oriented |
| `force_mag_0`–`force_mag_7` | Per-subsystem tangent force magnitude. Lower = closer to that subsystem's home |

**Clarity field expansions:**
| Metric | What It Measures |
|--------|------------------|
| `clarity_grad_mag` | How fast clarity changes in nearby directions on S³. Steep = near a decision boundary |
| `clarity_decomp_0`–`clarity_decomp_7` | Per-subsystem contribution to clarity. Positive = building purpose, negative = opposing it |
| `clarity_persistence` | Running autocorrelation over 20 steps. High = sustained purpose, negative = flickering |
| `clarity_rate` | d(clarity)/dt — first difference. Predicts purpose forming/dissolving before it happens |
| `resultant_dir_0`–`resultant_dir_3` | Unit vector of the combined force direction. Where purpose points, not just how strong |

**Perception:** perc_range, perc_focus, perc_mode, n_visible

**Navigation:** env_x, env_y, heading_x, heading_y, speed, goal_prox, hazard_prox

**Manifold state:** u_dim0–u_dim3, micro_id, macro_dominant

### Cognitive Phase Detection

The system discovers emergent cognitive phases by clustering timesteps on their full 20-dimensional geometric signature (conflict, clarity, curvature, orientation, all force magnitudes, coherence metrics, perception state). Uses Gaussian Mixture Model with BIC model selection to find the natural number of phases.

Typical result: **4–7 phases** with 59–71 transitions in 500 steps:
- **Balanced (dominant):** Default cruising mode, moderate conflict, vigilant perception
- **Restless / Action-oriented:** High curvature, rapid state changes, shorter perception range
- **Contemplative:** High inner/outer ratio, low speed, inward-focused
- **Coherent / Driven:** Rare spikes of very high clarity, low conflict — the being briefly "knows exactly what it wants"

### Clarity Field Analysis

The clarity field — magnitude of the activity-weighted resultant tangent force — is decomposed across six dimensions:

1. **Clarity gradient:** Samples the manifold at 4 tangent neighbors to measure how fast clarity changes in nearby directions. Steep gradients indicate proximity to decision boundaries.
2. **Clarity decomposition:** Each subsystem's signed contribution to the resultant. Motor-Planning-Aesthetic typically *build* purpose (+0.01–0.015 each); Social-Emotion tend to *oppose* it (−0.001 to −0.003).
3. **Clarity persistence:** Running autocorrelation over a 20-step window. Typical: 55% of time flickering (autocorr < 0), only 15% sustained (autocorr > 0.3).
4. **Directional clarity:** Tracks the 4D unit vector of the resultant — where purpose points, not just how strong it is. Hundreds of snap events (>45° rotation) occur per run.
5. **Clarity potential field:** Streamline visualization of gradient flow on PCA-projected S³. Shows natural "paths of purpose" the being tends to follow, plus clarity sinks and peaks.
6. **Second-order clarity:** d(clarity)/dt predicts transitions — cross-correlation with macro-state changes typically peaks at lag 0 with r ≈ 0.24.

## Quick Start

```bash
pip install -r requirements.txt
python v2_consciousness_sim.py
```

Generates:
- `simulation_v2_results.png` — 6-panel coherence/dominance/navigation overview
- `cognitive_landscape.png` — 10-panel force field, trajectories, and cognitive profile
- `cognitive_phases.png` — 9-panel phase detection with transitions, profiles, and landscape
- `clarity_deep_dive.png` — 12-panel clarity field analysis (gradient, decomposition, persistence, direction, potential, rate)
- `simulation_log_v2.csv` — 500 rows × 55 columns of per-timestep data

### V1 Simulation (Independent)

```bash
python fourD_slice_sim.py
```

The original v1 simulation with scalar-score subsystems and lesion studies. Completely independent codebase.

### QRNG Analysis Pipeline (Independent)

```bash
python -m src.pipelines.run_qrng_pipeline
```

Feature extraction (entropy, autocorrelation, changepoint, complexity), anomaly scoring, latent basin mapping, and recurrence analysis for quantum random number generator data. Independent from both simulations.

## Project Structure

```
FourD/
├── v2_consciousness_sim.py       # V2 simulation (2700+ lines, tangent force architecture)
├── fourD_slice_sim.py            # V1 simulation (750 lines, scalar scores)
├── config.py                     # Shared configuration
├── requirements.txt              # Dependencies
├── AUDIT_SUMMARY.md              # Architecture audit with priorities
├── V2_ARCHITECTURE_PLAN.md       # Design document
├── README_FEATURES.md            # Feature catalog
│
├── src/                          # QRNG analysis pipeline
│   ├── features/                 # Statistical feature extraction
│   ├── anomaly/                  # Anomaly scoring
│   ├── latent/                   # Basin mapping
│   ├── recurrence/               # RQA analysis
│   ├── viz/                      # Dashboard + heatmaps
│   └── pipelines/                # Pipeline orchestration
│
├── tests/                        # 28 tests (14 v1 sim + 14 pipeline)
├── notebooks/                    # Jupyter analysis
├── data/                         # Input data
├── outputs/                      # Pipeline outputs
└── docs/                         # Planning documents
```

## Theoretical Grounding

The simulation draws from:

1. **Integrated Information Theory (IIT):** Integration metric measures Φ-like information integration across macro basins
2. **Global Workspace Theory:** Subsystem competition for a shared manifold state mirrors competition for the global workspace
3. **Differential Geometry:** States on S³, forces as tangent vector fields, geodesic dynamics — not metaphorical, literally computed
4. **Attractor Dynamics:** Macro basins are attractors; decisions are basin-switching events; personality is the shape of the force landscape
5. **Predictive Processing:** Sensory gradient from PerceptionSlice closes the action-perception loop

The key insight: **neural networks (and brains) represent functions as geometric transformations, not symbolic operations.** This simulation takes that literally — subsystems *are* geometric forces, not labeled modules with scalar outputs.

---

## Roadmap

### Near-Term (Current Architecture Extensions)

**Causal Analysis Layer**
- ~~Cross-correlation: do clarity drops predict macro transitions?~~ ✅ Implemented (clarity rate × macro transition xcorr, r ≈ 0.24 at lag 0)
- Granger causality between geometric metrics and phase transitions
- Information-theoretic: transfer entropy between subsystem forces and trajectory curvature

**Wire to src/ Pipeline**
- Feed geometric time series (conflict, clarity, curvature) through the existing QRNG analysis tools
- Recurrence quantification analysis (RQA) on the manifold trajectory
- Changepoint detection on phase boundaries (compare to GMM results)
- Entropy profiles of the force magnitude time series

**Multi-Seed Profiling**
- Run N seeds, compare cognitive profiles and phase structures
- Classify "personality types" — do initial conditions create categorically different beings?
- Statistical tests on inter-seed variation in phase count, clarity, conflict distributions

### Medium-Term (CUDA / RTX Acceleration)

**GPU-Accelerated Simulation**
- Port ClosedManifold and BalancedSubsystems to CuPy or PyTorch tensors
- The core computation (tangent force projection, soft assignment) is matrix algebra —
  ideal for GPU parallelism
- Target: 10,000+ timesteps real-time on RTX, 100,000+ for batch analysis
- Enable sweeping CONFIG parameter space (alpha_pull, fatigue_rate, novelty_weight)
  across thousands of seeds simultaneously

**Batch Simulation Framework**
```
# Future API sketch
from fourd.gpu import BatchSimulation
batch = BatchSimulation(n_beings=1000, device='cuda:0')
batch.run(timesteps=10000)
profiles = batch.detect_phases()  # All 1000 beings in parallel
```

### Long-Term (Multi-Being & Rich Environment)

**Multiple Beings on Shared Torus**
- N beings (each with own S³ manifold + subsystem preferences) navigating the same toroidal world
- Social subsystem becomes meaningful: force modulated by proximity + state similarity of others
- Emergent social phenomena: flocking, avoidance, coalition formation, communication
- Each being's cognitive phases influenced by social interactions

**Being Differentiation**
- Instead of identical PREFERENCE_MATRIXes, sample each being's preferences from a distribution
- Some beings are "planners" (Planning preference region dominates), others are "intuitive"
- Natural emergence of personality diversity, specialization, and complementarity
- Genetics metaphor: preference matrices as genotypes, cognitive profiles as phenotypes

**Rich Environment Model**
- Replace static goals/hazards with dynamic entities (moving prey, spreading hazards)
- Resource dynamics: goals deplete, regenerate, interact
- Terrain effects: regions of the torus that amplify or dampen certain subsystem forces
- Day/night cycles, seasonal variation → environmental pressure on perception modes
- Ultimately: a self-sustaining ecosystem that runs without intervention

**Environment as Its Own Manifold**
- The torus gains its own geometric state (temperature, resource gradients, danger fields)
- Bidirectional coupling: being's actions modify environment, environment modifies forces
- Multi-scale dynamics: fast (individual steps) → medium (phase transitions) → slow (environmental evolution)

**Evolutionary Dynamics**
- Beings that reach goals survive; those that hit hazards don't
- Offspring inherit (mutated) preference matrices
- Natural selection on the geometric structure of consciousness
- Question: what cognitive architectures evolve under different environmental pressures?

### Speculative (Research Directions)

- **Manifold learning from data:** replace hand-designed S³ with manifold learned from actual neural recordings
- **Higher-dimensional consciousness:** S⁷ or S¹⁵ manifolds — what changes when the internal state has more room?
- **Quantum manifold dynamics:** replace classical state evolution with quantum walks on S³ — connect to QRNG pipeline
- **Formal IIT computation:** compute actual Φ from the force alignment matrix and transition probabilities
- **Consciousness phase transitions:** are there critical points in parameter space where the system undergoes a genuine phase transition in the physics sense?

## License

MIT. See [LICENSE](LICENSE).

## Tests

```bash
python -m pytest tests/ -q
# 28 passed
```