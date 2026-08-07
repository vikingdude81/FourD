# FourD — Consciousness as Geometry on S³

An emerging research platform for higher-dimensional consciousness-like dynamics.  
FourD combines a geometric consciousness simulation, GPU-accelerated ensemble experiments,
phase-transition analysis, and a QRNG / latent-analysis pipeline in a single repo.

> **Repo status (August 2026):** active development.  The GPU ensemble backend and
> experiment suites are implemented and runnable.  A second research thread (April–August
> 2026) grew out of Observer Patch Holography (OPH)-inspired boundary-negotiation work:
> a modular `UniversalEngine` (swappable manifold/topology/fatigue), basin-gateway and
> combinatorial-graph readouts, and — newest — a persistent **bearer state** layered on
> top of the engine to test whether interface geometry is *constitutive* (gates which
> competencies survive lesion/perturbation) rather than merely descriptive. See
> [Boundary Negotiation / Interface-Competency Thread](#boundary-negotiation--interface-competency-thread)
> below. The `src/` QRNG pipeline is partially bridged into this thread (see
> `rqa_boundary_analysis.py`, `latent_boundary_mapping.py`) but not yet fully unified.
> A planned `core/` package extraction and unified output schema are future work — see
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## What This Is

**The core idea:** consciousness isn't a thing that happens *to* a system — it's the
shape of what the system does when its internal degrees of freedom exceed what any single
subsystem can manage.  When 8 subsystems compete for control of a shared state on the
3-sphere (S³), their interaction creates a force landscape.  The being's trajectory
through that landscape *is* its stream of consciousness.

**What makes this different from typical agent sims:** subsystems aren't scorecards or
weights — they're tangent vector fields on S³.  Each subsystem generates a geometric
force at every point on the manifold, pulling the state along a great circle toward its
preferred region.  Forces curve with the manifold.  No information is lost.

---

## Repository Spine

| File / Directory | Role |
|---|---|
| `v2_consciousness_sim.py` | **Canonical model** — S³ dual-geometry, tangent-force, soft-assignment engine |
| `fourD_slice_sim.py` | **Legacy baseline** — V1 scalar-score model, lesion studies, interpretive reference |
| `gpu_ensemble_sim.py` | **GPU backend** — PyTorch/CUDA engine for large-scale ensemble runs and phase sweeps |
| `goldilocks_sweep.py` | High-resolution parameter sweep around the critical (Goldilocks) region |
| `critical_phenomena_suite.py` | Null/ablation controls, finite-size scaling, data collapse, coarse-graining |
| `universality_verification.py` | Deep-dive critical-exponent measurement and universality-class comparison |
| `mechanism_extraction.py` | Minimal reduced model, bifurcation analysis, layered mechanism ablation |
| `deep_analysis.py` | RQA, basin grammar, clarity dynamics, multi-seed robustness |
| `run_batch_simulations.py` | Parameter-sweep batch runner (wraps V2 simulation) |
| `src/` | **QRNG / latent analysis pipeline** — features, anomaly, latent mapping, recurrence, viz |

For a detailed breakdown of each subsystem and how they relate, see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Boundary Negotiation / Interface-Competency Thread

A second research thread, credited to Observer Patch Holography (OPH, by FloatingPragma —
https://github.com/FloatingPragma/observer-patch-holography) for its framing, asks a
different question than the canonical V2 model: not "what does this system's trajectory
look like" but "does *where information concentrates* (basin boundaries, transition
gateways) carry structure that's derivable, predictable, or constitutive." It runs on a
separate, lighter modular engine (`universality_test.py`'s `UniversalEngine`) with
swappable manifold (S³ / S² / flat R⁴), topology (cyclic / random / full / none), and
fatigue dynamics, so claims about "does geometry X matter" can be tested against matched
controls instead of only the canonical model.

| Script | Role |
|---|---|
| `oph_bridge_analysis.py` | OPH-inspired metrics bridged onto the V2 engine — starting point for this thread |
| `universality_test.py` | `UniversalEngine` — the swappable manifold/topology/fatigue core used by everything below |
| `boundary_negotiation_test.py` | 5-part validation: does clarity concentrate at basin boundaries, and is it universal or engine-specific? |
| `minimal_boundary_model.py` | Strips the effect down to its minimal substrate — reproduces on S¹/R¹ with just cyclic opponents + fatigue |
| `topology_dissection.py` | Decomposes "cyclic topology" into angle / pairing-structure / pair-count axes to find what's actually essential |
| `geometry_comparison.py` | S³ vs flat R⁴ head-to-head — found flat R⁴ has *stronger* boundary negotiation on the edge-clarity metric |
| `mechanism_extraction.py`, `critical_phenomena_suite.py`, `universality_verification.py` | Establish this is a genuine critical phase transition (3D-Ising-class exponents), not just a qualitative effect |
| `goldilocks_sweep.py` | Locates the critical parameter region empirically — `outputs/phase_cartography/goldilocks_report.txt` puts it around `fatigue_rate≈0.20–0.27`, `exploration_noise≈0.17–0.20` |
| `basin_gateway_analysis.py`, `basin_deep_dive.py`, `staged_gateway_control.py` | Basin-transition graph construction, gateway centrality, dwell-time/occupancy structure |
| `controller_ablation_multiseed.py` | Multi-seed ranking of controller variants with confidence intervals |
| `rqa_boundary_analysis.py`, `latent_boundary_mapping.py` | Bridge `src/recurrence` and `src/latent` (the QRNG pipeline's analysis modules) into boundary-transition analysis |
| `positive_geometry_readout.py` | Tests whether spanning-tree graph combinatorics (Foster's theorem), computed with zero access to simulation dynamics, predict the empirical gateway graph — found to work on flat4 (partial r=0.29, p<1e-7) but not on the canonical s3/cyclic geometry |
| `bearer_state_competency.py` | Adds a persistent **bearer state** to `UniversalEngine` that biases and is biased by dynamics; measures a competency vector (lesion recovery, adaptation speed, memory horizon) with immediate/cumulative-deficit decomposition so "0" can't conflate "no effect" with "instant recovery" |
| `qrng_developmental_capture.py` | Measures how long a one-shot perturbation's effect persists (`DC(Δ)`) under deterministic / PRNG / OS-CSPRNG injection, with a TOST equivalence test |
| `perturbation_concentration_sweep.py` | Isolates perturbation *shape* (concentrated vs. distributed) from entropy source — found this, not randomness quality, drives `DC(Δ)` differences |

**Open thread worth flagging:** the canonical `s3 + cyclic` configuration's default
`fatigue_rate=0.217` sits inside the empirically-identified Goldilocks/critical region
above, and `geometry_comparison.py`'s own S³ sweep shows a sharp clarity jump right around
`fatigue_rate≈0.18–0.20`. The August 2026 bearer-state competency runs found `s3/cyclic`
uniquely showing large, slow perturbation-recovery times (800+ steps) vs. near-zero
elsewhere — plausibly **critical slowing down** (a system near a real critical point
relaxes slowly after perturbation) rather than a manifold- or calibration-specific effect.
Not yet confirmed; the decisive test is a `fatigue_rate` sweep at fixed manifold/topology,
not yet run. See `outputs/bearer_state_competency/` for the raw numbers this is based on.

---

## Recommended Entry Points

### Run the canonical V2 model

```bash
pip install -r requirements.txt
python v2_consciousness_sim.py
```

Outputs:
- `simulation_v2_results.png` — 6-panel coherence/dominance/navigation overview
- `cognitive_landscape.png` — 10-panel force field, trajectories, cognitive profile
- `cognitive_phases.png` — 9-panel phase detection with transitions, profiles, landscape
- `clarity_deep_dive.png` — 12-panel clarity field analysis
- `simulation_log_v2.csv` — 500 rows × 55 columns of per-timestep data

### Run the legacy V1 baseline / lesion model

```bash
python fourD_slice_sim.py
```

V1 uses scalar-score subsystems and writes timestamped output directories.
Useful for lesion studies and comparison against V2 architecture.

### Run GPU phase-space / ensemble experiments

```bash
# Full parameter-space cartography (requires CUDA GPU)
python gpu_ensemble_sim.py

# High-resolution Goldilocks sweep around the critical region
python goldilocks_sweep.py

# Critical phenomena suite (null controls, FSS, data collapse, coarse-graining)
python critical_phenomena_suite.py

# Universality-class verification (critical exponents, 3D-Ising comparison)
python universality_verification.py

# Mechanism extraction (minimal model, bifurcation, adaptive phase diagram)
python mechanism_extraction.py

# Deep analysis (RQA, basin grammar, multi-seed robustness)
python deep_analysis.py
```

All GPU scripts write outputs under `outputs/`.
`gpu_ensemble_sim.py` auto-detects available VRAM and adjusts batch size.

### Run the boundary-negotiation / interface-competency thread

```bash
# Bearer-state competency vector (lesion recovery, adaptation, memory) across manifolds
python bearer_state_competency.py --device cuda:0 --steps 1800 --N 256 --seeds 10

# Developmental capture under deterministic / PRNG / OS-CSPRNG perturbation sources
python qrng_developmental_capture.py --device cuda:0 --steps 1800 --N 256 --seeds 16

# Does perturbation concentration (not entropy source) drive developmental capture?
python perturbation_concentration_sweep.py --device cuda:0 --steps 1800 --N 256 --seeds 16

# Combinatorial gateway readout: does spanning-tree graph structure predict empirical
# gateway importance? (run basin_gateway_analysis.py first to generate its input)
python basin_gateway_analysis.py --device cuda:0 --steps 2000 --N 128
python positive_geometry_readout.py
```

See [Boundary Negotiation / Interface-Competency Thread](#boundary-negotiation--interface-competency-thread)
above for what each script tests and how they connect.

### Run the QRNG analysis pipeline

```bash
python -m src.pipelines.run_qrng_pipeline \
  --input data/qrng_bits.npy \
  --output outputs/qrng_analysis/
```

Runs: bitstream loading → sliding windows → feature extraction → anomaly scoring →
latent basin mapping → recurrence analysis.

### Explore outputs / dashboard

```bash
bash run-dashboard.sh
# or directly:
python -m src.viz.dashboard \
  --features outputs/qrng_pipeline/sample_stream/window_features.csv \
  --latent  outputs/qrng_pipeline/sample_stream/latent_trajectory.csv
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Canonical Model  (v2_consciousness_sim.py)                       │
│  ─────────────────────────────────────────────────────────────   │
│  S³ Manifold · 600 micro-refs · 24 macro-basins                  │
│  8 subsystems → tangent force competition                         │
│  Soft macro assignment · Basin-escape · Toroidal environment      │
│  Outputs: 55 metrics/step, 4 figure sets, CSV log                │
└─────────────────────────┬────────────────────────────────────────┘
                           │ shares geometry / parameter choices
┌─────────────────────────▼────────────────────────────────────────┐
│  GPU Backend  (gpu_ensemble_sim.py)                               │
│  ─────────────────────────────────────────────────────────────   │
│  BatchConsciousnessEngine · PhaseCartographer                     │
│  PyTorch/CUDA · VRAM-aware batching                              │
│  10K–500K simultaneous beings per run                            │
│  Target: RTX 3050 (8 GB) → RTX 5090 (32 GB)                     │
└─────────────────────────┬────────────────────────────────────────┘
                           │ imported by all experiment suites
┌─────────────────────────▼────────────────────────────────────────┐
│  Experiment Suites  (top-level scripts)                           │
│  goldilocks_sweep · critical_phenomena_suite                      │
│  universality_verification · mechanism_extraction · deep_analysis │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Legacy Baseline  (fourD_slice_sim.py)                            │
│  Scalar-score V1 model · Lesion studies · Interpretive reference  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  QRNG / Latent Analysis Pipeline  (src/)                          │
│  features · anomaly · latent · recurrence · pipelines · viz       │
│  Bitstream → features → latent coordinator → RQA → dashboard      │
│  Currently analysed independently from the simulation tracks      │
└──────────────────────────────────────────────────────────────────┘
```

### The 8 Subsystems (V2 model)

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

Subsystems 0–3 form the "core four."  Subsystems 4–7 ("contrast four") are
geometrically distant — rarely dominant, but always shaping the trajectory.

### Key Metrics (55 columns per timestep)

**Coherence:** closure_coherence, integration, differentiation, path_coherence

**Geometric:**

| Metric | What It Measures |
|--------|------------------|
| `conflict_angle` | Angle between top-2 subsystem forces. 0° = agreement, 180° = torn |
| `clarity` | Resultant force magnitude. High = knows what it wants |
| `curvature` | Angular change per step. High = rapid state shifts |
| `inner_outer_ratio` | Perception / navigation dims. >1 = contemplative |
| `force_mag_0`–`force_mag_7` | Per-subsystem tangent force magnitude |
| `clarity_grad_mag` | How fast clarity changes near the current state |
| `clarity_decomp_0`–`clarity_decomp_7` | Per-subsystem signed clarity contribution |
| `clarity_persistence` | Running 20-step autocorrelation of clarity |
| `clarity_rate` | d(clarity)/dt — predicts transitions before they happen |
| `resultant_dir_0`–`resultant_dir_3` | Unit vector of combined force direction |

**Perception:** perc_range, perc_focus, perc_mode, n_visible  
**Navigation:** env_x, env_y, heading_x, heading_y, speed, goal_prox, hazard_prox  
**Manifold:** u_dim0–u_dim3, micro_id, macro_dominant

---

## Project Structure

```
FourD/
├── v2_consciousness_sim.py         # Canonical V2 model (tangent-force / S³)
├── fourD_slice_sim.py              # Legacy V1 model (scalar scores, lesion studies)
├── gpu_ensemble_sim.py             # GPU backend (BatchConsciousnessEngine, CUDA)
├── goldilocks_sweep.py             # High-res parameter sweep
├── critical_phenomena_suite.py     # Universality / criticality verification
├── universality_verification.py    # Critical-exponent deep dive
├── mechanism_extraction.py         # Minimal model + bifurcation analysis
├── deep_analysis.py                # RQA, basin grammar, multi-seed robustness
├── run_batch_simulations.py        # Batch runner for V2 parameter sweeps
├── config.py                       # Shared configuration
├── requirements.txt
│
├── src/                            # QRNG / latent analysis pipeline
│   ├── features/                   # Entropy, autocorrelation, changepoint, complexity
│   ├── anomaly/                    # Anomaly scoring + standardisation
│   ├── latent/                     # Basin mapping, latent coordinator
│   ├── recurrence/                 # Recurrence plots + RQA
│   ├── viz/                        # Dashboard + heatmaps
│   └── pipelines/                  # End-to-end pipeline orchestration
│
├── tests/                          # 28 tests (V1 sim + pipeline)
├── notebooks/                      # Jupyter analysis
├── data/                           # Input data
├── outputs/                        # All script outputs (gitignored)
└── docs/
    ├── ARCHITECTURE.md             # Subsystem reference + future roadmap
    └── planning/                   # Historical planning notes (may be stale)
```

---

## Theoretical Grounding

1. **Integrated Information Theory (IIT):** Integration metric measures Φ-like information integration across macro basins
2. **Global Workspace Theory:** Subsystem competition for a shared manifold mirrors competition for the global workspace
3. **Differential Geometry:** States on S³, forces as tangent vector fields, geodesic dynamics — literally computed
4. **Attractor Dynamics:** Macro basins are attractors; decisions are basin-switching events
5. **Predictive Processing:** Sensory gradient from PerceptionSlice closes the action-perception loop
6. **Critical Phenomena:** GPU experiment suites test whether the system exhibits genuine phase transitions with measurable universality-class exponents

---

## Status of Key Features

| Feature | Status |
|---|---|
| V2 canonical model (S³, tangent forces, soft assignment) | ✅ Implemented |
| Cognitive phase detection (GMM) | ✅ Implemented |
| Clarity field decomposition (gradient, persistence, rate) | ✅ Implemented |
| GPU ensemble backend (PyTorch/CUDA, VRAM-aware) | ✅ Implemented |
| Goldilocks / phase-space cartography sweep | ✅ Implemented |
| Critical phenomena suite (FSS, data collapse, coarse-graining) | ✅ Implemented |
| Universality-class verification | ✅ Implemented |
| Mechanism extraction / bifurcation analysis | ✅ Implemented |
| Multi-seed robustness analysis | ✅ Implemented |
| QRNG latent analysis pipeline | ✅ Implemented |
| Surrogate comparison / statistical tests | ✅ Implemented |
| `UniversalEngine` (swappable manifold/topology/fatigue) | ✅ Implemented |
| Boundary negotiation / gateway-graph analysis | ✅ Implemented |
| Bearer state + competency vector (lesion/adaptation/memory) | ✅ Implemented |
| Developmental capture (`DC(Δ)`) + perturbation-source comparison | ✅ Implemented |
| Combinatorial (spanning-tree) gateway readout | ✅ Implemented |
| Critical-slowing-down explanation for s3/cyclic's competency results | 🔧 Hypothesized, not yet tested |
| Matched-preference factorial (isolates manifold from calibration) | 🔧 Planned |
| Causal bearer-memory probe (decode + ablate, not just decay-time) | 🔧 Planned |
| Unified `core/` package (shared geometry, config, signatures) | 🔧 Planned |
| Standardised output schema across all scripts | 🔧 Planned |
| `src/` pipeline fully wired to boundary-negotiation thread (partially bridged via `rqa_boundary_analysis.py`, `latent_boundary_mapping.py`) | 🔧 Planned |
| Multi-being toroidal environment | 💡 Future |
| Manifold learning from neural-recording data | 💡 Future |

---

## Roadmap

### Near-Term
- **Decisive test:** sweep `fatigue_rate` at fixed manifold/topology (s3/cyclic) through the
  Goldilocks region identified by `goldilocks_sweep.py` (≈0.18–0.27) and measure whether
  `bearer_state_competency.py`'s lesion/adaptation recovery times peak there — tests the
  critical-slowing-down hypothesis directly, and is cheaper than the full matched-preference
  factorial below since it varies one already-understood parameter instead of building new
  preference generators.
- Matched-preference factorial across manifold × topology × preference-regime, using
  `topology_dissection.py`'s existing angle/structure/pair-count decomposition as the source
  of controlled preference variants rather than building a new generator from scratch.
- Extract shared geometry / config into a reusable `core/` package to prevent V2 ↔ GPU drift
- Standardise output directories and column schemas across all scripts

### Medium-Term
- Causal bearer-memory probe: decode a task-relevant variable from `b_t` and ablate it,
  rather than inferring memory purely from decay-time metrics (current `memory_horizon` is
  dominated by the fixed `bearer_decay` hyperparameter, not emergent memory)
- Re-run bearer-state competency experiments on `minimal_boundary_model.py`'s S¹/R¹
  substrates — 10–100x cheaper per run than full S³ at N=256, enabling much larger seed
  counts and parameter sweeps within the same GPU budget
- Reconcile `geometry_comparison.py`'s finding (flat R⁴ > S³ on edge-clarity boundary
  negotiation) with the bearer-competency finding (S³/cyclic > everything on lesion/
  adaptation recovery) — likely resolved once the criticality test above lands, since these
  may be measuring steady-state coupling strength vs. perturbation-response time, which
  needn't agree
- Multi-seed population studies with personality-type classification
- Formal Granger / transfer-entropy causal analysis between geometric metrics and phase transitions
- S⁷ / S¹⁵ manifold experiments

### Long-Term / Speculative
- Multiple beings on a shared toroidal world (social subsystem becomes meaningful)
- Evolutionary dynamics: beings that reach goals survive, offspring inherit preference matrices
- Manifold learning from actual neural recordings
- Quantum walks on S³ — formal connection to QRNG pipeline
- Real hardware QRNG capture wired into `data/qrng_bits.npy` (currently `os.urandom`-only)

---

## License

MIT. See [LICENSE](LICENSE).

## Tests

```bash
python -m pytest tests/ -q
# 28 passed
```