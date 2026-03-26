# FourD — Architecture Reference

This document describes the major subsystems of the FourD repository, how they
relate to each other, and where future refactoring work is expected.

---

## Subsystem Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  V2 Canonical Model                                                          │
│  v2_consciousness_sim.py  (~2700 lines)                                      │
│                                                                              │
│  - ClosedManifold: S³ with 600 Fibonacci micro-refs + 24 KMeans macro-basins │
│  - BalancedSubsystems: 8 tangent-force subsystems with fatigue + novelty     │
│  - PerceptionSlice: 20×20 toroidal environment, 3 goals, 3 hazards           │
│  - ConsciousnessSimV2: orchestrator, 3-stage step, 55-column CSV logging     │
│  - Cognitive phase detection via GMM + BIC                                   │
│  - Clarity field decomposition (gradient, persistence, rate, direction)      │
│  - Output: 4 figure sets + simulation_log_v2.csv                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                    geometry / parameter choices shared
                    (currently by duplication — extraction planned)
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│  GPU Backend                                                                 │
│  gpu_ensemble_sim.py  (~900 lines)                                           │
│                                                                              │
│  - generate_fibonacci_s3 / derive_macro_basins: S³ geometry (duplicated)    │
│  - BatchConsciousnessEngine: PyTorch/CUDA, VRAM-aware batch sizing           │
│    - Runs N beings in parallel on GPU tensors                                │
│    - Extracts per-being "consciousness signatures" (compact float vectors)   │
│  - PhaseCartographer: 5D parameter-space sweep, phase diagram generation     │
│  - SIGNATURE_NAMES, SUBSYSTEM_NAMES: shared naming constants                │
│                                                                              │
│  Runnable standalone:  python gpu_ensemble_sim.py [--device cuda:0]          │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ imported by all experiment suites
       ┌───────────────────────────┼────────────────────────────┐
       │                           │                            │
┌──────▼────────┐  ┌───────────────▼──────────────┐  ┌─────────▼────────────┐
│ goldilocks_   │  │ critical_phenomena_suite.py   │  │ universality_        │
│ sweep.py      │  │                               │  │ verification.py      │
│               │  │ - Null / ablation controls    │  │                      │
│ Fine-grained  │  │ - Subsystem-count FSS         │  │ - Susceptibility γ   │
│ sweep around  │  │ - Data collapse               │  │ - Finite-size ν      │
│ the critical  │  │ - Coarse-graining / RG        │  │ - Avalanche / 1/f    │
│ Goldilocks    │  │                               │  │ - 3D-Ising compare   │
│ region        │  │ Output: outputs/critical_     │  │                      │
│               │  │         phenomena/            │  │ Output: outputs/     │
│ Output:       │  └───────────────────────────────┘  │         universality/│
│ outputs/      │                                      └──────────────────────┘
│ goldilocks/   │  ┌───────────────────────────────┐
└───────────────┘  │ mechanism_extraction.py       │
                   │                               │
                   │ - Minimal S¹ reduced model    │
                   │ - Bifurcation analysis        │
                   │ - Layered ablation            │
                   │ - Adaptive phase diagram      │
                   │                               │
                   │ Output: outputs/mechanism/    │
                   └───────────────────────────────┘
                   ┌───────────────────────────────┐
                   │ deep_analysis.py              │
                   │                               │
                   │ - RQA on trajectory           │
                   │ - Basin transition grammar    │
                   │ - Clarity dynamics            │
                   │ - Multi-seed robustness       │
                   │   (100 seeds, GPU engine)     │
                   │                               │
                   │ Output: outputs/deep_analysis/│
                   └───────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Legacy V1 Baseline                                                          │
│  fourD_slice_sim.py  (~750 lines)                                            │
│                                                                              │
│  - Scalar-score subsystems (not tangent-force geometry)                      │
│  - 4D latent state with hyperspherical intent (not fully closed manifold)    │
│  - Lesion study infrastructure                                               │
│  - CLI output directories with timestamped summaries                         │
│  - Useful for: baseline comparison, lesion ablation, interpretive reference  │
│                                                                              │
│  Architecturally independent from V2 and GPU backend.                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  QRNG / Latent Analysis Pipeline                                             │
│  src/  (~15 modules)                                                         │
│                                                                              │
│  src/features/       Entropy, autocorrelation, changepoint, complexity       │
│  src/anomaly/        Anomaly scoring, standardisation                        │
│  src/latent/         Basin mapping, latent coordinator simulation            │
│  src/recurrence/     Recurrence plots, RQA metrics                          │
│  src/viz/            Dashboard, heatmaps                                     │
│  src/pipelines/      End-to-end orchestration, surrogate comparison          │
│                                                                              │
│  Entry point: python -m src.pipelines.run_qrng_pipeline                     │
│  Dashboard:   bash run-dashboard.sh                                          │
│                                                                              │
│  Currently analysed independently from the simulation tracks.                │
│  Future: share recurrence / latent tools with simulation outputs.            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How V1, V2, GPU, and Pipeline Relate

```
                   ┌──────────────────────────┐
                   │  V1 (fourD_slice_sim.py) │  ← interpretive baseline, lesion studies
                   └──────────────────────────┘
                              ↓  evolved into
                   ┌──────────────────────────┐
                   │  V2 (v2_consciousness_   │  ← canonical scientific model
                   │      sim.py)             │     S³, tangent forces, closed manifold
                   └──────────────┬───────────┘
                                  │  geometry / dynamics ported to GPU
                   ┌──────────────▼───────────┐
                   │  GPU Backend             │  ← accelerated engine
                   │  (gpu_ensemble_sim.py)   │     10K–500K beings in parallel
                   └──────────────┬───────────┘
                                  │  imported by
         ┌────────────────────────▼──────────────────────────┐
         │  Experiment suites (goldilocks, critical_phenomena, │
         │  universality, mechanism, deep_analysis)            │
         └─────────────────────────────────────────────────────┘

         ┌─────────────────────────────────────────────────────┐
         │  src/ QRNG pipeline                                  │
         │  Analyses external bitstream data through the same   │
         │  latent-coordinator abstraction (basin mapping, RQA) │
         │  Currently independent; planned: shared tools        │
         └─────────────────────────────────────────────────────┘
```

---

## Known Duplication / Technical Debt

### 1. Geometry duplicated: V2 ↔ GPU backend

`v2_consciousness_sim.py` and `gpu_ensemble_sim.py` both define:
- `generate_fibonacci_s3` (S³ lattice)
- `derive_macro_basins` (KMeans clustering)
- `PREFERENCE_MATRIX` / `PREFERENCE_MATRIX_NORMED`

**Risk:** parameter or algorithm drift between the two implementations.  
**Fix (planned):** extract into a `core/` package:
- `fourd/core/geometry.py`
- `fourd/core/preferences.py`
- `fourd/core/config.py`
- `fourd/core/signatures.py`

### 2. Experiment scripts import directly from `gpu_ensemble_sim.py`

All five experiment suites (`goldilocks_sweep`, `critical_phenomena_suite`,
`universality_verification`, `mechanism_extraction`, `deep_analysis`) import from
`gpu_ensemble_sim.py` directly.

This makes `gpu_ensemble_sim.py` a hard-dependency hub.  
**Fix (planned):** split into `fourd/gpu/engine.py`, `fourd/gpu/cartography.py`, etc.

### 3. QRNG pipeline is isolated from simulation analysis

The recurrence and latent tools in `src/` are useful for simulation outputs too, but
are currently wired only for QRNG bitstream inputs.  
**Fix (planned):** add simulation-compatible input adapters so the same RQA / anomaly
tools can run on `simulation_log_v2.csv` outputs.

### 4. Output directories are not standardised

V1 writes timestamped run directories; V2 writes figures directly to the project root;
GPU scripts write under `outputs/<script-name>/`.  
**Fix (planned):** define a shared output-root convention and update all scripts.

---

## Planned Target Structure

Once the `core/` extraction is done (a separate future PR), the repo will look like:

```
fourd/
  core/
    geometry.py        # Fibonacci S³, KMeans basins
    preferences.py     # PREFERENCE_MATRIX + normalisation
    config.py          # Default CONFIG + optimised params
    signatures.py      # SIGNATURE_NAMES, SUBSYSTEM_NAMES
  models/
    v1_baseline.py     # fourD_slice_sim logic
    v2_model.py        # v2_consciousness_sim logic
  gpu/
    engine.py          # BatchConsciousnessEngine
    cartography.py     # PhaseCartographer
    batch.py           # run_batch_simulations logic
  analysis/
    deep_analysis.py
    mechanism_extraction.py
    critical_phenomena.py
    universality.py
  pipelines/           # (currently src/pipelines/)
  recurrence/          # (currently src/recurrence/)
  viz/                 # (currently src/viz/)

scripts/               # thin entry-point wrappers
  run_v1.py
  run_v2.py
  run_goldilocks.py
  run_critical_suite.py
  run_universality.py
  run_qrng_pipeline.py

tests/
docs/
outputs/
```

This restructuring is **not** part of the current PR.  The goal here is to document
the existing layout accurately so a reader can understand the repo without needing to
reverse-engineer it.

---

## Planning Documents (Historical)

The `docs/planning/` directory contains notes and scaffolding plans from earlier
development stages.  These may be stale relative to the current codebase and are kept
for historical reference only.  Canonical descriptions of the current architecture live
in this file and in `README.md`.
