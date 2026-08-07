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

## Boundary Negotiation / Interface-Competency Thread (April–August 2026)

This thread runs on a separate, lighter engine (`universality_test.py`'s
`UniversalEngine`) rather than the V2 canonical model, specifically so that
"does geometry X matter" questions can be tested against matched controls
(swappable manifold / topology / fatigue) instead of only ever running the
one hand-tuned configuration. It grew out of Observer Patch Holography
(OPH, by FloatingPragma) framing, then extended in August 2026 into
persistent state and perturbation-response experiments. See the README's
[Boundary Negotiation / Interface-Competency Thread](../README.md#boundary-negotiation--interface-competency-thread)
table for the full script list; this section documents the reasoning chain
and the open hypothesis in more depth than the README index does.

### The chain of findings

1. `oph_bridge_analysis.py` → `boundary_negotiation_test.py` established that
   clarity concentrates at basin-transition boundaries rather than being
   uniform, and that this isn't an artifact of high transition rates (null
   model comparison).
2. `universality_test.py` generalized this across 10 manifold/topology/
   fatigue variants, finding the effect is not FourD-specific.
3. `minimal_boundary_model.py` found the *minimal* substrate that still
   reproduces it: S¹ or even R¹ with cyclic opponents + fatigue. No S³, no
   8 subsystems, no 600-point Fibonacci lattice required.
4. `topology_dissection.py` decomposed "cyclic topology" into independent
   axes (opposition angle, pairing structure, pair count) to find which
   sub-property is actually essential, rather than treating "cyclic" as
   one indivisible thing.
5. `mechanism_extraction.py`, `critical_phenomena_suite.py`, and
   `universality_verification.py` established that the underlying
   order/disorder transition (as `fatigue_rate` crosses a threshold) is a
   genuine critical phenomenon — finite-size scaling, data collapse, and
   critical exponents consistent with the 3D-Ising universality class, not
   just a qualitative regime change.
6. `goldilocks_sweep.py` located that critical region empirically. Its
   cached output (`outputs/phase_cartography/goldilocks_report.txt`) puts
   the top-10 "flourishing" configurations at `fatigue_rate≈0.20–0.27`,
   `exploration_noise≈0.17–0.20`. `geometry_comparison.py`'s independently
   cached S³ sweep (`outputs/geometry_comparison/geometry_comparison_results.json`,
   `part_a`) shows mean clarity jumping from ≈0.15 to ≈0.29 between
   `fatigue_rate=0.183` and `0.20` — a directly visible phase transition at
   almost exactly that boundary.
7. `basin_gateway_analysis.py` / `basin_deep_dive.py` / `staged_gateway_control.py`
   turned "where clarity concentrates" into an explicit transition graph
   (gateway centrality, dwell times, funnelness).
8. `positive_geometry_readout.py` (August 2026) asked whether that graph's
   own combinatorics — spanning-tree edge probabilities via Foster's
   theorem, computed with zero access to the clarity/dynamics data —
   predict which edges the simulation empirically marks as important
   gateways. Found yes on flat R⁴ (partial r=0.29, p<1e-7, controlling for
   raw traffic volume) but not on the canonical s3/cyclic geometry
   (partial r=0.01, p=0.89) — i.e. on flat4 a meaningful part of "what
   matters" is recoverable from topology alone; on s3/cyclic it isn't,
   suggesting significance there is generated dynamically rather than
   structurally.
9. `bearer_state_competency.py` (August 2026) added a persistent **bearer
   state** `b_t` to `UniversalEngine` — a variable that biases influences
   each step and is itself updated by activities each step, so it is
   constitutive (gates future dynamics) rather than a passive log. Measured
   a competency vector (lesion recovery, adaptation-to-environment-shift,
   memory horizon, self-maintenance) across manifold × topology × bearer
   on/off, with an immediate-deficit / cumulative-deficit / recovery-time
   decomposition (added after initial review) so that a recovery time of
   "0" can't conflate "nothing was disrupted" with "disrupted and
   instantly recovered" — those are opposite findings and the original
   single-number metric couldn't tell them apart.
10. `qrng_developmental_capture.py` and `perturbation_concentration_sweep.py`
    (August 2026) tested whether a one-shot perturbation's persistence
    (`DC(Δ)`) depends on its entropy source (deterministic / seeded PRNG /
    OS CSPRNG) or its concentration (localized vs. distributed at matched
    L2 norm). Found: concentration matters (r=0.28–0.38, p<0.02 on both
    manifolds tested), entropy source does not (PRNG vs. OS-CSPRNG
    equivalence tests were inconclusive-to-null; the one nominally
    significant result, s3 p=0.033, doesn't survive multiple-comparison
    correction across the ~6 tests run and shouldn't be read as a finding).

### Open hypothesis: is s3/cyclic's competency result about geometry, or about criticality?

Step 9's headline finding was that `s3 + cyclic` — the one configuration
using the hand-calibrated `PREFERENCE_MATRIX_NORMED` — showed uniquely
large, slow-recovering lesion and environment-shift disruption (800+ of
1800 steps) while every other manifold/topology combination (using
freshly-generated, uncalibrated synthetic preferences) recovered near-
instantly or showed no measurable deficit at all.

Two explanations are live and not yet distinguished:

- **H1 — geometry/calibration-specific:** something about S³ specifically,
  or about the hand-tuned preference matrix specifically, produces richer,
  more consequential dynamics that flat4/S² and synthetic preferences lack.
- **H2 — critical slowing down:** `s3/cyclic`'s default `fatigue_rate=0.217`
  happens to sit inside the Goldilocks/critical region located in step 6.
  Critical slowing down — relaxation times diverging near a real critical
  point — is a standard, well-understood phenomenon in systems already
  shown (step 5) to sit in a genuine universality class. Under H2, the
  large recovery times have nothing to do with S³ or calibration per se;
  any manifold/topology combination would show the same effect if its
  `fatigue_rate` were swept to its own critical region, and s3/cyclic
  looks special here only because it's the one condition that happened to
  already be tuned near criticality.

These aren't mutually exclusive (calibration could be *what puts s3/cyclic
near criticality* rather than an independent cause), but H2 makes a sharp,
falsifiable prediction that H1 alone doesn't: **recovery time should peak
as a function of `fatigue_rate` near the known transition (~0.18–0.27),
independent of which manifold/topology it's tested on.**

**Decisive test (see `criticality_sweep.py`):** sweep `fatigue_rate` at
fixed manifold/topology, holding everything else constant, and measure
`bearer_state_competency.py`'s `lesion_t_recovery` / `adapt_t_adapt`
against it. A peak near the known transition region supports H2. A flat
response (recovery time roughly constant across `fatigue_rate`, only
varying with manifold identity) would rule out criticality as the driver
and point back to H1 — at which point the matched-preference factorial
(controlling for calibration while varying manifold) becomes the right
next experiment instead.

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
