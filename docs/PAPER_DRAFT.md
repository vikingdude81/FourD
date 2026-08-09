# Bearer State, Bifurcation, and Graph-Recoverable Geometry in a Minimal Competitive-Fatigue Dynamical System

**Status:** draft / technical report. Not yet submitted anywhere. Numbers below are
pulled directly from `outputs/*/` JSON artifacts in this repository and are
reproducible via the commands cited in each section.

**Author:** *(fill in)*
**Affiliation:** *(fill in — independent research repository)*
**Code & data:** https://github.com/vikingdude81/FourD

---

## Abstract

We study a minimal multi-agent competitive dynamical system — N tangent-force
"subsystems" competing for control of a state on a manifold (S³, S², or flat
R⁴), modulated by activity-dependent fatigue — and ask three questions. First,
does the system's order/disorder transition, previously characterized as
critical (3D-Ising-class exponents), behave like a continuous critical point or
a discontinuous bifurcation under direct perturbation-response testing? We find
the latter: a saddle-node/transcritical bifurcation at a fatigue-rate threshold
that is empirically independent of the embedding manifold. Second, we add a
persistent **bearer state** — a variable that both biases and is biased by the
system's dynamics at every step, making it constitutive rather than a passive
log — and ask whether it changes the system's response to perturbation. It
does: the full system exhibits genuine (but transient) path-dependence/
hysteresis that a reduced 2–3 variable deterministic skeleton of the same
dynamics does not, while a functionally distinct phenomenon — near-permanent
inability to recover from a subsystem lesion at the same operating point — is
*not* explained by that hysteresis and remains open. Third, we test whether
the system's basin-transition graph, treated purely as combinatorial data (no
manifold coordinates), can recover the manifold's true metric structure via
effective-resistance embedding (Foster's theorem) — finding a mixed and
manifold-dependent result: on the compact manifold the graph recovers close to
the correct intrinsic dimensionality with weak metric fidelity, while on the
flat manifold it recovers no dimensional compression at all but stronger
metric fidelity. We report several corrected methodological errors found
during this work (an RNG-stream artifact that initially produced a spurious
"quantum-randomness" effect; a symbolic-computation performance bug; an
overstated claim of numerical agreement between two unrelated measurements)
as part of the record, since catching and correcting them is itself part of
what the results establish.

---

## 1. Introduction

This system originated as part of a broader, more speculative research
program (documented separately in this repository) inspired by discussions of
whether physical/computational architectures can act as "interfaces" onto
structured spaces of behavior or form — a framing associated with Michael
Levin's writing on developmental bioelectricity and cognition, and Donald
Hoffman's conscious-agent formalism. **This paper does not test or defend
that framing.** What it does is extract the one part of that broader
discussion that turned out to be genuinely tractable, falsifiable, and
already partly built: a concrete dynamical system with a real phase
transition, and the question of what persistent internal state does to that
system's response to perturbation, plus a narrow, literal version of "can
geometry be recovered from combinatorics alone" using a rigorous, unrelated
piece of graph theory (effective resistance / Foster's theorem) rather than
anything from the amplituhedron/positive-Grassmannian literature, which we
explicitly did not implement (see §7 and Appendix A for why not).

The system itself (`UniversalEngine`, `universality_test.py`) descends from a
separate, longer-running thread in this repository studying "boundary
negotiation" — the empirical finding that information (measured as clarity,
a resultant-force magnitude) concentrates disproportionately at transitions
between attractor basins rather than during stable dwelling, credited to
Observer Patch Holography (OPH) framing by FloatingPragma
(https://github.com/FloatingPragma/observer-patch-holography). That thread
established, across ten manifold/topology/fatigue variants
(`universality_test.py`), that the effect generalizes; found the minimal
substrate that reproduces it (a single subsystem-pair on S¹ or even R¹,
`minimal_boundary_model.py`); and — most relevantly here — established via
finite-size scaling, data collapse, and critical-exponent measurement
(`critical_phenomena_suite.py`, `universality_verification.py`) that the
underlying fatigue-driven order/disorder transition is a genuine phase
transition, not merely a qualitative regime change. This paper picks up from
that point and asks what that transition actually looks like under direct
perturbation.

## 2. Model

### 2.1 Base dynamics (`UniversalEngine`)

At each step, N=8 subsystems (by default) each generate a tangent force
toward a preferred direction on the current manifold. Per-subsystem
*influence* is a function of alignment with the current state; *activity* is
influence attenuated by an exponential fatigue penalty and renormalized
across subsystems (softmax-like competition); *fatigue* accumulates
proportional to activity and decays proportional to inactivity, clamped to
`[0, 3]`. The resultant force drives the state along the manifold (great-circle
geodesic on S³/S², clipped Euclidean step on flat R⁴). *Clarity* is the
resultant force's magnitude — high when the system's competing subsystems
agree on a direction, low when they're deadlocked. A soft-then-hard
assignment to one of 24 macro-basins (via softmax similarity to fixed macro
centers, `beta_macro=11.375`) gives a discrete symbolic trajectory over
basins, used throughout for transition-graph and recovery-time measurements.

Three structural axes are independently swappable: **manifold** (S³, S²,
flat R⁴ — `s3`, `s2`, `flat4`), **topology** (cyclic-opponent pairs, random
pairs, fully-competitive, uncoupled — governing the preference-vector
geometry), and **fatigue dynamics** (gradual, winner-take-all, none,
stochastic). The canonical configuration (`s3`, `cyclic`, `gradual`) uses a
hand-calibrated preference matrix (`PREFERENCE_MATRIX_NORMED`, four
antiparallel opponent pairs at 135° separation); all other manifold/topology
combinations use a generic, uncalibrated preference generator
(`generate_preferences`). This calibration asymmetry is a known confound,
discussed in §5.2 and not fully resolved by this work.

### 2.2 Bearer state (`BearerEngine`)

We extend `UniversalEngine` with a persistent vector `b_t ∈ R^{n_sub}`:

```
influences_t  = base_influences_t + bearer_weight · b_t          (perception, biased by b_t)
activities_t  = normalize(influences_t · fatigue_decay + noise)   (decision)
b_{t+1}       = (1 − bearer_decay) · b_t + bearer_gain · activities_t + ξ_t   (incorporation)
```

with `bearer_weight=0.6`, `bearer_decay=0.04`, `bearer_gain=0.35` throughout.
`b_t` feeds forward into the next step's influences and is itself updated by
the current step's activities — a closed loop, not a passive log — which is
the property we mean by "constitutive": anything incorporated into `b_t`
keeps shaping perception/decision until it decays, rather than merely being
recorded. `ξ_t` is an optional external perturbation injected at a specific
step (used in §6).

**Reproducibility note:** early versions of every paired perturbed/control
comparison in this study shared Torch's single global RNG stream across the
two engines being stepped in an interleaved loop, which silently broke the
"control" from being a matched no-perturbation replay after the first step
(see §8.1). All results reported here use the corrected version, in which
every `BearerEngine` instance owns a private `torch.Generator` seeded
independently.

## 3. Bearer state and competency vector (`bearer_state_competency.py`)

We measure four quantities per (manifold, topology, bearer on/off, seed):
`memory_horizon` (steps until a one-shot bearer perturbation's effect on
basin occupancy decays below 1/e of peak), `lesion_recovery` (response to
zeroing one subsystem's preference row and force contribution at the run's
midpoint), `adaptation_speed` (response to a random orthogonal rotation of
the macro-basin centers — a genuine spatial "environment shift," not a
label permutation, which would leave basin-occupancy entropy invariant by
construction and test nothing), and `self_maintenance` (inverse coefficient
of variation of clarity over an undisturbed run).

`lesion_recovery` and `adaptation_speed` are each decomposed into three
sub-quantities rather than reported as a single recovery time, because a
recovery time of 0 is ambiguous between "nothing was disrupted" and
"disrupted and instantly recovered" — opposite findings that a single number
cannot distinguish:

- `d_immediate` — deficit averaged over the first 20 post-event steps, relative
  to a 150-step pre-event baseline.
- `c_lesion` / `c_adapt` — cumulative shortfall (clipped at zero, so overshoot
  above baseline doesn't cancel a real dip) over up to 600 post-event steps.
- `t_recovery` / `t_adapt` — steps until the metric returns within tolerance of
  baseline, reported as NaN (not 0) when `d_immediate` is below a fixed
  no-effect threshold, so "no measurable disruption" and "instant recovery"
  remain distinguishable in the data.

**Result (N=256, 1800 steps, 10 seeds, 3 manifolds × 2 topologies × bearer
on/off; `outputs/bearer_state_competency/`):** the canonical `s3+cyclic`
configuration is the only one showing large, slow-recovering disruption from
both lesion and environment-shift — `lesion_t_recovery≈899` and
`adapt_t_adapt≈718–900` steps of a 1800-step run — while every other tested
configuration shows near-zero deficit or recovers within 0–300 steps. This
motivated the rest of the study: is that S³/cyclic-specific, or is it a
property of the underlying transition that any configuration would show if
tested at the right operating point?

## 4. The transition is a manifold-independent saddle-node bifurcation, not critical slowing down

### 4.1 Direct sweep (`criticality_sweep.py`)

Two competing explanations for §3's result: (H1) something about S³ geometry
or the calibrated preference matrix specifically; (H2) `s3+cyclic`'s default
`fatigue_rate=0.217` happens to sit inside a previously-located critical
region (`outputs/phase_cartography/goldilocks_report.txt` places top-flourishing
configurations at `fatigue_rate≈0.20–0.27`), and critical slowing down —
diverging relaxation time near a genuine critical point — produces the large
recovery times regardless of manifold.

We swept `fatigue_rate ∈ {0.05,…,0.40}` (12 values) on both `s3/cyclic` and
`flat4/cyclic`, N=256, 10 seeds
(`outputs/criticality_sweep/criticality_summary.csv`). Both manifolds
transition at essentially the same location (~0.18–0.20), which rules out H1
as originally stated — the transition's existence and location are not S³-
or calibration-specific. But the *shape* rules out H2 as originally stated
too: true critical slowing down predicts a peak that rises and decays
symmetrically around the critical point. Instead, `s3`'s `lesion_t_recovery`
jumps from unmeasurable (below ~0.18) to a near-ceiling value (899.2 of a
900-step post-lesion window) at `fatigue_rate=0.217` and **stays pinned at
that ceiling for every value tested up to 0.40** — no decay on the far side.
`flat4` differs again: a single-point spike exactly at `fatigue_rate=0.18`
(523.6 steps), then an immediate return to near-instant recovery (0 steps)
for every value ≥0.20.

### 4.2 Cross-validation against an independent reduced-model analysis

An unrelated, pre-existing analysis in this repository
(`mechanism_extraction.py`, a deterministic eigenvalue-crossing bifurcation
study of a 2–3 subsystem circle skeleton with no manifold embedding at all)
classifies this transition as **saddle-node/transcritical** — a
discontinuous jump onto a new branch, which matches the step-to-ceiling
shape observed directly. This is a genuine independent qualitative
cross-validation. It is *not* quantitative agreement: the reduced skeleton's
own eigenvalue-crossing point is `fatigue_rate≈0.0224`, roughly an order of
magnitude below the full engine's ~0.18–0.20, unsurprising given how much
that skeleton omits (manifold embedding, exploration noise, bearer state,
macro-basin assignment). An earlier draft of this analysis incorrectly
reported numerical agreement (`fatigue_rate≈0.1816`) by conflating the
skeleton's own computed value with a hardcoded reference constant
(`FR_C`, sourced from an unrelated full-engine measurement in
`deep_analysis.py`) that the plotting code merely drew as a comparison line;
this is corrected here and in the repository documentation (§8.3).

**Refined conclusion:** the transition's existence and location are general
properties of competition+fatigue dynamics (also independently supported by
`mechanism_extraction.py`'s layered ablation, which finds no transition
exists with competition or fatigue alone — only their combination,
`outputs/mechanism/layered_ablation_results.json`), not S³- or
calibration-specific. What *is* geometry-dependent is what happens after
crossing it: `s3` falls into a regime it does not leave for any tested
`fatigue_rate`; `flat4` only shows the effect exactly at the boundary.

## 5. Isolating the mechanism behind post-bifurcation regime stability

### 5.1 Compactness does not explain it (`compact_vs_noncompact_bifurcation.py`)

We built a matched pair of reduced deterministic skeletons sharing one
analysis harness: a compact circle (S¹, wraparound state) — essentially
`mechanism_extraction.py`'s existing model, reimplemented for direct
comparison — and a non-compact line (R¹, clamped rather than wrapped,
Gaussian-bump alignment in place of cosine since periodicity has no
non-compact analog). Both were run through the same eigenvalue-crossing
bifurcation classification and forward/backward `fatigue_rate` hysteresis
sweep. **Result:** both skeletons show the identical bifurcation type
(saddle-node/transcritical) and *neither* shows hysteresis
(`hysteresis_gap≈0` for both, `outputs/compact_vs_noncompact/`). Manifold
compactness, at this level of reduction, explains neither the transition
type nor the S³-vs-flat4 asymmetry in §4.1.

### 5.2 The full engine shows real but transient hysteresis; the lesion deficit is a separate, near-permanent phenomenon (`hysteresis_test.py`)

Since the reduced skeleton lacks a bearer state and lacks lesioning
(subsystem removal, as opposed to a smooth parameter sweep), we tested the
full `BearerEngine` directly, in two parts.

**Permanence.** We extended the post-lesion observation window from §4.1's
900–1500 steps to 6000, at `s3/cyclic`'s default `fatigue_rate=0.217`
(N=256, 10 seeds). **70% of runs had still not recovered 3000 steps
post-lesion** (`outputs/hysteresis_test/permanence_per_seed.csv`) — the
deficit is not merely slow on the original timescale, it is close to
permanent on a 3–6× longer one.

**Path-dependence.** Three histories were compared at the *same*
`fatigue_rate=0.30`: `direct` (constructed there from t=0), `ramp_up`
(started at 0.05, linearly ramped up through the transition), `ramp_down`
(started at 0.45, ramped down). All three are lesioned at the moment they
reach the target rate. **Result:** system state at the moment of lesion
differs sharply and consistently by history despite the identical
parameter value — mean clarity 0.109±0.004 (`ramp_up`), 0.241±0.008
(`direct`), 0.296±0.009 (`ramp_down`); one-way ANOVA F=1795, p=1.9×10⁻²⁹
(`outputs/hysteresis_test/path_dependence_summary.csv`). This is genuine
hysteresis, and it is not present in the reduced skeleton (§5.1) — the
natural candidate mechanism is the bearer-state feedback loop, which the
2–3 variable skeleton has no analog of.

Critically, this hysteresis is **transient**: a first version of this test
used a 300-step settling period between reaching the target rate and
lesioning, and found *zero* path-dependence. Direct inspection of engine
state (`fatigue.mean()`, `u_t`) immediately after the ramp versus after 300
further steps showed why — the fatigue variable saturates at its clamp
ceiling within a few hundred steps regardless of path, erasing the
history-dependent state before the settling window even completes. The
corrected test lesions immediately at ramp completion, while the transient
divergence still exists.

**These are two distinct phenomena, not one.** The pre-lesion
path-dependence is real but decays within roughly 300 steps at constant
`fatigue_rate`. The post-lesion deficit does not decay within 3000 steps.
The latter is not explained by the former, and remains open — the working
hypothesis (untested here) is that the lesioned subsystem's contribution may
be architecturally required for the post-bifurcation fixed point, making
full recovery structurally unavailable rather than merely slow (§8).

## 6. Perturbation shape, not entropy source, drives developmental capture (`qrng_developmental_capture.py`, `perturbation_concentration_sweep.py`)

Separately from the bifurcation work, we asked whether a one-shot
perturbation injected into `b_t` (see §2.2's `ξ_t`) persists longer or
shorter depending on its source: a fixed deterministic vector concentrated
on one subsystem, a seeded pseudo-random (PRNG) vector spread across all
subsystems, or an OS-CSPRNG-sourced vector (`os.urandom`, since no captured
hardware-QRNG data exists in this repository — this condition is explicitly
*not* a hardware quantum-random-number-generator measurement, and is labeled
`os_csprng` rather than `qrng` throughout the code to avoid overstating it).
`DC(Δ)` — the persistence horizon of the perturbation's effect on basin
occupancy — was measured across all three.

A first version of this comparison found a large, apparently significant
deterministic-vs-random effect. This measurement shared the RNG-stream
artifact described in §8.1: paired perturbed/control engines were silently
diverging for reasons unrelated to the injected perturbation. After the fix
(§8.1), the deterministic-vs-distributed effect persisted (N=256, 1800
steps, 16 seeds; e.g. `flat4`: deterministic vs. prng, p=0.0048) but the
*prng-vs-os_csprng* comparison — same perturbation shape, different entropy
source — did not (`s3` p=0.033, `flat4` p=0.74; the `s3` result does not
survive Bonferroni correction across the ~6 comparisons run in this study
and is not treated as a finding). A dedicated equivalence test (TOST,
bound=±10 steps) could not establish equivalence either
(`outputs/qrng_developmental_capture/source_comparison.json`), so this
remains "no detected effect," not "proven equivalent."

`perturbation_concentration_sweep.py` isolated the remaining variable
directly: perturbations interpolated between fully concentrated (one-hot)
and fully distributed (random direction) at matched magnitude, parametrized
by `κ = ||δ||₁² / (n·||δ||₂²)` (low κ = concentrated, κ→1 = distributed).
`DC(Δ)` horizon correlates with κ on both manifolds (`s3`: r=0.378,
p=0.0005; `flat4`: r=0.277, p=0.013; n=80 each,
`outputs/perturbation_concentration/concentration_summary.json`),
confirming the deterministic-vs-random effect is a perturbation-*shape*
effect, not an entropy-source effect.

## 7. Recovering manifold geometry from transition-graph combinatorics alone (`positive_geometry_readout.py`, `emergent_geometry_readout.py`)

### 7.1 Framing

Given a basin-transition graph — which macro-basins the system transitions
between and how often, with associated clarity statistics
(`basin_gateway_analysis.py`) — how much of the manifold's actual metric
structure can be recovered from the graph's combinatorics alone, with no
access to the coordinates that generated the dynamics?

We deliberately did not attempt to implement anything from the positive-
Grassmannian / amplituhedron literature (Arkani-Hamed & Trnka 2013;
Hoffman's proposed but unconstructed bridge from conscious-agent Markov
dynamics to that formalism) — no such construction exists to port in, and
attempting an ad hoc one risks producing something that looks like that
framework without being mathematically answerable to it. Instead we used a
narrower, rigorous, and *unrelated* piece of graph theory: the weighted
spanning-tree polytope of a graph, whose canonical form assigns each edge a
residue equal to `p_span(e) = w_e · R_e` (Foster's theorem — the probability
edge `e` appears in a weighted-uniform random spanning tree), where `R_e` is
effective resistance across `e`, computed via the Moore-Penrose pseudoinverse
of the graph Laplacian. This is legitimately "math deriving structure from
combinatorics alone" in a narrow, well-defined sense — `R_e` depends on the
entire graph's structure, not just edge `e`'s own weight — without invoking
anything not already established graph theory.

### 7.2 Does topology alone predict empirical significance? (`positive_geometry_readout.py`)

We asked whether `p_span(e)`, computed purely from transition-count
topology, predicts `basin_gateway_analysis.py`'s empirically-measured
`gateway_score(e)` (computed from clarity dynamics the graph construction
never saw). Because `gateway_score` already includes a traffic-volume
factor, we report both the raw correlation and the partial correlation
after regressing out `log(count+1)` from both sides, to isolate what global
graph structure explains beyond local traffic volume.

**Result:** on `flat4`, `p_span` predicts `gateway_score` beyond raw traffic
(partial r=0.292, p=5.4×10⁻⁸, n=334 edges). On the canonical `s3/cyclic`
geometry, it does not (partial r=0.010, p=0.89, n=222 edges). Topology alone
recovers a meaningful fraction of what the flat-manifold system treats as
significant; on the compact, calibrated geometry, significance appears to
be generated dynamically rather than being recoverable from static
structure.

### 7.3 Does the graph recover the true manifold metric and dimension? (`emergent_geometry_readout.py`)

Effective resistance has a known Euclidean interpretation: an embedding
exists (via the Laplacian pseudoinverse's eigenvectors, scaled by
√eigenvalue) in which `R_ij` is exactly squared Euclidean distance — the
resistive embedding. We treated `R` as a squared-distance matrix, applied
classical multidimensional scaling (double-centering), and compared the
result against the *true* geodesic (S³) or Euclidean (flat R⁴) distance
between the macro-basin centers that generated the dynamics, recomputed
deterministically and never fed to the graph construction.

**Result:** on `s3`, the graph's effective dimensionality (components
needed for 90% of the double-centered spectrum's variance) is 5 — close to
the true manifold dimension of 4 — but distance-reconstruction correlation
is weak (Pearson r=0.132–0.178, p<0.03). On `flat4`, effective
dimensionality is 16 (essentially no compression — the graph does not
"know" it is 4-dimensional) but distance correlation is markedly stronger
(r=0.343–0.426, p<10⁻⁸; `outputs/emergent_geometry/`). This mirrors §7.2's
asymmetry from a different angle: the two manifolds encode geometric/
semantic information in their transition graphs in qualitatively different
ways — `s3` compactly but noisily, `flat4` without compression but more
faithfully — not merely to different degrees of the same thing.

## 8. Methodological corrections made during this work

We report these because catching and correcting them is part of what makes
the surviving results trustworthy, and because the corrected numbers differ
substantially — sometimes qualitatively — from the initial, wrong ones.

**8.1 RNG-stream interleaving artifact.** Every paired perturbed/control
comparison in §3 and §6 initially built both engines with the same seed,
then stepped them in an interleaved loop (`pert.step(); ctrl.step()`
repeated). Since both engines drew from Torch's single global RNG stream,
`ctrl` consumed the *next* chunk of random draws relative to `pert` at every
step rather than a matched replay — silently breaking the counterfactual
comparison from the second step onward, independent of any injected
perturbation. This produced a spurious, apparently-significant
deterministic-vs-random-source effect in an early version of §6, and
inflated developmental-capture "horizon" values (200–480 steps) that
dropped to their correct range (20–100 steps) after the fix. Fixed by
giving every `BearerEngine` a private `torch.Generator`, seeded
independently, used for all step-time randomness.

**8.2 Symbolic-computation performance bug (§7.2 implementation).**
Computing the spanning-tree generating polynomial via SymPy with
`positive=True` symbol assumptions caused `.det()` to hang (>60s on a
6-node/15-edge subgraph, confirmed via minimal reproduction) due to
per-step assumption-resolution overhead in the default determinant
algorithm. Fixed by dropping the assumption and specifying the
division-free Berkowitz method explicitly (0.06s on the same input).

**8.3 Overstated cross-method numerical agreement (§4.2).** An earlier
draft of this analysis reported that the reduced-skeleton bifurcation
value and the empirically-measured transition location agreed to within
measurement resolution. They do not (0.0224 vs. ~0.18–0.20) — the earlier
draft conflated the skeleton's own computed eigenvalue-crossing point with
an unrelated hardcoded reference constant that the skeleton's own plotting
code merely draws as a comparison line. Only the qualitative bifurcation
*type* is supported by the cross-method comparison; the specific critical
value is not.

**8.4 Environment-shift metric validity (§3, `adaptation_speed`).** An
initial implementation of the "environment shift" perturbation permuted
macro-basin index labels rather than rotating basin *positions* in space.
Since occupancy-distribution entropy is invariant under a pure relabeling
by construction, this would have made `adaptation_speed` measure nothing,
regardless of the underlying dynamics. Caught before any results were
reported; fixed to apply a genuine random orthogonal rotation to basin
center coordinates.

## 9. Limitations

- The calibration confound flagged in §2.1 is only partially resolved.
  §4.1 shows the transition's *location* is not calibration-specific, but
  whether the calibrated preference matrix contributes to the
  post-bifurcation lock-in asymmetry (§4.1, §5) independent of manifold
  identity is untested. A matched-preference factorial — generating
  preference matrices for every manifold matched on the calibrated
  matrix's statistical properties (pairwise-angle distribution, norm,
  entropy) rather than reusing the hand-tuned matrix only for `s3` — would
  resolve this and is not yet built.
- All manifold comparisons use only two of the three available manifolds
  (`s3`, `flat4`) in the bifurcation/hysteresis work; `s2` is available in
  the engine but untested in §4–§5.
- The lesion protocol zeroes a single fixed subsystem (index 0). Whether
  the near-permanent deficit (§5.2) depends on which subsystem is lesioned,
  or on lesioning multiple subsystems, is untested.
- The `os_csprng` condition in §6 is not a hardware QRNG measurement; no
  real captured quantum-random bitstream exists in this repository at the
  time of writing.
- §7's graph-combinatorics analyses use a single simulation run per
  manifold (2000 steps, N=128 beings) to construct each transition graph;
  robustness across independent runs/seeds is untested.
- Sample sizes for statistical tests are moderate (10–16 seeds per
  condition in most experiments); several reported p-values, while
  individually below conventional thresholds, have not been corrected for
  the full multiple-comparisons burden of the overall study (see §6's
  explicit exception, where we do apply this reasoning).

## 10. Future Work

1. Direct test of the "architecturally required" hypothesis for the
   near-permanent lesion deficit (§5.2): does the lesioned subsystem's
   contribution enter the post-bifurcation fixed point's defining equations
   in a way that makes full recovery to baseline clarity structurally
   unavailable, as opposed to merely slow? Testable by comparing the
   deterministic skeleton's fixed-point clarity with and without the
   lesioned term, at `fatigue_rate` values past the transition.
2. Matched-preference factorial (§9) to fully close the calibration
   confound, reusing `topology_dissection.py`'s existing angle/structure/
   pair-count decomposition as a source of controlled preference variants.
3. Causal bearer-state probe: rather than inferring "memory" purely from
   decay-time metrics (§3's `memory_horizon`, which is dominated by the
   fixed `bearer_decay` hyperparameter), decode a task-relevant variable
   from `b_t` and test whether ablating it collapses task performance that
   requires it — decodability without behavioral consequence is not the
   same claim as causal necessity.
4. Extend §7's graph-combinatorics comparison with a model that predicts
   `gateway_score` from genuinely independent dynamic features (not
   features that are algebraic components of `gateway_score`'s own
   definition), to properly separate "predictable from topology" from
   "predictable from dynamics" as ChatGPT-provided review feedback on an
   earlier draft of this work correctly pointed out was not yet done
   rigorously.
5. Real hardware-QRNG capture data, to test the entropy-source question in
   §6 as originally intended rather than via an OS-CSPRNG stand-in.

## Acknowledgments

The boundary-negotiation framing this work builds on is credited to
Observer Patch Holography by FloatingPragma
(https://github.com/FloatingPragma/observer-patch-holography). An earlier
round of review feedback (from a ChatGPT session, reproduced and addressed
in this repository's commit history) correctly identified several
methodological issues addressed in §8 and §9 before they were caught
independently.

## Appendix A: Why the amplituhedron / positive-Grassmannian formalism was not implemented

The original motivating discussion for this thread referenced Hoffman's
proposal that conscious-agent network dynamics (formalized as Markov
kernels for perception/decision/action) might map onto decorated
permutations, the positive Grassmannian, and amplituhedron facets. This is,
by Hoffman's own framing, a conjectured bridge, not a constructed
mathematical result — there is no existing derivation to implement, adapt,
or test against. Attempting to build "an amplituhedron for this system"
without such a derivation would produce a construction with the vocabulary
of that formalism but no actual mathematical relationship to it, which
would be worse than not attempting it. §7's use of effective resistance and
the spanning-tree polytope is offered as the closest *rigorous* analog
available — a genuine (if much simpler) positive-geometry object with an
established canonical form — not as a substitute for the unconstructed
bridge.
