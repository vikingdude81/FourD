# Broken symmetry as a hostability condition for boundary-negotiation dynamics
### Tegmark bridge — full experiment record (S2, T1, T2), 2026-11-19

Status: working paper record for the `tegmark_bridge/` lane of the FourD program.
All numbers below are reproduced by the scripts in this directory; raw JSON in
`outputs/`. Figure: `outputs/t12_phase_maps.png`.

---

## 1. Framing

Tegmark's Mathematical Universe Hypothesis (arXiv:0704.0646): every
(mathematically consistent, restricted-to-decidable) structure physically
exists; observers are "self-aware substructures" (SAS) within some of them.
The standing objections our lane attacks: the measure problem (how are
structures weighted — Schmidhuber quant-ph/0011122 says by description length;
Müller 2412.02826 by algorithmic probability over observer-moments) and
unfalsifiability (Natal 2411.12562).

Our claim, now backed by three experiments: **structure existence-qua-hospitable
is a computable, falsifiable property.** A dynamical signature we call the
*boundary-negotiation signal* (agent collectives forming persistent basin
boundaries: transition events at which the population's basin assignment flips,
with post-transition preference-attachment clarity significantly above a
label-shuffled null — the canonical FourD readout `boundary_metrics`) is
**annihilated by exact central symmetry** and lives only in a window of broken
symmetry.

Literature file: `lit/REPORT.md` (parent-audited 2026-11-19; 11 indexed papers,
8 PDFs, 9 full-text extracts).

## 2. Protocol (identical across S2/T1/T2)

- Engine: FourD `UniversalEngine` (repo root `universality_test.py`), N=128
  agents on S³ manifold, steps=1000, gradual fatigue. The engine's "topology"
  enters only through the **preference substrate** P: a set of unit direction
  vectors (subsystems). Baseline FourD substrate: cyclic-8 simplex.
- Substrate substitutions are exact vertex sets of polytopes (unit
  circumradius) built in `vfd_bridge/graphs.py`, plus controls.
- Readout: `boundary_metrics(eng, n_shuffles=50)` →
  `null_z` (z-score of post-transition clarity vs shuffled null),
  `cohens_d`, `edge_clarity_r`, `transition_rate` (TR).
- Seeds: 3 per variant (torch+numpy), GPU (`cuda:0`, verified in output JSON).

## 3. S2 — the founding result (vfd_bridge, 2026-11-19 earlier today)

Substrates: full 24-cell (24 vertices), full 600-cell (120 vertices), count-
matched random sets, `antipodal12` (12 random dirs + their exact negations),
and `poly600_s8` (8-vertex subsample).

- Full polytopes: signal dead (null_z +0.2/−0.2, d≈0).
- Random 24/120: alive (z≈+16). → not subsystem count.
- **antipodal12: dead (z=+0.4).** → the mechanism is central symmetry
  (P = −P ⇒ exact pairwise force cancellation), not H4-specific structure.
- poly600_s8: alive (z≈+15).

## 4. T1 — hostability dial (`t1_sas_census.py`)

Random subsample of k of the 120 600-cell vertices, k = 4…120, 3 seeds.
The natural order parameter is the **antipodal-closure fraction** of the
subsample (fraction of rows whose exact negation is also present): it grows
monotonically with k and hits 1.0 only at k=120.

| k | antipodal_frac | null_z |
|---|---|---|
| 8–100 | 0.17–0.83 | +7.5 … +44.8 (alive; k=6 marginal, one seed negative) |
| 110 | 0.92 | +2.9 (dying) |
| 118 | 0.98 | +0.2 |
| 120 | 1.00 | +0.1 (dead) |

The collapse tracks antipodal closure, not size: healthy at 0.83, dying at
0.92–0.98, dead at 1.00.

**RETRACTION (self-audit same day):** T1 and S2 also reported a
"seed-sensitivity/history-bearing" criterion C2 (TR std across seeds > 0.01).
**C2 is confounded and withdrawn:** the subsample/random substrate was
re-drawn per seed, so TR variance across "seeds" measured *structure*
variance, not dynamical history-sensitivity. T2 (fixed structure per variant)
shows TR_std ≈ 0.001–0.002 for *every* structure, alive or dead — TR at this
scale is not a history measure. The null_z evidence above is unaffected (each
z is measured on a fixed structure). Any history claim must come from a proper
T3 design (fixed substrate, varied agent initializations; agreement statistics
of basin *identity*, not event rates).

## 5. T2 — mechanism pinning (`t2_mechanism.py`)

### Exp A — ε-restore on the dead 600-cell

prefs = V + ε·u (u fixed random direction), renormalized. ε=0 is the dead
substrate.

| ε | null_z | TR |
|---|---|---|
| 0 | +0.0 | 0.365 |
| 0.003 | −0.0 | 0.406 |
| 0.01 | +0.3 | 0.511 |
| 0.03 | +2.5 | 0.696 |
| **0.1** | **+17.7** | 0.999 |
| 0.3 | −0.0 | 1.000 |
| 1.0 | +0.6 | 1.000 |

Findings: (i) the signal **restores** from exact-cancellation death — the
mechanism is causal, not correlational; (ii) restoring the symmetry-breaking
is *necessary but not sufficient*: tiny ε stays dead (net induced polarity
below the dynamical noise floor), and large ε **boils** the system (TR→1.0:
one basin dominates, every agent pinned, no boundaries left to negotiate).
The bearer regime is a **Goldilocks window of net preference polarity** —
neither cancellation nor saturation. The full 600-cell at ε≈0.1 with z=+17.7
shows a maximal-symmetry structure is fully *recoverable* to hospitality with
a 10% symmetry-breaking field.

### Exp B — family zoo (3-D polytopes in an R⁴ hyperplane + 4-D hypercube)

| structure | antipodal? | null_z |
|---|---|---|
| octahedron | closed | +1.6 |
| cube-8 | closed | +1.3 |
| icosahedron | closed | −0.7 |
| dodecahedron-20 | closed | −1.7 |
| hypercube-16 {±1}⁴ | closed | +0.3 |
| **dodeca20 − 1 vertex** | one orphan antipode | **+36.8** |
| octa − 1 | orphan | +5.4 |
| cube8 − 1, icosa − 1, hypercube − 1 | orphan | +0.2, +0.8, +0.5 |
| tetrahedron | not closed (zero net sum) | +2.8 |

Findings: (i) **every centrally-closed polytope across families is dead** —
cross-dimensional confirmation of S2's antipodal12 pin; (ii) breaking closure
by a *single* orphaned antipode can fully restore the signal (dodeca−1 →
z=+36.8), but only when the orphan opens a *directional gap* against the rest
of the set — cube−1/icosa−1 stay dead (one orphan among few, or near-uniform
coverage → induced polarity/gap geometry below threshold). (iii) tetrahedron:
not centrally symmetric but zero net polarity; marginally dead (z=2.8).
Net polarity alone is not the full order parameter — **gap geometry** (where
the missing directions are relative to present ones) participates.

## 5.5 D1/D1b — mechanism autopsy (`d1_autopsy.py`, `d1b_fatigue.py`)

Why are centrally-closed substrates dead? Three measurements, one
confirmed mechanism, one killed hypothesis.

**(a) Attention deadlock (confirmed — the general mechanism).** After 1000
steps, accumulated per-subsystem fatigue: 600-cell and antipodal12 =
**exactly 0.00** (all agents, all subsystems); alive substrates (600cell_s8,
cyclic8) = up to 2.97. Reading the update rule:
`F += rate·a + 0.03·excess(a) − recovery·(1−a)`, so F stays pinned at 0
while attention `a_j` never crosses a concentration threshold
(≈ recovery/(rate+recovery)). In an antipodally closed set with affine
influence (0.5+0.3 cos), every attractor pull has a matched repeller of
equal magnitude, so expected attention is exactly uniform — **no subsystem
can win attention long enough to accumulate fatigue**. Fatigue is the fuel
of commitment rotation; without it, agents relax toward fixed axes and
transitions become pure noise crossings: TR seed-locked at 0.365,
post-transition clarity at chance (z≈0). Dead does not mean motionless —
it means unfueled.

**(b) Zero-force certificate (confirmed — sharper for true designs).** The
second-moment matrix M2 = Σ p pᵀ is isotropic **exactly** for the 24-cell
(dev2 = 0.0) and 600-cell (dev2 = 3×10⁻¹⁸): both are exact spherical
2-designs. Under uniform attention (from (a)), the net tangential preference
force is (M2u − u(uᵀM2u))/n, which **vanishes identically for every agent
configuration** when M2 ∝ I. So the full 24-/600-cell substrates are
provably force-free, not just empirically quiet. antipodal12 has dev2 =
0.46 (anisotropic) yet still dies → deadlock (a) is the general mechanism;
the zero-force theorem is the special knife-edge case. (dev4 values in
`d1_autopsy.json` carry Monte-Carlo error from the rotation-average
estimator and are indicative only — do not cite.)

**(c) Mirror-bounce hypothesis (KILLED).** Pre-registered expectation: dead
systems' transitions are basin↔antipodal-basin bounces. Measured bounce
fraction on s3: 0.000–0.003 everywhere, alive and dead. Withdrawn. (Odd
side-observation: on flat4 dodeca-family bounce ≈ 0.67 with z=+21 —
transitions there ARE mostly mirror-pair crossings yet the system is alive;
flagged for T4, not explained.)

**(d) Cross-engine map (flat4 vs s3, same substrates).** 600-cell dead in
BOTH engines (z +0.9 / +0.2) — engine-robust, as the theorems predict.
dodeca20: dead on s3 (z −1.6) but ALIVE on flat4 (+21.2) → that death is
engine/embedding-conditioned, not structural. dodeca20−1: alive in both
(+39.5 / +23.4) — the single-gap restoration is the most robust positive
result in the program.



## 6. Consolidated findings (what we now assert)

- **F1 (robust, cross-family):** exact central symmetry of the preference
  substrate annihilates the boundary-negotiation signal. Sufficient condition
  for death; verified on 600-cell, 24-cell, random antipodal pairs, octa,
  cube, icosa, dodeca, hypercube.
- **F2 (robust):** death is reversible and causal — symmetry-breaking field
  restores; the alive region is a window of net polarity (Goldilocks between
  cancellation and single-basin saturation), with the dead symmetric state a
  knife-edge at one end.
- **F3 (new, one-instance):** single-vertex gap opening (dodeca−1) suffices
  for full restoration in dense direction sets; whether it fires depends on
  gap geometry, not count.
- **F4 (mechanism, confirmed):** attention deadlock — antipodally closed
  substrates hold attention exactly uniform, so fatigue (commitment fuel)
  accumulates at exactly zero; without fatigue there is no commitment
  rotation, only noise crossings. Dead = unfueled, not motionless.
- **F5 (certificate):** 24-cell and 600-cell are exact spherical 2-designs
  (M2 isotropic to machine precision) ⇒ provably vanishing tangential
  preference force under uniform attention. Computable death certificate for
  any candidate structure: check central symmetry, then M2 isotropy.
- **F6 (engine map):** 600-cell death is engine-robust (s3 + flat4);
  dodeca20 death is engine-conditioned (alive on flat4); single-gap
  restoration (dodeca−1) is robust in both. Order-parameter status differs
  per structure — the certificate (F5) is the part that transfers.
- **R1 (withdrawn):** the S2/T1 "seed-deafness = no history" criterion C2 —
  confounded by per-seed structure re-draw; see §4.
- **R2 (withdrawn):** mirror-bounce hypothesis for dead-system transitions —
  measured bounce fraction ≈ 0 on s3 for all structures; see §5.5(c).

## 6.5 Provability ladder (what is claimable at each grade)

**Theorem grade (clean math, third-party checkable):**
- T-A: 24-cell/600-cell are exact spherical 2-designs; M2 = (n/4)I
  (verified to 3×10⁻¹⁸). Under uniform attention the net tangential
  preference force is identically zero for every configuration.
- T-B: in an antipodally closed substrate with affine influence
  0.5+0.3(u·p), the attention-weight multiset pairs exactly
  (w, mirrored), bounding attainable single-subsystem concentration by a
  factor ~4× the uniform share; with recovery-biased fatigue this pins
  F ≡ 0 whenever 4/n_sub·Z < threshold — observed exactly.
- Corollary (death certificate): any structure that is (i) centrally closed
  and (ii) n_sub above the concentration bound is force-free and
  fatigue-dead for this engine family. Computable on any candidate set in
  O(n·d²).

**Empirical grade (reproducible at fixed seeds, commands in §9):**
- T1 collapse dial, T2 restore window, family zoo, D1 autopsy tables.

**Prediction grade (falsifiable by outsiders):**
- Any attention+fatigue collective with symmetric substrate pairing should
  show the fatigue deadlock; the T4 phase boundary should show a boiling
  edge at TR→1; LPF-style market substrates (non-antipodal by inspection)
  should pass the certificate.

**NOT claimable yet:** history-bearing status (R1 pending T3), universality
beyond two manifold engines, anything market-empirical until the
certificate is actually run on LPF substrates (T6).

## 7. What this does to the Tegmark program

Tegmark: all structures exist, weight unknown (measure problem). Our result
carves the ensemble: **inhospitable ⊃ centrally-symmetric loci**, and this is
checkable by direct computation on any candidate structure — Natal's
unfalsifiability objection loses force against *this* part of MUH. It also
discriminates between measure proposals: uniform weighting is wrong (there are
provably dead structures); description-length (Schmidhuber) and
algorithmic-probability (Müller) measures don't currently know about the
symmetry axis at all. "Hospitable iff broken symmetry (with gap geometry)" is
a candidate **support condition** any measure must respect. Dual endpoint from
the VFD lane (vfd_bridge S1): designed-null structure lives AT maximal
symmetry; bearer/boundary structure lives at BROKEN symmetry — two ends of the
Scaling Law of Symmetry, now both measured.

## 8. Next experiments (queued, in order)

- **T3 — proper history test:** fixed substrate, vary only agent init/seeds;
  measure agreement of basin *identity* across runs (mutual information,
  path-space overlap of transition sequences). This is where history-bearing
  state is actually tested; the null is T2's finding that event *rates* are
  seed-deaf everywhere.
- **T4 — polarity–gap phase diagram:** interpolate continuously between
  cancellation (antipodal pairs) and saturation (all-aligned) along two axes
  (net polarity, isotropy/gap structure); locate the alive region boundary;
  expect a first-order-like edge at boiling (TR→1) and a soft edge at
  cancellation.
- **T5 — measure probe:** description-length (LZ on quantized vertex sets) vs
  hostability across a structure zoo — direct empirical comparison of our
  support condition against Schmidhuber-style measures.
- **T6 — bearer readout upgrade:** if T3 shows basin-identity persistence on
  windowed substrates, port the same test to the market-graph lane (LPF basin
  structure is empirically non-antipodal — candidate hospitable structure).

## 9. Reproduction

```
cd FourD
.venv/Scripts/python.exe vfd_bridge/verify_vfd_spectrum.py     # VFD table check
.venv/Scripts/python.exe vfd_bridge/s2_substrate_control.py    # S2
.venv/Scripts/python.exe tegmark_bridge/t1_sas_census.py       # T1 (see R1 caveat)
.venv/Scripts/python.exe tegmark_bridge/t2_mechanism.py        # T2
.venv/Scripts/python.exe tegmark_bridge/d1_autopsy.py          # D1 (isotropy + forensics)
.venv/Scripts/python.exe tegmark_bridge/d1b_fatigue.py         # D1b (fatigue deadlock)
```

## 10. Limitations

- One engine family; 'substrate' = preference geometry, not Laplacian-coupled
  node dynamics (that variant remains open and is the honest upgrade path
  toward VFD-style physics claims).
- 3 seeds per variant; boundary_metrics null uses label shuffling (50).
- Polytope vertices enter the engine as directions only — edge structure of
  the polytopes is unused here (S1 covers the edge/spectral side).
- Gap-geometry dependence (F3) is single-instance (dodeca); T4 must map it.
- C1 threshold (z>3) is conventional, not derived.

## 5.6 T6 — certificate on real market structure (LPF bridge, first shot)

Substrate: 504 daily log-return direction vectors, R^4 basket
(SPY/SMH/IWM/MU), local parquet cache, engine s3/N=128/1000 steps, 3 seeds.

Certificates (structure only):
- market_full: antiFrac=0.133, dev2=0.695, top-PC conc=0.652
- market_60: antiFrac=0.033 — passes the closure certificate easily
- market_60_closed (P+-P control): antiFrac=1.0

Engine states (state = HOSTS if z>3 and tr<0.9; BOILS if tr>=0.9):
- market_60:         z=+109.1 ±154, tr=1.000 -> BOILS (basin jitter)
- market_60_closed:  z=-0.9  ±0.5,  tr=0.351 -> DEAD  (closure kills market data too — certificate causal)
- market_resid_60 (SPY-beta stripped): z=+2.5, tr=1.000 -> BOILS
- market_white_60 (M2^-1/2 whitened):  z=-0.3, tr=1.000 -> BOILS
- protos_k8/16/24 (k-means regime prototypes): z=+2.3/-3.5/-12.1, tr 0.38-0.54 -> DEAD/negative
- random_60_ctrl:    z=+6.0 ±1.0, tr=0.749 -> HOSTS

Reading: (1) the certificate is **causal on empirical data** — antipodally
closing the exact same market sample flips it from active to dead.
(2) But raw market direction sets fail hostability for a *different*
reason than polytopes: cone concentration (conc~0.65) => near-degenerate
macro-sim distances => basin identity jitters every step (tr=1.0), and
whitening/residualizing/prototyping did not restore Goldilocks dynamics.
T6 v1 verdict: **null for the naive bridge**; closure is necessary, not
sufficient. Geometry that survives so far is well-spread open sets
(cyclic8, random, dodeca-1). Next T6b candidate: GEX/strike-day
option-landscape substrates (atlas_index) instead of return directions —
those are LPF's actual preference geometry.

## 5.7 N7 — cyclic orbits vs the certificate (Karkada et al. bridge)

Prompt: X-thread on arXiv:2602.15029 (Karkada/Korchinski/Nava/Wyart/Bahri):
translation symmetry in language co-occurrence statistics provably forces
Fourier geometry (month circles, rippled timelines, lat/long maps) even in
word2vec. Same representation-theory family as VFD/S1 (eigenmodes of a
symmetry-commuting operator) but constructive sign: symmetry of data
statistics *builds* geometry in learned representations.

Probe (outputs/ngon_certificate.json): regular Z_n orbits as FourD
preference substrates, n = 7,8,11,12,13,24 x jitter {0, .1, .25},
3 seeds x 1000 steps:
- NO cyclic orbit hosts (no z>3 anywhere).
- coplanar n-gons (jit=0): dead or anti (rank-2 degeneracy axis).
- full-rank jittered: still dead; n>=12 flips strongly ANTI (z to -14):
  post-transition clarity *below* chance.

Conclusion: the order parameter is not parity/antipodal closure alone but
**uniform closed coverage**: every direction answered at equal distance
(local force balance). Antipodal closure is the n=2 special case; the
month circle is the continuous-rotation case. Both cancel the net force;
negotiation needs a *gap*, not an orbit. Beautiful duality with the paper:
symmetry of statistics -> eigenmodes in the representation (gradient flow
seeks the symmetric solution); symmetry of the substrate -> no fuel for
agents (symmetric fixed point is absorbing). Same Schur coin, opposite sign.
Registered 2602.15029 + follow-up 2610.00647 in vfd_bridge/lit (PDFs on disk).

## 5.8 CERT2 — two geometry scalars, one 19-substrate corpus

Pure-geometry predictors vs freshly recomputed engine fate (3 seeds, 1000
steps, s3; outputs/cert2_corpus.json, cert2_scatter.png):

  Brel = ||net tangential force|| / gross force under uniform attention (fuel)
  conc = M2 top-eigenvalue share (cone congestion); drel = dmin/dmed (spacing)

Findings:
- **Brel = 0 exactly on every antipodally-closed or zero-sum-balanced set**
  (600cell, 24cell, dodeca20, anti12, ngon7/12, market60_closed) and 8/9
  of them are DEAD/ANTI. The one escape: **tetra (n=4) HOSTS at Brel=0** —
  with only 4 prefs, attention noise breaks the balance immediately. The
  theorem's force-cancellation kills dynamics only when balance is
  *many-body-pinned* (antipodal pairs / large designs), not when it is a
  fragile n=4 zero-sum. Certificate needs the pairing condition too.
- **Jitter axis is real and opposite**: market60 (conc .65, drel .025) and
  eps600_0.1 boil at tr≈0.95-1.0 — fuel is abundant there (Brel .08-.14);
  they die from basin degeneracy, not force cancellation.
- All HOSTS sit in the middle band: Brel in [.05,.22] with resolution
  intact (dodeca19, sub600_8) or near-random spacing (random60).
- ANTI remains a third, unsolved mode (protos24 z=-12, ngon12j z=-13.5):
  fuel present, resolution present, and transitions still anti-correlate
  with clarity. Open problem for CERT3.

Honest verdict: Brel(=0)+pairing is a proof-grade DEATH certificate;
conc+tr is an empirical BOIL certificate; neither yet a sufficiency
certificate for hosting. Two failure modes, two axes, one scatter.

## 5.9 E1 — entity watch: birth/death of substructures, directly observed

4000-step runs, basin-occupancy time series (outputs/e1_entities.json/png):

- cell600 / dodeca20 (closed): 0 entity births (occupancy never crosses
  20%), leader flips every 1.7 steps, occupancy entropy 2.98-3.02 (~max
  ln 24). Death signature = pure flicker, nothing ever born.
- sub600_8 (host): 8731 births, entity lives mean 2.8 / max 38 steps,
  entropy 1.34 -> a real ecology: entities are born, live, die.
- dodeca19 (strong host): ONE immortal dominant basin (0 leader changes
  in 4000 steps), entropy 0.79. Strong gap -> crystallization into a
  single persistent entity, not an ecology.
- random60 at long horizon: tr=0.98, z=-32.6 -> it boils at 4000 steps
  despite z=+6 at 1000. HORIZON MATTERS: 'HOSTS' labels are
  horizon-conditioned; logged as caveat for all short-run states.

Taxonomy emerges: closed=no births; Goldilocks=entity ecology (finite
lifespans); strong-gap=one immortal entity (order without negotiation).
'dies and resurrects' is now directly observable as occupancy events.

T7 (selection for hostability, evolving substrates) queued in background.

## 5.10 T7 — selection for hostability: what geometry does selection want?

Evolution over preference multisets (pop 12, 16 gens, fitness = engine z
with boil penalty, single fitness seed; closure mutation deliberately
available). Trace: outputs/t7_evolve.json.

- Fitness climbed 23.8 -> 45.5 and plateaued at gen 8.
- **Selection never kept symmetry**: antiFrac transiently 0.118 at gen 2,
  pruned to 0.0 by gen 5. Closure mutations are available and REJECTED.
  CERT2's axis is causally right: hostability-selection avoids closure.
- **Fuel was selected upward**: Brel 0.31 -> 0.40, the highest in any
  known substrate (dodeca19 is 0.052). Points pruned 16 -> 14.
- **Novel motif discovered by selection: duplicated preferences.**
  Final champion (n=14) has one exact duplicate pair (drel=0). Held-out
  validation (3 fresh seeds): z=+56.2 sd 2.2, tr=0.534 -> genuinely
  HOSTS, stronger than dodeca19 at equal horizon. Removing the duplicate:
  z stays +55.1 but tr jumps 0.534 -> 0.840 (boiling edge). The duplicate
  is a *stabilizer*: repeated preference = weighted attractor.
- Consequence: the substrate is a MULTISET, not a set. Hostability
  depends on multiplicities (weights), not just point positions.
  CERT2 scalars should be computed with multiplicity weights.
Caveats: one evolution run, pop 12; fitness single-seed (but champion
validated held-out). Plateau suggests diminishing returns at this budget.

## 5.11 E2 — the immortal entity is a structure property (18/18)

dodeca19, 4000 steps, 12 seeds x N=128 + 3 seeds x N=64 + 3 x N=256:
- Winner basin identical in 18/18 runs: basin 15, mean share 0.807-0.810
  of all agents, entropy 0.75-0.81, tr~0.263, z 45..125 (all HOST-grade).
- Strict 'zero leader changes' 10/18; the other 8 had exactly ONE flip
  (max_life 3999/4000) — a one-step initialization transient. Effective
  permanence: 18/18.
- Seed-deaf AND density-deaf (N 64->256 changes share by <0.005).
- Mechanism mapped: basin 15's center cos=0.978 to preference vertex 2,
  and cos=-0.706 to the DELETED vertex. The gap does not crown its own
  shadow — it kills the ANTIPODAL RIVAL of its neighbor, and that
  neighbor's basin inherits the world. Gap-as-rival-removal -> attractor
  monoculture. (Russell-zone reading: delete a repeller node, the attractor
  facing it takes the flow.)

## 5.12 T6b — market through the multiset door (weighted prototypes + entity watch)

Regime prototypes (kmeans on 504d SPY/SMH/IWM/MU directions) replicated
proportional to cluster occupancy (total n=60), engine s3 N=128 2000
steps x3 seeds. Outputs/t6b_weighted.json.

k    unw z          wtd z        wtd tr  wtd life_mean/max  wtd share
8    +4.7           +23.5        0.44    5.3 / 55           0.52
16   -4.7           -13.0        0.53    2.7 / 21           0.39
24   -20.0          -55.8 (ANTI) 0.29    1667 / 1999        0.79
raw60: z=+122 tr=1.0 (boil) life_mean 1.37, births 5020 = flicker-only

- Weighting CRYSTALLIZES: duplicates anchor a leader (share .14->.52 at
  k8, .58->.79 at k24; k24-wtd leader never dies in 2000 steps).
- But it does not rescue ANTI: k24 deepens (z -20 -> -55.8): the frozen
  leader makes off-leader transitions clarity-LOSING events. Crystallized
  market = immortal entity, anti-negotiation transitions.
- The genuine win: k=8 WTD HOSTS (z=+23.5, tr=0.44, real entity ecology
  mean life 5.3, max 55). Regime GRANULARITY was the hidden dial: coarse
  weighted regimes -> ecology; fine weighted regimes -> monoculture;
  raw daily directions -> flicker. Flicker->ecology->monoculture spans
  the full E1 taxonomy IN MARKET DATA.
- Brel fell under weighting (.06->.03) while z rose: fuel fraction is
  not monotone in hosting; resolution/weights interact. CERT3 note.

## 5.13 B1 — transplant test: the entity lives in the structure (decisive)

dodeca19 immortal entity, structure swapped to 19 random pts at t=1500,
restored at t=2500; B = bearer carried (u_t + fatigue preserved, fatigue
now mislabeling foreign prefs), C = bearer wiped (u_t + fatigue reset).
3 seeds each. Outputs/b1_transplant.json.

- AWAY phase: entity fully lost in BOTH (share15 = 0.000, a new leader
  arises from the random structure). Bearer continuity bought nothing.
- BACK phase: basin 15 re-crowns COMPLETELY in both — share 0.810
  (vs home 0.806-0.808), lead15_frac = 1.000 for the entire post-warmup
  window (<=100 steps to full recrown, zero later flips).
- B and C are numerically identical on every metric: the bearer's
  preserved history (including fatigue misattached to wrong preferences)
  has ZERO measurable effect on entity identity.
- Verdict: entity = fixed point of (structure x engine). The bearer does
  NOT carry entity identity across structure change. The bearer wall's
  slogan tested in the negative direction: change the structure and the
  entity changes, no matter what the bearer retains. History enters
  identity via SEDIMENTED STRUCTURE (T7 weights = crystallized history),
  not via agent state.
- Market reading: regime identity should live in the sedimented landscape
  (weighted attractors / king-node geometry), not in transient participant
  state. Regime survives participant turnover; regime dies when the
  landscape deforms.
- CAVEAT (bug, recorded): the mode-A guard was missing — 'A' rows also
  swapped structure (they duplicate B). True no-swap control is E2
  (18/18 home share 0.807-0.81), which the back-phase matches exactly.
  B-vs-C comparison (the actual dissociation) is unaffected.
- Follow-up candidate: graded transplant — rotate prefs by angle theta,
  plot recrown-time and entity-identity vs structural deformation;
  gives a quantitative identity-survival curve.

## 5.14 B2 — graded transplant: identity radius and translating entities

dodeca19 prefs rigidly rotated by theta in a fixed 2-plane at t=1500
(planes/rotation identical across seeds; basin centers fixed in space).
3 seeds x 8 angles. Outputs/b2_graded.json.

theta  away_share15  away_leader  recrown15
0      0.810         15           immediate (control clean)
15     0.688         15           immediate
30     0.384         15           167-212 steps
45     0.106          8           never
60     0.012          8           never
90     0.000          8           never
135    0.000          6           never
180    0.000          6           never

- Entity RIDES small deformations: leader stays basin 15 through 30 deg,
  share decaying gracefully (0.81 -> 0.69 -> 0.38). IDENTITY RADIUS
  between 30 and 45 degrees; the cliff is sharp.
- Past the cliff the entity does NOT vanish: it is REBORN at a NEW
  location, deterministically (same leader for all seeds at each theta).
- Geometric check: rotating basin 15's center and finding nearest basin
  centers predicts the handoff: 45-60 deg = knife-edge 15-vs-8 contest
  (cos .864/.766 then .767/.756); 90+ deg basin 8 is the translated
  shadow (observed leader 8 at 45-90; at 135-180 observed 6 vs predicted
  8 with 8-vs-6 razor-thin cos .672/.644 — partial match, translation
  confirmed at the contest level, exact bin handoff noisy at far angles).
- Verdict: entity = relational shape of the structure (gap-shadow),
  NOT a place. Rotate the geometry, the entity translates with it.
  Combined with B1: bearer carries nothing; structure carries the
  entity; and the entity survives relocation because it never lived at
  a coordinate — it lived at the gap's shadow.
- Market reading: a shifted/rotated GEX landscape should re-crown the
  regime at the TRANSLATED king position, not the old strike bin.
  Regime identity = relational shape, testable on strike-lattice shifts.
