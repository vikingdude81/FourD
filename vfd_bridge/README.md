# vfd_bridge — Vibrational Field Dynamics ↔ FourD

Bridge track started 2026-11-19 from the VFD ("Vibrational Field Dynamics",
vfd_org on X) infographic: spectral mode selection on the 600-cell graph.
**We do not adopt VFD's physical claims.** We verified their graph math and
use the construction as (a) a validated special case of the Scaling Law of
Symmetry program and (b) a symmetry-maximal control for FourD.

## Status

- `verify_vfd_spectrum.py` — **VERDICT: VFD table VERIFIED.** Built the
  600-cell 1-skeleton from H4 roots (120 vertices, 720 edges, 12-regular);
  L = 12I − A reproduces all 9 claimed eigenvalue families to 4 dp
  (λ = 0, 2.2918, 5.5279, 9, 12, 14, 14.4721, 15, 15.7082 with multiplicities
  1,4,9,16,25,36,9,16,4). H(λ)=7−λ/2 cancels λ=14 (mult 36) exactly;
  retained dim 83; ≤14 oscillator branches ω²₊,ᵣ = κλᵣ + {0,2g}. Run it — exit
  code 0 = pass.
- `graphs.py` — builders: 600-cell, 24-cell, cycle, grid, ER, BA, plus a
  documented **placeholder** `palindromagon()` (original 9-node construction
  not found in local repos — TODO recover from source archive).
- `filter_ladder.py` — experiment **S1** (spectral selection scaling),
  output `outputs/spectral_selection_scaling.csv`.

## S1 first results (2026-11-19)

VFD's trick is a **designed null**: one planted filter root deletes a whole
eigenvalue family. Across the symmetry ladder (polytopes → cycles/grids →
BA/ER scale-free, n = 9..256):

- 600-cell: planted root kills 30% of mode space, ΔH = 0.0775 nats.
- **Every other graph: 0 modes killed** by the raw VFD root.
- Adaptive root (7·mean_deg/6) fails even on the 24-cell (spectrum
  {0,6,8,10,12} — no 7d/6 hit). The planted root 14 = (7/6)·12 encodes H4
  structure; the filter is **not scale-covariant**.
- Symmetry signature is clean: max_mult_frac ≈ 0.30–0.375 for the two
  polytopes vs 0.01–0.2 for BA/ER; spectral entropy H_spec grows
  monotonically with complexity (1.88 at 600-cell → 5.26 at BA-256,
  5.41 at ER-256).
- Caveat rows: ba_49/ba_64 show max_mult_frac 0.17–0.20 — random duplicate
  eigenvalues (accidental multiplicity), not symmetry; distinguish before
  using as fit points.

Working reading: designed-null efficiency (Δ modes deleted per filter scalar)
is a function of spectral concentration = broken symmetry. VFD sits at the
degenerate max-symmetry endpoint; our scaling program owns the rest of the
curve.

## Planned experiments

- **S2 — symmetric-substrate control for UniversalEngine.** Add 600-cell /
  24-cell as substrates in the topology-swap suite. Note: current
  `generate_preferences(topology=...)` couples n_sub=8 subsystems; a
  120-node graph needs a new substrate-coupling mode (graph Laplacian
  coupling A into the engine, subsystems read node states). Prediction:
  maximally symmetric substrate flattens basin boundaries → bearer/gateway
  effects weaken; if they SURVIVE, the effect is dynamical, not geometric.
- **S3 — ordered filter gating (history-bearing).** Apply spectral families'
  cancellation in different *time orders* as dynamical gates on the engine;
  test whether construction order changes subsequent regime dynamics. VFD's
  filter is order-free (linear, one-shot); any order dependence is a
  history-bearing signature our program can claim.
- Collision experiment: run VFD-style spectral filtering on LPF regime data
  as a regime detector (future; needs basin-graph → Laplacian adapter).

## lit/

arXiv sweep on spectral mode selection / graph-Laplacian filtering /
coupled oscillators on symmetric graphs lives in `lit/` (master.json +
REPORT.md). PDFs/txt ignored by git.
