# tegmark_bridge — Mathematical Universe Hypothesis vs. our systems

Track opened 2026-11-19 after the VFD discussion: Tegmark's MUH / Level IV
multiverse says every mathematical structure physically exists and observers
are "self-aware substructures" (SAS) inside some of them. Tegmark's own
2007 paper (arXiv:0704.0646): *"a simulation is merely a special case of a
mathematical structure, hence included in the Level IV multiverse"* — running
it adds no existence. Standard critiques we take seriously: Woit ("empty,
radically untestable"), the measure problem (Schmidhuber: no a-priori weight
on all structures), Hut–Alford Gödel inconsistency, Ellis untestability.

## What our program adds

FourD's delta over Tegmark: existence is not binary per structure —
a structure only hosts **history-bearing** substructure (bearer, boundary
negotiation, seed-sensitivity) if it breaks certain symmetries. That makes a
*pilot measure*: hospitable vs inhospitable structures, measurable without
any physics.

## T1 — SAS census (RUN 2026-11-19, `t1_sas_census.py`)

Symmetry-breaking dial: UniversalEngine preference substrate = random
subsample of k of the 120 600-cell vertices; k=120 is the maximally
symmetric (antipodally closed) structure. 3 seeds × 1000 steps × N=128.
Criteria (pre-registered): C1 null_z > 3 (boundary signal), C2
transition-rate std across seeds > 0.01 (history-bearing / seed-sensitive).

Result (summary; full data `outputs/t1_sas_census.json`):

- k = 8..100: structure HOSTS (z from +7.5 to +44.8; k=6 marginal — one of
  three seeds negative; k=4 hosts but weak).
- k = 110: FAILS C1 (z +2.9). k = 118: z +0.2. k = 120: dead on both C1 and
  C2 (z +0.1, TR 0.361 ± 0.001 — completely seed-deaf).
- The collapse follows the **antipodal-closure fraction** of the vertex set:
  healthy at 0.72–0.83, dying at 0.92–0.98, dead at 1.00. The critical
  quantity identified in S2 (central symmetry P = −P ⇒ exact force
  cancellation) is the right order parameter; sheer structure size is not.

Reading: a first empirical slice of the "extropy landscape" over structures —
most of the neighborhood of the maximally-symmetric point is hospitable; the
inhospitable set is a measure-zero-ish symmetry-locked locus. If a measure
over structures is ever definable, broken symmetry looks like its support
condition, not uniform weighting over all structures (contra Tegmark, in
agreement with Schmidhuber's complexity-based intuition).

Caveats: 3 seeds, one engine family, 'substrate' = preference geometry;
k=6 noise; C2 threshold conventional. Replication with new seeds and more
structure families (platonic 3D sets, lattices, palindromagon once recovered)
before any claim.

## lit/

arXiv sweep (MUH, structural realism, measure problem, Gödel constraints,
Schmidhuber, SAS formalizations, digital-physics tests 2023-2026) lives in
`lit/` once the sweep lands.

## Question to resolve

User recalls an older local simulation that "generated lots of math and
polyhedrons" but never connected it to this — not found in
consciousness-sim / causal-emergence-lab / consciousness-emergence-* /
ai-math-studio greps. Possibly ChatGPT-era canvas (cf. lost Bearer chapter
sources). Ask user where it lived.
