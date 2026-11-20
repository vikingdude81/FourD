# AUDIT SUMMARY — FourD / tegmark_bridge experimental record

Purpose: a self-contained, checkable summary of everything tested in this lane,
HOW it was tested, and the exact numbers obtained — for cross-verification by
another model or a human auditor. Every claim cites its experiment. Known bugs
and withdrawn claims are listed, not hidden. Repo: private `vikingdude81/FourD`,
branch main, HEAD `cc2252f` (2026-11-19). Full record: `RESULTS.md` (631 lines),
metaphysical reading: `METAPHYSICS.md`, lit dossier: `lit_entities/REPORT.md`.

## 0. Shared protocol

- Engine: FourD `UniversalEngine` (`universality_test.py`). N agents on S^3
  (dim=4), gradual fatigue: F += rate*a + 0.03*excess(a) − recovery*(1−a);
  attention a = softmax-free activity = influences*exp(−F)+noise, renormalized;
  steering 0.707, fatigue_rate 0.217, exploration_noise 0.25, beta_macro 11.375,
  24 fixed macro-basin centers, basin = argmax softmax(beta*<u, center>).
- The "substrate" = preference matrix P (unit vectors = subsystem directions).
- Readouts: `null_z` = z-score of post-transition clarity vs label-shuffled null
  (50 shuffles); `transition_rate` (TR) = basin flips/step normalized; entity
  occupancy = per-basin agent counts.
- State labels (conventional thresholds): HOSTS if z>3 and tr<0.9;
  BOILS if tr>=0.9; DEAD if z<=3 and tr<0.9. ANTI = z strongly negative.
  Caveat: z>3 threshold is conventional, not derived.
- Seeds: 3 per variant (torch+numpy), cuda:0. Substrates: exact vertex sets of
  polytopes at unit circumradius (built in `vfd_bridge/graphs.py`).
- dodeca19 := 20 dodecahedron vertices embedded in R^4 (4th coord 0),
  normalized, with the LAST vertex deleted (19 rows). Basin numbering is the
  fixed 24-center grid; "basin 15" is the entity that crowns for dodeca19.

## 1. Results ledger (claim → protocol → numbers)

### VFD lane (control lane, not merged)
- **V1** Verified the external VFD claim's 600-cell spectrum table by
  independently building the 600-cell 1-skeleton and computing L=12I−A
  multiplicities. File: `vfd_bridge/verify_vfd_spectrum.py`. Result: table correct.

### S2 — substrate control (founding result)
- **S2** Full 24-cell/600-cell substrates: signal dead (null_z +0.2/−0.2, d≈0).
  Count-matched random sets: z≈+16. `antipodal12` (12 random dirs + exact
  negations): z=+0.4 → mechanism is central symmetry, not H4 structure.
  3 seeds each. File `vfd_bridge/s2_substrate_control.py`.

### T1 — hostability dial
- **T1** Random k-of-120 subsamples of 600-cell vertices, k=4..120. Antipodal-
  closure fraction predicts collapse: alive (z +7.5..+44.8) through frac 0.83;
  z=+2.9 at 0.92; z=+0.2 at 0.98; z=+0.1 at k=120 (frac 1.0).
- **RETRACTION R1 (recorded same day):** a companion "seed-sensitivity =
  history" criterion was confounded (subsample re-drawn per seed); withdrawn.
  T2 shows TR seed-std ≈0.001–0.002 for every structure.

### T2 — mechanism pinning
- **T2a** ε-restore on dead 600-cell: z(ε)= +0.0 (ε=0), +0.3 (0.01), +2.5 (0.03),
  **+17.7 (0.1)**, −0.0 (0.3, TR=1.0 boil), +0.6 (1.0). Alive region = Goldilocks
  window of net polarity between cancellation and saturation.
- **T2b** Family zoo (closed ⇒ dead): octa +1.6, cube +1.3, icosa −0.7,
  dodeca20 −1.7, hypercube-16 +0.3; **dodeca20 − 1 vertex: z=+36.8**; octa−1
  +5.4; cube−1 +0.2; icosa−1 +0.8; hypercube−1 +0.5; tetra +2.8 (zero-sum).
  Gap GEOMETRY (not count) determines whether one gap revives the system.

### D1/D1b — why closed substrates die
- **D1a attention deadlock (confirmed):** accumulated fatigue exactly 0.00 for
  600-cell and antipodal12 (all agents/subsystems); up to 2.97 for alive sets.
  Mechanism: antipodal pairing + affine influence (0.5+0.3cos) ⇒ attention
  stays uniform ⇒ no subsystem crosses the fatigue concentration threshold ⇒
  zero commitment rotation. TR seed-locked 0.365; clarity at chance.
- **D1b zero-force theorem (theorem-grade):** 24-cell and 600-cell are exact
  spherical 2-designs: M2=Σpp^T isotropic (dev2 = 0.0 and 3e−18). Under uniform
  attention net tangential force (M2u − u(u'M2u))/n ≡ 0 for every configuration.
  antipodal12 dev2=0.46 yet still dead ⇒ deadlock is general, zero-force is the
  knife-edge case. Death certificate: check central closure then M2 isotropy.
- **R2 (withdrawn):** mirror-bounce hypothesis for dead-system transitions:
  measured bounce fraction 0.000–0.003 (s3, all structures). Killed.
- **D1d engine map:** 600-cell dead in s3 AND flat4 (z +0.2/+0.9). dodeca20:
  dead s3 (−1.6), ALIVE flat4 (+21.2) → engine-conditioned. dodeca20−1: alive
  both (+39.5 / +23.4) — most robust positive result.

### T6 — market substrate (first pass: NULL verdict)
- 504 daily log-return direction vectors R^4 [SPY,SMH,IWM,MU] (local parquet),
  s3/N=128/1000 steps, 3 seeds. market_60: antiFrac 0.033 (passes closure cert)
  but z=+109.1±154, tr=1.000 → BOILS. antipodal-closed market control:
  z=−0.9, tr=0.351 → DEAD (certificate causal on real data). Whitening/residual
  did not fix boil (failure = cone congestion conc≈0.65, basin jitter).
  random_60_ctrl: z=+6.0 → HOSTS. Verdict recorded as null for naive bridge:
  closure necessary, not sufficient.

### N7 — cyclic orbits (Karkada bridge probe)
- Regular Z_n orbits n=7,8,11,12,13,24 × jitter {0,.1,.25}: none host;
  n≥12 strongly ANTI (z to −14). Order parameter upgraded to "uniform closed
  coverage" (local force balance), antipodal closure = n=2 case.

### CERT2 — geometry-only predictors (19-substrate corpus)
- Brel = ||net tangential force||/gross force (uniform attention); conc = M2
  top-eigenvalue share; drel = dmin/dmed (upper-triangle spacing — a diagonal
  bug was found and fixed during development).
- Brel=0 exactly on all closed/zero-sum sets; 8/9 DEAD/ANTI. **Exception:
  tetra (n=4) HOSTS at Brel=0 (z=+3.3)** ⇒ certificate needs many-body pairing.
  HOSTS band: Brel∈[.05,.22]. market60: Brel≈.14, conc≈.65, drel≈.025, BOILS.
  dodeca20→dodeca19: Brel 0→0.052, z −2.5→+35.7. ANTI unsolved (protos24
  z=−12, ngon12j z=−13.5).

### E1 — entity demographics (birth/death of substructures)
- 4000-step runs, entity birth = occupancy ≥20% of agents:
  cell600/dodeca20: 0 births, leader flips every 1.7 steps, entropy 2.98–3.02
  (max ln24≈3.18). sub600_8: 8731 births, lifespan mean 2.8/max 38, entropy 1.34
  (ecology). dodeca19: ONE immortal basin, 0 leader changes, entropy 0.79.
  random60: z=+6 at 1000 steps but tr=0.98/z=−32.6 at 4000 → HOSTS labels are
  horizon-conditioned (logged caveat).

### T7 — evolution for hostability
- Pop 12, 16 generations, fitness = engine z with boil penalty; closure
  mutation available. antiFrac 0.118 (gen2) → 0.0 (gen5): closure REJECTED.
  Brel 0.31→0.40. Champion n=14 with one EXACT duplicate pair; held-out 3
  seeds: z=+56.2±2.2, tr=0.534 (HOSTS). Removing duplicate: z +55.1 but
  tr 0.534→0.840. Consequence: substrates are MULTISETS; duplicates are
  stabilizers. Caveat: one evolution run, single fitness seed.

### E2 — repeatability of the immortal entity
- dodeca19, 4000 steps, 12 seeds×N=128 + 3×N=64 + 3×N=256: basin 15 wins
  18/18; share 0.807–0.810; strict zero-flip 10/18 (other 8: one init transient
  flip); seed- and density-deaf. Mechanism: basin-15 center cos 0.978 to
  preference vertex 2 and cos −0.706 to the DELETED vertex ⇒
  gap-as-rival-removal → attractor monoculture.

### T6b — weighted market prototypes
- kmeans regime prototypes replicated proportional to occupancy (total n=60),
  2000 steps ×3 seeds. k=8 weighted: z=+23.5, tr=0.44, life mean 5.3/max 55,
  share 0.52 → HOSTS. k=16: z=−13.0. k=24: z=−55.8 (ANTI), leader alive
  1667/1999 steps, share 0.79. raw60: z=+122, tr=1.0, 5020 births, mean life
  1.37 (flicker). Flicker→ecology→monoculture observed in market data via
  granularity × weighting. Caveat: single window, single basket.

### B1 — transplant (bearer test)
- dodeca19 home 0–1500; structure swapped to 19 random pts 1500–2500; restored
  2500–4000. B = bearer state carried; C = bearer wiped. 3 seeds.
  AWAY: share15 = 0.000 both (entity fully lost). BACK: basin 15 re-crowns to
  0.810 (home 0.806–0.808), lead15_frac 1.000, ≤100 steps. B ≡ C numerically
  ⇒ bearer carries zero identity. **Recorded bug: the intended no-swap control
  (mode A) also swapped (duplicated B); E2 serves as true control.**
- Known issue (recorded in code): first run crashed on `keepdims`→`keepdim`.

### B2 — graded transplant (identity radius)
- Rotate ALL preferences by θ in a fixed 2-plane (same plane all seeds) at
  t=1500; basin centers fixed. 3 seeds × 8 θ:
  θ=0: share 0.810, leader 15. 15°: 0.688/15. 30°: 0.384/15, recrown 167–212.
  45°: 0.106, leader 8, no recrown. 60°: 0.012/8. 90°: 0.000/8. 135°: 0.000/6.
  180°: 0.000/6. Identity radius ∈ (30°,45°); past the cliff the entity
  RE-INSTATIATES at a translated basin (deterministic across seeds). Geometric
  shadow check: nearest-rotated-center prediction matches 45–90° exactly;
  far angles (135–180°) are razor-thin contests (8-vs-6 cos .672/.644) —
  translation confirmed at contest level, exact bin handoff noisy.

### E3 — perturbation agency test (Hamon criterion)
- Kick at t=1500, dodeca19/basin15, 3 seeds, 4000 steps; recrown = leader for
  ≥100 consecutive steps; recovery = share ≥95% of home (0.807).
  teleport 25/50/100% of members to random S^3: min share 0.638/0.492/0.208;
  crown NEVER lost; 95% recovery in 5/5/7 steps.
  wind (displace whole crowd 80% toward far basin): min 0.143, recrown 5,
  recovery 8. noise ×4 for 50 steps: min 0.065, recrown at step 49, recovery 54.
  Verdict: full robustness, zero behavioral agency. Dev note (recorded):
  first launch crashed on occupancy-axis transpose + array truthiness; fixed
  before any results existed.

### E3b — lethality ladder (mortality surface)
- Stress made PERMANENT at t=1500; DEATH = no sustained recrown in 2500 steps.
  noise ×{1,2,4,8,16,32}: alive ×1 (0.810), ×2 (alive, share 0.375, crown
  never lost), DEAD ≥×4 (×4 lead15_post 0.655 without sustained recrown;
  ×32: 0.019). rotation {0,30,45,...,180}°: alive ≤30 (0.992), dead ≥45.
  steering {1.2,0.707,0.35,0.15,0.05,0}: alive everywhere; at 0.0 motion stops
  (frozen configuration = degenerate stasis, flagged as a caveat, not life).
  combos: (0°,×2) alive; **(30°,×2) DEAD** (both factors individually
  sublethal); (30°,×4) dead; (15°,×8) dead. ⇒ mortality is a region boundary;
  identity radius shrinks with engine chaos (critical noise ~4 at 0°, <2 at 30°).

### Literature lane (claims about OTHER papers — verifiable independently)
- VFD lit: papers registered incl. arXiv:2602.15029 (Karkada et al., symmetry
  in language statistics → representation geometry) and follow-up 2610.00647;
  PDFs on disk. Entity sweep: 12 queries, 142 indexed, 14 PDFs, excerpts quoted
  in `lit_entities/REPORT.md` (esp. 2407.21086: glider survives ∆t=0.2, dies
  0.17/0.26; 2402.10236; 2512.09352; 1802.01161; 2005.03742). A subagent
  hallucinated one arXiv id early in the program; it was removed on audit
  (recorded process note).
- **Not claimable yet (recorded):** history-bearing status (T3 not run);
  T4 phase diagram; T5 description-length probe; market entity identity
  (atlas test not run); universality beyond s3/flat4 engines.

## 2. Reproduction

```
cd FourD
.venv/Scripts/python.exe vfd_bridge/verify_vfd_spectrum.py
.venv/Scripts/python.exe vfd_bridge/s2_substrate_control.py
.venv/Scripts/python.exe tegmark_bridge/t1_sas_census.py
.venv/Scripts/python.exe tegmark_bridge/t2_mechanism.py
.venv/Scripts/python.exe tegmark_bridge/d1_autopsy.py
.venv/Scripts/python.exe tegmark_bridge/d1b_fatigue.py
.venv/Scripts/python.exe tegmark_bridge/t6_market_certificate.py
.venv/Scripts/python.exe tegmark_bridge/ngon_certificate.py
.venv/Scripts/python.exe tegmark_bridge/cert2_coverage.py
.venv/Scripts/python.exe tegmark_bridge/e1_entities.py
.venv/Scripts/python.exe tegmark_bridge/t7_evolve_hostability.py
.venv/Scripts/python.exe tegmark_bridge/e2_immortal.py
.venv/Scripts/python.exe tegmark_bridge/t6b_weighted_market.py
.venv/Scripts/python.exe tegmark_bridge/b1_transplant.py
.venv/Scripts/python.exe tegmark_bridge/b2_graded.py
.venv/Scripts/python.exe tegmark_bridge/e3_agency.py
.venv/Scripts/python.exe tegmark_bridge/e3b_lethality.py
```
Raw JSON in `tegmark_bridge/outputs/` (gitignored; regenerate locally).
Expected: seed-deterministic numbers within ±0.1 z on reruns (fixed seeds).

## 3. Weaknesses an auditor SHOULD press (our own list)

1. Entity = argmax basin of a 24-center grid: an observational discretization.
   Do results survive finer/coarser grids or kernel density entities?
2. Engine is a single (family of) attention+fatigue models; "structure × engine"
   claims are tested on 2 embeddings (s3, flat4), not engines.
3. HOSTS/DEAD thresholds (z>3, tr 0.9) conventional; bimodality not yet proven
   (E4 dip test queued).
4. B1 mode-A control bug (no-swap leg duplicated B); mitigated by E2 control.
5. T7: single evolution run; duplicate-stabilizer claim from one champion.
6. T6b market: single window/basket; k=8 HOSTS not certified across periods.
7. dodeca19 radius measured in one plane family; "identity radius" is a
   plane-specific number, not a global one.
8. Horizon-conditioned states (random60) mean any label at 1000 steps needs
   the 4000-step check.
