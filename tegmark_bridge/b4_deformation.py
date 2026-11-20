"""B4a — structural deformation ladder in the (frame-fixed) original geometry.
Two ladders, per frontier-audit design (2026-11-20):
1) GAP-FILL: restore the deleted vertex 20 with weight w = row-norm in [0,1].
   Endpoints verified: native dodeca19 (19 rows, no 20th) vs w=0 dead-20th
   (presence without weight) vs w=1 = native dodeca20.
2) VERTEX DISPLACEMENT: rotate ONE surviving vertex by graded alpha in the
   plane spanned by (vertex, basin-15 center) — near vertex (max cos to c15)
   vs far vertex (min cos to c15).
Readouts (frame = known unchanged geometry; no alignment fitting):
  - TV vs native dodeca19 mean occupancy (connects to B3)
  - cap concentration: fraction inside 60-deg cap around FIXED c15, and
    around the window peak direction (recorded, with its cos to c15)
  - leader basin, share15, tr (basin-transition rate)
"""
import sys, os, json
import numpy as np
import torch

ROOT = r"C:\Users\akbon\OneDrive\Documents\GitHub\FourD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tegmark_bridge"))
from d1_autopsy import make_engine

phi = (1 + 5 ** 0.5) / 2
dodeca = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)] +
    [[0, s1/phi, s2*phi] for s1 in (-1, 1) for s2 in (-1, 1)] +
    [[s1/phi, s2*phi, 0] for s1 in (-1, 1) for s2 in (-1, 1)] +
    [[s1*phi, 0, s2/phi] for s1 in (-1, 1) for s2 in (-1, 1)], float)
dodeca = np.c_[dodeca, np.zeros(20)]
dodeca /= np.linalg.norm(dodeca, axis=1)[:, None]
P19, V20 = dodeca[:-1], dodeca[-1]
STEPS = 4000; W0, W1 = 1000, 4000; NB = 24
COS60 = 0.5

def run(P_raw, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    eng = make_engine("cuda:0", 128, STEPS, P19, "s3")  # identical init
    Pt = torch.tensor(np.asarray(P_raw), dtype=torch.float32, device=eng.device)
    eng.prefs = Pt                      # NO row normalization: norm = weight
    eng.n_sub = Pt.shape[0]
    eng.fatigue = torch.zeros(eng.N, eng.n_sub, device=eng.device)
    C = eng.macro_centers.cpu().numpy(); c15 = C[15]
    occ = np.zeros(NB); cap_fix = cap_peak = 0.0; Rsum = np.zeros(4); nsnap = 0
    for t in range(STEPS):
        eng.step()
        if W0 <= t < W1:
            u = eng.u_t.cpu().numpy()
            occ += np.bincount((u @ C.T).argmax(1), minlength=NB)
            cap_fix += float((u @ c15 >= COS60).mean())
            Rsum += u.sum(0); nsnap += 1
    p = occ / occ.sum()
    R = Rsum / np.linalg.norm(Rsum)
    # recompute cap_peak with peak dir: cheap second pass not needed; use
    # same-window accumulation of |proj| (recompute via resultant proxy):
    # exact second pass:
    return p, float((p.argmax())), R, cap_fix / nsnap, float(p[15])

def full_run(P_raw, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    eng = make_engine("cuda:0", 128, STEPS, P19, "s3")
    Pt = torch.tensor(np.asarray(P_raw), dtype=torch.float32, device=eng.device)
    eng.prefs = Pt; eng.n_sub = Pt.shape[0]
    eng.fatigue = torch.zeros(eng.N, eng.n_sub, device=eng.device)
    C = eng.macro_centers.cpu().numpy(); c15 = C[15]
    occ = np.zeros(NB); cap_fix = cap_peak = 0.0; Rsum = np.zeros(4); nsnap = 0
    usnap = []
    for t in range(STEPS):
        eng.step()
        if W0 <= t < W1:
            u = eng.u_t.cpu().numpy()
            occ += np.bincount((u @ C.T).argmax(1), minlength=NB)
            cap_fix += float((u @ c15 >= COS60).mean())
            Rsum += u.sum(0); nsnap += 1
            if nsnap % 4 == 0:
                usnap.append(u)
    p = occ / occ.sum()
    U = np.concatenate(usnap); R = Rsum / np.linalg.norm(Rsum)
    cap_peak = float((U @ R >= COS60).mean())
    cap_fix = cap_fix / nsnap
    H = eng.hist_macro_basin[:, W0:W1].cpu().numpy()
    tr = float((H[:, 1:] != H[:, :-1]).mean())
    return p, int(p.argmax()), R, cap_fix, cap_peak, float(p[15]), tr

def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())

SEEDS = (9201, 9202, 9203)
# baseline: native dodeca19
native = [full_run(P19, sd) for sd in SEEDS]
p_home = np.mean([r[0] for r in native], axis=0)
base_tv = [round(tv(native[i][0], native[j][0]), 4)
           for i, j in [(0, 1), (1, 2), (0, 2)]]
print("native dodeca19 baseline:", flush=True)
for sd, r in zip(SEEDS, native):
    print(f"  sd{sd} leader={r[1]} share15={r[5]:.3f} tr={r[6]:.3f} "
          f"cap_fix={r[3]:.3f} cap_peak={r[4]:.3f}", flush=True)
print("baseline TV band:", base_tv, flush=True)

out = {"baseline_tv": base_tv, "rows": []}

def report(tag, P_raw):
    for sd in SEEDS:
        p, lead, R, cfix, cpeak, sh15, tr = full_run(P_raw, sd)
        row = dict(tag=tag, seed=sd, leader=lead, share15=round(sh15, 3),
                   tr=round(tr, 3), tv_vs_home=round(tv(p, p_home), 4),
                   cap_fix=round(cfix, 3), cap_peak=round(cpeak, 3),
                   cos_peak_home=round(float(R @ native[0][2]), 4))
        out["rows"].append(row)
        print(row, flush=True)

# ladder 1: gap-fill
report("native19_ctrl", P19)
report("dodeca20_native", dodeca)
report("gap_w0.00_dead", np.r_[P19, np.zeros((1, 4))])
for w in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
    report(f"gap_w{w:.2f}", np.r_[P19, [[w * x for x in V20]]])

# ladder 2: vertex displacement in (v, c15)-plane
# need c15: construct one engine to read macro centers (same seed init)
torch.manual_seed(9201); np.random.seed(9201)
_e = make_engine("cuda:0", 128, 10, P19, "s3")
c15 = _e.macro_centers.cpu().numpy()[15]
S19 = P19 @ c15
near, far = int(S19.argmax()), int(S19.argmin())
print(f"near vertex idx={near} cos(c15)={S19[near]:.3f}; "
      f"far idx={far} cos(c15)={S19[far]:.3f}", flush=True)
for vi, vname in [(near, "near"), (far, "far")]:
    v = P19[vi]
    e2 = c15 - (c15 @ v) * v
    e2 /= np.linalg.norm(e2)
    for adeg in (5, 10, 15, 20, 30, 45, 60, 90):
        a = np.radians(adeg)
        P = P19.copy()
        P[vi] = np.cos(a) * v + np.sin(a) * e2
        P /= np.linalg.norm(P, axis=1)[:, None]
        report(f"disp_{vname}{adeg}deg", P)

json.dump(out, open(os.path.join(ROOT, "tegmark_bridge", "outputs",
                                 "b4a_deformation.json"), "w"), indent=1)
print("WROTE b4a_deformation.json")
