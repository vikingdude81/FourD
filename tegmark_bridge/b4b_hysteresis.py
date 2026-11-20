"""B4b — hysteresis / route-dependence at partial gap fill.
Same deformed geometry P(w*), engine settings fixed, THREE approaches:
  A: warm-start conditioned on native dodeca19 (w=0), then hold at w*
  B: warm-start conditioned on native dodeca20 (w=1), then hold at w*
  C: fresh at w* throughout (control; condition and hold identical)
Condition phase 1000 steps; hold 4000 steps; measure window [2000,5000).
If A != B persists across seeds and long holds -> the geometry permits more
than one stable outcome and the route taken selects it (bearer/history
effect), which B3 ruled out for rigid rotation.
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
STEPS = 5000; TSW = 1000; W0, W1 = 2000, 5000; NB = 24; COS60 = 0.5

def Pw(w):
    return np.r_[P19, [[w * x for x in V20]]]

def run(w_star, seed, warm):  # warm in {'d19','d20','fresh'}
    torch.manual_seed(seed); np.random.seed(seed)
    P_cond = {"d19": P19, "d20": dodeca, "fresh": Pw(w_star)}[warm]
    eng = make_engine("cuda:0", 128, STEPS, P19, "s3")  # identical init
    def setP(P_raw):
        Pt = torch.tensor(np.asarray(P_raw), dtype=torch.float32, device=eng.device)
        old = eng.fatigue
        eng.prefs = Pt; eng.n_sub = Pt.shape[0]
        if warm == "d19":  # pad fatigue column for the new subsystem
            pad = torch.zeros(old.shape[0], Pt.shape[0] - old.shape[1],
                              device=old.device)
            eng.fatigue = torch.cat([old, pad], dim=1)
        else:
            eng.fatigue = torch.zeros(eng.N, Pt.shape[0], device=eng.device)
    setP(P_cond)
    C = eng.macro_centers.cpu().numpy(); c15 = C[15]
    occ = np.zeros(NB); cap_fix = 0.0; nsnap = 0; buckets = []
    cur_bucket = np.zeros(NB); bcount = 0
    for t in range(STEPS):
        if t == TSW and warm != "fresh":
            setP(Pw(w_star))
        eng.step()
        u = eng.u_t.cpu().numpy()
        lab = (u @ C.T).argmax(1)
        if t in range(TSW, STEPS) and (t - TSW) % 500 == 499:  # per-500 buckets
            buckets.append(cur_bucket / max(bcount, 1) / 128)
            cur_bucket = np.zeros(NB); bcount = 0
        if W0 <= t < W1:
            occ += np.bincount(lab, minlength=NB)
            cap_fix += float((u @ c15 >= COS60).mean())
            nsnap += 1
        if TSW <= t < W1:
            cur_bucket += np.bincount(lab, minlength=NB); bcount += 1
    buckets.append(cur_bucket / max(bcount, 1) / 128)
    p = occ / occ.sum()
    H = eng.hist_macro_basin[:, W0:W1].cpu().numpy()
    tr = float((H[:, 1:] != H[:, :-1]).mean())
    return p, int(p.argmax()), float(p[15]), tr, cap_fix / nsnap, buckets

def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())

# native references
torch.manual_seed(9201); np.random.seed(9201)
ref = {}
for name, P in [("d19", P19), ("d20", dodeca)]:
    torch.manual_seed(9201); np.random.seed(9201)
    eng = make_engine("cuda:0", 128, STEPS, P, "s3")
    C = eng.macro_centers.cpu().numpy(); c15 = C[15]
    occ = np.zeros(NB); cf = 0.0; n = 0
    for t in range(STEPS):
        eng.step()
        if W0 <= t < W1:
            u = eng.u_t.cpu().numpy()
            occ += np.bincount((u @ C.T).argmax(1), minlength=NB)
            cf += float((u @ c15 >= COS60).mean()); n += 1
    ref[name] = occ / occ.sum(); ref[name + "_cap"] = cf / n
print("ref d19 leader", int(ref['d19'].argmax()), "cap", round(ref['d19_cap'], 3),
      "| d20 leader", int(ref['d20'].argmax()), "cap", round(ref['d20_cap'], 3),
      flush=True)

out = {"rows": []}
for ws in (0.25, 0.5, 0.75):
    store = {}
    for warm in ("d19", "d20", "fresh"):
        for sd in (9201, 9202, 9203):
            p, lead, sh15, tr, cfix, buck = run(ws, sd, warm)
            key = (ws, warm)
            store.setdefault(warm, []).append(p)
            row = dict(w=ws, warm=warm, seed=sd, leader=lead,
                       share15=round(sh15, 3), tr=round(tr, 3),
                       cap_fix=round(cfix, 3),
                       tv_vs_d19=round(tv(p, ref["d19"]), 4),
                       tv_vs_d20=round(tv(p, ref["d20"]), 4),
                       buckets_share15=[round(float(b[15]), 3) for b in buck])
            out["rows"].append(row)
            print(row, flush=True)
    dAB = tv(np.mean(store["d19"], 0), np.mean(store["d20"], 0))
    print(f"== w*={ws}: TV(A-mean, B-mean) = {dAB:.4f}", flush=True)
    out.setdefault("AB_tv", {})[str(ws)] = dAB

json.dump(out, open(os.path.join(ROOT, "tegmark_bridge", "outputs",
                                 "b4b_hysteresis.json"), "w"), indent=1)
print("WROTE b4b_hysteresis.json")
