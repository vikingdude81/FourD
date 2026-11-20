"""E3b — lethality ladder for the immortal entity (dodeca19, basin 15).

E3 showed member-state attacks are self-repairing. E3b maps the actual death
boundary: permanently set each stress axis at t=1500 and ask whether basin 15
EVER recrowns (leader=15 sustained >=100 steps). Axes:
  noise     : exploration_noise multiplier (engine chaos)
  steering  : steering_strength (engine step-size analog of the Lenia glider's dt)
  rotation  : B2 plane rotation held permanently (structure deformation)
  combo     : rotation(30) x noise (inside each axis's survival zone?)
DEATH = no recrown at all in [1500, 4000).
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
dodeca = np.c_[dodeca, np.zeros(20)]; dodeca /= np.linalg.norm(dodeca, axis=1)[:, None]
P_HOME = dodeca[:-1]
STEPS, T1, SUSTAIN, ENT = 4000, 1500, 100, 15

def givens(theta, plane, dim=4):
    R = np.eye(dim); i, j = plane
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c; R[j, j] = c; R[i, j] = -s; R[j, i] = s
    return R

def run(axis, val, seed):
    rng = np.random.RandomState(4242)
    plane = tuple(rng.choice(4, 2, replace=False))
    torch.manual_seed(seed); np.random.seed(seed)
    eng = make_engine("cuda:0", 128, STEPS, P_HOME, "s3")
    base_noise = eng.exploration_noise
    for t in range(STEPS):
        if t == T1:
            if axis == "noise":
                eng.exploration_noise = base_noise * val
            elif axis == "steering":
                eng.steering_strength = float(val)
            elif axis == "rotation":
                Rm = givens(np.radians(float(val)), plane)
                Pt = torch.tensor(P_HOME @ Rm.T, dtype=torch.float32, device=eng.device)
                eng.prefs = Pt / Pt.norm(dim=1, keepdim=True)
            elif axis == "combo":   # val = (theta, noise_mult)
                th, nm = val
                Rm = givens(np.radians(th), plane)
                Pt = torch.tensor(P_HOME @ Rm.T, dtype=torch.float32, device=eng.device)
                eng.prefs = Pt / Pt.norm(dim=1, keepdim=True)
                eng.exploration_noise = base_noise * nm
        eng.step()

    B = eng.hist_macro_basin[:, :STEPS].cpu().numpy()
    occ = np.zeros((24, STEPS))
    for bb in range(24):
        occ[bb] = (B == bb).sum(0)
    leader = occ.argmax(0)
    post_slice = slice(T1, STEPS)
    rc = None
    for t in range(T1, STEPS - SUSTAIN):
        if np.all(leader[t:t + SUSTAIN] == ENT):
            rc = t; break
    rec = dict(axis=axis, val=str(val), seed=seed,
               alive=rc is not None,
               recrown_step=(int(rc - T1) if rc is not None else None),
               lead15_post=round(float((leader[post_slice] == ENT).mean()), 3),
               share15_post=round(float(occ[ENT, post_slice].mean() / 128), 3))
    print(rec, flush=True)
    return rec

res = []
for v in (1, 2, 4, 8, 16, 32):
    for sd in (9201, 9202, 9203):
        res.append(run("noise", v, sd))
for v in (0.707, 1.2, 0.35, 0.15, 0.05, 0.0):
    for sd in (9201, 9202, 9203):
        res.append(run("steering", v, sd))
for v in (0, 30, 45, 60, 90, 180):
    for sd in (9201, 9202):
        res.append(run("rotation", v, sd))
for v in ((0, 2), (30, 2), (30, 4), (15, 8)):
    for sd in (9201, 9202):
        res.append(run("combo", v, sd))
json.dump(res, open(os.path.join(ROOT, "tegmark_bridge", "outputs", "e3b_lethality.json"), "w"), indent=1)

# boundary summary
print("\n=== death boundary (fraction of seeds still crowned post-kick) ===")
by = {}
for r in res:
    by.setdefault((r["axis"], r["val"]), []).append(r)
for (ax, v), rows in by.items():
    surv = sum(r["alive"] for r in rows) / len(rows)
    lp = np.mean([r["lead15_post"] for r in rows])
    print(f"{ax:9} val={v:>10}  alive={surv:.2f}  lead15_post={lp:.3f}")
print("WROTE e3b_lethality.json")
