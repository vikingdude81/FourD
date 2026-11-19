"""B2 — graded transplant: entity identity vs structural deformation.

At t=1500 rotate ALL preference points of dodeca19 by angle theta in a
fixed random 2-plane of R4 (theta=0 = exact control: structure unchanged).
Basin centers are fixed in space, so the attractor geometry slides away
from basin 15's basin by an amount controlled by theta.

Metrics: does basin 15 hold through the deformation (entity rides the
structure), and if displaced, recrown time = first t after swap where
leader==15 persists 100 consecutive steps (None = never).
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
STEPS = 4000; T1 = 1500; SUSTAIN = 100

def givens(theta, plane, dim=4):
    R = np.eye(dim)
    i, j = plane
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c; R[j, j] = c; R[i, j] = -s; R[j, i] = s
    return R

def recrown_and_away(eng):
    B = eng.hist_macro_basin[:, T1 + 10:STEPS].cpu().numpy()
    occ = np.zeros((24, B.shape[1]))
    for bb in range(24):
        occ[bb] = (B == bb).sum(0)
    leader = occ.argmax(0)
    share15 = (B == 15).mean(0)
    rc = None
    for t in range(len(leader) - SUSTAIN):
        if np.all(leader[t:t + SUSTAIN] == 15):
            rc = t + SUSTAIN; break
    return dict(away_share15=round(float(share15.mean()), 3),
                away_top_leader=int(np.bincount(leader).argmax()),
                recrown_step=(int(rc + T1 + 10) if rc is not None else None),
                held_through=bool(rc is not None and rc <= SUSTAIN))

def run(theta_deg, seed):
    rng = np.random.RandomState(4242)
    plane = tuple(rng.choice(4, 2, replace=False))
    Rm = givens(np.radians(theta_deg), plane)
    torch.manual_seed(seed); np.random.seed(seed)
    eng = make_engine("cuda:0", 128, STEPS, P_HOME, "s3")
    P_rot = P_HOME @ Rm.T
    t = 0
    while t < STEPS:
        if t == T1:
            Pt = torch.tensor(P_rot, dtype=torch.float32, device=eng.device)
            Pt = Pt / Pt.norm(dim=1, keepdim=True)
            eng.prefs = Pt; eng.n_sub = Pt.shape[0]
        eng.step(); t += 1
    B0 = eng.hist_macro_basin[:, :T1].cpu().numpy()
    out = dict(theta=theta_deg, seed=seed,
               home_share15=round(float((B0 == 15).mean(0).mean()), 3),
               **recrown_and_away(eng))
    print(out, flush=True)
    return out

res = []
for th in (0, 15, 30, 45, 60, 90, 135, 180):
    for sd in (9201, 9202, 9203):
        res.append(run(th, sd))
json.dump(res, open(os.path.join(ROOT, "tegmark_bridge", "outputs", "b2_graded.json"), "w"), indent=1)
print("WROTE b2_graded.json")
