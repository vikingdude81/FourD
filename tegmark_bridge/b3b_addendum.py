"""B3b — addendum: where does co-rotating-frame equivalence break, and
wrong-transport controls (prove the TV collapse is not an artifact)."""
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
STEPS = 4000; T1 = 1500; W0, W1 = 1600, 2600
NB = 24

def givens(theta, plane, dim=4):
    R = np.eye(dim)
    i, j = plane
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c; R[j, j] = c; R[i, j] = -s; R[j, i] = s
    return R

def run(theta_deg, seed, transport="correct"):
    rng = np.random.RandomState(4242)
    plane = tuple(rng.choice(4, 2, replace=False))
    plane2 = tuple(rng.choice(4, 2, replace=False))
    while plane2 == plane:
        plane2 = tuple(rng.choice(4, 2, replace=False))
    th = np.radians(theta_deg)
    Rm = givens(th, plane)
    if transport == "correct":
        Rinv = givens(-th, plane)
    elif transport == "wrong_plane":
        Rinv = givens(-th, plane2)
    elif transport == "wrong_angle":
        Rinv = givens(-th / 3.0, plane)
    else:  # none
        Rinv = np.eye(4)
    torch.manual_seed(seed); np.random.seed(seed)
    eng = make_engine("cuda:0", 128, STEPS, P_HOME, "s3")
    P_rot = P_HOME @ Rm.T
    occ_tr = np.zeros((NB, 0))
    C = eng.macro_centers.cpu().numpy()
    for t in range(STEPS):
        if t == T1 and theta_deg != 0:
            Pt = torch.tensor(P_rot, dtype=torch.float32, device=eng.device)
            Pt = Pt / Pt.norm(dim=1, keepdim=True)
            eng.prefs = Pt; eng.n_sub = Pt.shape[0]
        eng.step()
        if W0 <= t < W1:
            u = eng.u_t.cpu().numpy() @ Rinv.T
            occ_tr = np.c_[occ_tr, np.bincount((u @ C.T).argmax(1), minlength=NB)]
    return occ_tr.sum(1) / occ_tr.sum()

def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())

ctrl = {sd: run(0, sd, "none") for sd in (9201, 9202, 9203)}
base_tv = [round(tv(ctrl[a], ctrl[b]), 4) for a, b in
           [(9201, 9202), (9202, 9203), (9201, 9203)]]
print("baseline TV:", base_tv, flush=True)
rows = []
for th, tp in [(135, "correct"), (180, "correct"),
               (45, "wrong_plane"), (45, "wrong_angle"), (45, "none")]:
    for sd in (9201, 9202, 9203):
        p = run(th, sd, tp)
        d = tv(p, ctrl[sd])
        rows.append((th, tp, sd, int(p.argmax()), round(float(p[15]), 3),
                     round(d, 4), d <= max(base_tv)))
        print(rows[-1], flush=True)
json.dump(dict(baseline_tv=base_tv, rows=rows), open(
    os.path.join(ROOT, "tegmark_bridge", "outputs", "b3b_addendum.json"), "w"), indent=1)
print("WROTE b3b_addendum.json")
