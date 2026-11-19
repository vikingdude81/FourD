"""E2 — repeatability of the immortal-entity regime (dodeca19, 4000 steps).

E1 showed zero leader changes for dodeca19 at seed 9001, N=128. Is the
monoculture (a) seed-robust, and (b) density-robust? Also: which basin
wins, and does the winner's identity depend on the seed?
"""
import sys, os, json
import numpy as np
import torch

ROOT = r"C:\Users\akbon\OneDrive\Documents\GitHub\FourD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tegmark_bridge"))
from universality_test import boundary_metrics
from d1_autopsy import make_engine

phi = (1 + 5 ** 0.5) / 2
dodeca = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)] +
    [[0, s1/phi, s2*phi] for s1 in (-1, 1) for s2 in (-1, 1)] +
    [[s1/phi, s2*phi, 0] for s1 in (-1, 1) for s2 in (-1, 1)] +
    [[s1*phi, 0, s2/phi] for s1 in (-1, 1) for s2 in (-1, 1)], float)
dodeca = np.c_[dodeca, np.zeros(20)]; dodeca /= np.linalg.norm(dodeca, axis=1)[:, None]
P = dodeca[:-1]
STEPS = 4000

def watch(P, N, seed):
    torch.manual_seed(seed); np.random.seed(seed % 2**31)
    eng = make_engine("cuda:0", N, STEPS, P, "s3")
    for _ in range(STEPS): eng.step()
    B = eng.hist_macro_basin[:, :STEPS].cpu().numpy()
    nb = int(B.max()) + 1
    occ = np.array([np.bincount(B[:, t], minlength=nb) for t in range(STEPS)]).T.astype(float)
    leader = occ.argmax(0)
    runs = []; cur = leader[0]; ln = 1
    for t in range(1, STEPS):
        if leader[t] == cur: ln += 1
        else: runs.append(ln); cur = leader[t]; ln = 1
    runs.append(ln)
    share = occ.max(0).mean() / N
    p = occ / N; ent = -(p * np.log(p + 1e-9)).sum(0).mean()
    mm = boundary_metrics(eng, n_shuffles=20)
    return dict(N=N, seed=seed, leader_changes=len(runs) - 1,
                max_life=int(np.max(runs)), immortal=bool(len(runs) == 1),
                top_basin=int(occ.mean(1).argmax()),
                mean_leader_share=round(float(share), 3),
                entropy=round(float(ent), 3), z=round(float(mm["null_z"]), 1),
                tr=round(float(mm["transition_rate"]), 3))

out = []
for sd in (9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010, 9011, 9012):
    r = watch(P, 128, sd); out.append(r); print("N128", r, flush=True)
for N in (64, 256):
    for sd in (9001, 9002, 9003):
        r = watch(P, N, sd); out.append(r); print(f"N{N}", r, flush=True)

json.dump(out, open(os.path.join(ROOT, "tegmark_bridge", "outputs", "e2_immortal.json"), "w"), indent=1)
imm = sum(r["immortal"] for r in out)
print(f"DONE immortal {imm}/{len(out)}; top basins:", sorted({r['top_basin'] for r in out}))
