"""E1 — entity watch: lifespans of persistent substructures inside substrates.

Question: on hospitable structures, do 'entities' emerge — basin occupancy
clusters with real lifespans (birth, dominance, death)? Dead substrates
should show flicker (no persistence); hosts should show long-lived
occupancy regimes. Observation of mathematical entities, not just rates.
"""
import sys, os, json
import numpy as np
import torch

ROOT = r"C:\Users\akbon\OneDrive\Documents\GitHub\FourD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tegmark_bridge"))
sys.path.insert(0, os.path.join(ROOT, "vfd_bridge"))
from universality_test import boundary_metrics
from d1_autopsy import make_engine
from graphs import six00_cell

phi = (1 + 5 ** 0.5) / 2
dodeca = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)] +
    [[0, s1/phi, s2*phi] for s1 in (-1, 1) for s2 in (-1, 1)] +
    [[s1/phi, s2*phi, 0] for s1 in (-1, 1) for s2 in (-1, 1)] +
    [[s1*phi, 0, s2/phi] for s1 in (-1, 1) for s2 in (-1, 1)], float)
dodeca = np.c_[dodeca, np.zeros(20)]; dodeca /= np.linalg.norm(dodeca, axis=1)[:, None]
_, m6 = six00_cell(); V600 = m6["vertices"].astype(np.float64)
r2 = np.random.RandomState(5)
sub8 = V600[r2.choice(120, 8, replace=False)]
r3 = np.random.RandomState(11)
rnd60 = r3.randn(60, 4); rnd60 /= np.linalg.norm(rnd60, axis=1, keepdims=True)

STEPS = 4000
def watch(name, P):
    torch.manual_seed(9001); np.random.seed(9001)
    eng = make_engine("cuda:0", 128, STEPS, P, "s3")
    for _ in range(STEPS): eng.step()
    B = eng.hist_macro_basin[:, :STEPS].cpu().numpy()   # agents x steps
    nb = int(B.max()) + 1
    occ = np.array([np.bincount(B[:, t], minlength=nb) for t in range(STEPS)]).T.astype(float)
    leader = occ.argmax(0)
    # leader regimes: contiguous runs of same leader = entity dominance lives
    runs = []; cur = leader[0]; ln = 1
    for t in range(1, STEPS):
        if leader[t] == cur: ln += 1
        else: runs.append(ln); cur = leader[t]; ln = 1
    runs.append(ln)
    # entity births: basin crosses 20% occupancy up; deaths cross below
    thr = 0.2 * 128
    births = int(((occ[:-1] < thr) & (occ[1:] >= thr)).sum())
    ent = -(occ / 128.0) * np.log((occ + 1e-9) / 128.0)
    ent = ent.sum(0)
    mm = boundary_metrics(eng, n_shuffles=50)
    stats = dict(name=name, n_basins=nb, n_leader_changes=len(runs) - 1,
                 mean_life=round(float(np.mean(runs)), 1),
                 max_life=round(float(np.max(runs)), 0),
                 median_life=round(float(np.median(runs)), 1),
                 births=births, occ_entropy=round(float(ent.mean()), 3),
                 z=round(float(mm["null_z"]), 1), tr=round(float(mm["transition_rate"]), 3))
    print(stats, flush=True)
    return stats, occ

out = {}
for name, P in [("dodeca19", dodeca[:-1]), ("dodeca20", dodeca),
                ("sub600_8", sub8), ("cell600", V600), ("random60", rnd60)]:
    s, occ = watch(name, P)
    out[name] = s
    if name in ("dodeca19", "dodeca20", "cell600"):
        np.save(os.path.join(ROOT, "tegmark_bridge", "outputs", f"e1_occ_{name}.npy"), occ)

json.dump(out, open(os.path.join(ROOT, "tegmark_bridge", "outputs", "e1_entities.json"), "w"), indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for a, nm in zip(ax, ("dodeca19", "dodeca20", "cell600")):
    occ = np.load(os.path.join(ROOT, "tegmark_bridge", "outputs", f"e1_occ_{nm}.npy"))
    a.imshow(occ, aspect="auto", cmap="viridis", vmax=60)
    s = out[nm]
    a.set_ylabel(f"{nm}\nlife {s['mean_life']:.0f} / chg {s['n_leader_changes']}")
ax[-1].set_xlabel("step"); fig.suptitle("E1 entity timelines: basin occupancy (rows) over 4000 steps")
fig.tight_layout()
p = os.path.join(ROOT, "tegmark_bridge", "outputs", "e1_entities.png")
fig.savefig(p, dpi=120); print("SAVED", p)
