"""T6b — market certificate through the MULTISET door (weighted prototypes).

T6 (v1) verdict: unweighted k-means regime prototypes died ANTI
(protos24 z=-12.1). T7 says hostability is a multiset property:
recurrence = weights = stabilizing attractors. Re-test market substrates
as WEIGHTED prototype multisets: prototype replicated proportional to
cluster occupancy. Also entity-watch (births/lifespans) on market
substrates: regime persistence is the market analog of entity lifespan.
"""
import sys, os, json
import numpy as np
import pandas as pd
import torch
from scipy.cluster.vq import kmeans2

ROOT = r"C:\Users\akbon\OneDrive\Documents\GitHub\FourD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tegmark_bridge"))
from universality_test import boundary_metrics
from d1_autopsy import make_engine

HIST = r"E:\lpf_data\history"
cols = {}
for t in ["SPY", "SMH", "IWM", "MU"]:
    df = pd.read_parquet(os.path.join(HIST, f"{t}.parquet"))
    cols[t] = np.log(df["close"].astype(float)).diff()
R = pd.DataFrame(cols).dropna().tail(504).values
V = R - R.mean(0)
Pm = V / np.linalg.norm(V, axis=1, keepdims=True)

def metrics(P):
    r = np.random.RandomState(7)
    X = r.randn(2048, 4); X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = X @ P.T
    F = P.sum(0)[None, :] - C.sum(1)[:, None] * X
    B1 = np.linalg.norm(F, axis=1).mean()
    per = np.linalg.norm(P[None] - C[:, :, None] * X[:, None, :], axis=2)
    return float(B1 / per.sum(1).mean())

def engine_watch(P, N=128, steps=2000, seeds=(3001, 3002, 3003)):
    res = []
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd % 2**31)
        eng = make_engine("cuda:0", N, steps, P, "s3")
        for _ in range(steps): eng.step()
        mm = boundary_metrics(eng, n_shuffles=30)
        B = eng.hist_macro_basin[:, :steps].cpu().numpy()
        nb = int(B.max()) + 1
        occ = np.array([np.bincount(B[:, t], minlength=nb) for t in range(steps)]).T.astype(float)
        leader = occ.argmax(0)
        runs = []; cur = leader[0]; ln = 1
        for t in range(1, steps):
            if leader[t] == cur: ln += 1
            else: runs.append(ln); cur = leader[t]; ln = 1
        runs.append(ln)
        thr = 0.2 * N
        births = int(((occ[:-1] < thr) & (occ[1:] >= thr)).sum())
        p = occ / N; ent = -(p * np.log(p + 1e-9)).sum(0).mean()
        res.append(dict(z=round(float(mm["null_z"]), 1),
                        tr=round(float(mm["transition_rate"]), 3),
                        births=births,
                        life_mean=round(float(np.mean(runs)), 1),
                        life_max=int(np.max(runs)),
                        leader_share=round(float(occ.max(0).mean() / N), 3),
                        entropy=round(float(ent), 3)))
    agg = {k: round(float(np.mean([r[k] for r in res])), 3) for k in res[0]}
    agg["z_sd"] = round(float(np.std([r["z"] for r in res])), 1)
    return agg

def weighted_protos(P, k, total=60):
    from scipy.cluster.vq import kmeans2
    cent, lab = kmeans2(P, k, seed=3, minit='++')
    cent = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    cnt = np.bincount(lab, minlength=k).astype(float)
    share = cnt / cnt.sum()
    alloc = np.maximum(1, np.floor(share * total)).astype(int)
    while alloc.sum() > total:
        alloc[alloc.argmax()] -= 1
    while alloc.sum() < total:
        alloc[(share * total - alloc).argmax()] += 1
    W = np.repeat(cent, alloc, axis=0)
    return W, alloc, cnt

out = {}
for k in (8, 16, 24):
    cent, lab = kmeans2(Pm, k, seed=3, minit='++')
    cent = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    W, alloc, cnt = weighted_protos(Pm, k)
    uw = engine_watch(cent)
    w = engine_watch(W)
    out[f"protos{k}_unw"] = dict(Brel=round(metrics(cent), 4), n=k, **uw)
    out[f"protos{k}_wtd"] = dict(Brel=round(metrics(W), 4), n=len(W), **w)
    print(f"k={k} UNW n={k:3d} Brel={out[f'protos{k}_unw']['Brel']:.3f} {uw}")
    print(f"k={k} WTD n={len(W):3d} Brel={out[f'protos{k}_wtd']['Brel']:.3f} {w}", flush=True)

# raw market sample entity watch too (was BOILS at 1000 steps; 2000 steps?)
r3 = np.random.RandomState(11); mkt = Pm[r3.choice(504, 60, replace=False)]
out["market60"] = dict(Brel=round(metrics(mkt), 4), **engine_watch(mkt))
print("market60", out["market60"], flush=True)

json.dump(out, open(os.path.join(ROOT, "tegmark_bridge", "outputs", "t6b_weighted.json"), "w"), indent=1)
print("WROTE t6b_weighted.json")
