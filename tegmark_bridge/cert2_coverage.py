"""CERT2 — the coverage certificate: does pure geometry predict engine fate?

Refined order parameter (from N7): NOT antipodal closure per se, but
uniform closed coverage -> local force balance -> no fuel. Two scalars:

  B = E_{x~U(S3)} || sum_p (p - (x.p)x) w_p ||   (net tangential force
      under uniform attention; B~0 = force-balanced = unfuelled)
      Also B2 = E_x min_p ||p - (x.p)x||  (coverage: every direction
      answered closely?)  -- use coverage radius directly.
  D = min pairwise angle / median pairwise angle (spacing resolution;
      small = congested cone = basin jitter axis)

Corpus spans everything measured so far; engine z recomputed consistently
(3 seeds x 1000 steps, N=128, s3) so metrics and labels share one run.
Prediction: z>3 (HOSTS) iff B high AND D not tiny.
"""
import json, os, sys
import numpy as np
import torch

ROOT = r"C:\Users\akbon\OneDrive\Documents\GitHub\FourD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tegmark_bridge"))
from universality_test import UniversalEngine, boundary_metrics
from d1_autopsy import make_engine

rng = np.random.RandomState(42)

# ---------- metrics ----------
def B_force(P, n_x=2048, seed=7):
    r = np.random.RandomState(seed)
    X = r.randn(n_x, 4); X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = X @ P.T                                  # n_x x S
    # net force F(x) = sum_p w_p (p - (x.p) x), uniform w
    F = P.sum(0)[None, :] - C.sum(1)[:, None] * X
    B1 = np.linalg.norm(F, axis=1).mean()
    # mean per-pair cancellation strength: ||F|| / sum ||p-(x.p)x||
    per = np.linalg.norm(P[None, :, :] - C[:, :, None] * X[:, None, :], axis=2)
    B2 = B1 / per.sum(1).mean()
    # coverage radius: max over x of min_p angle(x,p)
    cov = np.arccos(np.clip(C.max(1), -1, 1)).max()
    G = P @ P.T
    angs = np.arccos(np.clip(G, -1, 1))
    iu = np.triu_indices(len(P), 1)
    dmin = angs[iu].min(); dmed = np.median(angs[iu])
    M2 = P.T @ P
    w = np.linalg.eigvalsh(M2)
    conc = float(w.max() / w.sum())
    return dict(B=round(float(B1), 4), Brel=round(float(B2), 4),
                cover=round(float(cov), 4), dmin=round(float(dmin), 4),
                drel=round(float(dmin / dmed), 4), conc=round(conc, 4))

def engine_z(P, seeds=(2026, 2027, 2028)):
    zs, trs = [], []
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        eng = make_engine("cuda:0", 128, 1000, P, "s3")
        for _ in range(1000): eng.step()
        mm = boundary_metrics(eng, n_shuffles=50)
        zs.append(mm["null_z"]); trs.append(mm["transition_rate"])
    return float(np.mean(zs)), float(np.mean(trs))

# ---------- corpus ----------
sys.path.insert(0, os.path.join(ROOT, "vfd_bridge"))
from graphs import six00_cell, two4_cell
_, m6 = six00_cell(); V600 = m6['vertices'].astype(np.float64)
_, m24 = two4_cell(); V24 = m24['vertices'].astype(np.float64)
phi = (1 + 5 ** 0.5) / 2
dodeca = np.array(
    [[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)] +
    [[0,s1/phi,s2*phi] for s1 in (-1,1) for s2 in (-1,1)] +
    [[s1/phi,s2*phi,0] for s1 in (-1,1) for s2 in (-1,1)] +
    [[s1*phi,0,s2/phi] for s1 in (-1,1) for s2 in (-1,1)], float)
dodeca = np.c_[dodeca, np.zeros(20)]; dodeca /= np.linalg.norm(dodeca, axis=1)[:, None]
tet = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float)
tet = np.c_[tet, np.zeros(4)]; tet /= np.linalg.norm(tet, axis=1)[:, None]
b12 = np.random.RandomState(5).randn(12, 4); b12 /= np.linalg.norm(b12, axis=1, keepdims=True)

def ngon(n, jit=0.0, seed=99):
    k = np.arange(n); V = np.zeros((n, 4))
    V[:, 0] = np.cos(2*np.pi*k/n); V[:, 1] = np.sin(2*np.pi*k/n)
    if jit: V = V + jit*np.random.RandomState(seed).randn(n, 4)
    return V / np.linalg.norm(V, axis=1, keepdims=True)

r2 = np.random.RandomState(5)
u4 = r2.randn(4); u4 /= np.linalg.norm(u4)

import pandas as pd
HIST = r"E:\lpf_data\history"
cols = {}
for t in ["SPY", "SMH", "IWM", "MU"]:
    df = pd.read_parquet(os.path.join(HIST, f"{t}.parquet"))
    cols[t] = np.log(df["close"].astype(float)).diff()
R = pd.DataFrame(cols).dropna().tail(504).values
Vc = R - R.mean(0); Pm = Vc / np.linalg.norm(Vc, axis=1, keepdims=True)
r3 = np.random.RandomState(11); idx60 = r3.choice(504, 60, replace=False)
mkt = Pm[idx60]
rnd60 = r3.randn(60, 4); rnd60 /= np.linalg.norm(rnd60, axis=1, keepdims=True)
from scipy.cluster.vq import kmeans2
protos24 = kmeans2(Pm, 24, seed=3, minit='++')[0]
protos24 = protos24 / np.linalg.norm(protos24, axis=1, keepdims=True)
sub8 = V600[r2.choice(120, 8, replace=False)]
sub60 = V600[r2.choice(120, 60, replace=False)]

CORPUS = {
  "cell600": V600, "cell24": V24, "dodeca20": dodeca, "dodeca19": dodeca[:-1],
  "tetra": tet, "anti12": np.vstack([b12, -b12]),
  "ngon12": ngon(12), "ngon12j": ngon(12, 0.25), "ngon7": ngon(7), "ngon7j": ngon(7, 0.25),
  "ngon24j": ngon(24, 0.25),
  "eps600_0.03": V600 + 0.03*u4, "eps600_0.1": V600 + 0.1*u4,
  "sub600_60": sub60, "sub600_8": sub8,
  "market60": mkt, "market60_closed": np.vstack([mkt, -mkt]),
  "random60": rnd60, "protos24": protos24,
}

rows = []
for name, P in CORPUS.items():
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    met = B_force(P)
    z, tr = engine_z(P)
    state = "HOSTS" if (z > 3 and tr < 0.9) else ("BOILS" if tr >= 0.9 else ("DEAD" if abs(z) < 3 else "ANTI"))
    rows.append(dict(name=name, n=len(P), z=round(z, 1), tr=round(tr, 3), state=state, **met))
    print(f"{name:18s} n={len(P):3d} B={met['B']:7.3f} Brel={met['Brel']:6.4f} "
          f"cover={met['cover']:5.3f} drel={met['drel']:5.3f}  z={z:+7.1f} tr={tr:.3f} -> {state}", flush=True)

out = os.path.join(ROOT, "tegmark_bridge", "outputs", "cert2_corpus.json")
json.dump(rows, open(out, "w"), indent=1)
print("WROTE", out)
