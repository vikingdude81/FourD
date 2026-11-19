"""Cross-dimension probe: the Karkada et al. month-circle vs our closure certificate.

Claim under test: the Z12 month orbit (regular 12-gon) forces beautiful
embedding geometry in LLMs (constructive symmetry), but as a FourD
preference substrate it is antipodally closed -> force cancellation ->
boundary-negotiation dynamics DEAD. Odd-n orbits (weekdays, Z7) are open
-> alive. Same symmetry, opposite sign, decided by the certificate.
"""
import sys, os, json
import numpy as np, torch
ROOT = r"C:\Users\akbon\OneDrive\Documents\GitHub\FourD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tegmark_bridge"))
from universality_test import boundary_metrics
from d1_autopsy import make_engine

def ngon(n, jitter=0.0, seed=99):
    k = np.arange(n)
    V = np.zeros((n, 4))
    V[:, 0] = np.cos(2 * np.pi * k / n)
    V[:, 1] = np.sin(2 * np.pi * k / n)
    if jitter > 0:  # full-rank perturbation: breaks coplanarity, keeps parity structure
        rng = np.random.RandomState(seed)
        V = V + jitter * rng.randn(n, 4)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V

def anti_frac(P):
    G = P @ P.T
    n = len(P); c = 0
    for i in range(n):
        g = G[i].copy(); g[i] = 1.0
        if (g < -0.999).any(): c += 1
    return c / n

res = []
for jit in (0.0, 0.1, 0.25):
    for n, label in [(7, "weekdays_Z7"), (8, "octagon_Z8"), (11, "Z11"), (12, "months_Z12"),
                     (13, "Z13"), (24, "hours_Z24")]:
        P = ngon(n, jitter=jit)
        zs, trs = [], []
        for sd in (3001, 3002, 3003):
            torch.manual_seed(sd); np.random.seed(sd)
            eng = make_engine("cuda:0", 128, 1000, P, "s3")
            for _ in range(1000): eng.step()
            mm = boundary_metrics(eng, n_shuffles=50)
            zs.append(mm["null_z"]); trs.append(mm["transition_rate"])
        af = anti_frac(P)
        z, tr = float(np.mean(zs)), float(np.trunc(np.mean(trs) * 1000) / 1000)
        state = "HOSTS" if z > 3 else ("DEAD" if abs(z) < 3 else "ANTI")
        res.append(dict(n=n, label=label, jitter=jit, antiFrac=round(af, 3),
                        z=round(z, 1), tr=round(tr, 3), state=state))
        print(f"jit={jit:4.2f} {label:12s} n={n:2d} antiFrac={af:.3f}  z={z:+7.1f}  tr={tr:.3f} -> {state}", flush=True)

out = os.path.join(ROOT, "tegmark_bridge", "outputs", "ngon_certificate.json")
json.dump(res, open(out, "w"), indent=1)
print("WROTE", out)
