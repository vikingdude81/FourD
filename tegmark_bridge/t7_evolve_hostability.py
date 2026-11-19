"""T7 — selection for hostability: evolve substrates that HOST minds.

Tegmark's measure question made computational: run selection pressure over
structure space itself. Individuals = point-cloud preference substrates in
R^4 (n=16). Fitness = boundary-negotiation z with a boiling penalty
(tr>=0.85 -> score capped). Mutations: jitter a point, add/delete a point,
add its exact antipode (a 'closure mutation' — does selection ever choose
symmetry?), duplicate a point.

Prediction from CERT2: lineages should keep Brel>0 (fuel), avoid exact
antipodes, and settle in the middle-spacing band. If closed/high-conc
geometries ever win, CERT2's axes are wrong.
"""
import sys, os, json, time
import numpy as np
import torch

ROOT = r"C:\Users\akbon\OneDrive\Documents\GitHub\FourD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tegmark_bridge"))
from universality_test import boundary_metrics
from d1_autopsy import make_engine
# metrics inlined below (cert2_coverage must NOT be imported: it runs its corpus at import)
def metrics(P):
    r = np.random.RandomState(7)
    X = r.randn(2048, 4); X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = X @ P.T
    F = P.sum(0)[None, :] - C.sum(1)[:, None] * X
    B1 = np.linalg.norm(F, axis=1).mean()
    per = np.linalg.norm(P[None] - C[:, :, None] * X[:, None, :], axis=2)
    Brel = B1 / per.sum(1).mean()
    G = P @ P.T; angs = np.arccos(np.clip(G, -1, 1))
    iu = np.triu_indices(len(P), 1)
    drel = angs[iu].min() / np.median(angs[iu])
    anti = 0
    for i in range(len(P)):
        g = G[i].copy(); g[i] = 1.0
        if (g < -0.999).any(): anti += 1
    return float(Brel), float(drel), anti / len(P)

DEV = "cuda:0"; N_AG = 64; STEPS = 600

def fitness(P, seed=4242):
    torch.manual_seed(seed); np.random.seed(seed % 2**31)
    eng = make_engine(DEV, N_AG, STEPS, P, "s3")
    for _ in range(STEPS): eng.step()
    mm = boundary_metrics(eng, n_shuffles=20)
    z = mm["null_z"]; tr = mm["transition_rate"]
    score = z if tr < 0.85 else z - 40.0 * (tr - 0.85) / 0.15
    return float(score), float(z), float(tr)

def norm(P): return P / np.linalg.norm(P, axis=1, keepdims=True)

def mutate(P, rng):
    P = P.copy(); n = len(P); kind = rng.randint(6)
    if kind == 0:      # jitter one point
        i = rng.randint(n); P[i] += rng.randn(4) * 0.35
    elif kind == 1:    # global small jitter
        P += rng.randn(n, 4) * 0.12
    elif kind == 2:    # add random point
        P = np.vstack([P, rng.randn(1, 4)])
    elif kind == 3:    # delete random point
        i = rng.randint(n); P = np.delete(P, i, 0)
    elif kind == 4:    # closure mutation: add exact antipode of a random point
        i = rng.randint(n); P = np.vstack([P, -P[i:i+1]])
    else:              # duplicate a point
        i = rng.randint(n); P = np.vstack([P, P[i:i+1]])
    if len(P) < 6: P = np.vstack([P, rng.randn(6 - len(P), 4)])
    if len(P) > 40:
        keep = rng.choice(len(P), 40, replace=False); P = P[keep]
    return norm(P)

rng = np.random.RandomState(77)
POP = 12
pop = [norm(rng.randn(16, 4)) for _ in range(POP)]
results = []
t0 = time.time()
for gen in range(16):
    scored = []
    for P in pop:
        s, z, tr = fitness(P)
        scored.append((s, z, tr, P))
    scored.sort(key=lambda x: -x[0])
    best = scored[0]
    Brel, drel, af = metrics(best[3])
    results.append(dict(gen=gen, best=round(best[0], 2), z=round(best[1], 1),
                        tr=round(best[2], 3), n=len(best[3]),
                        Brel=round(Brel, 4), drel=round(drel, 4), antiFrac=round(af, 3),
                        secs=round(time.time() - t0)))
    print(results[-1], flush=True)
    elites = [x[3] for x in scored[:4]]
    pop = elites + [mutate(elites[rng.randint(4)], rng) for _ in range(POP - 4)]

out = os.path.join(ROOT, "tegmark_bridge", "outputs", "t7_evolve.json")
json.dump(dict(runs=results, final_best=[best[3].round(4).tolist()]), open(out, "w"), indent=1)
print("WROTE", out, "total_secs", round(time.time() - t0))
