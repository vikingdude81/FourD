"""Export demo assets for the kill-switch demo + precompute reference curves.

Writes tegmark_bridge/demo/data.json:
  - macro_centers (24x4), cyclic8 prefs
  - substrates: 600cell, 24cell, dodeca20, tetra, antipodal12
  - python reference curves: eps-scan z, gap-scan z (remove m single vertices)
  - engine params for the JS port
"""
import json, os, sys
import numpy as np
import torch

ROOT = r"C:\Users\akbon\OneDrive\Documents\GitHub\FourD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "vfd_bridge"))
from universality_test import UniversalEngine, boundary_metrics
from graphs import six00_cell, two4_cell

sys.path.insert(0, os.path.join(ROOT, "tegmark_bridge"))
from d1_autopsy import make_engine

DEV = 'cuda:0'
_, m6 = six00_cell(); V600 = m6['vertices'].astype(np.float64)
_, m24 = two4_cell(); V24 = m24['vertices'].astype(np.float64)
phi = (1 + 5 ** 0.5) / 2
dodeca = np.array(
    [[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)] +
    [[0,s1/phi,s2*phi] for s1 in (-1,1) for s2 in (-1,1)] +
    [[s1/phi,s2*phi,0] for s1 in (-1,1) for s2 in (-1,1)] +
    [[s1*phi,0,s2/phi] for s1 in (-1,1) for s2 in (-1,1)], float)
dodeca = np.c_[dodeca, np.zeros(20)]
dodeca /= np.linalg.norm(dodeca, axis=1)[:, None]
tet = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float)
tet = np.c_[tet, np.zeros(4)]
tet /= np.linalg.norm(tet, axis=1)[:, None]
rng = np.random.RandomState(5)
base = rng.randn(12, 4); base /= np.linalg.norm(base, axis=1, keepdims=True)
anti12 = np.vstack([base, -base])

def z_of(P, seed=4000):
    torch.manual_seed(seed); np.random.seed(seed)
    eng = make_engine(DEV, 128, 1000, P, 's3')
    for _ in range(1000):
        eng.step()
    mm = boundary_metrics(eng, n_shuffles=50)
    return round(mm['null_z'], 1), round(mm['transition_rate'], 3)

# reference curves (python engine, authoritative)
eps_grid, eps_z = [], []
u = rng.randn(4); u /= np.linalg.norm(u)
for e in [0.0, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
    z, tr = z_of(V600 + e * u)
    eps_grid.append(e); eps_z.append({'eps': e, 'z': z, 'tr': tr})
    print('eps', e, z, flush=True)

gap_grid, gap_z = [], []
perm = rng.permutation(len(V600))
for m in [0, 1, 2, 3, 5, 8, 12, 20, 30]:
    keep = np.ones(len(V600), bool); keep[perm[:m]] = False
    z, tr = z_of(V600[keep])
    gap_grid.append(m); gap_z.append({'removed': m, 'z': z, 'tr': tr})
    print('gap', m, z, flush=True)

eng0 = UniversalEngine(N=8, device=DEV, steps=2, manifold='s3', topology='cyclic')
cyc8 = eng0.prefs.cpu().numpy()
mc = eng0.macro_centers.cpu().numpy()

out = {
    'substrates': {
        'cell600': V600.tolist(), 'cell24': V24.tolist(),
        'dodeca20': dodeca.tolist(), 'tetra': tet.tolist(),
        'anti12': anti12.tolist(), 'cyclic8': cyc8.tolist(),
    },
    'macro_centers': mc.tolist(),
    'params': {'N': 128, 'steps': 1000, 'steering': 0.707, 'fatigue_rate': 0.217,
               'noise': 0.25, 'beta_macro': 11.375, 'recovery': 0.025,
               'floor': 0.05, 'novelty_w': 0.6, 'infl_a': 0.5, 'infl_b': 0.3},
    'ref_eps': eps_z, 'ref_gap': gap_z,
    'ref_static': {k: z_of(np.array(v))[0] for k, v in
                   [('cell600', V600), ('cell24', V24), ('dodeca20', dodeca),
                    ('dodeca19', dodeca[:-1]), ('tetra', tet), ('anti12', anti12)]},
}
dest = os.path.join(ROOT, 'tegmark_bridge', 'demo')
os.makedirs(dest, exist_ok=True)
json.dump(out, open(os.path.join(dest, 'data.json'), 'w'))
print('WROTE', os.path.join(dest, 'data.json'), out['ref_static'])
