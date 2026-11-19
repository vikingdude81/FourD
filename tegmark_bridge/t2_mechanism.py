"""T2 — mechanism pinning for the T1 collapse.

Two independent tests of the claim: the boundary-negotiation signal dies
under CENTRAL SYMMETRY (preference set closed under v -> -v) because the
preference force field cancels exactly.

Experiment A (epsilon-restore): start from the dead substrate (full 600-cell,
antipodal_frac=1.0) and perturb every vertex by eps*u along one fixed random
direction u, then renormalize. eps=0 is dead; eps infinitesimal already
breaks EXACT antipodal closure (antipodal_frac -> 0). If exact cancellation
is the mechanism, the signal should restore at small eps — a discontinuity
at eps=0 in the derivative, i.e. the dead state is a knife-edge.

Experiment B (family zoo): 3-D polytopes embedded in an R4 hyperplane.
EVERY centrally symmetric polytope (octahedron, cube, icosahedron,
dodecahedron, hypercube {±1}^4) is predicted DEAD; the tetrahedron (not
antipodally closed) predicted ALIVE. Cross-family confirmation that the
order parameter is antipodal closure, not H4 or dimension.

Readouts identical to T1 (C1: null_z > 3; C2: TR seed-std > 0.01).

Run: python tegmark_bridge/t2_mechanism.py [--seeds 3] [--steps 1000] [--N 128]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vfd_bridge"))

from universality_test import UniversalEngine, boundary_metrics
from graphs import six00_cell


def make_engine(device, N, steps, prefs_np):
    eng = UniversalEngine(N=N, device=device, steps=steps, manifold='s3',
                          topology='random', fatigue_type='gradual')
    P = torch.tensor(prefs_np, dtype=torch.float32, device=eng.device)
    P = P / (P.norm(dim=1, keepdim=True) + 1e-12)
    eng.prefs = P
    eng.n_sub = P.shape[0]
    eng.fatigue = torch.zeros(N, eng.n_sub, device=eng.device)
    return eng


def antipodal_frac(P):
    idx = {tuple(np.round(r, 6)) for r in P}
    return sum(1 for r in np.round(-P, 6) if tuple(r) in idx) / len(P)


def embed3(V3):
    n = np.linalg.norm(V3, axis=1)
    return np.c_[V3 / n[:, None], np.zeros(len(V3))]


def zoo():
    phi = (1 + 5 ** 0.5) / 2
    tet = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float)
    octa = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
    cube = np.array([[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)], float)
    icosa = np.array([
        [0, 1, phi],[0, 1, -phi],[0, -1, phi],[0, -1, -phi],
        [1, phi, 0],[1, -phi, 0],[-1, phi, 0],[-1, -phi, 0],
        [phi, 0, 1],[phi, 0, -1],[-phi, 0, 1],[-phi, 0, -1]], float)
    dodeca = np.array(
        [[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)] +
        [[0,s1/phi,s2*phi] for s1 in (-1,1) for s2 in (-1,1)] +
        [[s1/phi,s2*phi,0] for s1 in (-1,1) for s2 in (-1,1)] +
        [[s1*phi,0,s2/phi] for s1 in (-1,1) for s2 in (-1,1)], float)
    hyper = np.array([[a,b,c,d] for a in (-1,1) for b in (-1,1) for c in (-1,1) for d in (-1,1)], float)
    fams = {'tetra': embed3(tet), 'octa': embed3(octa), 'cube8': embed3(cube),
            'icosa': embed3(icosa), 'dodeca20': embed3(dodeca),
            'hypercube16': hyper / np.sqrt(4)}
    return fams


def run_variant(name, prefs, args, seeds):
    zs, ds, trs, afs = [], [], [], []
    for seed in range(seeds):
        torch.manual_seed(3000 + seed); np.random.seed(3000 + seed)
        eng = make_engine(args.device, args.N, args.steps, prefs)
        for _ in range(args.steps):
            eng.step()
        mm = boundary_metrics(eng, n_shuffles=50)
        zs.append(mm['null_z']); ds.append(mm['cohens_d']); trs.append(mm['transition_rate'])
        afs.append(antipodal_frac(prefs))
    row = {'name': name, 'n_sub': len(prefs),
           'antipodal_frac': float(np.mean(afs)),
           'null_z_mean': float(np.mean(zs)), 'null_z_std': float(np.std(zs)),
           'cohens_d_mean': float(np.mean(ds)),
           'TR_mean': float(np.mean(trs)), 'TR_std': float(np.std(trs)),
           'hosts': bool(np.mean(zs) > 3 and np.std(trs) > 0.01)}
    print(f"  {name:<22s} n={row['n_sub']:>3d} antiF={row['antipodal_frac']:.3f} "
          f"z={row['null_z_mean']:+6.1f}±{row['null_z_std']:4.1f} "
          f"TR={row['TR_mean']:.3f}±{row['TR_std']:.3f} "
          f"{'HOSTS' if row['hosts'] else 'no'}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--N', type=int, default=128)
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()

    rows = []
    print('== Exp A: epsilon-restore on the dead 600-cell substrate ==')
    _, m6 = six00_cell()
    V = m6['vertices']
    rng = np.random.RandomState(7)
    u = rng.randn(4); u /= np.linalg.norm(u)
    for eps in [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        prefs = V + eps * u
        rows.append(run_variant(f'eps={eps:g}', prefs, args, args.seeds))

    print('== Exp B: family zoo (3D polytopes in R4 hyperplane + hypercube) ==')
    for name, P in zoo().items():
        rows.append(run_variant(name, P, args, args.seeds))
        if name in ('octa', 'cube8', 'icosa', 'dodeca20', 'hypercube16'):
            rows.append(run_variant(name + '_minus1', P[:-1], args, args.seeds))

    out = os.path.join(HERE, 'outputs', 't2_mechanism.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'device': args.device, 'N': args.N, 'steps': args.steps,
               'seeds': args.seeds, 'rows': rows}, open(out, 'w'), indent=2)
    print(f'\nJSON: {out}')


if __name__ == '__main__':
    main()
