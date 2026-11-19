"""D1 — Mechanism autopsy: WHY are centrally-closed polytopes dead?

Three measurements per structure:

1. DESIGN CERTIFICATE. Second moment M2 = Σ p pᵀ (isotropic <=> spherical
   2-design) and the fourth-moment tensor T4 = Σ p⊗p⊗p⊗p projected onto its
   O(4)-invariant part (averaged over random rotations). Deviations dev2,
   dev4 quantify how much directional anisotropy the structure leaks at
   each order. Polytopes are near-perfect designs; gaps/random sets leak.

2. TRANSITION FORENSICS. The engine's macro basins (24 on S3) are
   themselves nearly antipodally symmetric. If a dead system's
   transitions are mostly MIRROR BOUNCES (basin a -> its antipode basin),
   then "dead" does not mean motionless — it means every negotiation is
   exactly cancelled by its mirror image, and any reflection-symmetric
   readout measures zero. Bounce fraction is the smoking gun.

3. CROSS-ENGINE CHECK. Same substrate, manifold='flat4' instead of 's3':
   is the death a property of the STRUCTURE or of the s3 dynamics?

Run: python tegmark_bridge/d1_autopsy.py [--seeds 1] [--steps 1000]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vfd_bridge"))

from universality_test import UniversalEngine, boundary_metrics
from graphs import six00_cell


def design_devs(P):
    """dev2, dev4: normalized anisotropy of 2nd and 4th moment tensors."""
    M2 = P.T @ P
    iso2 = (np.trace(M2) / 4) * np.eye(4)
    dev2 = np.linalg.norm(M2 - iso2) / np.linalg.norm(M2)
    T4 = np.einsum('ia,ib,ic,id->abcd', P, P, P, P)
    rng = np.random.RandomState(11)
    Tinv = np.zeros_like(T4)
    nrot = 120
    for _ in range(nrot):
        Q, _ = np.linalg.qr(rng.randn(4, 4))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        Tinv += np.einsum('ae,be,ce,de,abcd->abcd', Q, Q, Q, Q, T4)
    Tinv /= nrot
    dev4 = np.linalg.norm(T4 - Tinv) / np.linalg.norm(T4)
    return float(dev2), float(dev4)


def make_engine(device, N, steps, prefs_np, manifold, topology='random'):
    eng = UniversalEngine(N=N, device=device, steps=steps, manifold=manifold,
                          topology=topology, fatigue_type='gradual')
    if prefs_np is None:
        return eng
    P = torch.tensor(prefs_np, dtype=torch.float32, device=eng.device)
    P = P / (P.norm(dim=1, keepdim=True) + 1e-12)
    eng.prefs = P
    eng.n_sub = P.shape[0]
    eng.fatigue = torch.zeros(N, eng.n_sub, device=eng.device)
    return eng


def antipode_map(centers):
    """For each macro basin, index of the basin whose center is most antipodal."""
    C = centers.cpu().numpy() if torch.is_tensor(centers) else centers
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ C.T
    return np.argmin(S, axis=1), np.min(S, axis=1)


def forensics(eng):
    H = eng.hist_macro_basin.cpu().numpy()  # N x steps
    ant, quality = antipode_map(eng.macro_centers.cpu().numpy())
    trans = mirror = 0
    for i in range(H.shape[0]):
        h = H[i, 100:]
        idx = np.nonzero(h[1:] != h[:-1])[0]
        for k in idx:
            a, b = h[k], h[k + 1]
            trans += 1
            if ant[a] == b or ant[b] == a:
                mirror += 1
    bf = mirror / trans if trans else float('nan')
    return bf, trans


def run_struct(name, P, args, manifold):
    zs, bfs, trans_tot = [], [], 0
    dev2 = dev4 = float('nan')
    for seed in range(args.seeds):
        torch.manual_seed(4000 + seed); np.random.seed(4000 + seed)
        eng = make_engine(args.device, args.N, args.steps, P, manifold,
                          topology=('cyclic' if P is None else 'random'))
        for _ in range(args.steps):
            eng.step()
        mm = boundary_metrics(eng, n_shuffles=50)
        zs.append(mm['null_z'])
        bf, tr = forensics(eng)
        bfs.append(bf); trans_tot += tr
    if P is not None:
        dev2, dev4 = design_devs(np.asarray(P))
    else:  # cyclic baseline: read prefs off engine
        dev2, dev4 = design_devs(eng.prefs.cpu().numpy())
    row = {'name': name, 'manifold': manifold,
           'dev2': dev2, 'dev4': dev4,
           'null_z': float(np.mean(zs)), 'bounce_frac': float(np.mean(bfs)),
           'transitions': trans_tot}
    print(f"  {name:<20s} [{manifold}] dev2={dev2:.4f} dev4={dev4:.4f} "
          f"z={row['null_z']:+6.1f} bounce={row['bounce_frac']:.3f} "
          f"(n_tr={trans_tot})", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--N', type=int, default=128)
    ap.add_argument('--seeds', type=int, default=1)
    args = ap.parse_args()

    phi = (1 + 5 ** 0.5) / 2
    _, m6 = six00_cell()
    V = m6['vertices']
    rng = np.random.RandomState(5)
    base = rng.randn(12, 4); base /= np.linalg.norm(base, axis=1, keepdims=True)
    anti12 = np.vstack([base, -base])
    s8 = V[rng.choice(len(V), 8, replace=False)]
    dodeca3 = np.array(
        [[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)] +
        [[0,s1/phi,s2*phi] for s1 in (-1,1) for s2 in (-1,1)] +
        [[s1/phi,s2*phi,0] for s1 in (-1,1) for s2 in (-1,1)] +
        [[s1*phi,0,s2/phi] for s1 in (-1,1) for s2 in (-1,1)], float)
    dodeca3 /= np.linalg.norm(dodeca3, axis=1)[:, None]
    dodeca = np.c_[dodeca3, np.zeros(20)]
    eps01 = V + 0.1 * (rng.randn(4) / np.linalg.norm(rng.randn(4)))

    structs = [
        ('600cell_full', V), ('600cell_s8', s8), ('antipodal12', anti12),
        ('dodeca20', dodeca), ('dodeca20_minus1', dodeca[:-1]),
        ('eps0.1_600', eps01), ('cyclic8', None),
    ]
    rows = []
    print('== s3 (canonical engine) ==')
    for name, P in structs:
        rows.append(run_struct(name, P, args, 's3'))
    print('== flat4 (cross-engine check) ==')
    for name, P in [('600cell_full', V), ('dodeca20', dodeca),
                    ('dodeca20_minus1', dodeca[:-1]), ('600cell_s8', s8)]:
        rows.append(run_struct(name, P, args, 'flat4'))

    out = os.path.join(HERE, 'outputs', 'd1_autopsy.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'rows': rows, 'seeds': args.seeds}, open(out, 'w'), indent=2)
    print(f'\nJSON: {out}')


if __name__ == '__main__':
    main()
