"""T1 — SAS census: how much broken symmetry does a structure need to host
boundary-negotiation dynamics?

Tegmark's MUH says every mathematical structure "exists"; his measure problem
asks how to weigh them. Our S2 result (antipodal closure kills the
boundary-negotiation / bearer signal) suggests a concrete, measurable answer:
structures are NOT equally hospitable, and the hospitable axis is broken
symmetry. This experiment turns that into a curve.

Dial: take the 600-cell vertex set and subsample k vertices (seeded, without
replacement). k=120 is the maximally symmetric (antipodally closed) structure;
small k is generic/asymmetric. For each k, run the UniversalEngine with those
k vertices as the preference substrate and read out the canonical
boundary-negotiation metrics + seed-sensitivity of transition_rate.

Two hostability criteria (pre-registered):
  C1 (signal):     null_z mean > 3
  C2 (history):    transition_rate std across seeds > 0.01
                   (antipodal-closed S2 gave 0.365±0.001 = seed-deaf)
A structure "hosts" a bearer if C1 and C2 both pass.

Run: python tegmark_bridge/t1_sas_census.py [--device cuda:0] [--steps 1000] [--N 128] [--seeds 3]
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
sys.path.insert(0, ROOT)                       # repo root
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


def antipodal_closure_frac(P):
    """Fraction of rows whose exact negation is also present."""
    neg = -P
    idx = {tuple(np.round(r, 6)) for r in P}
    hits = sum(1 for r in np.round(neg, 6) if tuple(r) in idx)
    return hits / len(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--N', type=int, default=128)
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()

    _, m6 = six00_cell()
    V = m6['vertices']

    ks = [4, 6, 8, 12, 16, 24, 32, 48, 60, 80, 100, 110, 118, 120]
    results = []
    for k in ks:
        per_seed = []
        for seed in range(args.seeds):
            rng = np.random.RandomState(2000 + seed)
            torch.manual_seed(2000 + seed)
            np.random.seed(2000 + seed)
            if k == 120:
                prefs = V
            else:
                prefs = V[rng.choice(len(V), k, replace=False)]
            afrac = antipodal_closure_frac(np.round(prefs, 6))
            eng = make_engine(args.device, args.N, args.steps, prefs)
            t0 = time.time()
            for _ in range(args.steps):
                eng.step()
            mm = boundary_metrics(eng, n_shuffles=50)
            mm['antipodal_frac'] = afrac
            per_seed.append(mm)
            print(f'  k={k:>3d} seed{seed}: z={mm["null_z"]:+.1f} '
                  f'd={mm["cohens_d"]:+.3f} TR={mm["transition_rate"]:.3f} '
                  f'({time.time()-t0:.1f}s)', flush=True)
        trs = [s['transition_rate'] for s in per_seed]
        row = {
            'k': k,
            'antipodal_frac': float(np.mean([s['antipodal_frac'] for s in per_seed])),
            'null_z_mean': float(np.mean([s['null_z'] for s in per_seed])),
            'null_z_std': float(np.std([s['null_z'] for s in per_seed])),
            'cohens_d_mean': float(np.mean([s['cohens_d'] for s in per_seed])),
            'ecr_mean': float(np.mean([s['edge_clarity_r'] for s in per_seed])),
            'TR_mean': float(np.mean(trs)),
            'TR_std': float(np.std(trs)),
        }
        row['C1_signal'] = row['null_z_mean'] > 3
        row['C2_history'] = row['TR_std'] > 0.01
        row['hosts'] = row['C1_signal'] and row['C2_history']
        results.append(row)

    outdir = os.path.join(HERE, 'outputs')
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, 't1_sas_census.json')
    with open(out, 'w') as f:
        json.dump({'device': args.device, 'N': args.N, 'steps': args.steps,
                   'seeds': args.seeds, 'rows': results}, f, indent=2)

    print('\n' + '=' * 86)
    print('T1 SAS CENSUS — hostability vs symmetry-breaking dial (600-cell subsamples)')
    print(f'{"k":>4s} {"antiF":>6s} {"null_z":>14s} {"cohen_d":>8s} '
          f'{"TR":>7s} {"TRstd":>7s} {"C1":>3s} {"C2":>3s} {"HOSTS":>6s}')
    for r in results:
        print(f'{r["k"]:>4d} {r["antipodal_frac"]:>6.3f} '
              f'{r["null_z_mean"]:>+7.1f}±{r["null_z_std"]:.1f} '
              f'{r["cohens_d_mean"]:>+8.3f} {r["TR_mean"]:>7.3f} '
              f'{r["TR_std"]:>7.3f} '
              f'{"Y" if r["C1_signal"] else "n":>3s} '
              f'{"Y" if r["C2_history"] else "n":>3s} '
              f'{"YES" if r["hosts"] else "no":>6s}')
    print(f'\nJSON: {out}')


if __name__ == '__main__':
    main()
