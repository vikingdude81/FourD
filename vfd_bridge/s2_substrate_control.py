"""S2 — Maximal-symmetry substrate control for the universality suite.

In universality_test.py, "topology" enters only as the GEOMETRY of the n_sub=8
preference directions (cyclic / random / full / none). There is no graph
coupling in the dynamics. So the honest way to ask "does the boundary-
negotiation / bearer signal survive a maximally symmetric substrate?" is to
replace the 8-direction preference simplex with the VERTEX SETS of the two
maximal-symmetry polytopes — which are literally unit vectors in R^4 and can
serve directly as preference directions:

  poly24_24    prefs = all 24 24-cell vertices      (n_sub=24)
  poly600_120  prefs = all 120 600-cell vertices    (n_sub=120)
  rand24/rand120: same n_sub, random unit directions (decouples "many
               subsystems" from "symmetric arrangement")
  poly600_s8   8 vertices sampled from the 600-cell (decouples symmetric
               directions from subsystem count; n_sub matches baseline)
  cyclic8      FourD baseline (exact PREFERENCE_MATRIX_NORMED)

Readout = the suite's canonical boundary-negotiation metrics:
edge_clarity_r, cohens_d, null_z, transition_rate.

Pre-registered predictions (fix before looking):
  P1: poly120 boundary signal (null_z, cohens_d) is WITHIN the range spanned
      by rand24/rand120 — i.e., symmetry does not create or destroy the
      signal beyond what subsystem count does.
  P2: if poly600_120 shows BOUNDARY PRESENT (|null_z|>2 and |ecr|>0.5), the
      basin/gateway effect is dynamical, not a geometric artifact of the
      cyclic 8-simplex.
  P3: transition_rate for polytopes >= cyclic8 (symmetric direction sets have
      antipodal pairs -> weaker single-dominant lock-in).

Run:  python vfd_bridge/s2_substrate_control.py [--device cuda:0] [--steps 1000] [--N 128] [--seeds 3]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # repo root
sys.path.insert(0, HERE)                          # vfd_bridge

from universality_test import (PREFERENCE_MATRIX_NORMED, UniversalEngine,
                               boundary_metrics)
from graphs import six00_cell, two4_cell


def make_engine(device, N, steps, prefs_np, topology_fallback='random'):
    """Engine with overridden preference substrate (+ matching fatigue bank)."""
    eng = UniversalEngine(N=N, device=device, steps=steps, manifold='s3',
                          topology=topology_fallback, fatigue_type='gradual')
    P = torch.tensor(prefs_np, dtype=torch.float32, device=eng.device)
    # normalize to unit length (prefs live on S^3 directions)
    P = P / (P.norm(dim=1, keepdim=True) + 1e-12)
    eng.prefs = P
    eng.n_sub = P.shape[0]
    eng.fatigue = torch.zeros(N, eng.n_sub, device=eng.device)
    return eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--N', type=int, default=128)
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()

    # vertex substrates -----------------------------------------------------
    _, m6 = six00_cell()
    _, m24 = two4_cell()
    v600 = np.asarray(m6['vertices'], dtype=np.float64)
    v24 = np.asarray(m24['vertices'], dtype=np.float64)

    variants = {
        'cyclic8': PREFERENCE_MATRIX_NORMED,
        'poly600_s8': None,   # filled per-seed below
        'rand8': None,
        'poly24_24': v24,
        'rand24': None,
        'poly600_120': v600,
        'rand120': None,
        'antipodal12': None,   # 12 random dirs + their antipodes: central symmetry only
    }

    results = {}
    for name, fixed in variants.items():
        accs = []
        for seed in range(args.seeds):
            rng = np.random.RandomState(1000 + seed)
            torch.manual_seed(1000 + seed)
            np.random.seed(1000 + seed)
            if name == 'poly600_s8':
                idx = rng.choice(len(v600), 8, replace=False)
                prefs = v600[idx]
            elif name == 'rand8':
                prefs = rng.randn(8, 4)
            elif name == 'rand24':
                prefs = rng.randn(24, 4)
            elif name == 'rand120':
                prefs = rng.randn(120, 4)
            elif name == 'antipodal12':
                base = rng.randn(12, 4)
                prefs = np.vstack([base, -base])
            else:
                prefs = fixed
            t0 = time.time()
            eng = make_engine(args.device, args.N, args.steps, prefs)
            for _ in range(args.steps):
                eng.step()
            m = boundary_metrics(eng, n_shuffles=100)
            m['wall'] = round(time.time() - t0, 1)
            accs.append(m)
            print(f'  {name:<12s} seed{seed}: r={m["edge_clarity_r"]:+.3f} '
                  f'd={m["cohens_d"]:+.3f} z={m["null_z"]:+.1f} '
                  f'TR={m["transition_rate"]:.3f} ({m["wall"]}s)', flush=True)
        results[name] = {
            k: {'mean': float(np.mean([a[k] for a in accs])),
                'std': float(np.std([a[k] for a in accs]))}
            for k in ('edge_clarity_r', 'cohens_d', 'clarity_gap', 'null_z',
                      'transition_rate')
        }

    outdir = os.path.join(HERE, 'outputs')
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, 's2_substrate.json')
    with open(out, 'w') as f:
        json.dump({'device': args.device, 'N': args.N, 'steps': args.steps,
                   'seeds': args.seeds, 'results': results}, f, indent=2)

    print('\n' + '=' * 78)
    print('S2 SUMMARY (mean over seeds)')
    print(f'{"variant":<12s} {"ecr":>14s} {"cohen_d":>14s} {"null_z":>14s} {"trans_rate":>14s}')
    for name, r in results.items():
        print(f'{name:<12s} '
              f'{r["edge_clarity_r"]["mean"]:>+7.3f}±{r["edge_clarity_r"]["std"]:.3f} '
              f'{r["cohens_d"]["mean"]:>+7.3f}±{r["cohens_d"]["std"]:.3f} '
              f'{r["null_z"]["mean"]:>+7.1f}±{r["null_z"]["std"]:.1f} '
              f'{r["transition_rate"]["mean"]:>8.3f}±{r["transition_rate"]["std"]:.3f}')
    print(f'\nJSON: {out}')


if __name__ == '__main__':
    main()
