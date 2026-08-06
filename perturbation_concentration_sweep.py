#!/usr/bin/env python3
"""
Perturbation Concentration Sweep
===================================

qrng_developmental_capture.py found a significant deterministic-vs-random
difference in DC(Delta) horizon, but "deterministic" and "random" differ in
two entangled ways: entropy source, AND perturbation *shape* (one subsystem
concentrated vs. all subsystems evenly loaded, at matched L2 norm). The
prng-vs-os_csprng comparison (same shape, different entropy source) found no
detectable difference. This script isolates the remaining variable directly:
does developmental capture depend on how *concentrated* a fixed-magnitude
perturbation is, independent of its entropy source?

Concentration is quantified as

    kappa = ||delta||_1^2 / (n * ||delta||_2^2)

which is 1/n for a fully concentrated one-hot vector and 1 for a uniform
(all components equal magnitude) vector -- i.e. kappa increases with how
*distributed* the perturbation is (higher kappa = more spread out).

For each of several interpolation weights alpha in [0, 1] (alpha=1: pure
one-hot toward subsystem 0; alpha=0: pure random unit direction), the
injected vector is

    v = normalize(alpha * e_0 + (1 - alpha) * r),   r ~ random unit vector

kappa is measured post-hoc from the realized v (not from alpha directly,
since a random draw's exact concentration varies seed to seed), and DC(Delta)
horizon is regressed against measured kappa.

Usage:
    python perturbation_concentration_sweep.py [--device cuda:0] [--steps 1800] [--N 96] [--seeds 8]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from bearer_state_competency import resolve_device
from qrng_developmental_capture import build_pair, divergence_curve, horizon_from_curve

OUT_DIR = os.path.join('outputs', 'perturbation_concentration')

ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]


def make_concentration_xi(alpha: float, N: int, n_sub: int, device: str, seed: int, strength: float):
    rng = np.random.RandomState(seed)
    r = rng.standard_normal((N, n_sub)).astype(np.float32)
    r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-8)
    e0 = np.zeros((N, n_sub), dtype=np.float32)
    e0[:, 0] = 1.0
    v = alpha * e0 + (1 - alpha) * r
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

    l1 = np.sum(np.abs(v), axis=1)
    l2 = np.linalg.norm(v, axis=1)
    kappa = float(np.mean((l1 ** 2) / (n_sub * (l2 ** 2) + 1e-12)))

    return torch.tensor(v, device=device) * strength, kappa


def run_condition(manifold, topology, alpha, device, steps, N, seed, t0, strength):
    eng_pert, eng_ctrl = build_pair(manifold, topology, device, steps, N, seed)
    xi, kappa = make_concentration_xi(alpha, N, eng_pert.n_sub, device, seed, strength)
    eng_pert.schedule_perturbation(t0, xi)
    for _ in range(steps):
        eng_pert.step()
        eng_ctrl.step()
    deltas, divs = divergence_curve(eng_pert, eng_ctrl, t0, steps)
    horizon = horizon_from_curve(deltas, divs)
    peak = float(divs.max()) if len(divs) else 0.0
    return kappa, horizon, peak


def run_sweep(device, steps, N, seeds, outdir, t0=None, strength=3.0):
    os.makedirs(outdir, exist_ok=True)
    device = resolve_device(device)
    if t0 is None:
        t0 = steps // 2

    manifolds = ['s3', 'flat4']
    rows = []
    total = len(manifolds) * len(ALPHAS) * seeds
    done = 0
    t_start = time.time()

    for manifold in manifolds:
        for alpha in ALPHAS:
            for seed in range(seeds):
                kappa, horizon, peak = run_condition(
                    manifold, 'cyclic', alpha, device, steps, N, seed, t0, strength)
                rows.append(dict(manifold=manifold, alpha=alpha, kappa=kappa,
                                  seed=seed, horizon=horizon, peak_divergence=peak))
                done += 1
                elapsed = time.time() - t_start
                print(f'  [{done}/{total}] {manifold:6s} alpha={alpha:.2f} kappa={kappa:.3f} '
                      f'seed={seed}  horizon={horizon:.0f}  ({elapsed:.0f}s elapsed)')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'concentration_per_seed.csv'), index=False)

    results = {}
    for manifold in manifolds:
        sub = df[df.manifold == manifold]
        r, p = stats.pearsonr(sub['kappa'], sub['horizon'])
        rs, ps = stats.spearmanr(sub['kappa'], sub['horizon'])
        results[manifold] = dict(pearson_r=float(r), pearson_p=float(p),
                                  spearman_r=float(rs), spearman_p=float(ps))
        print(f'\n{manifold}: corr(kappa, horizon) pearson r={r:.3f} p={p:.4g}  '
              f'spearman r={rs:.3f} p={ps:.4g}')

    with open(os.path.join(outdir, 'concentration_summary.json'), 'w') as f:
        json.dump({'t0': t0, 'strength': strength, 'steps': steps, 'N': N, 'seeds': seeds,
                    'correlations': results}, f, indent=2)

    plot_results(df, manifolds, outdir)
    return df, results


def plot_results(df, manifolds, outdir):
    fig, axes = plt.subplots(1, len(manifolds), figsize=(6 * len(manifolds), 4.5))
    if len(manifolds) == 1:
        axes = [axes]
    for ax, manifold in zip(axes, manifolds):
        sub = df[df.manifold == manifold]
        ax.scatter(sub['kappa'], sub['horizon'], s=18, alpha=0.6, c='#4C72B0')
        ax.set_xlabel('kappa (1/n = concentrated, 1 = distributed)')
        ax.set_ylabel('DC(Delta) horizon')
        ax.set_title(f'manifold={manifold}')
    fig.suptitle('Does perturbation concentration predict developmental-capture horizon?')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'concentration_correlation.png'), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--steps', type=int, default=1800)
    ap.add_argument('--N', type=int, default=96)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--strength', type=float, default=3.0)
    ap.add_argument('--outdir', default=OUT_DIR)
    args = ap.parse_args()

    print('Perturbation Concentration Sweep')
    print(f'  device={args.device}  steps={args.steps}  N={args.N}  seeds={args.seeds}  '
          f'strength={args.strength}')
    run_sweep(args.device, args.steps, args.N, args.seeds, args.outdir, strength=args.strength)
    print(f'\nOutputs written to {args.outdir}/')


if __name__ == '__main__':
    main()
