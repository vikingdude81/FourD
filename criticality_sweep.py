#!/usr/bin/env python3
"""
Criticality Sweep — Does Perturbation-Recovery Time Peak at the Known Transition?
=====================================================================================

Decisive test for the open hypothesis documented in
docs/ARCHITECTURE.md ("Boundary Negotiation / Interface-Competency Thread").

bearer_state_competency.py found that s3+cyclic -- the one configuration
using the hand-calibrated PREFERENCE_MATRIX_NORMED -- showed uniquely large,
slow lesion/environment-shift recovery times (800+ of 1800 steps) versus
near-zero everywhere else. Two explanations are live:

  H1 (geometry/calibration-specific): something about S3 or the calibrated
     preferences specifically produces richer dynamics.
  H2 (critical slowing down): s3/cyclic's default fatigue_rate=0.217 happens
     to sit inside the Goldilocks/critical region goldilocks_sweep.py
     already located (~0.20-0.27), and critical slowing down -- a standard
     consequence of sitting near a genuine critical point, which
     critical_phenomena_suite.py / universality_verification.py already
     showed this system has (3D-Ising-class exponents) -- would produce
     large recovery times regardless of manifold identity.

H2 makes a sharp, falsifiable prediction H1 doesn't: recovery time should
peak as a function of fatigue_rate near the known transition (~0.18-0.27),
on ANY manifold/topology, not just s3/cyclic. This script sweeps
fatigue_rate at fixed manifold/topology (both s3/cyclic -- where the
transition is already located -- and flat4/cyclic, as a manifold where no
transition has been located, to see whether the same sweep produces a peak
there too or stays flat) and measures lesion_t_recovery / adapt_t_adapt
against it.

A peak near ~0.18-0.27 on both manifolds supports H2 (criticality, not
geometry, drives the effect). A peak only on s3, or no peak anywhere,
supports H1 (something s3/calibration-specific) and points back to the
matched-preference factorial as the next experiment instead.

Usage:
    python criticality_sweep.py [--device cuda:0] [--steps 1800] [--N 256] [--seeds 10]
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
from scipy import stats

from bearer_state_competency import lesion_recovery, adaptation_speed, resolve_device

OUT_DIR = os.path.join('outputs', 'criticality_sweep')

# Denser sampling around the known transition (~0.18-0.27 per
# goldilocks_sweep.py / geometry_comparison.py), sparser outside it.
FATIGUE_RATES = [0.05, 0.10, 0.15, 0.18, 0.20, 0.217, 0.23, 0.25, 0.27, 0.30, 0.35, 0.40]
MANIFOLDS = ['s3', 'flat4']
TOPOLOGY = 'cyclic'


def run_condition(manifold, fatigue_rate, device, steps, N, seed):
    lesion = lesion_recovery(manifold, TOPOLOGY, True, device, steps, N, seed,
                              fatigue_rate=fatigue_rate)
    adapt = adaptation_speed(manifold, TOPOLOGY, True, device, steps, N, seed,
                              fatigue_rate=fatigue_rate)
    return dict(
        lesion_d_immediate=lesion['d_immediate'], lesion_c_lesion=lesion['c_lesion'],
        lesion_t_recovery=lesion['t_recovery'],
        adapt_d_immediate=adapt['d_immediate'], adapt_c_adapt=adapt['c_adapt'],
        adapt_t_adapt=adapt['t_adapt'],
    )


def run_sweep(device, steps, N, seeds, outdir):
    os.makedirs(outdir, exist_ok=True)
    device = resolve_device(device)
    rows = []
    total = len(MANIFOLDS) * len(FATIGUE_RATES) * seeds
    done = 0
    t_start = time.time()

    for manifold in MANIFOLDS:
        for fr in FATIGUE_RATES:
            for seed in range(seeds):
                metrics = run_condition(manifold, fr, device, steps, N, seed)
                rows.append(dict(manifold=manifold, fatigue_rate=fr, seed=seed, **metrics))
                done += 1
                elapsed = time.time() - t_start
                print(f'  [{done}/{total}] {manifold:6s} fr={fr:.3f} seed={seed}  '
                      f"t_recovery={metrics['lesion_t_recovery']}  "
                      f"t_adapt={metrics['adapt_t_adapt']}  ({elapsed:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'criticality_per_seed.csv'), index=False)

    summary = (df.groupby(['manifold', 'fatigue_rate'])
               [['lesion_d_immediate', 'lesion_c_lesion', 'lesion_t_recovery',
                 'adapt_d_immediate', 'adapt_c_adapt', 'adapt_t_adapt']]
               .agg(['mean', 'std', lambda s: s.notna().mean()]))
    summary.columns = ['_'.join(c) if c[1] != '<lambda_0>' else f'{c[0]}_frac_measurable'
                        for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(outdir, 'criticality_summary.csv'), index=False)

    # Peak-location test: does the fatigue_rate with the max mean t_recovery
    # (among measurable/non-NaN cases) fall inside the known transition band?
    peak_report = {}
    for manifold in MANIFOLDS:
        sub = summary[summary.manifold == manifold]
        for metric in ['lesion_t_recovery', 'adapt_t_adapt']:
            col = f'{metric}_mean'
            if sub[col].notna().any():
                peak_row = sub.loc[sub[col].idxmax()]
                peak_report[f'{manifold}_{metric}'] = dict(
                    peak_fatigue_rate=float(peak_row['fatigue_rate']),
                    peak_value=float(peak_row[col]),
                    in_known_transition_band=bool(0.18 <= peak_row['fatigue_rate'] <= 0.27),
                )
            else:
                peak_report[f'{manifold}_{metric}'] = dict(
                    peak_fatigue_rate=None, peak_value=None,
                    in_known_transition_band=None,
                    note='no measurable (non-NaN) recovery events at any fatigue_rate',
                )

    with open(os.path.join(outdir, 'criticality_summary.json'), 'w') as f:
        json.dump({
            'steps': steps, 'N': N, 'seeds': seeds,
            'known_transition_band': [0.18, 0.27],
            'peak_report': peak_report,
        }, f, indent=2)

    print('\nPeak-location report (does recovery time peak inside the known ~0.18-0.27 band?):')
    for k, v in peak_report.items():
        print(f'  {k}: {v}')

    plot_results(df, summary, outdir)
    return df, summary, peak_report


def plot_results(df, summary, outdir):
    metrics = ['lesion_t_recovery', 'adapt_t_adapt']
    fig, axes = plt.subplots(len(metrics), len(MANIFOLDS), figsize=(6 * len(MANIFOLDS), 4.5 * len(metrics)),
                              squeeze=False)
    for row, metric in enumerate(metrics):
        for col, manifold in enumerate(MANIFOLDS):
            ax = axes[row][col]
            sub = summary[summary.manifold == manifold].sort_values('fatigue_rate')
            ax.errorbar(sub['fatigue_rate'], sub[f'{metric}_mean'], yerr=sub[f'{metric}_std'],
                        marker='o', color='#4C72B0', capsize=3)
            ax.axvspan(0.18, 0.27, alpha=0.15, color='#DD5555', label='known transition band')
            ax.set_xlabel('fatigue_rate')
            ax.set_ylabel(metric)
            ax.set_title(f'{manifold} / cyclic')
            ax.legend(fontsize=8)
    fig.suptitle('Does perturbation-recovery time peak at the known critical fatigue_rate?')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'criticality_sweep.png'), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--steps', type=int, default=1800)
    ap.add_argument('--N', type=int, default=256)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--outdir', default=OUT_DIR)
    args = ap.parse_args()

    print('Criticality Sweep')
    print(f'  device={args.device}  steps={args.steps}  N={args.N}  seeds={args.seeds}')
    print(f'  fatigue_rate values: {FATIGUE_RATES}')
    run_sweep(args.device, args.steps, args.N, args.seeds, args.outdir)
    print(f'\nOutputs written to {args.outdir}/')


if __name__ == '__main__':
    main()
