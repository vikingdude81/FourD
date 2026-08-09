#!/usr/bin/env python3
"""
Full-Engine Hysteresis Test
==============================

compact_vs_noncompact_bifurcation.py found that compactness alone does not
explain the s3-vs-flat4 asymmetry seen in criticality_sweep.py -- both
compact (S1) and non-compact (R1) reduced skeletons show the same
bifurcation type and, importantly, NEITHER shows hysteresis. That's a clean
negative result for "manifold compactness explains the lock-in," and it
means whatever produces s3's permanent lock-in must live in something the
reduced skeleton doesn't have: the bearer-state feedback loop, the full
8-subsystem/macro-basin-assignment layer, or lesioning (removing an entire
subsystem) being a qualitatively different intervention than a smooth
fatigue_rate sweep.

This script tests two things directly on the FULL engine rather than a
proxy:

  Part A -- Permanence: is s3's post-lesion deficit at fatigue_rate=0.217
  genuinely permanent, or does it eventually resolve given a much longer
  observation window (criticality_sweep.py only looked 900-1500 steps past
  the lesion)?

  Part B -- Path-dependence: does the state the engine is in when lesioned
  depend on how it got there? Three histories all converge to the same
  fatigue_rate=0.30 before the lesion event: 'direct' (built there from
  t=0), 'ramp_up' (started below the transition, fatigue_rate linearly
  ramped up through it), 'ramp_down' (started above, ramped down). If
  lesion recovery differs across histories despite all three sitting at the
  same fatigue_rate when lesioned, that IS hysteresis/multistability in the
  full engine -- and would explain the tension with the reduced skeleton's
  has_hysteresis=false, since the reduced skeleton has no bearer state and
  no lesioning.

Usage:
    python hysteresis_test.py [--device cuda:0] [--N 256] [--seeds 10]
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

from bearer_state_competency import BearerEngine, deficit_metrics, resolve_device

OUT_DIR = os.path.join('outputs', 'hysteresis_test')

MANIFOLD = 's3'
TOPOLOGY = 'cyclic'


def build_raw_engine(device, steps, N, seed, start_fr):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return BearerEngine(N=N, device=device, steps=steps, manifold=MANIFOLD, topology=TOPOLOGY,
                         fatigue_type='gradual', use_bearer=True, rng_seed=seed,
                         fatigue_rate=start_fr)


# ============================================================================
# PART A: PERMANENCE (long-window recovery at the default fatigue_rate)
# ============================================================================

def run_permanence(device, steps, N, seed, lesion_at=None, fatigue_rate=0.217):
    if lesion_at is None:
        lesion_at = steps // 2
    eng = build_raw_engine(device, steps, N, seed, fatigue_rate)
    eng.schedule_lesion(lesion_at, sub_idx=0)
    for _ in range(steps):
        eng.step()
    clarity = eng.hist_clarity[:, :steps].cpu().numpy().mean(axis=0)
    return deficit_metrics(clarity, lesion_at, steps)


# ============================================================================
# PART B: PATH-DEPENDENCE (three histories converging to the same fr)
# ============================================================================

HISTORY_START_FR = {'direct': None, 'ramp_up': 0.05, 'ramp_down': 0.45}


def fr_schedule(history, t, ramp_steps, target_fr):
    if history == 'direct':
        return target_fr
    start = HISTORY_START_FR[history]
    if t >= ramp_steps:
        return target_fr
    return start + (target_fr - start) * (t / ramp_steps)


def run_history(device, steps, N, seed, history, target_fr, lesion_at, ramp_steps):
    start_fr = target_fr if history == 'direct' else HISTORY_START_FR[history]
    eng = build_raw_engine(device, steps, N, seed, start_fr)
    eng.schedule_lesion(lesion_at, sub_idx=0)
    for t in range(steps):
        eng.fatigue_rate = fr_schedule(history, t, ramp_steps, target_fr)
        eng.step()
    clarity = eng.hist_clarity[:, :steps].cpu().numpy().mean(axis=0)
    metrics = deficit_metrics(clarity, lesion_at, steps)
    metrics['clarity_at_lesion'] = float(clarity[lesion_at - 1])
    metrics['clarity_curve'] = clarity.tolist()
    return metrics


def run_all(device, N, seeds, outdir,
            permanence_steps=6000, permanence_fr=0.217,
            history_steps=1800, target_fr=0.30, ramp_steps=100, settle_steps=0):
    # NOTE: a first version of this test used a 300-step settling gap between
    # reaching target_fr and lesioning, and found zero path-dependence. A
    # diagnostic run showed why: fatigue saturates at its clamp ceiling
    # (fatigue.clamp(0,3)) within a few hundred steps regardless of path,
    # erasing history *before* the settling window even ends -- states
    # right after the ramp (fatigue.mean 1.62 vs 0.41 vs 2.59 for
    # direct/ramp_up/ramp_down at ramp_steps=100) differ substantially, but
    # 300 more steps at constant fr washes that out. settle_steps=0 (the new
    # default) lesions immediately at ramp completion, while the transient
    # divergence still exists, which is the fair test of path-dependence.
    os.makedirs(outdir, exist_ok=True)
    device = resolve_device(device)
    t_start = time.time()

    # --- Part A ---
    print('Part A: Permanence (extended window)')
    perm_rows = []
    lesion_at = permanence_steps // 2
    for seed in range(seeds):
        m = run_permanence(device, permanence_steps, N, seed, lesion_at, permanence_fr)
        perm_rows.append(dict(seed=seed, **m))
        print(f'  seed={seed}  t_recovery={m["t_recovery"]}  d_immediate={m["d_immediate"]:.4f}  '
              f'({time.time() - t_start:.0f}s elapsed)')
    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(os.path.join(outdir, 'permanence_per_seed.csv'), index=False)
    frac_never_recovered = float((perm_df['t_recovery'] >= (permanence_steps - lesion_at) - 1).mean())
    print(f'  fraction never recovering within {permanence_steps - lesion_at} steps: {frac_never_recovered:.2f}')

    # --- Part B ---
    print('\nPart B: Path-dependence')
    hist_rows = []
    lesion_at_hist = ramp_steps + settle_steps
    histories = ['direct', 'ramp_up', 'ramp_down']
    curves = {}
    for history in histories:
        for seed in range(seeds):
            m = run_history(device, history_steps, N, seed, history, target_fr,
                             lesion_at_hist, ramp_steps)
            curve = m.pop('clarity_curve')
            curves[(history, seed)] = curve
            hist_rows.append(dict(history=history, seed=seed, **m))
            print(f'  {history:10s} seed={seed}  t_recovery={m["t_recovery"]}  '
                  f'd_immediate={m["d_immediate"]:.4f}  clarity_at_lesion={m["clarity_at_lesion"]:.4f}  '
                  f'({time.time() - t_start:.0f}s elapsed)')
    hist_df = pd.DataFrame(hist_rows)
    hist_df.to_csv(os.path.join(outdir, 'path_dependence_per_seed.csv'), index=False)

    summary = hist_df.groupby('history')[['d_immediate', 'c_lesion', 't_recovery', 'clarity_at_lesion']].agg(
        ['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(outdir, 'path_dependence_summary.csv'), index=False)

    # Does t_recovery differ across histories at the SAME target fatigue_rate?
    groups = [hist_df[hist_df.history == h]['t_recovery'].dropna().values for h in histories]
    if all(len(g) > 1 for g in groups):
        f_stat, p_val = stats.f_oneway(*groups)
    else:
        f_stat, p_val = float('nan'), float('nan')

    # clarity_at_lesion differing across histories (all at the same fr) is
    # the more direct multistability signature: same parameter, different
    # state, purely because of how it got there.
    clarity_groups = [hist_df[hist_df.history == h]['clarity_at_lesion'].values for h in histories]
    f_clarity, p_clarity = stats.f_oneway(*clarity_groups)

    result = dict(
        permanence=dict(steps=permanence_steps, fatigue_rate=permanence_fr,
                         lesion_at=lesion_at, frac_never_recovered=frac_never_recovered),
        path_dependence=dict(
            target_fr=target_fr, ramp_steps=ramp_steps, history_steps=history_steps,
            t_recovery_anova_f=float(f_stat), t_recovery_anova_p=float(p_val),
            clarity_at_lesion_anova_f=float(f_clarity), clarity_at_lesion_anova_p=float(p_clarity),
        ),
    )
    with open(os.path.join(outdir, 'hysteresis_test_summary.json'), 'w') as f:
        json.dump(result, f, indent=2)

    plot_results(perm_df, hist_df, curves, histories, seeds, outdir)
    return perm_df, hist_df, result


def plot_results(perm_df, hist_df, curves, histories, seeds, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1 = axes[0]
    colors = {'direct': '#8172B3', 'ramp_up': '#4C72B0', 'ramp_down': '#DD5555'}
    for history in histories:
        all_curves = [np.array(curves[(history, seed)]) for seed in range(seeds)]
        mean_curve = np.mean(np.stack(all_curves), axis=0)
        ax1.plot(mean_curve, label=history, color=colors[history])
    ax1.set_xlabel('step')
    ax1.set_ylabel('mean clarity')
    ax1.set_title('Clarity trajectories by history (all converge to same target fr)')
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    for history in histories:
        vals = hist_df[hist_df.history == history]['clarity_at_lesion'].values
        ax2.scatter([history] * len(vals), vals, alpha=0.6, color=colors[history])
    ax2.set_ylabel('clarity at lesion moment (same target fr for all)')
    ax2.set_title('Does history change state at a fixed fatigue_rate?')

    fig.suptitle('Full-engine hysteresis / path-dependence test (s3/cyclic)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hysteresis_test.png'), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--N', type=int, default=256)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--outdir', default=OUT_DIR)
    args = ap.parse_args()

    print('Full-Engine Hysteresis Test')
    print(f'  device={args.device}  N={args.N}  seeds={args.seeds}')
    perm_df, hist_df, result = run_all(args.device, args.N, args.seeds, args.outdir)

    print('\n=== Part A: Permanence ===')
    print(json.dumps(result['permanence'], indent=2))
    print('\n=== Part B: Path-dependence ===')
    print(json.dumps(result['path_dependence'], indent=2))
    print(f'\nOutputs written to {args.outdir}/')


if __name__ == '__main__':
    main()
