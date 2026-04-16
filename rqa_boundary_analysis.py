#!/usr/bin/env python3
"""
RQA at Boundary Transitions — Temporal Structure of Consciousness Switching
===========================================================================

Uses Recurrence Quantification Analysis (src/recurrence) to characterize
the *temporal microstructure* of boundary transitions vs dwelling periods:

  Part A: Recurrence comparison — RQA metrics at boundary windows vs dwelling
  Part B: Topology comparison — RQA structure for boundary-present vs absent variants
  Part C: Determinism gradient — Is boundary negotiation deterministic or stochastic?
  Part D: Recurrence time distribution — Characteristic timescales of switching

Bridges the dormant src/recurrence module into the main research thread.

OPH Credit: Framework adapted from Observer Patch Holography by FloatingPragma.
  https://github.com/FloatingPragma/observer-patch-holography

Usage:
    python rqa_boundary_analysis.py [--device cuda:0] [--steps 2000] [--N 64]
"""

from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.recurrence.embedding import time_delay_embedding, estimate_delay
from src.recurrence.recurrence_plot import recurrence_matrix
from src.recurrence.rqa import compute_rqa, determinism, laminarity, recurrence_rate

from universality_test import UniversalEngine, boundary_metrics

OUT_DIR = os.path.join('outputs', 'rqa_boundary')


# ============================================================================
# PART A: RQA AT BOUNDARY WINDOWS VS DWELLING WINDOWS
# ============================================================================

def part_a_boundary_vs_dwelling_rqa(device='cuda:0', steps=2000, N=64):
    """
    Compare RQA metrics in time windows centred on boundary transitions
    vs windows during stable dwelling periods.
    """
    print('\n  ── Part A: Boundary vs Dwelling RQA ──')

    engine = UniversalEngine(
        N=N, device=device, steps=steps,
        manifold='s3', topology='cyclic', fatigue_type='gradual',
    )
    for t in range(steps):
        engine.step()

    basins = engine.hist_macro_basin[:, :steps].cpu().numpy()
    clarity = engine.hist_clarity[:, :steps].cpu().numpy()

    # Identify transition timesteps per being
    transitions = basins[:, 1:] != basins[:, :-1]  # (N, steps-1)

    # Extract sliding windows around transitions vs dwelling
    # Note: transition rate ~0.37, so transitions are frequent.
    # Use small windows and small exclusion zones.
    half_win = 15
    min_gap = 8  # minimum timesteps between sampled windows
    dwell_radius = 5  # exclusion radius around transitions for dwelling windows

    boundary_rqa = {'rr': [], 'det': [], 'lam': [], 'avg_diag': []}
    dwelling_rqa = {'rr': [], 'det': [], 'lam': [], 'avg_diag': []}

    n_beings_sampled = min(N, 32)  # Sample subset for speed
    for b in range(n_beings_sampled):
        c = clarity[b]
        tr = transitions[b]

        # Find transition indices
        tr_idx = np.where(tr)[0]
        if len(tr_idx) < 3:
            continue

        # --- Boundary windows ---
        last_sampled = -min_gap
        for idx in tr_idx:
            if idx < half_win or idx > steps - half_win - 1:
                continue
            if idx - last_sampled < min_gap:
                continue
            last_sampled = idx

            window = c[idx - half_win:idx + half_win]
            delay = max(1, estimate_delay(window, max_lag=10))
            emb = time_delay_embedding(window, dimension=3, delay=delay)
            if emb.shape[0] < 10:
                continue
            rmat = recurrence_matrix(emb, threshold_percentile=15.0)
            rqa = compute_rqa(rmat, min_length=2)
            for k in boundary_rqa:
                boundary_rqa[k].append(rqa[k])

        # --- Dwelling windows: furthest from any transition ---
        # Use a smaller exclusion radius to ensure we find dwelling periods
        dwell_mask = np.ones(steps - 1, dtype=bool)
        for idx in tr_idx:
            lo = max(0, idx - dwell_radius)
            hi = min(steps - 1, idx + dwell_radius)
            dwell_mask[lo:hi] = False

        dwell_candidates = np.where(dwell_mask)[0]
        dwell_candidates = dwell_candidates[
            (dwell_candidates > half_win) & (dwell_candidates < steps - half_win)
        ]
        # Sample same number as boundary windows
        n_boundary = len(boundary_rqa['rr']) // max(1, b)  # avg per being so far
        n_target = max(n_boundary, 20)
        if len(dwell_candidates) > 0:
            rng = np.random.RandomState(b)
            sample_idx = rng.choice(
                dwell_candidates, size=min(n_target, len(dwell_candidates)),
                replace=False
            )
            for idx in sample_idx:
                window = c[idx - half_win:idx + half_win]
                delay = max(1, estimate_delay(window, max_lag=10))
                emb = time_delay_embedding(window, dimension=3, delay=delay)
                if emb.shape[0] < 10:
                    continue
                rmat = recurrence_matrix(emb, threshold_percentile=15.0)
                rqa = compute_rqa(rmat, min_length=2)
                for k in dwelling_rqa:
                    dwelling_rqa[k].append(rqa[k])

    # Statistical comparison
    print(f'\n    Windows sampled: {len(boundary_rqa["rr"])} boundary, '
          f'{len(dwelling_rqa["rr"])} dwelling')

    results = {}
    for metric in ['rr', 'det', 'lam', 'avg_diag']:
        bvals = np.array(boundary_rqa[metric])
        dvals = np.array(dwelling_rqa[metric])
        if len(bvals) > 5 and len(dvals) > 5:
            t_stat, p_val = stats.ttest_ind(bvals, dvals, equal_var=False)
            d = (bvals.mean() - dvals.mean()) / np.sqrt(
                (bvals.std()**2 + dvals.std()**2) / 2 + 1e-15)
            results[metric] = {
                'boundary_mean': float(bvals.mean()),
                'dwelling_mean': float(dvals.mean()),
                't_stat': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(d),
            }
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f'    {metric:>9s}: boundary={bvals.mean():.4f}  dwelling={dvals.mean():.4f}  '
                  f'd={d:+.3f}  p={p_val:.2e}  {sig}')

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metric_labels = {
        'rr': 'Recurrence Rate', 'det': 'Determinism',
        'lam': 'Laminarity', 'avg_diag': 'Avg Diagonal Length',
    }
    for ax, metric in zip(axes, ['rr', 'det', 'lam', 'avg_diag']):
        bvals = boundary_rqa[metric]
        dvals = dwelling_rqa[metric]
        if len(bvals) < 2 or len(dvals) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10)
            ax.set_title(metric_labels[metric], fontsize=10, fontweight='bold')
            continue
        parts = ax.violinplot([bvals, dvals], positions=[0, 1], showmeans=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Boundary', 'Dwelling'], fontsize=9)
        ax.set_ylabel(metric_labels[metric], fontsize=9)
        if metric in results:
            p = results[metric]['p_value']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.set_title(f'{metric_labels[metric]}\n({sig}, d={results[metric]["cohens_d"]:+.2f})',
                         fontsize=10, fontweight='bold')

    plt.suptitle('RQA: Boundary Transition Windows vs Dwelling Windows',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/part_a_boundary_vs_dwelling_rqa.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {OUT_DIR}/part_a_boundary_vs_dwelling_rqa.png')

    return results


# ============================================================================
# PART B: RQA ACROSS TOPOLOGY VARIANTS
# ============================================================================

def part_b_topology_rqa_comparison(device='cuda:0', steps=2000, N=64):
    """
    Compare RQA structure for boundary-present vs boundary-absent topologies.
    Uses the strongest (flat R⁴/cyclic) and weakest (no opponents) from
    universality test results.
    """
    print('\n  ── Part B: Topology RQA Comparison ──')

    variants = [
        ('Baseline S³ cyclic',  's3',    'cyclic', 'gradual'),
        ('Flat R⁴ cyclic',     'flat4', 'cyclic', 'gradual'),
        ('S³ no opponents',     's3',    'none',   'gradual'),
        ('S³ random opponents', 's3',    'random', 'gradual'),
    ]

    all_rqa = {}
    for name, manifold, topology, fatigue in variants:
        print(f'    Running {name}...', end='', flush=True)

        engine = UniversalEngine(
            N=N, device=device, steps=steps,
            manifold=manifold, topology=topology, fatigue_type=fatigue,
        )
        for t in range(steps):
            engine.step()

        clarity = engine.hist_clarity[:, :steps].cpu().numpy()

        # Compute RQA for each being's full clarity trajectory
        being_rqa = {'rr': [], 'det': [], 'lam': [], 'avg_diag': []}
        for b in range(min(N, 32)):
            c = clarity[b]
            delay = max(1, estimate_delay(c, max_lag=30))
            emb = time_delay_embedding(c, dimension=3, delay=delay)
            if emb.shape[0] < 20:
                continue
            rmat = recurrence_matrix(emb, threshold_percentile=10.0)
            rqa = compute_rqa(rmat, min_length=2)
            for k in being_rqa:
                being_rqa[k].append(rqa[k])

        means = {k: float(np.mean(v)) for k, v in being_rqa.items()}
        stds = {k: float(np.std(v)) for k, v in being_rqa.items()}
        all_rqa[name] = {'means': means, 'stds': stds, 'raw': being_rqa}

        print(f'  RR={means["rr"]:.4f}  DET={means["det"]:.4f}  '
              f'LAM={means["lam"]:.4f}')

    # Plot grouped bars
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    variant_names = list(all_rqa.keys())
    x = np.arange(len(variant_names))
    colors = ['#2d2d2d', '#55A868', '#4C72B0', '#DD5555']

    for ax, metric, label in zip(axes,
                                  ['rr', 'det', 'lam', 'avg_diag'],
                                  ['Recurrence Rate', 'Determinism',
                                   'Laminarity', 'Avg Diag Length']):
        means = [all_rqa[n]['means'][metric] for n in variant_names]
        stds = [all_rqa[n]['stds'][metric] for n in variant_names]
        ax.bar(x, means, yerr=stds, color=colors, edgecolor='white',
               capsize=4, linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(' ', '\n') for n in variant_names],
                           fontsize=7)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')

    plt.suptitle('RQA Metrics Across Topology Variants', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/part_b_topology_rqa.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {OUT_DIR}/part_b_topology_rqa.png')

    return {name: v['means'] for name, v in all_rqa.items()}


# ============================================================================
# PART C: DETERMINISM GRADIENT — ARE TRANSITIONS DETERMINISTIC?
# ============================================================================

def part_c_determinism_analysis(device='cuda:0', steps=2000, N=64):
    """
    Measure determinism as a function of proximity to basin transitions.
    If boundary negotiation is structured, DET should peak near transitions.
    """
    print('\n  ── Part C: Determinism Gradient near Transitions ──')

    engine = UniversalEngine(
        N=N, device=device, steps=steps,
        manifold='s3', topology='cyclic', fatigue_type='gradual',
    )
    for t in range(steps):
        engine.step()

    basins = engine.hist_macro_basin[:, :steps].cpu().numpy()
    clarity = engine.hist_clarity[:, :steps].cpu().numpy()

    # For each distance-from-transition bin, compute average DET
    max_dist = 60
    dist_bins = np.arange(0, max_dist + 1)
    det_by_dist = {d: [] for d in dist_bins}
    rr_by_dist = {d: [] for d in dist_bins}

    n_sampled = min(N, 32)
    for b in range(n_sampled):
        tr = basins[b, 1:] != basins[b, :-1]
        tr_idx = np.where(tr)[0]
        if len(tr_idx) < 2:
            continue

        c = clarity[b]
        # Compute distance to nearest transition for each timestep
        dists = np.full(steps, max_dist + 1)
        for idx in tr_idx:
            for d in range(max_dist + 1):
                for t in [idx - d, idx + d]:
                    if 0 <= t < steps:
                        dists[t] = min(dists[t], d)

        # For each distance bin, collect local RQA
        win = 20
        for t_center in range(win, steps - win, win // 2):
            d = dists[t_center]
            if d > max_dist:
                continue
            window = c[t_center - win:t_center + win]
            delay = max(1, estimate_delay(window, max_lag=8))
            emb = time_delay_embedding(window, dimension=3, delay=delay)
            if emb.shape[0] < 8:
                continue
            rmat = recurrence_matrix(emb, threshold_percentile=15.0)
            rqa = compute_rqa(rmat)
            det_by_dist[d].append(rqa['det'])
            rr_by_dist[d].append(rqa['rr'])

    # Average per bin
    dists_plot, det_mean, det_se = [], [], []
    rr_mean, rr_se = [], []
    for d in dist_bins:
        vals_d = det_by_dist[d]
        vals_r = rr_by_dist[d]
        if len(vals_d) >= 3:
            dists_plot.append(d)
            det_mean.append(np.mean(vals_d))
            det_se.append(np.std(vals_d) / np.sqrt(len(vals_d)))
            rr_mean.append(np.mean(vals_r))
            rr_se.append(np.std(vals_r) / np.sqrt(len(vals_r)))

    dists_plot = np.array(dists_plot)
    det_mean = np.array(det_mean)
    det_se = np.array(det_se)
    rr_mean = np.array(rr_mean)
    rr_se = np.array(rr_se)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.fill_between(dists_plot, det_mean - det_se, det_mean + det_se,
                     alpha=0.2, color='#4C72B0')
    ax1.plot(dists_plot, det_mean, '-', color='#4C72B0', linewidth=2)
    ax1.set_xlabel('Distance from Nearest Transition (timesteps)', fontsize=10)
    ax1.set_ylabel('Determinism (DET)', fontsize=10)
    ax1.set_title('Determinism vs Proximity to Basin Transition',
                  fontsize=11, fontweight='bold')
    ax1.axvline(0, color='red', ls='--', alpha=0.5, label='Transition event')
    ax1.legend(fontsize=9)

    ax2.fill_between(dists_plot, rr_mean - rr_se, rr_mean + rr_se,
                     alpha=0.2, color='#DD5555')
    ax2.plot(dists_plot, rr_mean, '-', color='#DD5555', linewidth=2)
    ax2.set_xlabel('Distance from Nearest Transition (timesteps)', fontsize=10)
    ax2.set_ylabel('Recurrence Rate (RR)', fontsize=10)
    ax2.set_title('Recurrence Rate vs Proximity to Basin Transition',
                  fontsize=11, fontweight='bold')
    ax2.axvline(0, color='red', ls='--', alpha=0.5)

    plt.suptitle('Does Deterministic Structure Peak at Boundaries?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/part_c_determinism_gradient.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {OUT_DIR}/part_c_determinism_gradient.png')

    # Correlation: distance vs DET
    if len(dists_plot) >= 5:
        rho, p = stats.spearmanr(dists_plot, det_mean)
        print(f'    Distance ↔ DET: ρ = {rho:+.3f} (p = {p:.3e})')
        rho_rr, p_rr = stats.spearmanr(dists_plot, rr_mean)
        print(f'    Distance ↔ RR:  ρ = {rho_rr:+.3f} (p = {p_rr:.3e})')

    return {
        'distances': dists_plot.tolist(),
        'det_mean': det_mean.tolist(),
        'rr_mean': rr_mean.tolist(),
    }


# ============================================================================
# PART D: RECURRENCE TIME DISTRIBUTION
# ============================================================================

def part_d_recurrence_times(device='cuda:0', steps=2000, N=64):
    """
    Compute the distribution of return times between basin transitions
    and compare with recurrence times from RQA.
    """
    print('\n  ── Part D: Recurrence Time Distributions ──')

    engine = UniversalEngine(
        N=N, device=device, steps=steps,
        manifold='s3', topology='cyclic', fatigue_type='gradual',
    )
    for t in range(steps):
        engine.step()

    basins = engine.hist_macro_basin[:, :steps].cpu().numpy()

    # --- Inter-transition intervals ---
    all_intervals = []
    for b in range(N):
        tr_idx = np.where(basins[b, 1:] != basins[b, :-1])[0]
        if len(tr_idx) >= 2:
            intervals = np.diff(tr_idx)
            all_intervals.extend(intervals.tolist())

    all_intervals = np.array(all_intervals)

    # --- Return times: time to revisit same basin ---
    return_times = []
    for b in range(min(N, 32)):
        unique_basins = np.unique(basins[b])
        for basin_id in unique_basins[:5]:  # Sample up to 5 basins per being
            visits = np.where(basins[b] == basin_id)[0]
            if len(visits) >= 2:
                gaps = np.diff(visits)
                # Only count gaps where being left and returned
                long_gaps = gaps[gaps > 3]
                return_times.extend(long_gaps.tolist())

    return_times = np.array(return_times) if len(return_times) > 0 else np.array([0])

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    if len(all_intervals) > 10:
        ax1.hist(all_intervals, bins=50, density=True, color='#4C72B0',
                 alpha=0.7, edgecolor='white')
        ax1.set_xlabel('Inter-Transition Interval (timesteps)', fontsize=10)
        ax1.set_ylabel('Density', fontsize=10)
        ax1.set_title(f'Inter-Transition Intervals\n'
                      f'μ={np.mean(all_intervals):.1f}, σ={np.std(all_intervals):.1f}',
                      fontsize=10, fontweight='bold')
        ax1.axvline(np.mean(all_intervals), color='red', ls='--', alpha=0.7)

    if len(return_times) > 10:
        ax2.hist(return_times, bins=50, density=True, color='#55A868',
                 alpha=0.7, edgecolor='white')
        ax2.set_xlabel('Basin Return Time (timesteps)', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title(f'Basin Return Times\n'
                      f'μ={np.mean(return_times):.1f}, σ={np.std(return_times):.1f}',
                      fontsize=10, fontweight='bold')
        ax2.axvline(np.mean(return_times), color='red', ls='--', alpha=0.7)

    # Log-log survival plot for inter-transition intervals
    if len(all_intervals) > 10:
        sorted_int = np.sort(all_intervals)
        survival = 1 - np.arange(1, len(sorted_int) + 1) / len(sorted_int)
        mask = survival > 0
        ax3.loglog(sorted_int[mask], survival[mask], '.', color='#4C72B0',
                   markersize=2, alpha=0.5)
        ax3.set_xlabel('Interval τ', fontsize=10)
        ax3.set_ylabel('P(T > τ)', fontsize=10)
        ax3.set_title('Survival Distribution\n(linear = power law)',
                      fontsize=10, fontweight='bold')

        # Fit power law in log-log
        log_x = np.log(sorted_int[mask])
        log_y = np.log(survival[mask])
        valid = np.isfinite(log_x) & np.isfinite(log_y)
        if valid.sum() > 10:
            slope, intercept, r, _, _ = stats.linregress(log_x[valid], log_y[valid])
            fit_x = np.linspace(log_x[valid].min(), log_x[valid].max(), 100)
            ax3.loglog(np.exp(fit_x), np.exp(intercept + slope * fit_x),
                       '--', color='red', linewidth=2,
                       label=f'α={-slope:.2f} (R²={r**2:.3f})')
            ax3.legend(fontsize=9)

    plt.suptitle('Temporal Scales of Boundary Dynamics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/part_d_recurrence_times.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {OUT_DIR}/part_d_recurrence_times.png')

    results = {}
    if len(all_intervals) > 0:
        results['inter_transition'] = {
            'mean': float(np.mean(all_intervals)),
            'median': float(np.median(all_intervals)),
            'std': float(np.std(all_intervals)),
            'n_samples': len(all_intervals),
        }
    if len(return_times) > 0:
        results['return_times'] = {
            'mean': float(np.mean(return_times)),
            'median': float(np.median(return_times)),
            'std': float(np.std(return_times)),
            'n_samples': len(return_times),
        }

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='RQA Boundary Analysis')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--N', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print('╔══════════════════════════════════════════════════════════════════════╗')
    print('║  RQA BOUNDARY ANALYSIS — Temporal Structure of Switching           ║')
    print('║                                                                    ║')
    print('║  Activating src/recurrence for boundary transition analysis        ║')
    print('║  OPH by FloatingPragma                                             ║')
    print('║  https://github.com/FloatingPragma/observer-patch-holography       ║')
    print('╚══════════════════════════════════════════════════════════════════════╝')

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'\nGPU: {props.name} ({props.total_memory / 1e9:.1f} GB)')

    t_start = time.time()

    results = {}
    results['part_a'] = part_a_boundary_vs_dwelling_rqa(args.device, args.steps, args.N)
    results['part_b'] = part_b_topology_rqa_comparison(args.device, args.steps, args.N)
    results['part_c'] = part_c_determinism_analysis(args.device, args.steps, args.N)
    results['part_d'] = part_d_recurrence_times(args.device, args.steps, args.N)

    out_path = f'{OUT_DIR}/rqa_boundary_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  All results saved to {out_path}')

    elapsed = time.time() - t_start
    print(f'  Total wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
