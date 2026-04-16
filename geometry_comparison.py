#!/usr/bin/env python3
"""
S³ vs Flat R⁴ Deep Comparison — Does Geometry Buy Anything?
============================================================

The universality test showed flat R⁴ has *stronger* boundary negotiation
than S³ (d=+1.44 vs d=+0.10).  This script investigates whether S³
provides benefits the boundary metrics don't capture:

  Part A: Phase transition comparison — Sharper or smoother on each manifold?
  Part B: Basin grammar complexity — Richer symbolic dynamics on S³?
  Part C: Multi-seed robustness — More reproducible on S³?
  Part D: Parameter sensitivity — Wider "Goldilocks zone" on S³?

OPH Credit: Framework adapted from Observer Patch Holography by FloatingPragma.
  https://github.com/FloatingPragma/observer-patch-holography

Usage:
    python geometry_comparison.py [--device cuda:0] [--steps 2000] [--N 128]
"""

from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import Counter
from scipy import stats

from universality_test import UniversalEngine, boundary_metrics

OUT_DIR = os.path.join('outputs', 'geometry_comparison')


# ============================================================================
# PART A: PHASE TRANSITION SHARPNESS
# ============================================================================

def part_a_phase_transition(device='cuda:0', steps=1500, N=128):
    """
    Sweep fatigue_rate across the critical region for both manifolds.
    Compare transition sharpness (slope of order parameter at fr_c).
    """
    print('\n  ── Part A: Phase Transition Comparison ──')

    fr_values = np.linspace(0.05, 0.45, 25)
    manifolds = ['s3', 'flat4']
    manifold_labels = {'s3': 'S³', 'flat4': 'Flat R⁴'}

    results = {}
    for manifold in manifolds:
        clarity_means = []
        transition_rates = []
        boundary_rs = []
        print(f'    {manifold_labels[manifold]}:', end='', flush=True)

        for fr in fr_values:
            engine = UniversalEngine(
                N=N, device=device, steps=steps,
                manifold=manifold, topology='cyclic', fatigue_type='gradual',
                fatigue_rate=float(fr),
            )
            for t in range(steps):
                engine.step()

            clarity = engine.hist_clarity[:, :steps].cpu().numpy()
            basins = engine.hist_macro_basin[:, :steps].cpu().numpy()

            warmup = steps // 4
            clarity_means.append(float(clarity[:, warmup:].mean()))

            tr = (basins[:, warmup + 1:] != basins[:, warmup:-1]).mean()
            transition_rates.append(float(tr))

            # Quick boundary metric (edge-clarity r only)
            metrics = boundary_metrics(engine, n_shuffles=20)
            boundary_rs.append(metrics['edge_clarity_r'])

            print('.', end='', flush=True)

        results[manifold] = {
            'fr_values': fr_values.tolist(),
            'clarity': clarity_means,
            'transition_rate': transition_rates,
            'boundary_r': boundary_rs,
        }
        print(f' done')

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'s3': '#4C72B0', 'flat4': '#55A868'}

    for manifold in manifolds:
        r = results[manifold]
        label = manifold_labels[manifold]
        c = colors[manifold]
        axes[0].plot(r['fr_values'], r['clarity'], 'o-', color=c,
                     label=label, markersize=4, linewidth=1.5)
        axes[1].plot(r['fr_values'], r['transition_rate'], 'o-', color=c,
                     label=label, markersize=4, linewidth=1.5)
        axes[2].plot(r['fr_values'], r['boundary_r'], 'o-', color=c,
                     label=label, markersize=4, linewidth=1.5)

    axes[0].set_xlabel('Fatigue Rate', fontsize=10)
    axes[0].set_ylabel('Mean Clarity', fontsize=10)
    axes[0].set_title('Order Parameter (Clarity)', fontsize=11, fontweight='bold')
    axes[0].legend()
    axes[0].axvline(0.182, color='gray', ls=':', alpha=0.5, label='fr_c')

    axes[1].set_xlabel('Fatigue Rate', fontsize=10)
    axes[1].set_ylabel('Transition Rate', fontsize=10)
    axes[1].set_title('Basin Transition Rate', fontsize=11, fontweight='bold')
    axes[1].axvline(0.182, color='gray', ls=':', alpha=0.5)

    axes[2].set_xlabel('Fatigue Rate', fontsize=10)
    axes[2].set_ylabel('Edge↔Clarity (r)', fontsize=10)
    axes[2].set_title('Boundary Negotiation Strength', fontsize=11, fontweight='bold')
    axes[2].axhline(0, color='k', linewidth=0.3)
    axes[2].axvline(0.182, color='gray', ls=':', alpha=0.5)

    plt.suptitle('Phase Transition: S³ vs Flat R⁴',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/part_a_phase_transition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {OUT_DIR}/part_a_phase_transition.png')

    # Compute transition sharpness: max derivative of clarity wrt fr
    for manifold in manifolds:
        c = np.array(results[manifold]['clarity'])
        fr = np.array(results[manifold]['fr_values'])
        dc_dfr = np.gradient(c, fr)
        max_slope = np.max(np.abs(dc_dfr))
        peak_fr = fr[np.argmax(np.abs(dc_dfr))]
        print(f'    {manifold_labels[manifold]}: max |dC/dfr| = {max_slope:.4f} at fr = {peak_fr:.3f}')
        results[manifold]['max_slope'] = float(max_slope)
        results[manifold]['peak_fr'] = float(peak_fr)

    return results


# ============================================================================
# PART B: BASIN GRAMMAR COMPLEXITY
# ============================================================================

def part_b_basin_grammar(device='cuda:0', steps=2000, N=128):
    """
    Compare the complexity of basin visitation sequences (symbolic dynamics)
    between S³ and flat R⁴.
    """
    print('\n  ── Part B: Basin Grammar Complexity ──')

    manifolds = ['s3', 'flat4']
    manifold_labels = {'s3': 'S³', 'flat4': 'Flat R⁴'}

    results = {}
    for manifold in manifolds:
        print(f'    Running {manifold_labels[manifold]}...', end='', flush=True)

        engine = UniversalEngine(
            N=N, device=device, steps=steps,
            manifold=manifold, topology='cyclic', fatigue_type='gradual',
        )
        for t in range(steps):
            engine.step()

        basins = engine.hist_macro_basin[:, :steps].cpu().numpy()
        warmup = steps // 4

        # --- Metrics per being ---
        unique_basins_visited = []
        entropy_sequence = []
        bigram_entropy = []
        trigram_entropy = []

        for b in range(N):
            seq = basins[b, warmup:]

            # Unique basins
            unique_basins_visited.append(len(np.unique(seq)))

            # Unigram entropy
            counts = Counter(seq)
            total = sum(counts.values())
            probs = np.array([c / total for c in counts.values()])
            h1 = -np.sum(probs * np.log2(probs + 1e-15))
            entropy_sequence.append(h1)

            # Bigram entropy
            bigrams = list(zip(seq[:-1], seq[1:]))
            bg_counts = Counter(bigrams)
            total_bg = sum(bg_counts.values())
            bg_probs = np.array([c / total_bg for c in bg_counts.values()])
            h2 = -np.sum(bg_probs * np.log2(bg_probs + 1e-15))
            bigram_entropy.append(h2)

            # Trigram entropy
            trigrams = list(zip(seq[:-2], seq[1:-1], seq[2:]))
            tg_counts = Counter(trigrams)
            total_tg = sum(tg_counts.values())
            tg_probs = np.array([c / total_tg for c in tg_counts.values()])
            h3 = -np.sum(tg_probs * np.log2(tg_probs + 1e-15))
            trigram_entropy.append(h3)

        results[manifold] = {
            'unique_basins': float(np.mean(unique_basins_visited)),
            'unigram_entropy': float(np.mean(entropy_sequence)),
            'bigram_entropy': float(np.mean(bigram_entropy)),
            'trigram_entropy': float(np.mean(trigram_entropy)),
            # Excess entropy: information rate beyond unigram
            'excess_entropy': float(np.mean(bigram_entropy) - np.mean(entropy_sequence)),
        }

        print(f'  H1={results[manifold]["unigram_entropy"]:.3f}  '
              f'H2={results[manifold]["bigram_entropy"]:.3f}  '
              f'H3={results[manifold]["trigram_entropy"]:.3f}  '
              f'unique={results[manifold]["unique_basins"]:.1f}')

    # Statistical comparison
    print(f'\n    Comparison:')
    for metric in ['unigram_entropy', 'bigram_entropy', 'trigram_entropy', 'unique_basins']:
        s3_val = results['s3'][metric]
        r4_val = results['flat4'][metric]
        diff = r4_val - s3_val
        print(f'      {metric:<20s}: S³={s3_val:.3f}  R⁴={r4_val:.3f}  Δ={diff:+.3f}')

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metrics = ['unique_basins', 'unigram_entropy', 'bigram_entropy', 'trigram_entropy']
    titles = ['Unique Basins', 'Unigram H', 'Bigram H', 'Trigram H']
    x = np.arange(2)
    width = 0.5

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [results['s3'][metric], results['flat4'][metric]]
        ax.bar(x, vals, width, color=['#4C72B0', '#55A868'], edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(['S³', 'Flat R⁴'], fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Basin Grammar Complexity: S³ vs Flat R⁴',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/part_b_basin_grammar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {OUT_DIR}/part_b_basin_grammar.png')

    return results


# ============================================================================
# PART C: MULTI-SEED ROBUSTNESS
# ============================================================================

def part_c_multi_seed(device='cuda:0', steps=1500, N=64, n_seeds=10):
    """
    Run multiple random seeds and compare variance of key metrics
    across manifolds.  Lower variance = more robust.
    """
    print('\n  ── Part C: Multi-Seed Robustness ──')

    manifolds = ['s3', 'flat4']
    manifold_labels = {'s3': 'S³', 'flat4': 'Flat R⁴'}

    results = {}
    for manifold in manifolds:
        print(f'    {manifold_labels[manifold]}:', end='', flush=True)
        seed_metrics = {'edge_clarity_r': [], 'cohens_d': [], 'null_z': [],
                        'transition_rate': [], 'mean_clarity': []}

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            engine = UniversalEngine(
                N=N, device=device, steps=steps,
                manifold=manifold, topology='cyclic', fatigue_type='gradual',
            )
            for t in range(steps):
                engine.step()

            metrics = boundary_metrics(engine, n_shuffles=30)
            for k in ['edge_clarity_r', 'cohens_d', 'null_z', 'transition_rate']:
                seed_metrics[k].append(metrics[k])

            warmup = steps // 4
            mc = float(engine.hist_clarity[:, warmup:steps].cpu().numpy().mean())
            seed_metrics['mean_clarity'].append(mc)

            print('.', end='', flush=True)

        results[manifold] = {}
        for k, vals in seed_metrics.items():
            results[manifold][k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'cv': float(np.std(vals) / (np.abs(np.mean(vals)) + 1e-15)),
                'values': vals,
            }
        print(f' done')

    # Print comparison
    print(f'\n    {"Metric":<20s}  {"S³ μ±σ":>15s}  {"S³ CV":>8s}  '
          f'{"R⁴ μ±σ":>15s}  {"R⁴ CV":>8s}  {"More Robust":>12s}')
    print(f'    {"-"*20}  {"-"*15}  {"-"*8}  {"-"*15}  {"-"*8}  {"-"*12}')

    for k in ['edge_clarity_r', 'cohens_d', 'null_z', 'transition_rate', 'mean_clarity']:
        s3 = results['s3'][k]
        r4 = results['flat4'][k]
        more_robust = 'S³' if s3['cv'] < r4['cv'] else 'R⁴'
        print(f'    {k:<20s}  {s3["mean"]:>+7.3f}±{s3["std"]:.3f}  {s3["cv"]:>8.3f}  '
              f'{r4["mean"]:>+7.3f}±{r4["std"]:.3f}  {r4["cv"]:>8.3f}  {more_robust:>12s}')

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    metric_keys = ['edge_clarity_r', 'cohens_d', 'null_z', 'transition_rate', 'mean_clarity']
    metric_labels = ['Edge↔Clarity', "Cohen's d", 'Null z', 'Trans Rate', 'Mean Clarity']

    for ax, k, label in zip(axes, metric_keys, metric_labels):
        s3_vals = results['s3'][k]['values']
        r4_vals = results['flat4'][k]['values']
        parts = ax.violinplot([s3_vals, r4_vals], positions=[0, 1], showmeans=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['S³', 'R⁴'], fontsize=10)
        ax.set_title(label, fontsize=10, fontweight='bold')

    plt.suptitle(f'Multi-Seed Robustness ({n_seeds} Seeds)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/part_c_multi_seed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {OUT_DIR}/part_c_multi_seed.png')

    return {m: {k: {kk: vv for kk, vv in v.items() if kk != 'values'}
                for k, v in r.items()} for m, r in results.items()}


# ============================================================================
# PART D: PARAMETER SENSITIVITY (GOLDILOCKS ZONE WIDTH)
# ============================================================================

def part_d_parameter_sensitivity(device='cuda:0', steps=1000, N=64):
    """
    Sweep 2 key parameters (fatigue_rate, steering_strength) on both
    manifolds and compare the width of the "productive region" where
    boundary negotiation is present.
    """
    print('\n  ── Part D: Parameter Sensitivity (Goldilocks Zone) ──')

    fr_range = np.linspace(0.05, 0.45, 12)
    ss_range = np.linspace(0.2, 1.2, 12)
    manifolds = ['s3', 'flat4']
    manifold_labels = {'s3': 'S³', 'flat4': 'Flat R⁴'}

    results = {}
    for manifold in manifolds:
        print(f'    {manifold_labels[manifold]}:', end='', flush=True)
        grid = np.zeros((len(fr_range), len(ss_range)))

        for i, fr in enumerate(fr_range):
            for j, ss in enumerate(ss_range):
                engine = UniversalEngine(
                    N=N, device=device, steps=steps,
                    manifold=manifold, topology='cyclic', fatigue_type='gradual',
                    fatigue_rate=float(fr), steering_strength=float(ss),
                )
                for t in range(steps):
                    engine.step()

                metrics = boundary_metrics(engine, n_shuffles=20)
                grid[i, j] = metrics['edge_clarity_r']
            print('.', end='', flush=True)

        # Count cells where boundary negotiation is present (r > 0.3)
        n_present = int((grid > 0.3).sum())
        n_strong = int((grid > 0.5).sum())
        n_total = grid.size

        results[manifold] = {
            'grid': grid.tolist(),
            'fr_range': fr_range.tolist(),
            'ss_range': ss_range.tolist(),
            'n_present': n_present,
            'n_strong': n_strong,
            'fraction_present': n_present / n_total,
            'fraction_strong': n_strong / n_total,
        }
        print(f'  present={n_present}/{n_total}  strong={n_strong}/{n_total}')

    # Plot heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, manifold in zip(axes, manifolds):
        grid = np.array(results[manifold]['grid'])
        im = ax.imshow(grid, aspect='auto', origin='lower',
                        extent=[ss_range[0], ss_range[-1], fr_range[0], fr_range[-1]],
                        cmap='RdYlGn', vmin=-0.5, vmax=1.0)
        ax.set_xlabel('Steering Strength', fontsize=10)
        ax.set_ylabel('Fatigue Rate', fontsize=10)
        frac = results[manifold]['fraction_present']
        ax.set_title(f'{manifold_labels[manifold]}\n'
                     f'{results[manifold]["n_present"]}/{grid.size} cells present '
                     f'({frac:.0%})',
                     fontsize=11, fontweight='bold')
        # Mark optimal point
        ax.plot(0.707, 0.217, '*', color='white', markersize=15,
                markeredgecolor='black', markeredgewidth=1)
        plt.colorbar(im, ax=ax, label='Edge↔Clarity (r)')

    plt.suptitle('Goldilocks Zone: S³ vs Flat R⁴',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/part_d_parameter_sensitivity.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {OUT_DIR}/part_d_parameter_sensitivity.png')

    # Clean results for JSON
    return {m: {k: v for k, v in r.items() if k != 'grid'}
            for m, r in results.items()}


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='S³ vs Flat R⁴ Comparison')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--N', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print('╔══════════════════════════════════════════════════════════════════════╗')
    print('║  GEOMETRY COMPARISON — Does S³ Buy Anything Over Flat R⁴?          ║')
    print('║                                                                    ║')
    print('║  Phase transition, basin grammar, robustness, parameter sensitivity║')
    print('║  OPH by FloatingPragma                                             ║')
    print('║  https://github.com/FloatingPragma/observer-patch-holography       ║')
    print('╚══════════════════════════════════════════════════════════════════════╝')

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'\nGPU: {props.name} ({props.total_memory / 1e9:.1f} GB)')

    t_start = time.time()

    results = {}
    results['part_a'] = part_a_phase_transition(args.device, args.steps, args.N)
    results['part_b'] = part_b_basin_grammar(args.device, args.steps, args.N)
    results['part_c'] = part_c_multi_seed(args.device, min(args.steps, 1500), min(args.N, 64))
    results['part_d'] = part_d_parameter_sensitivity(args.device, min(args.steps, 1000),
                                                      min(args.N, 64))

    out_path = f'{OUT_DIR}/geometry_comparison_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  All results saved to {out_path}')

    elapsed = time.time() - t_start
    print(f'  Total wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
