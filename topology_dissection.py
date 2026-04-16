#!/usr/bin/env python3
"""
Topology Dissection — What About Cyclic Opponents Is Essential?
===============================================================

The universality test showed boundary negotiation requires cyclic opponent
topology specifically.  This script decomposes that topology into its
sub-properties to find the *minimal structural requirement*:

  ANTI-PARALLELISM axis:  How much opposition is needed?
    - fourd_baseline    : Real FourD preferences (135° / cos=-0.707)
    - full_anti_180     : Exact anti-parallel (180° / cos=-1.0)
    - partial_anti_120  : 120° opposition (cos=-0.5)
    - partial_anti_90   : Orthogonal (90° / cos=0)
    - partial_anti_60   : Weak co-alignment (60° / cos=+0.5)
    - aligned_pairs     : Near-aligned (≈0° / cos≈+1)

  STRUCTURE axis:  Does the pairing pattern matter?
    - shuffled_pairs    : Same 8 FourD vectors, random pairing order
    - chain             : Interleaved cardinal-bridge sequence
    - star              : Motor vs 7 opponents
    - one_pair_rest_clust: Motor↔Emotion pair + 6 clustered

  PAIR COUNT axis:  Does the number of opponent pairs matter?
    - pairs_1           : Motor↔Emotion only + 6 neutral
    - pairs_2           : 2 FourD pairs + 4 neutral
    - pairs_4           : All 4 FourD pairs (= baseline)

OPH Credit: Framework adapted from Observer Patch Holography by FloatingPragma.
  https://github.com/FloatingPragma/observer-patch-holography

Usage:
    python topology_dissection.py [--device cuda:0] [--steps 1000] [--N 128]
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
from scipy import stats

# Reuse engine infrastructure from universality test
from universality_test import (
    UniversalEngine,
    boundary_metrics,
    GROUP_COLORS,
)
from gpu_ensemble_sim import PREFERENCE_MATRIX_NORMED


# ============================================================================
# EXTENDED PREFERENCE GENERATION
# ============================================================================

def _rotate_opponents(prefs, pairs, target_angle_deg):
    """Rotate opponent vectors to achieve a specific angle with their primary.

    The target_angle_deg is the angle BETWEEN the pair members:
      180° = fully anti-parallel, 90° = orthogonal, 0° = aligned.
    """
    result = prefs.copy()
    angle_rad = np.deg2rad(target_angle_deg)

    for pi, oi in pairs:
        primary = prefs[pi].copy()
        opponent = prefs[oi].copy()

        # Get orthogonal component of opponent relative to primary
        proj = np.dot(opponent, primary) * primary
        ortho = opponent - proj
        ortho_norm = np.linalg.norm(ortho)
        if ortho_norm > 1e-8:
            ortho /= ortho_norm
        else:
            # Exactly anti-parallel: pick arbitrary orthogonal direction
            for d in range(len(primary)):
                candidate = np.zeros_like(primary)
                candidate[d] = 1.0
                ortho = candidate - np.dot(candidate, primary) * primary
                if np.linalg.norm(ortho) > 1e-6:
                    ortho /= np.linalg.norm(ortho)
                    break

        # Construct new opponent at target angle from primary
        new_opp = np.cos(angle_rad) * primary + np.sin(angle_rad) * ortho
        result[oi] = new_opp / (np.linalg.norm(new_opp) + 1e-8)

    return result


def _keep_n_pairs(prefs, pairs, n_keep, rng, dim):
    """Keep n_keep opponent pairs from FourD, cluster the rest."""
    result = prefs.copy()

    # Find a neutral direction orthogonal to kept pair primaries
    kept_primaries = [prefs[pairs[i][0]] for i in range(n_keep)]
    neutral = rng.randn(dim)
    for kp in kept_primaries:
        neutral -= np.dot(neutral, kp) * kp
    norm = np.linalg.norm(neutral)
    if norm < 1e-6:
        neutral = rng.randn(dim)
    neutral /= np.linalg.norm(neutral) + 1e-8

    # Replace non-kept subsystems with clustered directions
    kept_indices = set()
    for i in range(n_keep):
        kept_indices.add(pairs[i][0])
        kept_indices.add(pairs[i][1])

    for i in range(len(prefs)):
        if i not in kept_indices:
            result[i] = neutral + rng.randn(dim) * 0.15
            result[i] /= np.linalg.norm(result[i]) + 1e-8

    return result


# The 4 natural opponent pairs in PREFERENCE_MATRIX_NORMED:
#   Motor(0) ↔ Emotion(4):    cos = -0.707 (135°)
#   Planning(1) ↔ Social(5):  cos = -0.707 (135°)
#   Attention(2) ↔ Intuition(6): cos = -0.707 (135°)
#   Memory(3) ↔ Aesthetic(7): cos = -0.707 (135°)
FOURD_PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]


def generate_dissection_preferences(n_sub, dim, variant, seed=42):
    """Generate preference matrices derived from the real FourD preferences.

    The real PREFERENCE_MATRIX_NORMED has a specific structure:
      - 4 "cardinal" subsystems axis-aligned: Motor[+x], Planning[+y],
        Attention[+z], Memory[+w]
      - 4 "bridge" subsystems spanning two axes: Emotion[-x,+y]/√2,
        Social[-y,+z]/√2, Intuition[-z,+w]/√2, Aesthetic[+x,-w]/√2
      - Natural opponent pairs at 135° (cos=-0.707)

    All variants preserve or systematically modify this structure.
    """
    rng = np.random.RandomState(seed)
    real = PREFERENCE_MATRIX_NORMED.copy()  # (8, 4)

    if variant == 'fourd_baseline':
        # The actual FourD preferences — this IS the baseline (135° opposition)
        return real

    # --- ANTI-PARALLELISM AXIS ---
    # Modify the opposition angle while keeping primary directions fixed

    elif variant == 'full_anti_180':
        # Push opponents to exactly anti-parallel (cos = -1.0, 180°)
        prefs = real.copy()
        for pi, oi in FOURD_PAIRS:
            prefs[oi] = -prefs[pi]
        return prefs

    elif variant == 'partial_anti_120':
        # Reduce opposition to 120° (cos = -0.5)
        return _rotate_opponents(real, FOURD_PAIRS, target_angle_deg=120)

    elif variant == 'partial_anti_90':
        # Orthogonal pairs (cos = 0, 90°) — no opposition at all
        return _rotate_opponents(real, FOURD_PAIRS, target_angle_deg=90)

    elif variant == 'partial_anti_60':
        # Weak co-alignment (cos = +0.5, 60°)
        return _rotate_opponents(real, FOURD_PAIRS, target_angle_deg=60)

    elif variant == 'aligned_pairs':
        # Near-aligned (≈0°): opponents ≈ same direction as primary
        prefs = real.copy()
        for pi, oi in FOURD_PAIRS:
            prefs[oi] = prefs[pi] + rng.randn(dim) * 0.05
            prefs[oi] /= np.linalg.norm(prefs[oi]) + 1e-8
        return prefs

    # --- STRUCTURE AXIS ---
    # Same 8 FourD vectors but rearranged into different competition topologies

    elif variant == 'shuffled_pairs':
        # Keep all 8 FourD vectors but randomly reassign pairing order
        prefs = real.copy()
        idx = rng.permutation(8)
        return prefs[idx]

    elif variant == 'chain':
        # Interleave cardinals and bridges so consecutive subsystems oppose:
        # Motor, Emotion, Planning, Social, Attention, Intuition, Memory, Aesthetic
        order = [0, 4, 1, 5, 2, 6, 3, 7]
        return real[order]

    elif variant == 'star':
        # Motor vs everyone: keep Motor, rotate all others toward anti-Motor
        prefs = real.copy()
        motor = prefs[0].copy()
        for i in range(1, n_sub):
            v = -motor + rng.randn(dim) * 0.4
            prefs[i] = v / (np.linalg.norm(v) + 1e-8)
        return prefs

    elif variant == 'one_pair_rest_clust':
        # Keep only Motor↔Emotion pair, cluster the other 6
        prefs = real.copy()
        neutral = rng.randn(dim)
        neutral -= np.dot(neutral, prefs[0]) * prefs[0]  # orthogonal to Motor
        neutral /= np.linalg.norm(neutral) + 1e-8
        for i in [1, 2, 3, 5, 6, 7]:
            prefs[i] = neutral + rng.randn(dim) * 0.15
            prefs[i] /= np.linalg.norm(prefs[i]) + 1e-8
        return prefs

    # --- PAIR COUNT AXIS ---
    # Keep N of the 4 FourD pairs, cluster the rest

    elif variant == 'pairs_1':
        # Only Motor↔Emotion pair, rest clustered
        return _keep_n_pairs(real, FOURD_PAIRS, n_keep=1, rng=rng, dim=dim)

    elif variant == 'pairs_2':
        # Motor↔Emotion + Planning↔Social, rest clustered
        return _keep_n_pairs(real, FOURD_PAIRS, n_keep=2, rng=rng, dim=dim)

    elif variant == 'pairs_4':
        # All 4 pairs = baseline
        return real

    raise ValueError(f'Unknown dissection variant: {variant}')


# ============================================================================
# MODIFIED ENGINE WITH CUSTOM TOPOLOGY
# ============================================================================

class DissectionEngine(UniversalEngine):
    """UniversalEngine with custom preference injection."""

    def __init__(self, N, device='cuda:0', steps=1000,
                 manifold='s3', topology='cyclic', fatigue_type='gradual',
                 custom_prefs=None, **kwargs):
        super().__init__(N, device=device, steps=steps,
                         manifold=manifold, topology=topology,
                         fatigue_type=fatigue_type, **kwargs)
        if custom_prefs is not None:
            self.prefs = torch.tensor(
                custom_prefs, dtype=torch.float32, device=self.device)


# ============================================================================
# VARIANT DEFINITIONS
# ============================================================================

DISSECTION_VARIANTS = [
    # (name, variant_key, group)
    # --- Baseline ---
    ('FourD baseline (135°)', 'fourd_baseline',      'baseline'),

    # --- Anti-parallelism axis ---
    ('Full anti (180°)',      'full_anti_180',        'anti_angle'),
    ('120° opposition',       'partial_anti_120',     'anti_angle'),
    ('Orthogonal (90°)',      'partial_anti_90',      'anti_angle'),
    ('60° co-alignment',      'partial_anti_60',      'anti_angle'),
    ('Aligned (≈0°)',         'aligned_pairs',        'anti_angle'),

    # --- Structure axis ---
    ('Shuffled pairs',        'shuffled_pairs',       'structure'),
    ('Ring chain',            'chain',                'structure'),
    ('Star (1 vs 7)',         'star',                 'structure'),
    ('1 pair + clustered',    'one_pair_rest_clust',  'structure'),

    # --- Pair count axis ---
    ('1 pair + 6 neutral',    'pairs_1',              'pair_count'),
    ('2 pairs + 4 neutral',   'pairs_2',              'pair_count'),
    ('4 pairs (= baseline)',  'pairs_4',              'pair_count'),
]

DISSECTION_COLORS = {
    'baseline':    '#2d2d2d',
    'anti_angle':  '#E07020',   # Orange for opposition angle
    'structure':   '#4C72B0',   # Blue for structure
    'pair_count':  '#55A868',   # Green for pair count
}


# ============================================================================
# RUN ALL DISSECTION VARIANTS
# ============================================================================

def run_dissection(device='cuda:0', steps=1000, N=128,
                   outdir='outputs/topology_dissection'):
    """Run all dissection variants and collect boundary metrics."""
    os.makedirs(outdir, exist_ok=True)

    print(f'\n  Running {len(DISSECTION_VARIANTS)} topology dissection variants '
          f'({N} beings × {steps} steps each)')
    print(f'  {"Variant":<25s}  {"Key":>22s}  {"Group":>12s}')
    print(f'  {"-"*25}  {"-"*22}  {"-"*12}')
    for name, key, group in DISSECTION_VARIANTS:
        print(f'  {name:<25s}  {key:>22s}  {group:>12s}')
    print()

    all_results = {}

    for i, (name, variant_key, group) in enumerate(DISSECTION_VARIANTS):
        print(f'  [{i+1}/{len(DISSECTION_VARIANTS)}] {name}...', end='', flush=True)
        t0 = time.time()

        # Generate custom preferences for this variant
        custom_prefs = generate_dissection_preferences(8, 4, variant_key, seed=42)

        # Measure pairwise cosine similarities for diagnostics
        cos_sim = custom_prefs @ custom_prefs.T
        np.fill_diagonal(cos_sim, 0)
        min_cos = float(cos_sim.min())
        mean_cos = float(cos_sim.mean())

        engine = DissectionEngine(
            N=N, device=device, steps=steps,
            manifold='s3', topology='cyclic', fatigue_type='gradual',
            custom_prefs=custom_prefs,
        )

        for t in range(steps):
            engine.step()

        metrics = boundary_metrics(engine, n_shuffles=100)
        elapsed = time.time() - t0

        all_results[name] = {
            'variant_key': variant_key,
            'group': group,
            'min_cosine_sim': round(min_cos, 4),
            'mean_cosine_sim': round(mean_cos, 4),
            **metrics,
            'wall_time': round(elapsed, 1),
        }

        print(f'  r={metrics["edge_clarity_r"]:+.3f}  '
              f'd={metrics["cohens_d"]:+.3f}  '
              f'z={metrics["null_z"]:+.1f}  '
              f'cos_min={min_cos:+.3f}  '
              f'({elapsed:.1f}s)')

    return all_results


# ============================================================================
# ANALYSIS: OPPOSITION ANGLE GRADIENT
# ============================================================================

def opposition_angle_analysis(results, outdir='outputs/topology_dissection'):
    """
    Analyse the anti-parallelism gradient:
    180° → 120° → 60° → 0° and correlate with boundary metrics.
    """
    angle_variants = [
        ('Full anti (180°)',      180),
        ('FourD baseline (135°)', 135),
        ('120° opposition',       120),
        ('Orthogonal (90°)',       90),
        ('60° co-alignment',       60),
        ('Aligned (≈0°)',           0),
    ]

    angles, rs, ds, zs = [], [], [], []
    for name, angle in angle_variants:
        if name in results:
            angles.append(angle)
            rs.append(results[name]['edge_clarity_r'])
            ds.append(results[name]['cohens_d'])
            zs.append(results[name]['null_z'])

    fig, axes_fig = plt.subplots(1, 3, figsize=(15, 5))

    for ax, vals, ylabel, title in zip(axes_fig,
                                        [rs, ds, zs],
                                        ['Edge↔Clarity (r)', "Cohen's d", 'Null z-score'],
                                        ['Temporal Coupling', 'Clarity Boost', 'Signal Strength']):
        ax.plot(angles, vals, 'o-', color='#E07020', markersize=8, linewidth=2)
        ax.axhline(0, color='k', linewidth=0.3)
        ax.set_xlabel('Opposition Angle (degrees)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks([0, 60, 120, 180])
        ax.invert_xaxis()

        # Spearman correlation
        if len(angles) >= 4:
            rho, p = stats.spearmanr(angles, vals)
            ax.text(0.05, 0.95, f'ρ={rho:+.3f} p={p:.3f}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Opposition Angle → Boundary Negotiation Gradient',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{outdir}/opposition_angle_gradient.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved opposition angle gradient to {outdir}/opposition_angle_gradient.png')


# ============================================================================
# ANALYSIS: PAIR COUNT SCALING
# ============================================================================

def pair_count_analysis(results, outdir='outputs/topology_dissection'):
    """Analyse how boundary negotiation scales with number of opponent pairs."""
    pair_variants = [
        ('1 pair + 6 neutral',   1),
        ('2 pairs + 4 neutral',  2),
        ('4 pairs (= baseline)', 4),
    ]

    counts, rs, ds, zs = [], [], [], []
    for name, count in pair_variants:
        if name in results:
            counts.append(count)
            rs.append(results[name]['edge_clarity_r'])
            ds.append(results[name]['cohens_d'])
            zs.append(results[name]['null_z'])

    fig, axes_fig = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vals, ylabel in zip(axes_fig,
                                 [rs, ds, zs],
                                 ['Edge↔Clarity (r)', "Cohen's d", 'Null z-score']):
        ax.bar(counts, vals, color='#55A868', edgecolor='white', width=0.6)
        ax.set_xlabel('Number of Opponent Pairs', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(counts)
        ax.axhline(0, color='k', linewidth=0.3)

    plt.suptitle('Opponent Pair Count → Boundary Negotiation Scaling',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{outdir}/pair_count_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved pair count scaling to {outdir}/pair_count_scaling.png')


# ============================================================================
# COMPARISON PLOT
# ============================================================================

def dissection_comparison_plot(results, outdir='outputs/topology_dissection'):
    """4-panel comparison like universality_test but for dissection variants."""
    names = list(results.keys())
    n = len(names)
    groups = [results[nm]['group'] for nm in names]
    colors = [DISSECTION_COLORS[g] for g in groups]

    metrics = {
        'edge_clarity_r': 'Edge ↔ Clarity (r)',
        'cohens_d': "Transition Clarity Boost (Cohen's d)",
        'null_z': 'Null Model Deviation (z-score)',
        'transition_rate': 'Basin Transition Rate',
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for ax_idx, (key, title) in enumerate(metrics.items()):
        ax = axes[ax_idx]
        vals = [results[nm][key] for nm in names]
        short = [nm.replace('FourD baseline (135°)', 'FourD\n(135°)')
                  .replace('Full anti (180°)', 'Full anti\n(180°)')
                  .replace('120° opposition', '120°\nopp.')
                  .replace('Orthogonal (90°)', 'Orthog.\n(90°)')
                  .replace('60° co-alignment', '60°\nco-align')
                  .replace('Aligned (≈0°)', 'Aligned\n(≈0°)')
                  .replace('Shuffled pairs', 'Shuffled\npairs')
                  .replace('Ring chain', 'Ring\nchain')
                  .replace('Star (1 vs 7)', 'Star\n(1v7)')
                  .replace('1 pair + clustered', '1 pair\n+clust')
                  .replace('1 pair + 6 neutral', '1 pair\n+6 neut')
                  .replace('2 pairs + 4 neutral', '2 pairs\n+4 neut')
                  .replace('4 pairs (= baseline)', '4 pairs\n(base)')
                 for nm in names]

        bars = ax.bar(range(n), vals, color=colors, edgecolor='white', linewidth=0.5)
        baseline_val = results['FourD baseline (135°)'][key]
        ax.axhline(baseline_val, color='#2d2d2d', ls='--', alpha=0.4, linewidth=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(short, fontsize=6.5, ha='center')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axhline(0, color='k', linewidth=0.3)

        for bar, val in zip(bars, vals):
            y = bar.get_height()
            if abs(val) > 0.001:
                fmt = f'{val:+.2f}' if abs(val) < 100 else f'{val:+.0f}'
                ax.text(bar.get_x() + bar.get_width() / 2, y,
                        fmt, ha='center', va='bottom' if y >= 0 else 'top',
                        fontsize=6, fontweight='bold')

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=DISSECTION_COLORS['baseline'], label='Baseline'),
        Patch(facecolor=DISSECTION_COLORS['anti_angle'], label='Opposition angle'),
        Patch(facecolor=DISSECTION_COLORS['structure'], label='Graph structure'),
        Patch(facecolor=DISSECTION_COLORS['pair_count'], label='Pair count'),
    ]
    fig.legend(handles=legend, loc='upper center', ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{outdir}/topology_dissection_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved comparison to {outdir}/topology_dissection_comparison.png')


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_dissection_summary(results):
    """Print formatted summary with verdicts."""
    print('\n' + '=' * 110)
    print('TOPOLOGY DISSECTION — What About Cyclic Opponents Is Essential?')
    print('=' * 110)

    print(f'\n  {"Variant":<25s}  {"Edge↔Clarity":>12s}  {"Cohen d":>9s}  '
          f'{"Null z":>8s}  {"Trans Rate":>10s}  {"cos_min":>8s}  {"Verdict":>18s}')
    print(f'  {"-"*25}  {"-"*12}  {"-"*9}  {"-"*8}  {"-"*10}  {"-"*8}  {"-"*18}')

    verdicts = {}
    for name, r in results.items():
        ecr = r['edge_clarity_r']
        cd = r['cohens_d']
        nz = r['null_z']
        tr = r['transition_rate']
        cmin = r['min_cosine_sim']

        if tr < 0.005:
            verdict = 'NO TRANSITIONS'
        elif ecr > 0.5 and abs(nz) > 5:
            verdict = 'STRONG'
        elif ecr > 0.3 and abs(nz) > 2:
            verdict = 'MODERATE'
        elif abs(ecr) > 0.15 or abs(nz) > 2:
            verdict = 'WEAK'
        else:
            verdict = 'ABSENT'

        verdicts[name] = verdict
        print(f'  {name:<25s}  {ecr:>+12.4f}  {cd:>+9.4f}  '
              f'{nz:>+8.1f}  {tr:>10.4f}  {cmin:>+8.3f}  {verdict:>18s}')

    # Per-axis conclusions
    print('\n  PER-AXIS CONCLUSIONS:')

    # Anti-parallelism
    angle_names = [n for n, r in results.items() if r['group'] == 'anti_angle']
    angle_strong = sum(1 for n in angle_names if verdicts[n] in ('STRONG', 'MODERATE'))
    print(f'    Opposition angle:  {angle_strong}/{len(angle_names)} variants show boundary negotiation')
    baseline_r = results.get('FourD baseline (135°)', {}).get('edge_clarity_r', 0)
    aligned_r = results.get('Aligned (≈0°)', {}).get('edge_clarity_r', 0)
    if aligned_r > 0.3:
        print(f'    → Anti-parallelism NOT necessary (even aligned pairs show it)')
    elif aligned_r < 0.15:
        print(f'    → Anti-parallelism IS essential (aligned pairs fail)')
    else:
        print(f'    → Anti-parallelism helps but is not strictly required')

    # Structure
    struct_names = [n for n, r in results.items() if r['group'] == 'structure']
    struct_strong = sum(1 for n in struct_names if verdicts[n] in ('STRONG', 'MODERATE'))
    print(f'    Graph structure:  {struct_strong}/{len(struct_names)} variants show boundary negotiation')

    # Pair count
    pair_names = [n for n, r in results.items() if r['group'] == 'pair_count']
    pair_strong = sum(1 for n in pair_names if verdicts[n] in ('STRONG', 'MODERATE'))
    print(f'    Pair count:       {pair_strong}/{len(pair_names)} variants show boundary negotiation')

    return verdicts


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Topology Dissection')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--outdir', default='outputs/topology_dissection')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print('╔══════════════════════════════════════════════════════════════════════╗')
    print('║  TOPOLOGY DISSECTION — What Makes Cyclic Opponents Special?        ║')
    print('║                                                                    ║')
    print('║  Decomposing: opposition angle, graph structure, pair count        ║')
    print('║  OPH by FloatingPragma                                             ║')
    print('║  https://github.com/FloatingPragma/observer-patch-holography       ║')
    print('╚══════════════════════════════════════════════════════════════════════╝')

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'\nGPU: {props.name} ({props.total_memory / 1e9:.1f} GB)')

    t_start = time.time()

    results = run_dissection(args.device, args.steps, args.N, args.outdir)
    verdicts = print_dissection_summary(results)
    dissection_comparison_plot(results, args.outdir)
    opposition_angle_analysis(results, args.outdir)
    pair_count_analysis(results, args.outdir)

    out_path = f'{args.outdir}/topology_dissection_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Results saved to {out_path}')

    elapsed = time.time() - t_start
    print(f'\nTotal wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
