#!/usr/bin/env python3
"""
Minimal Boundary Model — The Simplest System With Boundary Negotiation
======================================================================

Strips the FourD engine to its absolute essentials to find the *minimal*
system that still produces boundary-dominated information concentration:

  Model 1: Circle S¹ + cyclic opponents + fatigue  (2D, simplest)
  Model 2: Line R¹ + cyclic opponents + fatigue     (1D!)
  Model 3: S¹ + opponents, NO fatigue               (is fatigue needed?)
  Model 4: S¹ + fatigue, NO opponents               (are opponents needed?)
  Model 5: Discrete 4-state + cyclic transitions     (pure combinatorial)
  Model 6: S¹ + 2 subsystems only (minimal count)
  Model 7: S¹ + 4 subsystems (half of FourD)

Then identifies the *boundary* between models that show the effect and
those that don't — the phase boundary OF the model space.

OPH Credit: Framework adapted from Observer Patch Holography by FloatingPragma.
  https://github.com/FloatingPragma/observer-patch-holography

Usage:
    python minimal_boundary_model.py [--steps 2000] [--N 256]
"""

from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT_DIR = os.path.join('outputs', 'minimal_model')


# ============================================================================
# MINIMAL ENGINE (Pure NumPy, no GPU needed)
# ============================================================================

class MinimalEngine:
    """
    Stripped-down consciousness engine with only:
      - Position state (scalar or vector)
      - Subsystem preferences (directional biases)
      - Fatigue (optional resource depletion)
      - Competition (via preference opposition)

    Parameters:
        n_sub:       Number of subsystems
        n_sims:      Number of parallel simulations
        dim:         Dimensionality of state (1, 2, etc.)
        manifold:    'circle' (S¹), 'line' (R¹), 'plane' (R²)
        use_fatigue: Whether to include fatigue dynamics
        use_opponents: Whether preferences oppose each other
        fatigue_rate: Rate of fatigue accumulation
        steering:    Steering strength
        noise:       Exploration noise
        n_basins:    Number of macro basins for assignment
    """

    def __init__(self, n_sub=8, n_sims=256, dim=2, manifold='circle',
                 use_fatigue=True, use_opponents=True,
                 fatigue_rate=0.217, steering=0.707, noise=0.25,
                 recovery=0.025, n_basins=8, seed=42):
        self.n_sub = n_sub
        self.N = n_sims
        self.dim = dim
        self.manifold = manifold
        self.use_fatigue = use_fatigue
        self.use_opponents = use_opponents
        self.fr = fatigue_rate
        self.ss = steering
        self.noise = noise
        self.rec = recovery
        self.n_basins = n_basins

        rng = np.random.RandomState(seed)

        # State
        if manifold == 'circle':
            self.theta = rng.uniform(0, 2 * np.pi, n_sims)
        elif manifold == 'line':
            self.pos = rng.randn(n_sims) * 0.3
        elif manifold == 'plane':
            self.pos = rng.randn(n_sims, 2) * 0.3

        # Preferences
        if use_opponents and n_sub >= 2:
            n_pairs = n_sub // 2
            if manifold == 'circle':
                angles = np.linspace(0, np.pi, n_pairs, endpoint=False)
                self.pref_angles = np.zeros(n_sub)
                for i in range(n_pairs):
                    self.pref_angles[i] = angles[i]
                    self.pref_angles[n_pairs + i] = angles[i] + np.pi
            elif manifold == 'line':
                self.pref_dirs = np.zeros(n_sub)
                for i in range(n_pairs):
                    self.pref_dirs[i] = 1.0 * (i + 1) / n_pairs
                    self.pref_dirs[n_pairs + i] = -1.0 * (i + 1) / n_pairs
            elif manifold == 'plane':
                angles = np.linspace(0, np.pi, n_pairs, endpoint=False)
                self.pref_vecs = np.zeros((n_sub, 2))
                for i in range(n_pairs):
                    self.pref_vecs[i] = [np.cos(angles[i]), np.sin(angles[i])]
                    self.pref_vecs[n_pairs + i] = [-np.cos(angles[i]), -np.sin(angles[i])]
        else:
            if manifold == 'circle':
                self.pref_angles = rng.uniform(0, np.pi / 4, n_sub)
            elif manifold == 'line':
                self.pref_dirs = rng.randn(n_sub) * 0.1 + 0.5
            elif manifold == 'plane':
                base = np.array([1.0, 0.0])
                self.pref_vecs = np.tile(base, (n_sub, 1))
                self.pref_vecs += rng.randn(n_sub, 2) * 0.1

        # Fatigue
        self.fatigue = np.zeros((n_sims, n_sub))

        # Basin centers
        if manifold == 'circle':
            self.basin_angles = np.linspace(0, 2 * np.pi, n_basins, endpoint=False)
        elif manifold == 'line':
            self.basin_centers = np.linspace(-2, 2, n_basins)
        elif manifold == 'plane':
            theta = np.linspace(0, 2 * np.pi, n_basins, endpoint=False)
            self.basin_centers_2d = np.column_stack([np.cos(theta), np.sin(theta)]) * 0.8

        # History
        self.history = {'clarity': [], 'basin': [], 'dominant_sub': []}

    def step(self):
        """Single timestep update."""
        # --- Influences ---
        if self.manifold == 'circle':
            influences = np.zeros((self.N, self.n_sub))
            for s in range(self.n_sub):
                influences[:, s] = 0.5 + 0.3 * np.cos(self.theta - self.pref_angles[s])
        elif self.manifold == 'line':
            influences = np.zeros((self.N, self.n_sub))
            for s in range(self.n_sub):
                influences[:, s] = 0.5 + 0.3 * np.tanh(self.pos * self.pref_dirs[s])
        elif self.manifold == 'plane':
            influences = np.zeros((self.N, self.n_sub))
            for s in range(self.n_sub):
                dot = np.sum(self.pos * self.pref_vecs[s], axis=1)
                influences[:, s] = 0.5 + 0.3 * np.tanh(dot)

        # --- Fatigue → activities ---
        if self.use_fatigue:
            effective = influences * np.exp(-self.fatigue)
        else:
            effective = influences.copy()

        noise_val = self.noise * np.random.randn(self.N, self.n_sub)
        effective = np.maximum(effective + noise_val, 0.05)
        activities = effective / (effective.sum(axis=1, keepdims=True) + 1e-8)

        # --- Update fatigue ---
        if self.use_fatigue:
            self.fatigue += self.fr * activities
            excess = np.maximum(activities - 1.0 / self.n_sub, 0.02) - 0.02
            self.fatigue += 0.03 * excess
            inactive_recovery = (1.0 - activities) * self.rec
            self.fatigue = np.clip(self.fatigue - inactive_recovery, 0, 3)

        # --- Forces & drive ---
        if self.manifold == 'circle':
            forces = np.zeros((self.N, self.n_sub))
            for s in range(self.n_sub):
                forces[:, s] = -np.sin(self.theta - self.pref_angles[s])
            drive = np.sum(activities * forces, axis=1)
            raw_noise = self.noise * np.random.randn(self.N)
            self.theta = (self.theta + self.ss * (drive + raw_noise)) % (2 * np.pi)
        elif self.manifold == 'line':
            forces = np.zeros((self.N, self.n_sub))
            for s in range(self.n_sub):
                forces[:, s] = self.pref_dirs[s] - self.pos
            drive = np.sum(activities * forces, axis=1)
            raw_noise = self.noise * np.random.randn(self.N)
            self.pos = np.clip(self.pos + self.ss * (drive + raw_noise), -3, 3)
        elif self.manifold == 'plane':
            forces = np.zeros((self.N, self.n_sub, 2))
            for s in range(self.n_sub):
                forces[:, s, :] = self.pref_vecs[s] - self.pos
            drive = np.einsum('ns,nsd->nd', activities, forces)
            raw_noise = self.noise * np.random.randn(self.N, 2)
            self.pos = np.clip(self.pos + self.ss * (drive + raw_noise), -3, 3)

        # --- Macro basin assignment ---
        if self.manifold == 'circle':
            dists = np.abs(np.subtract.outer(self.theta, self.basin_angles))
            dists = np.minimum(dists, 2 * np.pi - dists)
            basin = dists.argmin(axis=1)
        elif self.manifold == 'line':
            dists = np.abs(self.pos[:, None] - self.basin_centers[None, :])
            basin = dists.argmin(axis=1)
        elif self.manifold == 'plane':
            dists = np.sum((self.pos[:, None, :] - self.basin_centers_2d[None, :, :]) ** 2, axis=2)
            basin = dists.argmin(axis=1)

        # --- Clarity ---
        resultant = np.zeros(self.N)
        if self.manifold == 'circle':
            for s in range(self.n_sub):
                resultant += activities[:, s] * forces[:, s]
            clarity = np.abs(resultant)
        elif self.manifold == 'line':
            for s in range(self.n_sub):
                resultant += activities[:, s] * forces[:, s]
            clarity = np.abs(resultant)
        elif self.manifold == 'plane':
            res_vec = np.einsum('ns,nsd->nd', activities, forces)
            clarity = np.linalg.norm(res_vec, axis=1)

        dominant = activities.argmax(axis=1)

        self.history['clarity'].append(clarity.copy())
        self.history['basin'].append(basin.copy())
        self.history['dominant_sub'].append(dominant.copy())

    def get_arrays(self):
        """Convert history lists to numpy arrays."""
        return {
            'clarity': np.array(self.history['clarity']).T,    # (N, T)
            'basin': np.array(self.history['basin']).T,        # (N, T)
            'dominant_sub': np.array(self.history['dominant_sub']).T,
        }


# ============================================================================
# BOUNDARY METRICS (adapted for minimal engine)
# ============================================================================

def minimal_boundary_metrics(h, n_shuffles=100):
    """Compute boundary metrics from minimal engine history arrays."""
    basins = h['basin']
    clarity = h['clarity']
    N, steps = basins.shape

    # Metric 1: Edge-clarity temporal correlation
    window, stride = 50, 10
    n_win = max(1, (steps - window) // stride)
    edge_fracs = np.zeros(n_win)
    win_clarity = np.zeros(n_win)

    for w in range(n_win):
        t0 = w * stride
        t1 = t0 + window
        b = basins[:, t0:t1]
        c = clarity[:, t0:t1]
        trans = b[:, 1:] != b[:, :-1]
        trans_mask = np.concatenate([np.zeros((N, 1), dtype=bool), trans], axis=1)
        c_e = c[trans_mask]
        c_b = c[~trans_mask]
        if len(c_e) > 1 and len(c_b) > 1:
            s_e = np.var(c_e)
            ef = (s_e * len(c_e)) / (np.var(c) * c.size + 1e-15)
        else:
            ef = 0.0
        edge_fracs[w] = ef
        win_clarity[w] = c.mean()

    if np.std(edge_fracs) > 1e-10 and np.std(win_clarity) > 1e-10:
        edge_clarity_r, _ = stats.pearsonr(edge_fracs, win_clarity)
    else:
        edge_clarity_r = 0.0

    # Metric 2: transition clarity gap
    transitions = basins[:, 1:] != basins[:, :-1]
    c_trans = clarity[:, 1:][transitions]
    c_dwell = clarity[:, 1:][~transitions]

    if len(c_trans) > 10 and len(c_dwell) > 10:
        gap = float(c_trans.mean() - c_dwell.mean())
        pooled = np.sqrt((c_trans.std() ** 2 + c_dwell.std() ** 2) / 2)
        cohens_d = gap / (pooled + 1e-15)
    else:
        gap = 0.0
        cohens_d = 0.0

    # Metric 3: null z-score
    null_gaps = []
    for _ in range(n_shuffles):
        b_shuf = basins.copy()
        for i in range(N):
            np.random.shuffle(b_shuf[i])
        tr_shuf = b_shuf[:, 1:] != b_shuf[:, :-1]
        c_e = clarity[:, 1:][tr_shuf]
        c_b = clarity[:, 1:][~tr_shuf]
        if len(c_e) > 1 and len(c_b) > 1:
            null_gaps.append(float(c_e.mean() - c_b.mean()))
        else:
            null_gaps.append(0.0)
    null_gaps = np.array(null_gaps)
    null_z = (gap - null_gaps.mean()) / (null_gaps.std() + 1e-15)

    trans_rate = float(transitions.mean())

    return dict(
        edge_clarity_r=float(edge_clarity_r),
        cohens_d=float(cohens_d),
        null_z=float(null_z),
        transition_rate=float(trans_rate),
    )


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

MODELS = [
    # (name, kwargs, group)
    ('S¹ 8-sub full',
     dict(n_sub=8, dim=2, manifold='circle', use_fatigue=True, use_opponents=True),
     'reference'),

    ('R¹ 8-sub full',
     dict(n_sub=8, dim=1, manifold='line', use_fatigue=True, use_opponents=True),
     'manifold'),

    ('R² 8-sub full',
     dict(n_sub=8, dim=2, manifold='plane', use_fatigue=True, use_opponents=True),
     'manifold'),

    ('S¹ no fatigue',
     dict(n_sub=8, dim=2, manifold='circle', use_fatigue=False, use_opponents=True),
     'ablation'),

    ('S¹ no opponents',
     dict(n_sub=8, dim=2, manifold='circle', use_fatigue=True, use_opponents=False),
     'ablation'),

    ('S¹ 2-subsystems',
     dict(n_sub=2, dim=2, manifold='circle', use_fatigue=True, use_opponents=True, n_basins=4),
     'minimal'),

    ('S¹ 4-subsystems',
     dict(n_sub=4, dim=2, manifold='circle', use_fatigue=True, use_opponents=True, n_basins=6),
     'minimal'),
]

MODEL_COLORS = {
    'reference': '#2d2d2d',
    'manifold':  '#55A868',
    'ablation':  '#DD5555',
    'minimal':   '#4C72B0',
}


# ============================================================================
# RUN ALL MODELS
# ============================================================================

def run_all_models(steps=2000, N=256):
    """Run all minimal models and collect metrics."""
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'\n  Running {len(MODELS)} minimal models ({N} beings × {steps} steps)')
    print(f'  {"Model":<25s}  {"Manifold":>8s}  {"N_sub":>5s}  {"Fatigue":>7s}  {"Opps":>4s}')
    print(f'  {"-"*25}  {"-"*8}  {"-"*5}  {"-"*7}  {"-"*4}')
    for name, kwargs, _ in MODELS:
        print(f'  {name:<25s}  {kwargs["manifold"]:>8s}  '
              f'{kwargs["n_sub"]:>5d}  {str(kwargs["use_fatigue"]):>7s}  '
              f'{str(kwargs["use_opponents"]):>4s}')
    print()

    all_results = {}

    for i, (name, kwargs, group) in enumerate(MODELS):
        print(f'  [{i+1}/{len(MODELS)}] {name}...', end='', flush=True)
        t0 = time.time()

        engine = MinimalEngine(n_sims=N, **kwargs)
        for t in range(steps):
            engine.step()

        h = engine.get_arrays()
        metrics = minimal_boundary_metrics(h, n_shuffles=100)
        elapsed = time.time() - t0

        all_results[name] = {
            'group': group,
            **kwargs,
            **metrics,
            'wall_time': round(elapsed, 1),
        }

        # Verdict
        ecr = metrics['edge_clarity_r']
        nz = metrics['null_z']
        tr = metrics['transition_rate']
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

        all_results[name]['verdict'] = verdict

        print(f'  r={ecr:+.3f}  d={metrics["cohens_d"]:+.3f}  '
              f'z={nz:+.1f}  TR={tr:.3f}  → {verdict}  ({elapsed:.1f}s)')

    return all_results


# ============================================================================
# SUBSYSTEM COUNT GRADIENT
# ============================================================================

def subsystem_count_gradient(steps=2000, N=256):
    """Test boundary negotiation with 2, 3, 4, 5, 6, 7, 8 subsystems."""
    print('\n  ── Subsystem Count Gradient ──')

    counts = list(range(2, 9))
    rs, ds, zs = [], [], []

    for n_sub in counts:
        print(f'    n_sub={n_sub}...', end='', flush=True)
        engine = MinimalEngine(
            n_sub=n_sub, n_sims=N, dim=2, manifold='circle',
            use_fatigue=True, use_opponents=True,
            n_basins=max(4, n_sub), seed=42,
        )
        for t in range(steps):
            engine.step()

        h = engine.get_arrays()
        metrics = minimal_boundary_metrics(h, n_shuffles=50)
        rs.append(metrics['edge_clarity_r'])
        ds.append(metrics['cohens_d'])
        zs.append(metrics['null_z'])
        print(f'  r={metrics["edge_clarity_r"]:+.3f}  '
              f'd={metrics["cohens_d"]:+.3f}  z={metrics["null_z"]:+.1f}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vals, label in zip(axes, [rs, ds, zs],
                                ['Edge↔Clarity (r)', "Cohen's d", 'Null z']):
        ax.plot(counts, vals, 'o-', color='#4C72B0', markersize=8, linewidth=2)
        ax.set_xlabel('Number of Subsystems', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticks(counts)
        ax.axhline(0, color='k', linewidth=0.3)

    plt.suptitle('Boundary Negotiation vs Subsystem Count (S¹, cyclic, fatigue)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/subsystem_count_gradient.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {OUT_DIR}/subsystem_count_gradient.png')

    return {
        'counts': counts,
        'edge_clarity_r': rs,
        'cohens_d': ds,
        'null_z': zs,
    }


# ============================================================================
# COMPARISON PLOT
# ============================================================================

def comparison_plot(results):
    """4-panel comparison bar chart."""
    names = list(results.keys())
    n = len(names)
    colors = [MODEL_COLORS[results[nm]['group']] for nm in names]

    metrics = {
        'edge_clarity_r': 'Edge ↔ Clarity (r)',
        'cohens_d': "Cohen's d",
        'null_z': 'Null z-score',
        'transition_rate': 'Transition Rate',
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax_idx, (key, title) in enumerate(metrics.items()):
        ax = axes[ax_idx]
        vals = [results[nm][key] for nm in names]
        short = [nm.replace('-subsystems', '\nsub')
                  .replace(' full', '\nfull')
                  .replace(' no ', '\nno ')
                 for nm in names]
        bars = ax.bar(range(n), vals, color=colors, edgecolor='white')
        ax.set_xticks(range(n))
        ax.set_xticklabels(short, fontsize=7, ha='center')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axhline(0, color='k', linewidth=0.3)

        for bar, val in zip(bars, vals):
            y = bar.get_height()
            if abs(val) > 0.001:
                fmt = f'{val:+.2f}' if abs(val) < 100 else f'{val:+.0f}'
                ax.text(bar.get_x() + bar.get_width() / 2, y,
                        fmt, ha='center', va='bottom' if y >= 0 else 'top',
                        fontsize=7, fontweight='bold')

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=MODEL_COLORS['reference'], label='Reference'),
        Patch(facecolor=MODEL_COLORS['manifold'], label='Manifold variants'),
        Patch(facecolor=MODEL_COLORS['ablation'], label='Ablations'),
        Patch(facecolor=MODEL_COLORS['minimal'], label='Minimal count'),
    ]
    fig.legend(handles=legend, loc='upper center', ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{OUT_DIR}/minimal_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved: {OUT_DIR}/minimal_model_comparison.png')


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_summary(results):
    """Print formatted results."""
    print('\n' + '=' * 100)
    print('MINIMAL BOUNDARY MODEL — What Is the Simplest System?')
    print('=' * 100)

    print(f'\n  {"Model":<25s}  {"r":>8s}  {"d":>8s}  {"z":>8s}  '
          f'{"TR":>8s}  {"Verdict":>12s}')
    print(f'  {"-"*25}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*12}')

    for name, r in results.items():
        print(f'  {name:<25s}  {r["edge_clarity_r"]:>+8.3f}  '
              f'{r["cohens_d"]:>+8.3f}  {r["null_z"]:>+8.1f}  '
              f'{r["transition_rate"]:>8.3f}  {r["verdict"]:>12s}')

    # Find minimal model
    strong = [n for n, r in results.items() if r['verdict'] in ('STRONG', 'MODERATE')]
    if strong:
        print(f'\n  Models with boundary negotiation: {", ".join(strong)}')
        # Find smallest by n_sub
        smallest = min(strong, key=lambda n: results[n]['n_sub'])
        print(f'  Minimal model: {smallest} ({results[smallest]["n_sub"]} subsystems, '
              f'{results[smallest]["manifold"]} manifold)')


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Minimal Boundary Model')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--N', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print('╔══════════════════════════════════════════════════════════════════════╗')
    print('║  MINIMAL BOUNDARY MODEL — How Simple Can It Get?                   ║')
    print('║                                                                    ║')
    print('║  Testing S¹, R¹, R² with subsystem/fatigue/opponent ablations      ║')
    print('║  OPH by FloatingPragma                                             ║')
    print('║  https://github.com/FloatingPragma/observer-patch-holography       ║')
    print('╚══════════════════════════════════════════════════════════════════════╝')

    t_start = time.time()

    results = run_all_models(args.steps, args.N)
    print_summary(results)
    comparison_plot(results)

    sub_gradient = subsystem_count_gradient(args.steps, args.N)

    # Combine and save
    all_output = {'models': results, 'subsystem_gradient': sub_gradient}
    out_path = f'{OUT_DIR}/minimal_model_results.json'
    # Clean non-serializable fields
    for name in results:
        for k in list(results[name].keys()):
            if isinstance(results[name][k], bool):
                results[name][k] = int(results[name][k])

    with open(out_path, 'w') as f:
        json.dump(all_output, f, indent=2, default=str)
    print(f'\n  Results saved to {out_path}')

    elapsed = time.time() - t_start
    print(f'  Total wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
