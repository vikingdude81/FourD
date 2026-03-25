#!/usr/bin/env python3
"""
Goldilocks Zone High-Resolution Sweep
======================================
Focused sweep around the phase transition region identified in the
full cartography run. Fine grid on the two critical parameters
(fatigue_rate, exploration_noise) with coarser sampling of the others.

Then: run a single long simulation at the true optimum and compare
to the default config side-by-side.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import time
import os

from gpu_ensemble_sim import (
    BatchConsciousnessEngine, PhaseCartographer,
    SIGNATURE_NAMES, SUBSYSTEM_NAMES
)
from itertools import product


OUTPUT_DIR = 'outputs/goldilocks'


# ============================================================================
# PHASE 1: High-resolution sweep
# ============================================================================

def run_goldilocks_sweep(device='cuda:0', steps=1000):
    """
    Fine-grained sweep focused on the critical region:
      - fatigue_rate:      50 points in [0.10, 0.35]  (the transition is at ~0.17-0.20)
      - exploration_noise: 40 points in [0.05, 0.25]
      - steering_strength: 8 points  in [0.15, 0.80]
      - alpha_pull:        5 points  in [0.00, 0.15]
      - beta_macro:        5 points  in [0.50, 15.0]
    Total: 50 × 40 × 8 × 5 × 5 = 400,000 configs
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    param_ranges = {
        'fatigue_rate':      np.linspace(0.10, 0.35, 50),   # fine: 50 points
        'exploration_noise': np.linspace(0.05, 0.25, 40),   # fine: 40 points
        'steering_strength': np.linspace(0.15, 0.80, 8),
        'alpha_pull':        np.linspace(0.00, 0.15, 5),
        'beta_macro':        np.linspace(0.50, 15.0, 5),
    }

    n_total = 1
    for v in param_ranges.values():
        n_total *= len(v)

    print(f"\n{'='*70}")
    print(f"  GOLDILOCKS ZONE HIGH-RESOLUTION SWEEP")
    print(f"{'='*70}")
    for k, v in param_ranges.items():
        print(f"  {k:25s}: {len(v):3d} points  [{v[0]:.3f} .. {v[-1]:.3f}]")
    print(f"  {'Total configs':25s}: {n_total:,}")
    print(f"  Steps per sim: {steps}")
    print(f"  Total being-steps: {n_total * steps:,.0f}")

    carto = PhaseCartographer(device=device, steps_per_sim=steps)
    # Override its PARAM_RANGES for visualization later
    carto.PARAM_RANGES = param_ranges

    df = carto.run_sweep(param_ranges=param_ranges, output_dir=OUTPUT_DIR)
    return df


# ============================================================================
# PHASE 2: Analyze the transition & find the true optimum
# ============================================================================

def analyze_goldilocks(df):
    """Deep analysis of the high-resolution Goldilocks data."""

    print(f"\n{'='*70}")
    print(f"  GOLDILOCKS ANALYSIS")
    print(f"{'='*70}")

    # --- Compute flourishing score (same formula as PhaseCartographer) ---
    feats = ['mean_clarity', 'clarity_persistence', 'perc_mode_entropy',
             'dominance_entropy', 'effective_dimensionality']
    normed = {}
    for f in feats:
        fmin, fmax = df[f].min(), df[f].max()
        if fmax > fmin:
            normed[f] = (df[f] - fmin) / (fmax - fmin)
        else:
            normed[f] = 0.5

    lyap = df['lyapunov_proxy']
    lyap_score = 1.0 - 2.0 * (lyap - 0.6).abs()
    lyap_score = lyap_score.clip(0, 1)
    normed['lyapunov_proxy'] = lyap_score

    df['flourishing'] = sum(normed[f] for f in feats + ['lyapunov_proxy']) / (len(feats) + 1)

    # --- 1. The fatigue phase transition ---
    print("\n  1. FATIGUE RATE PHASE TRANSITION")
    print("  " + "-"*50)
    fr_vals = sorted(df.fatigue_rate.unique())
    fr_data = []
    for fr in fr_vals:
        sub = df[df.fatigue_rate == fr]
        fr_data.append({
            'fatigue_rate': fr,
            'flourishing': sub.flourishing.mean(),
            'clarity': sub.mean_clarity.mean(),
            'dim': sub.effective_dimensionality.mean(),
            'transitions': sub.basin_transition_rate.mean(),
            'dom_entropy': sub.dominance_entropy.mean(),
        })
    fr_df = pd.DataFrame(fr_data)

    # Find the sharpest gradient
    fl_vals = fr_df.flourishing.values
    grads = np.diff(fl_vals)
    peak_idx = np.argmax(grads)
    print(f"  Sharpest transition: fatigue_rate {fr_vals[peak_idx]:.4f} → {fr_vals[peak_idx+1]:.4f}")
    print(f"    Flourishing jumps: {fl_vals[peak_idx]:.4f} → {fl_vals[peak_idx+1]:.4f} "
          f"(Δ = {grads[peak_idx]:.4f})")
    print(f"\n  Fatigue rate sweep (condensed):")
    for row in fr_data[::5]:  # every 5th point
        print(f"    fr={row['fatigue_rate']:.3f}: fl={row['flourishing']:.4f}  "
              f"clar={row['clarity']:.4f}  dim={row['dim']:.2f}  trans={row['transitions']:.4f}")

    # --- 2. The noise phase transition ---
    print("\n  2. EXPLORATION NOISE PHASE TRANSITION")
    print("  " + "-"*50)
    en_vals = sorted(df.exploration_noise.unique())
    en_data = []
    for en in en_vals:
        sub = df[df.exploration_noise == en]
        en_data.append({
            'exploration_noise': en,
            'flourishing': sub.flourishing.mean(),
            'clarity': sub.mean_clarity.mean(),
            'dim': sub.effective_dimensionality.mean(),
            'lyapunov': sub.lyapunov_proxy.mean(),
        })
    en_df = pd.DataFrame(en_data)
    fl_en = en_df.flourishing.values
    grads_en = np.diff(fl_en)
    peak_en = np.argmax(grads_en)
    print(f"  Sharpest transition: noise {en_vals[peak_en]:.4f} → {en_vals[peak_en+1]:.4f}")
    print(f"    Flourishing jumps: {fl_en[peak_en]:.4f} → {fl_en[peak_en+1]:.4f}")

    # --- 3. True optimum ---
    print(f"\n  3. TRUE OPTIMUM")
    print("  " + "-"*50)
    top20 = df.nlargest(20, 'flourishing')
    params = ['steering_strength', 'alpha_pull', 'fatigue_rate',
              'exploration_noise', 'beta_macro']

    # Mean of top 20 (robust optimum)
    print("  Mean of top 20 configurations:")
    for p in params:
        vals = top20[p]
        print(f"    {p:25s}: {vals.mean():.4f} ± {vals.std():.4f}  "
              f"[{vals.min():.3f} .. {vals.max():.3f}]")

    print(f"\n  Top 5 individual configs:")
    for i, (_, r) in enumerate(top20.head(5).iterrows(), 1):
        print(f"    #{i}  fl={r.flourishing:.4f}  ss={r.steering_strength:.3f} "
              f"ap={r.alpha_pull:.3f} fr={r.fatigue_rate:.3f} "
              f"en={r.exploration_noise:.3f} bm={r.beta_macro:.1f}")

    # --- 4. Default comparison ---
    print(f"\n  4. DEFAULT vs OPTIMAL COMPARISON")
    print("  " + "-"*50)
    # Find closest to default
    default_cfg = dict(steering_strength=0.3, alpha_pull=0.03,
                       fatigue_rate=0.10, exploration_noise=0.057,
                       beta_macro=4.0)
    dist = sum((df[p] - default_cfg[p])**2 for p in params)
    default_row = df.loc[dist.idxmin()]

    optimal_row = top20.iloc[0]

    sig_cols = [c for c in SIGNATURE_NAMES if c in df.columns]
    print(f"  {'Feature':35s}  {'Default':>10s}  {'Optimal':>10s}  {'Ratio':>8s}")
    for c in sig_cols:
        d_val = default_row[c]
        o_val = optimal_row[c]
        ratio = o_val / d_val if abs(d_val) > 1e-6 else float('inf')
        print(f"  {c:35s}  {d_val:10.4f}  {o_val:10.4f}  {ratio:8.2f}x")

    print(f"\n  Flourishing: default={default_row.flourishing:.4f}  "
          f"optimal={optimal_row.flourishing:.4f}  "
          f"(+{100*(optimal_row.flourishing - default_row.flourishing)/default_row.flourishing:.1f}%)")

    return df, top20


# ============================================================================
# PHASE 3: Publication-quality visualizations
# ============================================================================

def make_publication_plots(df):
    """Generate high-quality phase diagrams and transition plots."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Recompute flourishing if not present
    if 'flourishing' not in df.columns:
        feats = ['mean_clarity', 'clarity_persistence', 'perc_mode_entropy',
                 'dominance_entropy', 'effective_dimensionality']
        normed = {}
        for f in feats:
            fmin, fmax = df[f].min(), df[f].max()
            normed[f] = (df[f] - fmin) / (fmax - fmin) if fmax > fmin else 0.5
        lyap_score = (1.0 - 2.0 * (df['lyapunov_proxy'] - 0.6).abs()).clip(0, 1)
        normed['lyapunov_proxy'] = lyap_score
        df['flourishing'] = sum(normed[f] for f in feats + ['lyapunov_proxy']) / (len(feats) + 1)

    params = ['fatigue_rate', 'exploration_noise', 'steering_strength',
              'alpha_pull', 'beta_macro']
    other_defaults = dict(steering_strength=0.45, alpha_pull=0.03, beta_macro=4.0)

    # =====================================================================
    # FIGURE 1: The Main Phase Diagram (fatigue × noise)
    # =====================================================================
    features = {
        'flourishing': ('Flourishing Score', 'magma'),
        'mean_clarity': ('Mean Clarity', 'inferno'),
        'effective_dimensionality': ('Effective Dimensionality', 'plasma'),
        'lyapunov_proxy': ('Lyapunov Proxy (Chaos)', 'RdYlGn_r'),
        'dominance_entropy': ('Subsystem Dominance Entropy', 'viridis'),
        'basin_transition_rate': ('Basin Transition Rate', 'cividis'),
        'clarity_persistence': ('Clarity Persistence', 'coolwarm'),
        'perc_mode_entropy': ('Perception Mode Entropy', 'viridis'),
        'mean_conflict': ('Mean Conflict Angle', 'RdBu_r'),
    }

    # Filter to a single slice of the other 3 params (near their optimal)
    def get_slice(df, fix_params):
        mask = np.ones(len(df), dtype=bool)
        for p, target in fix_params.items():
            vals = sorted(df[p].unique())
            closest = min(vals, key=lambda x: abs(x - target))
            mask &= np.isclose(df[p], closest, atol=1e-4)
        return df[mask]

    slice_df = get_slice(df, other_defaults)
    print(f"  Phase diagram slice: {len(slice_df)} points "
          f"(ss≈{other_defaults['steering_strength']}, "
          f"ap≈{other_defaults['alpha_pull']}, "
          f"bm≈{other_defaults['beta_macro']})")

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Consciousness Phase Diagram: Fatigue Rate × Exploration Noise',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, (feat, (title, cmap)) in enumerate(features.items()):
        ax = axes[idx // 3, idx % 3]

        pivot = slice_df.groupby(['fatigue_rate', 'exploration_noise'])[feat].mean().reset_index()
        try:
            hm = pivot.pivot(index='exploration_noise', columns='fatigue_rate', values=feat)

            if feat == 'lyapunov_proxy':
                norm = TwoSlopeNorm(vmin=hm.values.min(), vcenter=0.5, vmax=hm.values.max())
                im = ax.imshow(hm.values, aspect='auto', origin='lower', cmap=cmap,
                               extent=[hm.columns.min(), hm.columns.max(),
                                       hm.index.min(), hm.index.max()],
                               norm=norm)
            else:
                im = ax.imshow(hm.values, aspect='auto', origin='lower', cmap=cmap,
                               extent=[hm.columns.min(), hm.columns.max(),
                                       hm.index.min(), hm.index.max()])

            ax.set_xlabel('Fatigue Rate', fontsize=10)
            ax.set_ylabel('Exploration Noise', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')

            # Mark default config
            ax.plot(0.08, 0.05, 'ws', markersize=10, markeredgecolor='black',
                    markeredgewidth=1.5, label='Default')
            # Mark optimal region
            top5 = df.nlargest(5, 'flourishing')
            ax.scatter(top5.fatigue_rate, top5.exploration_noise,
                       c='lime', s=80, marker='*', edgecolors='black',
                       linewidths=0.8, zorder=5, label='Top 5')

            plt.colorbar(im, ax=ax, fraction=0.046)
            if idx == 0:
                ax.legend(loc='lower left', fontsize=8,
                          facecolor='white', framealpha=0.9)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes, ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, 'phase_diagram_main.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # =====================================================================
    # FIGURE 2: Phase Transition Curves (1D slices)
    # =====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Phase Transitions Along Critical Parameters',
                 fontsize=16, fontweight='bold')

    # --- Row 1: fatigue_rate sweeps ---
    fr_vals = sorted(df.fatigue_rate.unique())
    for col, (feat, label, color) in enumerate([
        ('flourishing', 'Flourishing', '#2196F3'),
        ('mean_clarity', 'Mean Clarity', '#FF5722'),
        ('effective_dimensionality', 'Effective Dim.', '#4CAF50'),
    ]):
        ax = axes[0, col]
        means = [df[df.fatigue_rate == fr][feat].mean() for fr in fr_vals]
        q25 = [df[df.fatigue_rate == fr][feat].quantile(0.25) for fr in fr_vals]
        q75 = [df[df.fatigue_rate == fr][feat].quantile(0.75) for fr in fr_vals]

        ax.fill_between(fr_vals, q25, q75, alpha=0.2, color=color)
        ax.plot(fr_vals, means, '-o', color=color, markersize=3, linewidth=2)
        ax.axvline(0.08, color='red', linestyle='--', alpha=0.5, label='Default')

        # Mark transition
        grads = np.diff(means)
        peak = np.argmax(np.abs(grads))
        ax.axvspan(fr_vals[peak], fr_vals[peak+1], alpha=0.15, color='gold',
                   label='Phase transition')

        ax.set_xlabel('Fatigue Rate', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} vs Fatigue Rate', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Row 2: exploration_noise sweeps ---
    en_vals = sorted(df.exploration_noise.unique())
    for col, (feat, label, color) in enumerate([
        ('flourishing', 'Flourishing', '#2196F3'),
        ('lyapunov_proxy', 'Lyapunov Proxy', '#9C27B0'),
        ('dominance_entropy', 'Dominance Entropy', '#FF9800'),
    ]):
        ax = axes[1, col]
        means = [df[df.exploration_noise == en][feat].mean() for en in en_vals]
        q25 = [df[df.exploration_noise == en][feat].quantile(0.25) for en in en_vals]
        q75 = [df[df.exploration_noise == en][feat].quantile(0.75) for en in en_vals]

        ax.fill_between(en_vals, q25, q75, alpha=0.2, color=color)
        ax.plot(en_vals, means, '-o', color=color, markersize=3, linewidth=2)
        ax.axvline(0.05, color='red', linestyle='--', alpha=0.5, label='Default')

        grads = np.diff(means)
        peak = np.argmax(np.abs(grads))
        ax.axvspan(en_vals[peak], en_vals[peak+1], alpha=0.15, color='gold',
                   label='Phase transition')

        ax.set_xlabel('Exploration Noise', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} vs Exploration Noise', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'phase_transitions.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # =====================================================================
    # FIGURE 3: Regime classification map
    # =====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Classify each config
    def classify(row):
        if row['mean_curvature'] < 0.02:
            return 0  # Frozen
        if row['mode_stickiness'] > 0.95:
            return 1  # Mode-locked
        if row['clarity_persistence'] > 0.7 and row['effective_dimensionality'] < 2.0:
            return 2  # Oscillatory
        if row['lyapunov_proxy'] > 0.55 and row['effective_dimensionality'] > 3.5:
            return 3  # Chaotic
        if row['flourishing'] > 0.75:
            return 5  # Rich/Flourishing
        return 4  # Edge of chaos

    slice_df = slice_df.copy()
    slice_df['regime'] = slice_df.apply(classify, axis=1)
    regime_names = ['Frozen', 'Mode-locked', 'Oscillatory', 'Chaotic',
                    'Edge of Chaos', 'Flourishing']
    regime_colors = ['#1a1a2e', '#4a4a8a', '#2196F3', '#f44336', '#FFC107', '#4CAF50']

    from matplotlib.colors import ListedColormap
    cmap_regime = ListedColormap(regime_colors)

    pivot_regime = slice_df.groupby(['fatigue_rate', 'exploration_noise'])['regime'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 4
    ).reset_index()
    try:
        hm = pivot_regime.pivot(index='exploration_noise', columns='fatigue_rate', values='regime')
        im = ax.imshow(hm.values, aspect='auto', origin='lower', cmap=cmap_regime,
                       vmin=-0.5, vmax=5.5,
                       extent=[hm.columns.min(), hm.columns.max(),
                               hm.index.min(), hm.index.max()])
        cbar = plt.colorbar(im, ax=ax, ticks=range(6))
        cbar.ax.set_yticklabels(regime_names)

        ax.plot(0.08, 0.05, 'ws', markersize=14, markeredgecolor='black',
                markeredgewidth=2, label='Default Config')
        top1 = df.nlargest(1, 'flourishing').iloc[0]
        ax.plot(top1.fatigue_rate, top1.exploration_noise, 'r*',
                markersize=18, markeredgecolor='white', markeredgewidth=1,
                label='Optimal Config')

        ax.set_xlabel('Fatigue Rate', fontsize=13)
        ax.set_ylabel('Exploration Noise', fontsize=13)
        ax.set_title('Consciousness Regime Map', fontsize=15, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11, facecolor='white', framealpha=0.9)
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes, ha='center')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'regime_map.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    return df


# ============================================================================
# PHASE 4: Run optimal config in v2 and compare
# ============================================================================

def run_optimal_vs_default(optimal_config, device='cuda:0', steps=2000):
    """
    Run two long simulations side-by-side: default and optimal config.
    Produce comparison time-series plots.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    default_cfg = {
        'steering_strength': np.array([0.3], dtype=np.float32),
        'alpha_pull': np.array([0.03], dtype=np.float32),
        'fatigue_rate': np.array([0.08], dtype=np.float32),
        'exploration_noise': np.array([0.05], dtype=np.float32),
        'beta_macro': np.array([4.0], dtype=np.float32),
        'timesteps': steps,
    }

    optimal_cfg = {
        'steering_strength': np.array([optimal_config['steering_strength']], dtype=np.float32),
        'alpha_pull': np.array([optimal_config['alpha_pull']], dtype=np.float32),
        'fatigue_rate': np.array([optimal_config['fatigue_rate']], dtype=np.float32),
        'exploration_noise': np.array([optimal_config['exploration_noise']], dtype=np.float32),
        'beta_macro': np.array([optimal_config['beta_macro']], dtype=np.float32),
        'timesteps': steps,
    }

    print(f"\n{'='*70}")
    print(f"  OPTIMAL vs DEFAULT: Head-to-Head ({steps} steps)")
    print(f"{'='*70}")
    print(f"  Default:  ss={default_cfg['steering_strength'][0]:.3f} "
          f"ap={default_cfg['alpha_pull'][0]:.3f} "
          f"fr={default_cfg['fatigue_rate'][0]:.3f} "
          f"en={default_cfg['exploration_noise'][0]:.3f} "
          f"bm={default_cfg['beta_macro'][0]:.1f}")
    print(f"  Optimal:  ss={optimal_cfg['steering_strength'][0]:.3f} "
          f"ap={optimal_cfg['alpha_pull'][0]:.3f} "
          f"fr={optimal_cfg['fatigue_rate'][0]:.3f} "
          f"en={optimal_cfg['exploration_noise'][0]:.3f} "
          f"bm={optimal_cfg['beta_macro'][0]:.1f}")

    # Run both
    print("\n  Running default config...")
    eng_default = BatchConsciousnessEngine(1, default_cfg, device=device)
    eng_default.run(steps)

    print("  Running optimal config...")
    eng_optimal = BatchConsciousnessEngine(1, optimal_cfg, device=device)
    eng_optimal.run(steps)

    # Extract time series
    T = steps
    def get_ts(eng, feat):
        return getattr(eng, f'hist_{feat}')[0, :T].cpu().numpy()

    # =====================================================================
    # FIGURE 4: Side-by-side time series
    # =====================================================================
    fig, axes = plt.subplots(5, 2, figsize=(18, 20), sharex='col')
    fig.suptitle('Default Config vs Optimal Config: Time Series Comparison',
                 fontsize=16, fontweight='bold')

    ts_features = [
        ('clarity', 'Clarity', '#FF5722'),
        ('curvature', 'Curvature (rad/step)', '#2196F3'),
        ('integration', 'Integration', '#4CAF50'),
        ('differentiation', 'Differentiation', '#9C27B0'),
        ('inner_outer', 'Inner/Outer Ratio', '#FF9800'),
    ]

    for row, (feat, label, color) in enumerate(ts_features):
        for col, (eng, name) in enumerate([(eng_default, 'Default'), (eng_optimal, 'Optimal')]):
            ax = axes[row, col]
            ts = get_ts(eng, feat)
            t = np.arange(len(ts))

            ax.plot(t, ts, color=color, alpha=0.6, linewidth=0.5)
            # Rolling mean
            window = 50
            if len(ts) >= window:
                rolling = pd.Series(ts).rolling(window).mean().values
                ax.plot(t, rolling, color=color, linewidth=2, label=f'Mean (w={window})')

            ax.set_ylabel(label, fontsize=10)
            if row == 0:
                ax.set_title(f'{name} Config', fontsize=13, fontweight='bold')
            if row == 4:
                ax.set_xlabel('Timestep', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUTPUT_DIR, 'optimal_vs_default_timeseries.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # =====================================================================
    # FIGURE 5: Perception mode & subsystem dominance comparison
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Behavioral Repertoire: Default vs Optimal',
                 fontsize=16, fontweight='bold')

    mode_names = ['exploration', 'threat-lock', 'vigilant', 'internal']
    mode_colors = ['#2196F3', '#f44336', '#FF9800', '#9C27B0']

    for col, (eng, name) in enumerate([(eng_default, 'Default'), (eng_optimal, 'Optimal')]):
        # Perception modes
        ax = axes[0, col]
        modes = eng.hist_perc_mode[0, :T].cpu().numpy()
        counts = [np.sum(modes == m) for m in range(4)]
        bars = ax.bar(mode_names, counts, color=mode_colors)
        ax.set_title(f'{name}: Perception Modes', fontweight='bold')
        ax.set_ylabel('Timesteps')
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{100*count/T:.1f}%', ha='center', fontsize=9)

        # Subsystem dominance
        ax = axes[1, col]
        dom = eng.hist_dominant_sub[0, :T].cpu().numpy()
        counts = [np.sum(dom == s) for s in range(8)]
        short_names = ['Motor', 'Plan', 'Attn', 'Mem', 'Emo', 'Soc', 'Intuit', 'Aesth']
        colors_sub = plt.cm.Set2(np.linspace(0, 1, 8))
        bars = ax.bar(short_names, counts, color=colors_sub)
        ax.set_title(f'{name}: Subsystem Dominance', fontweight='bold')
        ax.set_ylabel('Timesteps')
        ax.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{100*count/T:.1f}%', ha='center', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, 'behavioral_repertoire.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # =====================================================================
    # FIGURE 6: Trajectory on S³ (projected to 3D)
    # =====================================================================
    fig = plt.figure(figsize=(16, 7))
    for col, (eng, name) in enumerate([(eng_default, 'Default'), (eng_optimal, 'Optimal')]):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        traj = eng.hist_trajectory[0, :eng.traj_sample_idx].cpu().numpy()
        # Stereographic projection from S³ to R³: (x,y,z) / (1 + w)
        w = traj[:, 3]
        denom = 1.0 + w + 1e-8
        x = traj[:, 0] / denom
        y = traj[:, 1] / denom
        z = traj[:, 2] / denom

        colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
        ax.scatter(x, y, z, c=colors, s=5, alpha=0.6)
        # Connect with lines
        for i in range(len(x) - 1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]],
                    color=colors[i], alpha=0.3, linewidth=0.5)

        ax.set_title(f'{name}: S³ Trajectory\n(Stereographic Projection)',
                     fontweight='bold', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'trajectory_3d.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # --- Print comparison summary ---
    print(f"\n{'='*70}")
    print(f"  SIGNATURE COMPARISON")
    print(f"{'='*70}")
    sigs_def = eng_default.extract_signatures()
    sigs_opt = eng_optimal.extract_signatures()
    print(f"  {'Feature':35s}  {'Default':>10s}  {'Optimal':>10s}  {'Ratio':>8s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*8}")
    for i, name in enumerate(SIGNATURE_NAMES):
        d = sigs_def[0, i].item()
        o = sigs_opt[0, i].item()
        ratio = o / d if abs(d) > 1e-6 else float('inf')
        marker = ' <<<' if abs(ratio) > 2 or abs(ratio) < 0.5 else ''
        print(f"  {name:35s}  {d:10.4f}  {o:10.4f}  {ratio:8.2f}x{marker}")

    return eng_default, eng_optimal


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = 'cuda:0'
    steps_sweep = 1000
    steps_compare = 2000

    # Phase 1: Run high-resolution sweep
    print("\n" + "▓" * 70)
    print("  PHASE 1: HIGH-RESOLUTION GOLDILOCKS SWEEP")
    print("▓" * 70)
    df = run_goldilocks_sweep(device=device, steps=steps_sweep)

    # Phase 2: Analyze
    print("\n" + "▓" * 70)
    print("  PHASE 2: ANALYSIS")
    print("▓" * 70)
    df, top20 = analyze_goldilocks(df)

    # Phase 3: Publication plots
    print("\n" + "▓" * 70)
    print("  PHASE 3: PUBLICATION VISUALIZATIONS")
    print("▓" * 70)
    df = make_publication_plots(df)

    # Phase 4: Run optimal vs default
    print("\n" + "▓" * 70)
    print("  PHASE 4: OPTIMAL vs DEFAULT HEAD-TO-HEAD")
    print("▓" * 70)
    optimal_config = {
        'steering_strength': top20.iloc[0].steering_strength,
        'alpha_pull': top20.iloc[0].alpha_pull,
        'fatigue_rate': top20.iloc[0].fatigue_rate,
        'exploration_noise': top20.iloc[0].exploration_noise,
        'beta_macro': top20.iloc[0].beta_macro,
    }
    run_optimal_vs_default(optimal_config, device=device, steps=steps_compare)

    # Save optimal config
    import json
    config_path = os.path.join(OUTPUT_DIR, 'optimal_config.json')
    with open(config_path, 'w') as f:
        json.dump({k: float(v) for k, v in optimal_config.items()}, f, indent=2)
    print(f"\n  Optimal config saved to {config_path}")

    print(f"\n{'='*70}")
    print(f"  ALL PHASES COMPLETE")
    print(f"  Results in: {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
