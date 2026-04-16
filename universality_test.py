#!/usr/bin/env python3
"""
Universality Test — Boundary Negotiation Across Engine Architectures
====================================================================

Tests whether boundary-dominated information concentration is a universal
property of multi-agent competition systems, or specific to the FourD
engine architecture.

Varies three structural axes independently:
  TOPOLOGY:    cyclic opponents | random pairs | fully connected | none
  FATIGUE:     gradual accumulation | winner-take-all | none | stochastic
  MANIFOLD:    S³ (4-sphere) | S² (3-sphere) | flat R⁴

For each of 10 variants, computes three core boundary negotiation metrics:
  - Edge-clarity temporal correlation (r)
  - Transition-conditioned clarity gap (Cohen's d)
  - Null model deviation (z-score)

OPH Credit: Framework adapted from Observer Patch Holography by FloatingPragma.
  https://github.com/FloatingPragma/observer-patch-holography

Usage:
    python universality_test.py [--device cuda:0] [--steps 1000]
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
import torch.nn.functional as F
from scipy import stats
from sklearn.cluster import KMeans

from gpu_ensemble_sim import (
    PREFERENCE_MATRIX_NORMED,
    generate_fibonacci_s3,
    derive_macro_basins,
)


# ============================================================================
# PREFERENCE GENERATION
# ============================================================================

def generate_preferences(n_sub, dim, topology, seed=42):
    """
    Generate (n_sub, dim) preference matrix for a given topology.

    Topologies:
      'cyclic':  4 opponent pairs, each anti-parallel
      'random':  random opponent pairings, anti-parallel
      'full':    maximally separated directions (high competition)
      'none':    clustered directions (low competition)
    """
    rng = np.random.RandomState(seed)

    if topology == 'cyclic':
        n_pairs = n_sub // 2
        # Generate n_pairs directions, Gram-Schmidt orthogonalize
        raw = rng.randn(n_pairs, dim)
        vecs = np.zeros_like(raw)
        for i in range(n_pairs):
            v = raw[i].copy()
            for j in range(i):
                v -= np.dot(v, vecs[j]) * vecs[j]
            norm = np.linalg.norm(v)
            vecs[i] = v / (norm + 1e-8) if norm > 1e-8 else rng.randn(dim)
            vecs[i] /= np.linalg.norm(vecs[i])
        prefs = np.zeros((n_sub, dim))
        for i in range(n_pairs):
            prefs[i] = vecs[i]
            prefs[i + n_pairs] = -vecs[i]
        return prefs

    elif topology == 'random':
        # Random pairing: shuffle indices then pair consecutively
        indices = rng.permutation(n_sub)
        n_pairs = n_sub // 2
        raw = rng.randn(n_pairs, dim)
        for i in range(n_pairs):
            raw[i] /= np.linalg.norm(raw[i]) + 1e-8
        prefs = np.zeros((n_sub, dim))
        for i in range(n_pairs):
            prefs[indices[2 * i]] = raw[i]
            prefs[indices[2 * i + 1]] = -raw[i]
        return prefs

    elif topology == 'full':
        # Maximally separated: use vertices of cross-polytope + extras
        prefs = np.zeros((n_sub, dim))
        for i in range(min(n_sub, 2 * dim)):
            axis = i // 2
            sign = 1 if i % 2 == 0 else -1
            if axis < dim:
                prefs[i, axis] = sign
        # Fill remaining with random unit vectors
        for i in range(2 * dim, n_sub):
            v = rng.randn(dim)
            prefs[i] = v / (np.linalg.norm(v) + 1e-8)
        return prefs

    elif topology == 'none':
        # Clustered: small perturbations around a single direction
        base = np.zeros(dim)
        base[0] = 1.0
        prefs = np.zeros((n_sub, dim))
        for i in range(n_sub):
            noise = rng.randn(dim) * 0.15
            v = base + noise
            prefs[i] = v / (np.linalg.norm(v) + 1e-8)
        return prefs

    raise ValueError(f'Unknown topology: {topology}')


# ============================================================================
# MANIFOLD HELPERS
# ============================================================================

def fibonacci_s2(n):
    """Fibonacci lattice on S²."""
    golden = (1 + np.sqrt(5)) / 2
    points = np.zeros((n, 3))
    for i in range(n):
        theta = 2 * np.pi * i / golden
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        points[i, 0] = np.sin(phi) * np.cos(theta)
        points[i, 1] = np.sin(phi) * np.sin(theta)
        points[i, 2] = np.cos(phi)
    norms = np.linalg.norm(points, axis=1, keepdims=True) + 1e-8
    return points / norms


def make_macro_centers(manifold, n_macro=24, device='cuda:0'):
    """Generate macro basin centers for a given manifold."""
    if manifold == 's3':
        micro = generate_fibonacci_s3(600)
        centers = derive_macro_basins(micro, n_macro)
    elif manifold == 's2':
        micro = fibonacci_s2(300)
        km = KMeans(n_clusters=n_macro, random_state=0, n_init=10)
        km.fit(micro)
        centers = km.cluster_centers_
        norms = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
        centers = centers / norms
    elif manifold == 'flat4':
        samples = np.random.RandomState(0).randn(500, 4) * 0.8
        km = KMeans(n_clusters=n_macro, random_state=0, n_init=10)
        km.fit(samples)
        centers = km.cluster_centers_
    else:
        raise ValueError(f'Unknown manifold: {manifold}')
    return torch.tensor(centers, dtype=torch.float32, device=device)


# ============================================================================
# UNIVERSAL ENGINE
# ============================================================================

class UniversalEngine:
    """
    Modular consciousness engine with swappable topology, fatigue, and manifold.

    Produces the same history tensor interface as BatchConsciousnessEngine
    (hist_clarity, hist_macro_basin, hist_dominant_sub) for use with
    boundary negotiation metrics.
    """

    def __init__(self, N, device='cuda:0', steps=1000,
                 manifold='s3', topology='cyclic', fatigue_type='gradual',
                 steering_strength=0.707, fatigue_rate=0.217,
                 exploration_noise=0.25, beta_macro=11.375):
        self.N = N
        self.device = torch.device(device)
        self.manifold_type = manifold
        self.topology_type = topology
        self.fatigue_type = fatigue_type
        self.n_sub = 8
        self.max_steps = steps

        # Dimension from manifold
        self.dim = {'s3': 4, 's2': 3, 'flat4': 4}[manifold]

        # Preferences
        if manifold == 's3' and topology == 'cyclic':
            # Use exact FourD preferences for baseline
            self.prefs = torch.tensor(
                PREFERENCE_MATRIX_NORMED, dtype=torch.float32, device=self.device)
        else:
            prefs_np = generate_preferences(self.n_sub, self.dim, topology)
            self.prefs = torch.tensor(
                prefs_np, dtype=torch.float32, device=self.device)

        # Macro basin centers
        self.macro_centers = make_macro_centers(manifold, 24, device)
        self.n_macro = 24

        # Scalar parameters
        self.steering_strength = steering_strength
        self.fatigue_rate = fatigue_rate
        self.exploration_noise = exploration_noise
        self.beta_macro = beta_macro
        self.recovery_rate = 0.025
        self.floor_value = 0.05
        self.novelty_weight = 0.6

        # State
        dev = self.device
        if manifold in ('s3', 's2'):
            init = torch.randn(N, self.dim, device=dev)
            self.u_t = F.normalize(init, dim=1)
        else:  # flat4
            self.u_t = torch.randn(N, self.dim, device=dev) * 0.3

        self.fatigue = torch.zeros(N, self.n_sub, device=dev)

        # History
        self.hist_clarity = torch.zeros(N, steps, device=dev)
        self.hist_macro_basin = torch.zeros(N, steps, dtype=torch.int32, device=dev)
        self.hist_dominant_sub = torch.zeros(N, steps, dtype=torch.int32, device=dev)
        self.step_count = 0

    @torch.no_grad()
    def step(self):
        t = self.step_count
        N, dev = self.N, self.device
        u = self.u_t

        # === INFLUENCES ===
        if self.manifold_type == 'flat4':
            u_dir = F.normalize(u, dim=1)
            influences = torch.einsum('nd,sd->ns', u_dir, self.prefs)
        else:
            influences = torch.einsum('nd,sd->ns', u, self.prefs)
        influences = 0.5 + 0.3 * influences

        # === FATIGUE → ACTIVITIES ===
        effective = influences * torch.exp(-self.fatigue)
        noise = self.exploration_noise * torch.randn(N, self.n_sub, device=dev)
        effective = (effective + noise).clamp(min=self.floor_value)
        activities = effective / (effective.sum(dim=1, keepdim=True) + 1e-8)

        # === UPDATE FATIGUE ===
        if self.fatigue_type == 'gradual':
            self.fatigue = self.fatigue + self.fatigue_rate * activities
            excess = (activities - 1.0 / self.n_sub).clamp(min=0.02) - 0.02
            self.fatigue = self.fatigue + 0.03 * excess
            inactive = (1.0 - activities) * self.recovery_rate
            self.fatigue = (self.fatigue - inactive).clamp(0, 3)
        elif self.fatigue_type == 'wta':
            winner = activities.argmax(dim=1)
            mask = F.one_hot(winner, self.n_sub).float()
            self.fatigue = self.fatigue + self.fatigue_rate * mask
            self.fatigue = (self.fatigue - self.recovery_rate).clamp(0, 3)
        elif self.fatigue_type == 'none':
            pass  # fatigue stays at zero
        elif self.fatigue_type == 'stochastic':
            spike = (torch.rand(N, self.n_sub, device=dev) < 0.3 * activities).float()
            self.fatigue = self.fatigue + self.fatigue_rate * spike
            self.fatigue = (self.fatigue - self.recovery_rate).clamp(0, 3)

        # === FORCES ===
        if self.manifold_type in ('s3', 's2'):
            radial = torch.einsum('sd,nd->ns', self.prefs, u)
            forces = self.prefs.unsqueeze(0) - radial.unsqueeze(2) * u.unsqueeze(1)
        else:  # flat4
            forces = self.prefs.unsqueeze(0) - u.unsqueeze(1)

        # === DRIVE ===
        activity_force = torch.einsum('ns,nsd->nd', activities, forces)

        rest_scores = torch.exp(-self.fatigue)
        novelty_force = torch.einsum('ns,nsd->nd', rest_scores, forces)
        mean_rest = rest_scores.mean(dim=1, keepdim=True)
        forces_mean = forces.mean(dim=1)
        novelty_force = novelty_force - mean_rest * forces_mean

        nw = self.novelty_weight
        drive = (1 - nw) * activity_force + nw * novelty_force

        # Noise (tangent-projected for spheres)
        raw_noise = self.exploration_noise * torch.randn(N, self.dim, device=dev)
        if self.manifold_type in ('s3', 's2'):
            noise_rad = (raw_noise * u).sum(dim=1, keepdim=True)
            drive = drive + raw_noise - noise_rad * u
        else:
            drive = drive + raw_noise

        # === UPDATE POSITION ===
        new_u = u + self.steering_strength * drive
        if self.manifold_type in ('s3', 's2'):
            self.u_t = F.normalize(new_u, dim=1)
        else:
            norm = new_u.norm(dim=1, keepdim=True)
            scale = torch.where(norm > 2.0, 2.0 / norm, torch.ones_like(norm))
            self.u_t = new_u * scale

        # === MACRO BASIN ASSIGNMENT ===
        if self.manifold_type in ('s3', 's2'):
            macro_sim = torch.einsum('nd,md->nm', self.u_t, self.macro_centers)
        else:
            dists = ((self.u_t.unsqueeze(1) - self.macro_centers.unsqueeze(0)) ** 2).sum(-1)
            macro_sim = -dists

        macro_weights = F.softmax(self.beta_macro * macro_sim, dim=1)
        dominant_basin = macro_weights.argmax(dim=1).int()

        # === CLARITY ===
        resultant = torch.einsum('ns,nsd->nd', activities, forces)
        clarity = resultant.norm(dim=1)
        dominant_sub = activities.argmax(dim=1).int()

        # === STORE HISTORY ===
        if t < self.max_steps:
            self.hist_clarity[:, t] = clarity
            self.hist_macro_basin[:, t] = dominant_basin
            self.hist_dominant_sub[:, t] = dominant_sub

        self.step_count += 1


# ============================================================================
# BOUNDARY NEGOTIATION METRICS
# ============================================================================

def boundary_metrics(engine, n_shuffles=100):
    """
    Compute 3 core boundary negotiation metrics from engine history.

    Returns:
        edge_clarity_r:  Pearson r of edge entropy fraction vs clarity over time
        cohens_d:        Cohen's d for clarity at transitions vs dwelling
        null_z:          z-score of real clarity gap vs shuffled null
        transition_rate: fraction of timesteps with basin changes
    """
    steps = engine.step_count
    basins = engine.hist_macro_basin[:, :steps].cpu().numpy()
    clarity = engine.hist_clarity[:, :steps].cpu().numpy()
    N = basins.shape[0]

    # --- Metric 1: Edge-clarity temporal correlation ---
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
        trans_mask = np.concatenate(
            [np.zeros((N, 1), dtype=bool), trans], axis=1)
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

    # --- Metric 2: transition-conditioned clarity gap ---
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

    # --- Metric 3: Null model z-score ---
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
        clarity_gap=float(gap),
        null_z=float(null_z),
        transition_rate=float(trans_rate),
    )


# ============================================================================
# VARIANT DEFINITIONS
# ============================================================================

VARIANTS = [
    # (name, manifold, topology, fatigue_type, color_group)
    # --- Baseline ---
    ('Baseline (FourD)',         's3',    'cyclic',  'gradual',    'baseline'),
    # --- Topology sweep (S³ + gradual fatigue) ---
    ('Random opponents',         's3',    'random',  'gradual',    'topology'),
    ('Full competition',         's3',    'full',    'gradual',    'topology'),
    ('No opponents',             's3',    'none',    'gradual',    'topology'),
    # --- Fatigue sweep (S³ + cyclic opponents) ---
    ('Winner-take-all',          's3',    'cyclic',  'wta',        'fatigue'),
    ('No fatigue',               's3',    'cyclic',  'none',       'fatigue'),
    ('Stochastic fatigue',       's3',    'cyclic',  'stochastic', 'fatigue'),
    # --- Manifold sweep (cyclic + gradual fatigue) ---
    ('S² manifold',              's2',    'cyclic',  'gradual',    'manifold'),
    ('Flat R⁴',                  'flat4', 'cyclic',  'gradual',    'manifold'),
    # --- Max departure ---
    ('Max departure',            's2',    'random',  'stochastic', 'control'),
]

GROUP_COLORS = {
    'baseline': '#2d2d2d',
    'topology': '#4C72B0',
    'fatigue':  '#DD5555',
    'manifold': '#55A868',
    'control':  '#8172B3',
}


# ============================================================================
# RUN ALL VARIANTS
# ============================================================================

def run_all_variants(device='cuda:0', steps=1000, N=128, outdir='outputs/universality'):
    """Run all 10 variants and collect boundary metrics."""
    os.makedirs(outdir, exist_ok=True)

    print(f'\n  Running {len(VARIANTS)} engine variants ({N} beings × {steps} steps each)')
    print(f'  {"Variant":<25s}  {"Manifold":>8s}  {"Topology":>8s}  {"Fatigue":>10s}')
    print(f'  {"-"*25}  {"-"*8}  {"-"*8}  {"-"*10}')
    for name, mf, tp, ft, _ in VARIANTS:
        print(f'  {name:<25s}  {mf:>8s}  {tp:>8s}  {ft:>10s}')
    print()

    all_results = {}

    for i, (name, manifold, topology, fatigue_type, group) in enumerate(VARIANTS):
        print(f'  [{i+1}/{len(VARIANTS)}] {name}...', end='', flush=True)
        t0 = time.time()

        engine = UniversalEngine(
            N=N, device=device, steps=steps,
            manifold=manifold, topology=topology, fatigue_type=fatigue_type,
        )

        for t in range(steps):
            engine.step()

        metrics = boundary_metrics(engine, n_shuffles=100)
        elapsed = time.time() - t0

        all_results[name] = {
            'manifold': manifold,
            'topology': topology,
            'fatigue_type': fatigue_type,
            'group': group,
            **metrics,
            'wall_time': round(elapsed, 1),
        }

        print(f'  r={metrics["edge_clarity_r"]:+.3f}  '
              f'd={metrics["cohens_d"]:+.3f}  '
              f'z={metrics["null_z"]:+.1f}  '
              f'TR={metrics["transition_rate"]:.3f}  '
              f'({elapsed:.1f}s)')

    return all_results


# ============================================================================
# COMPARISON PLOTS
# ============================================================================

def comparison_plots(results, outdir='outputs/universality'):
    """Generate 4-panel comparison figure."""
    names = list(results.keys())
    n = len(names)
    groups = [results[nm]['group'] for nm in names]
    colors = [GROUP_COLORS[g] for g in groups]

    metrics = {
        'edge_clarity_r': 'Edge ↔ Clarity (r)',
        'cohens_d': "Transition Clarity Boost (Cohen's d)",
        'null_z': 'Null Model Deviation (z-score)',
        'transition_rate': 'Basin Transition Rate',
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for ax_idx, (key, title) in enumerate(metrics.items()):
        ax = axes[ax_idx]
        vals = [results[nm][key] for nm in names]
        short_names = [nm.replace(' (FourD)', '\n(FourD)').replace(' opponents', '\nopp.')
                       .replace(' competition', '\ncomp.').replace(' fatigue', '\nfat.')
                       .replace('Winner-take-all', 'WTA')
                       .replace('Stochastic', 'Stochastic\n')
                       .replace('Max departure', 'Max\ndeparture')
                       for nm in names]

        bars = ax.bar(range(n), vals, color=colors, edgecolor='white', linewidth=0.5)

        # Baseline reference line
        baseline_val = results['Baseline (FourD)'][key]
        ax.axhline(baseline_val, color='#2d2d2d', ls='--', alpha=0.4, linewidth=1)

        ax.set_xticks(range(n))
        ax.set_xticklabels(short_names, fontsize=7, ha='center')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axhline(0, color='k', linewidth=0.3)

        # Value labels
        for bar, val in zip(bars, vals):
            y = bar.get_height()
            if abs(val) > 0.001:
                fmt = f'{val:+.2f}' if abs(val) < 100 else f'{val:+.0f}'
                ax.text(bar.get_x() + bar.get_width() / 2, y,
                        fmt, ha='center', va='bottom' if y >= 0 else 'top',
                        fontsize=7, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GROUP_COLORS['baseline'], label='Baseline'),
        Patch(facecolor=GROUP_COLORS['topology'], label='Topology variants'),
        Patch(facecolor=GROUP_COLORS['fatigue'], label='Fatigue variants'),
        Patch(facecolor=GROUP_COLORS['manifold'], label='Manifold variants'),
        Patch(facecolor=GROUP_COLORS['control'], label='Max departure'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{outdir}/universality_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved comparison plot to {outdir}/universality_comparison.png')


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_summary(results):
    """Print formatted summary table."""
    print('\n' + '=' * 100)
    print('UNIVERSALITY RESULTS — BOUNDARY NEGOTIATION HYPOTHESIS')
    print('=' * 100)

    print(f'\n  {"Variant":<25s}  {"Edge↔Clarity":>12s}  {"Cohen d":>9s}  '
          f'{"Null z":>8s}  {"Trans Rate":>10s}  {"Verdict":>18s}')
    print(f'  {"-"*25}  {"-"*12}  {"-"*9}  {"-"*8}  {"-"*10}  {"-"*18}')

    verdicts = {}
    for name, r in results.items():
        ecr = r['edge_clarity_r']
        cd = r['cohens_d']
        nz = r['null_z']
        tr = r['transition_rate']

        # Verdict logic
        if tr < 0.005:
            verdict = 'NO TRANSITIONS'
        elif abs(ecr) > 0.5 and abs(nz) > 2:
            verdict = 'BOUNDARY PRESENT'
        elif abs(ecr) > 0.3 or abs(nz) > 2:
            verdict = 'PARTIAL'
        else:
            verdict = 'ABSENT'

        verdicts[name] = verdict

        print(f'  {name:<25s}  {ecr:>+12.4f}  {cd:>+9.4f}  '
              f'{nz:>+8.1f}  {tr:>10.4f}  {verdict:>18s}')

    # Summary counts
    present = sum(1 for v in verdicts.values() if v == 'BOUNDARY PRESENT')
    partial = sum(1 for v in verdicts.values() if v == 'PARTIAL')
    absent = sum(1 for v in verdicts.values() if v in ('ABSENT', 'NO TRANSITIONS'))

    print(f'\n  Boundary negotiation detected:  {present} present, {partial} partial, {absent} absent')

    # Axis-level verdicts
    print('\n  PER-AXIS CONCLUSIONS:')

    topo_names = [n for n, r in results.items() if r['group'] == 'topology']
    topo_present = sum(1 for n in topo_names if verdicts[n] == 'BOUNDARY PRESENT')
    print(f'    Topology:  {topo_present}/{len(topo_names)} variants show boundary negotiation')
    if topo_present == len(topo_names):
        print(f'    → Topology is NOT necessary (any opponent structure works)')
    elif topo_present == 0:
        print(f'    → Topology IS critical (all alternatives lose the pattern)')
    else:
        print(f'    → Topology matters partially (some structures preserve it)')

    fat_names = [n for n, r in results.items() if r['group'] == 'fatigue']
    fat_present = sum(1 for n in fat_names if verdicts[n] in ('BOUNDARY PRESENT', 'PARTIAL'))
    print(f'    Fatigue:   {fat_present}/{len(fat_names)} variants show boundary negotiation')
    if 'No fatigue' in verdicts and verdicts['No fatigue'] in ('ABSENT', 'NO TRANSITIONS'):
        print(f'    → Fatigue IS necessary (without it, boundary negotiation collapses)')

    mf_names = [n for n, r in results.items() if r['group'] == 'manifold']
    mf_present = sum(1 for n in mf_names if verdicts[n] == 'BOUNDARY PRESENT')
    print(f'    Manifold:  {mf_present}/{len(mf_names)} variants show boundary negotiation')
    if mf_present == len(mf_names):
        print(f'    → Manifold geometry is NOT necessary (structure survives on different spaces)')
    elif mf_present == 0:
        print(f'    → Manifold geometry IS critical (S³ specifically required)')
    else:
        print(f'    → Manifold matters partially')

    return verdicts


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Universality Test')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--outdir', default='outputs/universality')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print('╔══════════════════════════════════════════════════════════════════════╗')
    print('║  UNIVERSALITY TEST — Is Boundary Negotiation Universal?            ║')
    print('║                                                                    ║')
    print('║  Testing across topology, fatigue, and manifold variations         ║')
    print('║  OPH by FloatingPragma                                             ║')
    print('║  https://github.com/FloatingPragma/observer-patch-holography       ║')
    print('╚══════════════════════════════════════════════════════════════════════╝')

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'\nGPU: {props.name} ({props.total_memory / 1e9:.1f} GB)')

    t_start = time.time()

    results = run_all_variants(args.device, args.steps, args.N, args.outdir)
    verdicts = print_summary(results)
    comparison_plots(results, args.outdir)

    # Save JSON
    out_path = f'{args.outdir}/universality_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Results saved to {out_path}')

    elapsed = time.time() - t_start
    print(f'\nTotal wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
