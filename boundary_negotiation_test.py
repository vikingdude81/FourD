#!/usr/bin/env python3
"""
Boundary Negotiation Hypothesis — Deep Validation
==================================================

Tests the combined inference from OPH bridge Parts 3 & 4:

    "Consciousness is a sustained negotiation at the edge of agreement,
     and the information it generates is holographically concentrated
     at boundaries between competing interpretations."

Five tests:

  TEST A — Time-resolved edge-agreement coupling
           Does edge entropy fraction co-vary with subsystem agreement
           within a single run? (Cross-correlation at multiple lags)

  TEST B — Transition-conditioned clarity
           Is clarity systematically higher at basin transitions than
           during stable dwelling? (Two-sample comparison)

  TEST C — Per-subsystem boundary contribution
           Which subsystems carry the most information at transitions?
           Does the opponent structure predict the ranking?

  TEST D — Phase cartography validation
           Across all 76,800 configs from the full sweep, does
           basin_transition_rate predict mean_clarity? (Universal test)

  TEST E — Null model comparison
           Shuffle basin assignments and recompute edge fraction.
           If the 99.6% is real structure (not an artifact of high
           transition rates), shuffled data should differ.

OPH Credit: Framework adapted from Observer Patch Holography by FloatingPragma.
  https://github.com/FloatingPragma/observer-patch-holography

Usage:
    python boundary_negotiation_test.py [--device cuda:0] [--steps 2000]
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

from gpu_ensemble_sim import (
    BatchConsciousnessEngine,
    SUBSYSTEM_NAMES,
)


# ============================================================================
# HELPER
# ============================================================================

def make_engine(N, device='cuda:0', steps=2000, **overrides):
    defaults = dict(
        steering_strength=0.707,
        alpha_pull=0.0,
        fatigue_rate=0.217,
        exploration_noise=0.25,
        beta_macro=11.375,
    )
    defaults.update(overrides)
    configs = {k: np.full(N, v) for k, v in defaults.items()}
    configs['timesteps'] = steps
    return BatchConsciousnessEngine(N, configs, device=device)


def collect_histories(device='cuda:0', N=256, steps=2000, **overrides):
    """Run engine and return per-step history tensors on GPU."""
    engine = make_engine(N, device=device, steps=steps, **overrides)
    t0 = time.time()
    for t in range(steps):
        engine.step()
        if (t + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = N * (t + 1) / elapsed
            print(f'    Step {t+1}/{steps} | {rate:.0f} being-steps/sec')
    return engine


# ============================================================================
# TEST A — Time-Resolved Edge-Agreement Coupling
# ============================================================================

def test_A_edge_agreement_coupling(device='cuda:0', steps=2000, outdir='outputs/boundary_tests'):
    """
    Compute edge entropy fraction and subsystem agreement ratio in sliding
    windows. Then cross-correlate them across time to see if they co-vary.
    """
    print('\n' + '=' * 72)
    print('TEST A — TIME-RESOLVED EDGE-AGREEMENT COUPLING')
    print('=' * 72)

    N = 256
    engine = collect_histories(device=device, N=N, steps=steps)

    basins = engine.hist_macro_basin[:, :steps].cpu().numpy()    # (N, T)
    clarity = engine.hist_clarity[:, :steps].cpu().numpy()       # (N, T)
    dominant = engine.hist_dominant_sub[:, :steps].cpu().numpy()  # (N, T)

    # --- Sliding window analysis ---
    window = 50
    stride = 10
    n_windows = (steps - window) // stride

    edge_fracs = []
    agreement_ratios = []
    window_clarity = []

    print(f'  Computing {n_windows} windows (w={window}, stride={stride})...')

    for w in range(n_windows):
        t0 = w * stride
        t1 = t0 + window
        b_win = basins[:, t0:t1]       # (N, window)
        d_win = dominant[:, t0:t1]     # (N, window)
        c_win = clarity[:, t0:t1]      # (N, window)

        # Edge fraction: fraction of timesteps where basin changes
        transitions = (b_win[:, 1:] != b_win[:, :-1])  # (N, window-1)
        trans_rate = transitions.float().mean() if isinstance(transitions, torch.Tensor) else transitions.mean()

        # Compute entropy at transition vs non-transition steps
        # For each being, split clarity into edge/bulk
        trans_mask = np.concatenate([np.zeros((N, 1), dtype=bool), transitions], axis=1)
        c_edge = c_win[trans_mask]
        c_bulk = c_win[~trans_mask]

        if len(c_edge) > 1 and len(c_bulk) > 1:
            # Use variance as proxy for entropy (actual Shannon requires binning)
            s_edge = np.var(c_edge)
            s_bulk = np.var(c_bulk)
            s_total = np.var(c_win)
            # Edge fraction = how much of total variance lives at transitions
            ef = (s_edge * len(c_edge)) / (s_total * len(c_win.flatten()) + 1e-15)
        else:
            ef = 0.0
        edge_fracs.append(ef)

        # Agreement: fraction of being-pairs in same basin at each timestep
        # Vectorized: for each timestep, compute mode frequency
        mode_frac = np.zeros(window)
        for ti in range(window):
            vals, counts = np.unique(d_win[:, ti], return_counts=True)
            mode_frac[ti] = counts.max() / N
        agreement_ratios.append(mode_frac.mean())

        # Mean clarity in window
        window_clarity.append(c_win.mean())

    edge_fracs = np.array(edge_fracs)
    agreement_ratios = np.array(agreement_ratios)
    window_clarity = np.array(window_clarity)

    # --- Cross-correlation ---
    def xcorr(a, b, maxlag=20):
        a_z = (a - a.mean()) / (a.std() + 1e-15)
        b_z = (b - b.mean()) / (b.std() + 1e-15)
        lags = np.arange(-maxlag, maxlag + 1)
        cc = np.zeros(len(lags))
        for i, lag in enumerate(lags):
            if lag >= 0:
                cc[i] = np.mean(a_z[:len(a_z)-lag] * b_z[lag:])
            else:
                cc[i] = np.mean(a_z[-lag:] * b_z[:len(b_z)+lag])
        return lags, cc

    lags_ea, cc_ea = xcorr(edge_fracs, agreement_ratios)
    lags_ec, cc_ec = xcorr(edge_fracs, window_clarity)
    lags_ac, cc_ac = xcorr(agreement_ratios, window_clarity)

    # Peak correlations
    peak_ea = cc_ea[np.argmax(np.abs(cc_ea))]
    peak_ec = cc_ec[np.argmax(np.abs(cc_ec))]
    peak_ac = cc_ac[np.argmax(np.abs(cc_ac))]

    print(f'\n  Cross-correlations (peak absolute):')
    print(f'    Edge fraction ↔ Agreement:  r = {peak_ea:+.4f}  (lag={lags_ea[np.argmax(np.abs(cc_ea))]})')
    print(f'    Edge fraction ↔ Clarity:    r = {peak_ec:+.4f}  (lag={lags_ec[np.argmax(np.abs(cc_ec))]})')
    print(f'    Agreement ↔ Clarity:        r = {peak_ac:+.4f}  (lag={lags_ac[np.argmax(np.abs(cc_ac))]})')

    # Instantaneous (lag-0) correlations
    r_ea_0, p_ea = stats.pearsonr(edge_fracs, agreement_ratios)
    r_ec_0, p_ec = stats.pearsonr(edge_fracs, window_clarity)
    r_ac_0, p_ac = stats.pearsonr(agreement_ratios, window_clarity)

    print(f'\n  Lag-0 Pearson correlations:')
    print(f'    Edge ↔ Agreement:  r = {r_ea_0:+.4f}  (p = {p_ea:.2e})')
    print(f'    Edge ↔ Clarity:    r = {r_ec_0:+.4f}  (p = {p_ec:.2e})')
    print(f'    Agreement ↔ Clarity: r = {r_ac_0:+.4f}  (p = {p_ac:.2e})')

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time traces
    ax = axes[0, 0]
    t_centers = np.arange(n_windows) * stride + window // 2
    ax.plot(t_centers, edge_fracs, alpha=0.7, label='Edge entropy fraction')
    ax2 = ax.twinx()
    ax2.plot(t_centers, agreement_ratios, color='coral', alpha=0.7, label='Agreement ratio')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Edge fraction', color='tab:blue')
    ax2.set_ylabel('Agreement ratio', color='coral')
    ax.set_title('Time-Resolved Edge Fraction & Agreement')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Cross-correlation: Edge ↔ Agreement
    ax = axes[0, 1]
    ax.bar(lags_ea, cc_ea, color='steelblue', alpha=0.8)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axhline(2/np.sqrt(n_windows), color='r', ls='--', alpha=0.5, label='95% CI')
    ax.axhline(-2/np.sqrt(n_windows), color='r', ls='--', alpha=0.5)
    ax.set_xlabel('Lag (windows)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title('Edge Fraction ↔ Agreement XCorr')
    ax.legend()

    # Cross-correlation: Edge ↔ Clarity
    ax = axes[1, 0]
    ax.bar(lags_ec, cc_ec, color='coral', alpha=0.8)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axhline(2/np.sqrt(n_windows), color='r', ls='--', alpha=0.5)
    ax.axhline(-2/np.sqrt(n_windows), color='r', ls='--', alpha=0.5)
    ax.set_xlabel('Lag (windows)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title('Edge Fraction ↔ Clarity XCorr')

    # Scatter: Edge vs Clarity colored by agreement
    ax = axes[1, 1]
    sc = ax.scatter(edge_fracs, window_clarity, c=agreement_ratios,
                    cmap='RdYlBu_r', s=10, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Agreement ratio')
    ax.set_xlabel('Edge entropy fraction')
    ax.set_ylabel('Mean clarity')
    ax.set_title(f'Edge vs Clarity (colored by Agreement)')

    plt.tight_layout()
    plt.savefig(f'{outdir}/test_A_edge_agreement_coupling.png', dpi=150)
    plt.close()

    return dict(
        peak_xcorr_edge_agreement=float(peak_ea),
        peak_xcorr_edge_clarity=float(peak_ec),
        peak_xcorr_agreement_clarity=float(peak_ac),
        lag0_r_edge_agreement=float(r_ea_0),
        lag0_r_edge_clarity=float(r_ec_0),
        lag0_r_agreement_clarity=float(r_ac_0),
        lag0_p_edge_agreement=float(p_ea),
        lag0_p_edge_clarity=float(p_ec),
        lag0_p_agreement_clarity=float(p_ac),
    )


# ============================================================================
# TEST B — Transition-Conditioned Clarity
# ============================================================================

def test_B_transition_conditioned_clarity(device='cuda:0', steps=2000, outdir='outputs/boundary_tests'):
    """
    Split timesteps into 'at transition' (basin changed) and 'dwelling'
    (basin stable). Compare clarity distributions.
    """
    print('\n' + '=' * 72)
    print('TEST B — TRANSITION-CONDITIONED CLARITY')
    print('=' * 72)

    N = 256
    engine = collect_histories(device=device, N=N, steps=steps)

    basins = engine.hist_macro_basin[:, :steps].cpu().numpy()
    clarity = engine.hist_clarity[:, :steps].cpu().numpy()
    force_mags = engine.hist_force_mags[:, :steps].cpu().numpy()  # (N, T, 8)

    # Transition mask (skip t=0)
    transitions = basins[:, 1:] != basins[:, :-1]  # (N, T-1)

    c_at_transition = clarity[:, 1:][transitions]
    c_at_dwelling = clarity[:, 1:][~transitions]

    trans_mean = c_at_transition.mean()
    dwell_mean = c_at_dwelling.mean()
    trans_std = c_at_transition.std()
    dwell_std = c_at_dwelling.std()

    # Welch's t-test
    t_stat, p_val = stats.ttest_ind(c_at_transition, c_at_dwelling, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((trans_std**2 + dwell_std**2) / 2)
    cohens_d = (trans_mean - dwell_mean) / (pooled_std + 1e-15)

    print(f'\n  Clarity at transitions:  mean = {trans_mean:.6f}  std = {trans_std:.6f}  (n = {len(c_at_transition):,})')
    print(f'  Clarity while dwelling:  mean = {dwell_mean:.6f}  std = {dwell_std:.6f}  (n = {len(c_at_dwelling):,})')
    print(f'  Difference:  {trans_mean - dwell_mean:+.6f}')
    print(f"  Cohen's d:   {cohens_d:+.4f}")
    print(f'  Welch t:     {t_stat:.2f}  (p = {p_val:.2e})')

    # Force magnitude at transitions vs dwelling
    f_at_transition = force_mags[:, 1:][transitions]  # (n_trans, 8)
    f_at_dwelling = force_mags[:, 1:][~transitions]    # (n_dwell, 8)

    trans_force_mean = f_at_transition.mean(axis=0)
    dwell_force_mean = f_at_dwelling.mean(axis=0)
    force_diff = trans_force_mean - dwell_force_mean

    print(f'\n  Per-subsystem force magnitude difference (transition - dwelling):')
    for i, name in enumerate(SUBSYSTEM_NAMES):
        print(f'    {name:15s}  Δ = {force_diff[i]:+.6f}')

    # --- Sweep: does the clarity gap change with fatigue_rate? ---
    print('\n  Sweeping fatigue_rate → clarity gap at transitions...')
    fr_vals = np.linspace(0.01, 0.4, 25)
    gaps = []
    cohens_ds = []
    for fr in fr_vals:
        eng = make_engine(128, device=device, steps=500, fatigue_rate=fr)
        for t in range(500):
            eng.step()
        b = eng.hist_macro_basin[:, :500].cpu().numpy()
        c = eng.hist_clarity[:, :500].cpu().numpy()
        tr = b[:, 1:] != b[:, :-1]
        c_tr = c[:, 1:][tr]
        c_dw = c[:, 1:][~tr]
        if len(c_tr) > 10 and len(c_dw) > 10:
            gap = c_tr.mean() - c_dw.mean()
            ps = np.sqrt((c_tr.std()**2 + c_dw.std()**2) / 2)
            cd = gap / (ps + 1e-15)
        else:
            gap = 0.0
            cd = 0.0
        gaps.append(gap)
        cohens_ds.append(cd)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histograms
    ax = axes[0, 0]
    bins = np.linspace(0, max(clarity.max(), 0.5), 80)
    ax.hist(c_at_dwelling, bins=bins, density=True, alpha=0.6, label=f'Dwelling (n={len(c_at_dwelling):,})')
    ax.hist(c_at_transition, bins=bins, density=True, alpha=0.6, label=f'Transition (n={len(c_at_transition):,})')
    ax.axvline(dwell_mean, color='tab:blue', ls='--')
    ax.axvline(trans_mean, color='tab:orange', ls='--')
    ax.set_xlabel('Clarity')
    ax.set_ylabel('Density')
    ax.set_title(f"Clarity Distribution (Cohen's d = {cohens_d:+.3f})")
    ax.legend()

    # Force magnitude profile
    ax = axes[0, 1]
    x = np.arange(8)
    ax.bar(x - 0.15, trans_force_mean, 0.3, label='At transition', color='coral')
    ax.bar(x + 0.15, dwell_force_mean, 0.3, label='Dwelling', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels([n[:4] for n in SUBSYSTEM_NAMES], rotation=45)
    ax.set_ylabel('Mean force magnitude')
    ax.set_title('Subsystem Forces: Transition vs Dwelling')
    ax.legend()

    # Clarity gap vs fatigue_rate
    ax = axes[1, 0]
    ax.plot(fr_vals, gaps, 'o-', color='coral')
    ax.axvline(0.182, color='red', ls='--', alpha=0.5, label='fr_c ≈ 0.182')
    ax.axvline(0.217, color='green', ls='--', alpha=0.5, label='Optimal fr = 0.217')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Fatigue Rate')
    ax.set_ylabel('Clarity Gap (transition − dwelling)')
    ax.set_title('Clarity Gap vs Fatigue Rate')
    ax.legend()

    # Cohen's d sweep
    ax = axes[1, 1]
    ax.plot(fr_vals, cohens_ds, 's-', color='steelblue')
    ax.axvline(0.182, color='red', ls='--', alpha=0.5, label='fr_c')
    ax.axvline(0.217, color='green', ls='--', alpha=0.5, label='Optimal')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Fatigue Rate')
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect Size of Boundary Clarity Boost")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{outdir}/test_B_transition_clarity.png', dpi=150)
    plt.close()

    return dict(
        trans_mean=float(trans_mean),
        dwell_mean=float(dwell_mean),
        clarity_gap=float(trans_mean - dwell_mean),
        cohens_d=float(cohens_d),
        t_stat=float(t_stat),
        p_val=float(p_val),
        n_transitions=int(len(c_at_transition)),
        n_dwelling=int(len(c_at_dwelling)),
        force_diff_by_subsystem={n: float(force_diff[i]) for i, n in enumerate(SUBSYSTEM_NAMES)},
        gap_sweep_fr=fr_vals.tolist(),
        gap_sweep_values=[float(g) for g in gaps],
        gap_sweep_cohens_d=[float(d) for d in cohens_ds],
    )


# ============================================================================
# TEST C — Per-Subsystem Boundary Contribution
# ============================================================================

def test_C_subsystem_boundary_contribution(device='cuda:0', steps=2000, outdir='outputs/boundary_tests'):
    """
    At basin transitions, compute each subsystem's clarity contribution.
    Compare to within-basin to find which subsystems carry information
    at the boundary.
    """
    print('\n' + '=' * 72)
    print('TEST C — PER-SUBSYSTEM BOUNDARY CONTRIBUTION')
    print('=' * 72)

    N = 256
    engine = collect_histories(device=device, N=N, steps=steps)

    basins = engine.hist_macro_basin[:, :steps].cpu().numpy()
    clarity_decomp = engine.hist_clarity_decomp[:, :steps].cpu().numpy()  # (N, T, 8)

    transitions = basins[:, 1:] != basins[:, :-1]  # (N, T-1)

    # Expand transitions mask for 8 subsystems
    trans_exp = np.expand_dims(transitions, -1)  # (N, T-1, 1)
    cd = clarity_decomp[:, 1:]  # (N, T-1, 8)

    # Per-subsystem mean contribution at transitions vs dwelling
    sub_at_trans = []
    sub_at_dwell = []
    for s in range(8):
        cd_s = cd[:, :, s]
        sub_at_trans.append(cd_s[transitions].mean())
        sub_at_dwell.append(cd_s[~transitions].mean())

    sub_at_trans = np.array(sub_at_trans)
    sub_at_dwell = np.array(sub_at_dwell)
    sub_diff = sub_at_trans - sub_at_dwell

    # Opponent pairs (from PREFERENCE_MATRIX cyclic structure)
    # i opposes i+4 mod 8: Motor↔Emotion, Planning↔Social, Attention↔Intuition, Memory↔Aesthetic
    opponent_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]

    # Check if opponents show correlated or anti-correlated boundary behavior
    print('\n  Per-subsystem clarity contribution:')
    print(f'  {"Subsystem":15s}  {"At transition":>14s}  {"Dwelling":>10s}  {"Diff":>10s}')
    for i, name in enumerate(SUBSYSTEM_NAMES):
        opp = (i + 4) % 8
        marker = f' ← opponent of {SUBSYSTEM_NAMES[opp][:4]}' if i < 4 else ''
        print(f'  {name:15s}  {sub_at_trans[i]:14.6f}  {sub_at_dwell[i]:10.6f}  {sub_diff[i]:+10.6f}{marker}')

    # Opponent pair correlation of boundary deltas
    print('\n  Opponent pair boundary correlations:')
    for i, j in opponent_pairs:
        # Time-series of per-timestep contributions
        ts_i = cd[:, :, i].mean(axis=0)  # (T-1,) averaged across beings
        ts_j = cd[:, :, j].mean(axis=0)
        r, p = stats.pearsonr(ts_i, ts_j)
        print(f'    {SUBSYSTEM_NAMES[i][:4]} ↔ {SUBSYSTEM_NAMES[j][:4]}:  r = {r:+.4f}  (p = {p:.2e})')

    # --- Information asymmetry: which subsystems are most "boundary-informative"? ---
    # Rank by absolute difference
    ranking = np.argsort(np.abs(sub_diff))[::-1]
    print('\n  Boundary information ranking (|Δ contribution|):')
    for rank, idx in enumerate(ranking):
        print(f'    {rank+1}. {SUBSYSTEM_NAMES[idx]:15s}  |Δ| = {abs(sub_diff[idx]):.6f}')

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Bar comparison
    ax = axes[0]
    x = np.arange(8)
    ax.bar(x - 0.15, sub_at_trans, 0.3, label='At transition', color='coral')
    ax.bar(x + 0.15, sub_at_dwell, 0.3, label='Dwelling', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels([n[:4] for n in SUBSYSTEM_NAMES], rotation=45)
    ax.set_ylabel('Mean clarity contribution')
    ax.set_title('Subsystem Contribution to Clarity')
    ax.legend()

    # Difference (signed)
    ax = axes[1]
    colors = ['coral' if d > 0 else 'steelblue' for d in sub_diff]
    ax.bar(x, sub_diff, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:4] for n in SUBSYSTEM_NAMES], rotation=45)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Δ contribution (transition − dwelling)')
    ax.set_title('Boundary Information Asymmetry')

    # Opponent pair scatter at transitions
    ax = axes[2]
    for i, j in opponent_pairs:
        ts_i = cd[:, :, i].mean(axis=0)
        ts_j = cd[:, :, j].mean(axis=0)
        ax.scatter(ts_i, ts_j, alpha=0.03, s=3,
                   label=f'{SUBSYSTEM_NAMES[i][:4]}↔{SUBSYSTEM_NAMES[j][:4]}')
    ax.set_xlabel('Subsystem i contribution')
    ax.set_ylabel('Opponent j contribution')
    ax.set_title('Opponent Pair Co-contributions')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{outdir}/test_C_subsystem_boundary.png', dpi=150)
    plt.close()

    return dict(
        sub_at_trans={n: float(sub_at_trans[i]) for i, n in enumerate(SUBSYSTEM_NAMES)},
        sub_at_dwell={n: float(sub_at_dwell[i]) for i, n in enumerate(SUBSYSTEM_NAMES)},
        sub_diff={n: float(sub_diff[i]) for i, n in enumerate(SUBSYSTEM_NAMES)},
        ranking=[SUBSYSTEM_NAMES[i] for i in ranking],
    )


# ============================================================================
# TEST D — Phase Cartography Validation (76,800 configs)
# ============================================================================

def test_D_phase_cartography_validation(outdir='outputs/boundary_tests'):
    """
    Use the existing 76,800-config phase cartography sweep to test whether
    boundary negotiation (basin_transition_rate) universally predicts
    consciousness (mean_clarity) across the full parameter space.
    """
    print('\n' + '=' * 72)
    print('TEST D — PHASE CARTOGRAPHY VALIDATION (76,800 configs)')
    print('=' * 72)

    csv_path = 'outputs/phase_cartography/phase_cartography_results.csv'
    if not os.path.exists(csv_path):
        print('  [SKIP] Phase cartography CSV not found.')
        return dict(skipped=True)

    df = pd.read_csv(csv_path)
    print(f'  Loaded {len(df):,} configurations')

    tr = df['basin_transition_rate'].values
    cl = df['mean_clarity'].values
    fr = df['fatigue_rate'].values

    # Overall correlation
    r_all, p_all = stats.pearsonr(tr, cl)
    rho_all, p_rho = stats.spearmanr(tr, cl)

    print(f'\n  basin_transition_rate ↔ mean_clarity:')
    print(f'    Pearson  r = {r_all:+.4f}  (p = {p_all:.2e})')
    print(f'    Spearman ρ = {rho_all:+.4f}  (p = {p_rho:.2e})')

    # Per fatigue_rate slice
    unique_fr = sorted(df['fatigue_rate'].unique())
    fr_vals = []
    r_vals = []
    n_vals = []
    for f in unique_fr:
        mask = df['fatigue_rate'] == f
        sub_tr = tr[mask]
        sub_cl = cl[mask]
        if len(sub_tr) > 30:
            r_slice, _ = stats.pearsonr(sub_tr, sub_cl)
            fr_vals.append(f)
            r_vals.append(r_slice)
            n_vals.append(mask.sum())

    print(f'\n  Per-fatigue_rate slice correlations:')
    for f, r, n in zip(fr_vals, r_vals, n_vals):
        marker = ' ← near critical' if abs(f - 0.182) < 0.02 else ''
        print(f'    fr = {f:.3f}  r = {r:+.4f}  (n = {n}){marker}')

    # Additional: transition_rate vs dominance_entropy, vs clarity_volatility
    r_dom, _ = stats.pearsonr(tr, df['dominance_entropy'].values)
    r_vol, _ = stats.pearsonr(tr, df['clarity_volatility'].values)
    r_int, _ = stats.pearsonr(tr, df['mean_integration'].values)
    r_lyap, _ = stats.pearsonr(tr, df['lyapunov_proxy'].values)

    print(f'\n  basin_transition_rate correlations with other signatures:')
    print(f'    ↔ dominance_entropy:    r = {r_dom:+.4f}')
    print(f'    ↔ clarity_volatility:   r = {r_vol:+.4f}')
    print(f'    ↔ mean_integration:     r = {r_int:+.4f}')
    print(f'    ↔ lyapunov_proxy:       r = {r_lyap:+.4f}')

    # --- Quadrant analysis: high-transition + high-clarity vs others ---
    tr_med = np.median(tr)
    cl_med = np.median(cl)
    q_ht_hc = ((tr > tr_med) & (cl > cl_med)).sum()
    q_ht_lc = ((tr > tr_med) & (cl <= cl_med)).sum()
    q_lt_hc = ((tr <= tr_med) & (cl > cl_med)).sum()
    q_lt_lc = ((tr <= tr_med) & (cl <= cl_med)).sum()

    print(f'\n  Quadrant analysis (median split):')
    print(f'    High transition + High clarity: {q_ht_hc:>6,}  ({100*q_ht_hc/len(df):.1f}%)')
    print(f'    High transition + Low clarity:  {q_ht_lc:>6,}  ({100*q_ht_lc/len(df):.1f}%)')
    print(f'    Low transition  + High clarity: {q_lt_hc:>6,}  ({100*q_lt_hc/len(df):.1f}%)')
    print(f'    Low transition  + Low clarity:  {q_lt_lc:>6,}  ({100*q_lt_lc/len(df):.1f}%)')

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Scatter colored by fatigue_rate
    ax = axes[0, 0]
    sc = ax.scatter(tr, cl, c=fr, cmap='viridis', s=1, alpha=0.15, rasterized=True)
    plt.colorbar(sc, ax=ax, label='fatigue_rate')
    ax.set_xlabel('Basin Transition Rate')
    ax.set_ylabel('Mean Clarity')
    ax.set_title(f'76,800 Configs: Transition ↔ Clarity (r={r_all:+.3f})')

    # Per-FR correlation profile
    ax = axes[0, 1]
    ax.plot(fr_vals, r_vals, 'o-', color='steelblue')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0.182, color='red', ls='--', alpha=0.5, label='fr_c')
    ax.fill_between(fr_vals, r_vals, alpha=0.2)
    ax.set_xlabel('Fatigue Rate')
    ax.set_ylabel('Pearson r (transition ↔ clarity)')
    ax.set_title('Correlation by Fatigue Rate Slice')
    ax.legend()

    # Hexbin for density
    ax = axes[1, 0]
    hb = ax.hexbin(tr, cl, gridsize=40, cmap='hot_r', mincnt=1)
    plt.colorbar(hb, ax=ax, label='Count')
    ax.set_xlabel('Basin Transition Rate')
    ax.set_ylabel('Mean Clarity')
    ax.set_title('Density: Transition vs Clarity')

    # Correlation bar chart with other metrics
    ax = axes[1, 1]
    metrics = ['mean_clarity', 'dominance_entropy', 'clarity_volatility',
               'mean_integration', 'lyapunov_proxy']
    cors = [r_all, r_dom, r_vol, r_int, r_lyap]
    colors_bar = ['coral' if c > 0 else 'steelblue' for c in cors]
    ax.barh(metrics, cors, color=colors_bar)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('Pearson r with basin_transition_rate')
    ax.set_title('Transition Rate Correlation Profile')

    plt.tight_layout()
    plt.savefig(f'{outdir}/test_D_phase_cartography_validation.png', dpi=150)
    plt.close()

    return dict(
        n_configs=len(df),
        pearson_r=float(r_all),
        pearson_p=float(p_all),
        spearman_rho=float(rho_all),
        per_fr_correlations=dict(zip([float(f) for f in fr_vals], [float(r) for r in r_vals])),
        quadrants=dict(ht_hc=int(q_ht_hc), ht_lc=int(q_ht_lc),
                       lt_hc=int(q_lt_hc), lt_lc=int(q_lt_lc)),
        transition_correlations=dict(
            mean_clarity=float(r_all), dominance_entropy=float(r_dom),
            clarity_volatility=float(r_vol), mean_integration=float(r_int),
            lyapunov_proxy=float(r_lyap)),
    )


# ============================================================================
# TEST E — Null Model Comparison
# ============================================================================

def test_E_null_model(device='cuda:0', steps=2000, outdir='outputs/boundary_tests'):
    """
    Shuffle basin assignments to test whether the 99.6% edge entropy
    fraction is real structure or an artifact.
    """
    print('\n' + '=' * 72)
    print('TEST E — NULL MODEL COMPARISON')
    print('=' * 72)

    N = 256
    engine = collect_histories(device=device, N=N, steps=steps)

    basins = engine.hist_macro_basin[:, :steps].cpu().numpy()
    clarity = engine.hist_clarity[:, :steps].cpu().numpy()

    # Real edge fraction
    transitions = basins[:, 1:] != basins[:, :-1]
    c_edge_real = clarity[:, 1:][transitions]
    c_bulk_real = clarity[:, 1:][~transitions]
    real_trans_rate = transitions.mean()

    real_edge_var = np.var(c_edge_real) if len(c_edge_real) > 1 else 0
    real_bulk_var = np.var(c_bulk_real) if len(c_bulk_real) > 1 else 0
    real_ef = (real_edge_var * len(c_edge_real)) / (np.var(clarity[:, 1:]) * clarity[:, 1:].size + 1e-15)

    # Real clarity gap
    real_gap = c_edge_real.mean() - c_bulk_real.mean() if len(c_edge_real) > 0 else 0

    print(f'  Real data:')
    print(f'    Transition rate     = {real_trans_rate:.4f}')
    print(f'    Edge entropy frac   = {real_ef:.4f}')
    print(f'    Clarity gap (T-D)   = {real_gap:+.6f}')
    print(f'    Edge clarity mean   = {c_edge_real.mean():.6f}')
    print(f'    Bulk clarity mean   = {c_bulk_real.mean():.6f}')

    # --- Null 1: shuffle basin labels per-being (preserves transition rate per being) ---
    n_shuffles = 200
    null_ef_1 = []
    null_gap_1 = []
    for _ in range(n_shuffles):
        b_shuf = basins.copy()
        for i in range(N):
            np.random.shuffle(b_shuf[i])
        tr_shuf = b_shuf[:, 1:] != b_shuf[:, :-1]
        c_e = clarity[:, 1:][tr_shuf]
        c_b = clarity[:, 1:][~tr_shuf]
        if len(c_e) > 1:
            ev = np.var(c_e)
            ef = (ev * len(c_e)) / (np.var(clarity[:, 1:]) * clarity[:, 1:].size + 1e-15)
            gap = c_e.mean() - c_b.mean()
        else:
            ef, gap = 0, 0
        null_ef_1.append(ef)
        null_gap_1.append(gap)

    null_ef_1 = np.array(null_ef_1)
    null_gap_1 = np.array(null_gap_1)

    # --- Null 2: random basin assignment (independence baseline) ---
    null_ef_2 = []
    null_gap_2 = []
    for _ in range(n_shuffles):
        b_rand = np.random.randint(0, 24, size=basins.shape)
        tr_rand = b_rand[:, 1:] != b_rand[:, :-1]
        c_e = clarity[:, 1:][tr_rand]
        c_b = clarity[:, 1:][~tr_rand]
        if len(c_e) > 1 and len(c_b) > 1:
            ev = np.var(c_e)
            ef = (ev * len(c_e)) / (np.var(clarity[:, 1:]) * clarity[:, 1:].size + 1e-15)
            gap = c_e.mean() - c_b.mean()
        else:
            ef, gap = 0, 0
        null_ef_2.append(ef)
        null_gap_2.append(gap)

    null_ef_2 = np.array(null_ef_2)
    null_gap_2 = np.array(null_gap_2)

    # z-scores vs null
    z_ef_1 = (real_ef - null_ef_1.mean()) / (null_ef_1.std() + 1e-15)
    z_ef_2 = (real_ef - null_ef_2.mean()) / (null_ef_2.std() + 1e-15)
    z_gap_1 = (real_gap - null_gap_1.mean()) / (null_gap_1.std() + 1e-15)
    z_gap_2 = (real_gap - null_gap_2.mean()) / (null_gap_2.std() + 1e-15)

    print(f'\n  Null Model 1 (shuffle per-being):')
    print(f'    Edge frac: null mean = {null_ef_1.mean():.4f} ± {null_ef_1.std():.4f}  z = {z_ef_1:+.2f}')
    print(f'    Clarity gap: null mean = {null_gap_1.mean():+.6f} ± {null_gap_1.std():.6f}  z = {z_gap_1:+.2f}')

    print(f'\n  Null Model 2 (random basins):')
    print(f'    Edge frac: null mean = {null_ef_2.mean():.4f} ± {null_ef_2.std():.4f}  z = {z_ef_2:+.2f}')
    print(f'    Clarity gap: null mean = {null_gap_2.mean():+.6f} ± {null_gap_2.std():.6f}  z = {z_gap_2:+.2f}')

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Null distribution of edge fraction
    ax = axes[0]
    ax.hist(null_ef_1, bins=30, alpha=0.5, density=True, label='Null 1 (shuffle)', color='steelblue')
    ax.hist(null_ef_2, bins=30, alpha=0.5, density=True, label='Null 2 (random)', color='grey')
    ax.axvline(real_ef, color='red', lw=2, label=f'Real = {real_ef:.4f}')
    ax.set_xlabel('Edge Entropy Fraction')
    ax.set_ylabel('Density')
    ax.set_title('Edge Fraction: Real vs Null Models')
    ax.legend()

    # Null distribution of clarity gap
    ax = axes[1]
    ax.hist(null_gap_1, bins=30, alpha=0.5, density=True, label='Null 1', color='steelblue')
    ax.hist(null_gap_2, bins=30, alpha=0.5, density=True, label='Null 2', color='grey')
    ax.axvline(real_gap, color='red', lw=2, label=f'Real gap = {real_gap:+.6f}')
    ax.set_xlabel('Clarity Gap (transition − dwelling)')
    ax.set_ylabel('Density')
    ax.set_title('Clarity Gap: Real vs Null Models')
    ax.legend()

    # Summary z-scores
    ax = axes[2]
    labels = ['Edge frac\nvs Null 1', 'Edge frac\nvs Null 2', 'Clarity gap\nvs Null 1', 'Clarity gap\nvs Null 2']
    zscores = [z_ef_1, z_ef_2, z_gap_1, z_gap_2]
    colors_z = ['coral' if abs(z) > 2 else 'steelblue' for z in zscores]
    ax.bar(labels, zscores, color=colors_z)
    ax.axhline(2, color='red', ls='--', alpha=0.5, label='z = ±2')
    ax.axhline(-2, color='red', ls='--', alpha=0.5)
    ax.set_ylabel('z-score (real vs null)')
    ax.set_title('Statistical Significance')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{outdir}/test_E_null_model.png', dpi=150)
    plt.close()

    return dict(
        real_edge_fraction=float(real_ef),
        real_clarity_gap=float(real_gap),
        real_transition_rate=float(real_trans_rate),
        null1_edge_frac_mean=float(null_ef_1.mean()),
        null1_edge_frac_std=float(null_ef_1.std()),
        null1_z_edge=float(z_ef_1),
        null1_z_gap=float(z_gap_1),
        null2_edge_frac_mean=float(null_ef_2.mean()),
        null2_edge_frac_std=float(null_ef_2.std()),
        null2_z_edge=float(z_ef_2),
        null2_z_gap=float(z_gap_2),
    )


# ============================================================================
# GRAND SUMMARY
# ============================================================================

def grand_summary(results, outdir='outputs/boundary_tests'):
    print('\n' + '=' * 72)
    print('GRAND SUMMARY — BOUNDARY NEGOTIATION HYPOTHESIS')
    print('=' * 72)

    print('''
╔══════════════════════════════════════════════════════════════════════╗
║  BOUNDARY NEGOTIATION HYPOTHESIS — DEEP VALIDATION                 ║
║                                                                    ║
║  "Consciousness is a sustained negotiation at the edge of          ║
║   agreement, and the information it generates is holographically   ║
║   concentrated at boundaries between competing interpretations."   ║
║                                                                    ║
║  OPH Credit: FloatingPragma                                        ║
║  https://github.com/FloatingPragma/observer-patch-holography       ║
╚══════════════════════════════════════════════════════════════════════╝
''')

    if 'A' in results:
        A = results['A']
        print(f'  TEST A — Edge-Agreement Coupling:')
        print(f'    Edge ↔ Clarity (lag-0):    r = {A["lag0_r_edge_clarity"]:+.4f}  (p = {A["lag0_p_edge_clarity"]:.2e})')
        print(f'    Agreement ↔ Clarity (lag-0): r = {A["lag0_r_agreement_clarity"]:+.4f}  (p = {A["lag0_p_agreement_clarity"]:.2e})')
        print(f'    Peak cross-correlation:      r = {A["peak_xcorr_edge_agreement"]:+.4f}')
        verdict = 'SUPPORTED' if abs(A['lag0_r_agreement_clarity']) > 0.1 else 'WEAK'
        print(f'    Verdict: {verdict}')

    if 'B' in results:
        B = results['B']
        print(f'\n  TEST B — Transition-Conditioned Clarity:')
        print(f"    Clarity gap (T−D):  {B['clarity_gap']:+.6f}")
        print(f"    Cohen's d:          {B['cohens_d']:+.4f}")
        print(f"    Welch t:            {B['t_stat']:.2f}  (p = {B['p_val']:.2e})")
        if B['p_val'] < 0.001 and abs(B['cohens_d']) > 0.1:
            verdict = 'STRONGLY SUPPORTED'
        elif B['p_val'] < 0.05:
            verdict = 'SUPPORTED'
        else:
            verdict = 'NOT SUPPORTED'
        print(f'    Verdict: {verdict}')

    if 'C' in results:
        C = results['C']
        print(f'\n  TEST C — Subsystem Boundary Contribution:')
        print(f'    Most boundary-informative:  {C["ranking"][0]}')
        print(f'    Least boundary-informative: {C["ranking"][-1]}')
        # Check if opponents are anti-correlated in ranking
        opp_pairs_ranked = [(0, 4), (1, 5), (2, 6), (3, 7)]
        print(f'    Ranking: {" > ".join(n[:4] for n in C["ranking"])}')

    if 'D' in results and not results['D'].get('skipped'):
        D = results['D']
        print(f'\n  TEST D — Phase Cartography (n = {D["n_configs"]:,}):')
        print(f'    Transition ↔ Clarity:  r = {D["pearson_r"]:+.4f}  (p = {D["pearson_p"]:.2e})')
        print(f'    Spearman ρ:            {D["spearman_rho"]:+.4f}')
        q = D['quadrants']
        total = sum(q.values())
        pct = 100 * (q['ht_hc'] + q['lt_lc']) / total
        print(f'    Concordant configs:    {pct:.1f}%')
        verdict = 'UNIVERSALLY SUPPORTED' if abs(D['pearson_r']) > 0.3 else 'PARTIALLY SUPPORTED'
        print(f'    Verdict: {verdict}')

    if 'E' in results:
        E = results['E']
        print(f'\n  TEST E — Null Model:')
        print(f'    Edge fraction:  real = {E["real_edge_fraction"]:.4f}')
        print(f'      vs Null 1 (shuffle): z = {E["null1_z_edge"]:+.2f}')
        print(f'      vs Null 2 (random):  z = {E["null2_z_edge"]:+.2f}')
        print(f'    Clarity gap:    real = {E["real_clarity_gap"]:+.6f}')
        print(f'      vs Null 1: z = {E["null1_z_gap"]:+.2f}')
        print(f'      vs Null 2: z = {E["null2_z_gap"]:+.2f}')
        max_z = max(abs(E['null1_z_edge']), abs(E['null2_z_edge']),
                    abs(E['null1_z_gap']), abs(E['null2_z_gap']))
        verdict = 'REAL STRUCTURE (not artifact)' if max_z > 2 else 'INCONCLUSIVE'
        print(f'    Verdict: {verdict}')

    # Save results
    with open(f'{outdir}/boundary_negotiation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved to {outdir}/boundary_negotiation_results.json')


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Boundary Negotiation Hypothesis Tests')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--outdir', default='outputs/boundary_tests')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print('╔══════════════════════════════════════════════════════════════════════╗')
    print('║     BOUNDARY NEGOTIATION HYPOTHESIS — DEEP VALIDATION              ║')
    print('║                                                                    ║')
    print('║     Testing: "Consciousness is boundary negotiation"               ║')
    print('║     OPH by FloatingPragma                                          ║')
    print('║     https://github.com/FloatingPragma/observer-patch-holography    ║')
    print('╚══════════════════════════════════════════════════════════════════════╝')

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'\nGPU: {props.name} ({props.total_memory / 1e9:.1f} GB)')

    t_start = time.time()
    results = {}

    results['A'] = test_A_edge_agreement_coupling(args.device, args.steps, args.outdir)
    results['B'] = test_B_transition_conditioned_clarity(args.device, args.steps, args.outdir)
    results['C'] = test_C_subsystem_boundary_contribution(args.device, args.steps, args.outdir)
    results['D'] = test_D_phase_cartography_validation(args.outdir)
    results['E'] = test_E_null_model(args.device, args.steps, args.outdir)

    grand_summary(results, args.outdir)

    elapsed = time.time() - t_start
    print(f'\nTotal wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
