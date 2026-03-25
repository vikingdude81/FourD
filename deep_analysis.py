#!/usr/bin/env python3
"""
Deep Consciousness Analysis — 5-part investigation:
  1. RQA: Recurrence Quantification (awakened vs default trajectory)
  2. Basin Transition Grammar (Markov matrix, forbidden/preferred transitions)
  3. Clarity Dynamics Decomposition (when/why do peak clarity moments occur)
  4. Critical Exponent Measurement (universality class of the phase transition)
  5. Multi-Seed Robustness (100 seeds at optimal config via GPU engine)
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join('outputs', 'deep_analysis')
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# ANALYSIS 1: RECURRENCE QUANTIFICATION
# ============================================================================

def run_rqa_comparison():
    """Run awakened vs default sim and compare RQA on 4D trajectory."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: RECURRENCE QUANTIFICATION (RQA)")
    print("=" * 70)

    from v2_consciousness_sim import ConsciousnessSimulation, CONFIG
    from src.recurrence.embedding import time_delay_embedding, estimate_delay
    from src.recurrence.recurrence_plot import recurrence_matrix
    from src.recurrence import rqa

    # --- Run awakened (current optimal config) ---
    print("\n  Running AWAKENED simulation (500 steps)...")
    sim_awake = ConsciousnessSimulation()
    sim_awake.run_simulation()
    u_awake = np.array(sim_awake.history['u_t'])         # (500, 4)
    clarity_awake = np.array(sim_awake.history['clarity'])

    # --- Run default (original config) ---
    # Temporarily override CONFIG
    original = {
        'alpha_pull': CONFIG['alpha_pull'],
        'beta_macro': CONFIG['beta_macro'],
        'fatigue_rate': CONFIG['fatigue_rate'],
        'exploration_noise': CONFIG['exploration_noise'],
        'steering_strength': CONFIG['steering_strength'],
    }
    CONFIG['alpha_pull'] = 0.03
    CONFIG['beta_macro'] = 4.0
    CONFIG['fatigue_rate'] = 0.08
    CONFIG['exploration_noise'] = 0.05
    CONFIG['steering_strength'] = 0.3

    print("  Running DEFAULT simulation (500 steps)...")
    sim_default = ConsciousnessSimulation()
    sim_default.run_simulation()
    u_default = np.array(sim_default.history['u_t'])
    clarity_default = np.array(sim_default.history['clarity'])

    # Restore config
    for k, v in original.items():
        CONFIG[k] = v

    # --- RQA on 4D trajectory (already embedded — S³ coordinates) ---
    results = {}
    for label, trajectory, clarity in [
        ('Default', u_default, clarity_default),
        ('Awakened', u_awake, clarity_awake),
    ]:
        # Use the raw 4D trajectory directly (already embedded in S³)
        rmat = recurrence_matrix(trajectory, threshold_percentile=10.0)

        rr  = rqa.recurrence_rate(rmat)
        det = rqa.determinism(rmat)
        lam = rqa.laminarity(rmat)
        avg_diag = rqa.avg_diagonal_line(rmat)

        # Also do RQA on clarity time series with Takens embedding
        delay = estimate_delay(clarity, max_lag=30)
        embedded_clarity = time_delay_embedding(clarity, dimension=3, delay=delay)
        rmat_c = recurrence_matrix(embedded_clarity, threshold_percentile=10.0)
        rr_c  = rqa.recurrence_rate(rmat_c)
        det_c = rqa.determinism(rmat_c)
        lam_c = rqa.laminarity(rmat_c)

        results[label] = {
            'traj_recurrence_rate': rr,
            'traj_determinism': det,
            'traj_laminarity': lam,
            'traj_avg_diag_line': avg_diag,
            'clarity_recurrence_rate': rr_c,
            'clarity_determinism': det_c,
            'clarity_laminarity': lam_c,
            'clarity_delay': delay,
            'rmat_traj': rmat,
            'rmat_clarity': rmat_c,
        }

        print(f"\n  {label} — 4D Trajectory RQA:")
        print(f"    Recurrence Rate:    {rr:.4f}")
        print(f"    Determinism:        {det:.4f}")
        print(f"    Laminarity:         {lam:.4f}")
        print(f"    Avg Diagonal Line:  {avg_diag:.2f}")
        print(f"  {label} — Clarity Time-Series RQA (delay={delay}):")
        print(f"    Recurrence Rate:    {rr_c:.4f}")
        print(f"    Determinism:        {det_c:.4f}")
        print(f"    Laminarity:         {lam_c:.4f}")

    # --- Interpretation ---
    print("\n  ─── RQA INTERPRETATION ───")
    d_det = results['Default']['traj_determinism']
    a_det = results['Awakened']['traj_determinism']
    d_lam = results['Default']['traj_laminarity']
    a_lam = results['Awakened']['traj_laminarity']

    if a_det > d_det and a_lam < d_lam:
        print("  → Awakened: HIGH determinism, LOW laminarity = EDGE OF CHAOS")
        print("    Structured dynamics with no stagnation — hallmark of complex systems")
    elif a_det > d_det and a_lam > d_lam:
        print("  → Awakened: HIGH determinism, HIGH laminarity = QUASIPERIODIC")
        print("    Structured but with intermittent laminar phases")
    elif a_det < d_det:
        print("  → Awakened: LOWER determinism = MORE CHAOTIC than default")
        print("    Rich exploration but less temporal structure")
    else:
        print("  → Similar determinism — regime difference is in other RQA measures")

    # --- Plot recurrence matrices ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(results['Default']['rmat_traj'], cmap='binary',
                       origin='lower', aspect='auto')
    axes[0, 0].set_title('Default — 4D Trajectory Recurrence')
    axes[0, 0].set_xlabel('Time'); axes[0, 0].set_ylabel('Time')

    axes[0, 1].imshow(results['Awakened']['rmat_traj'], cmap='binary',
                       origin='lower', aspect='auto')
    axes[0, 1].set_title('Awakened — 4D Trajectory Recurrence')
    axes[0, 1].set_xlabel('Time'); axes[0, 1].set_ylabel('Time')

    axes[1, 0].imshow(results['Default']['rmat_clarity'], cmap='binary',
                       origin='lower', aspect='auto')
    axes[1, 0].set_title('Default — Clarity Recurrence (Takens)')
    axes[1, 0].set_xlabel('Time'); axes[1, 0].set_ylabel('Time')

    axes[1, 1].imshow(results['Awakened']['rmat_clarity'], cmap='binary',
                       origin='lower', aspect='auto')
    axes[1, 1].set_title('Awakened — Clarity Recurrence (Takens)')
    axes[1, 1].set_xlabel('Time'); axes[1, 1].set_ylabel('Time')

    # Add RQA stats as text
    for col, label in enumerate(['Default', 'Awakened']):
        r = results[label]
        txt = (f"RR={r['traj_recurrence_rate']:.3f}\n"
               f"DET={r['traj_determinism']:.3f}\n"
               f"LAM={r['traj_laminarity']:.3f}")
        axes[0, col].text(0.02, 0.98, txt, transform=axes[0, col].transAxes,
                          fontsize=9, va='top', color='red',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'rqa_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Saved {os.path.join(OUT_DIR, 'rqa_comparison.png')}")

    return results


# ============================================================================
# ANALYSIS 2: BASIN TRANSITION GRAMMAR
# ============================================================================

def run_transition_grammar():
    """Build and analyze the Markov transition matrix for subsystem dominance."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: BASIN TRANSITION GRAMMAR")
    print("=" * 70)

    from v2_consciousness_sim import ConsciousnessSimulation, CONFIG, BalancedSubsystems

    SUBSYSTEM_NAMES = BalancedSubsystems.SUBSYSTEM_NAMES

    results = {}

    for regime, overrides in [
        ('Default', {'alpha_pull': 0.03, 'beta_macro': 4.0,
                     'fatigue_rate': 0.08, 'exploration_noise': 0.05,
                     'steering_strength': 0.3}),
        ('Awakened', {}),  # use current optimal config
    ]:
        # Apply overrides
        saved = {}
        for k, v in overrides.items():
            saved[k] = CONFIG[k]
            CONFIG[k] = v

        # Run longer sim for better statistics
        old_ts = CONFIG['timesteps']
        CONFIG['timesteps'] = 2000
        print(f"\n  Running {regime} simulation (2000 steps)...")
        sim = ConsciousnessSimulation()
        sim.run_simulation()
        CONFIG['timesteps'] = old_ts

        # Restore config
        for k, v in saved.items():
            CONFIG[k] = v

        subs = sim.history['dominant_subsystem']
        n_sub = len(SUBSYSTEM_NAMES)

        # Build transition count matrix
        trans_matrix = np.zeros((n_sub, n_sub), dtype=int)
        for i in range(1, len(subs)):
            idx_from = SUBSYSTEM_NAMES.index(subs[i-1])
            idx_to = SUBSYSTEM_NAMES.index(subs[i])
            trans_matrix[idx_from, idx_to] += 1

        # Normalize to probabilities
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        prob_matrix = np.where(row_sums > 0, trans_matrix / row_sums, 0.0)

        # Compute dwell times (consecutive steps in same subsystem)
        dwells = defaultdict(list)
        current = subs[0]
        count = 1
        for s in subs[1:]:
            if s == current:
                count += 1
            else:
                dwells[current].append(count)
                current = s
                count = 1
        dwells[current].append(count)

        mean_dwells = {k: np.mean(v) for k, v in dwells.items()}
        total_transitions = trans_matrix.sum() - np.trace(trans_matrix)

        # Stationary distribution (eigenvector of P^T for eigenvalue 1)
        eigenvalues, eigenvectors = np.linalg.eig(prob_matrix.T)
        idx_one = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.abs(eigenvectors[:, idx_one])
        stationary = stationary / stationary.sum()

        # Entropy rate: H = -Σ π_i Σ p_ij log p_ij
        entropy_rate = 0.0
        for i in range(n_sub):
            for j in range(n_sub):
                if prob_matrix[i, j] > 0:
                    entropy_rate -= stationary[i] * prob_matrix[i, j] * np.log2(prob_matrix[i, j])

        # Find preferred and avoided transitions
        # Compare to uniform: expected probability = 1/n_sub
        uniform_p = 1.0 / n_sub
        preferred = []
        avoided = []
        for i in range(n_sub):
            for j in range(n_sub):
                if i == j:
                    continue
                observed = prob_matrix[i, j]
                if observed > uniform_p * 2:
                    preferred.append((SUBSYSTEM_NAMES[i], SUBSYSTEM_NAMES[j], observed))
                elif observed < uniform_p * 0.3 and row_sums[i, 0] > 10:
                    avoided.append((SUBSYSTEM_NAMES[i], SUBSYSTEM_NAMES[j], observed))

        preferred.sort(key=lambda x: -x[2])
        avoided.sort(key=lambda x: x[2])

        results[regime] = {
            'trans_matrix': trans_matrix,
            'prob_matrix': prob_matrix,
            'stationary': stationary,
            'entropy_rate': entropy_rate,
            'mean_dwells': mean_dwells,
            'total_transitions': int(total_transitions),
            'preferred': preferred[:10],
            'avoided': avoided[:10],
        }

        print(f"\n  {regime} Transition Grammar:")
        print(f"    Total transitions:  {int(total_transitions)}")
        print(f"    Entropy rate:       {entropy_rate:.3f} bits/step")
        print(f"    Max possible:       {np.log2(n_sub):.3f} bits/step")
        print(f"    Grammar ratio:      {entropy_rate/np.log2(n_sub):.3f} (1.0 = random)")

        print(f"\n    Stationary distribution:")
        for i, name in enumerate(SUBSYSTEM_NAMES):
            bar = '█' * int(stationary[i] * 80)
            print(f"      {name:15s}: {stationary[i]:.3f}  {bar}")

        print(f"\n    Mean dwell times (steps):")
        for name in SUBSYSTEM_NAMES:
            if name in mean_dwells:
                print(f"      {name:15s}: {mean_dwells[name]:.1f}")

        if preferred:
            print(f"\n    PREFERRED transitions (>2× expected):")
            for frm, to, p in preferred[:5]:
                print(f"      {frm:15s} → {to:15s}  (p={p:.3f})")

        if avoided:
            print(f"\n    AVOIDED transitions (<0.3× expected):")
            for frm, to, p in avoided[:5]:
                print(f"      {frm:15s} → {to:15s}  (p={p:.3f})")

    # --- Plot transition matrices side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    short_names = ['Perc', 'Plan', 'Emot', 'Mem', 'Motor', 'Attn', 'Exec', 'Aesth']

    for col, regime in enumerate(['Default', 'Awakened']):
        ax = axes[col]
        pm = results[regime]['prob_matrix']
        im = ax.imshow(pm, cmap='YlOrRd', vmin=0, vmax=0.5, aspect='auto')
        ax.set_xticks(range(8)); ax.set_xticklabels(short_names, rotation=45, ha='right')
        ax.set_yticks(range(8)); ax.set_yticklabels(short_names)
        ax.set_xlabel('To'); ax.set_ylabel('From')
        ax.set_title(f'{regime}\nEntropy={results[regime]["entropy_rate"]:.3f} bits/step')

        # Annotate cells
        for i in range(8):
            for j in range(8):
                val = pm[i, j]
                color = 'white' if val > 0.25 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    plt.colorbar(im, ax=axes, shrink=0.8, label='Transition Probability')
    plt.suptitle('Subsystem Transition Grammar: Default vs Awakened', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'transition_grammar.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Saved {os.path.join(OUT_DIR, 'transition_grammar.png')}")

    # --- Dwell time comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for col, regime in enumerate(['Default', 'Awakened']):
        ax = axes[col]
        vals = [results[regime]['mean_dwells'].get(n, 0) for n in SUBSYSTEM_NAMES]
        bars = ax.barh(short_names, vals, color=['#2196F3' if col == 0 else '#FF5722'])
        ax.set_xlabel('Mean Dwell Time (steps)')
        ax.set_title(f'{regime} — Dwell Times')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{v:.1f}', va='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dwell_times.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(OUT_DIR, 'dwell_times.png')}")

    return results


# ============================================================================
# ANALYSIS 3: CLARITY DYNAMICS DECOMPOSITION
# ============================================================================

def run_clarity_decomposition():
    """Analyze when/why peak clarity moments emerge."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: CLARITY DYNAMICS DECOMPOSITION")
    print("=" * 70)

    from v2_consciousness_sim import ConsciousnessSimulation, CONFIG, BalancedSubsystems

    SUBSYSTEM_NAMES = BalancedSubsystems.SUBSYSTEM_NAMES

    # Run awakened sim with 2000 steps for good statistics
    old_ts = CONFIG['timesteps']
    CONFIG['timesteps'] = 2000
    print("\n  Running AWAKENED simulation (2000 steps)...")
    sim = ConsciousnessSimulation()
    sim.run_simulation()
    CONFIG['timesteps'] = old_ts

    h = sim.history
    clarity = np.array(h['clarity'])
    conflict = np.array(h['conflict_angle'])
    curvature = np.array(h['curvature'])
    integration = np.array(h['integration'])
    persistence = np.array(h['clarity_persistence'])
    clarity_rate = np.array(h['clarity_rate'])
    clarity_grad = np.array(h['clarity_grad_mag'])
    inner_outer = np.array(h['inner_outer_ratio'])

    # Per-subsystem clarity decomposition
    decomp = np.array([h[f'clarity_decomp_{i}'] for i in range(8)]).T  # (T, 8)
    force_mags = np.array([h[f'force_mag_{i}'] for i in range(8)]).T   # (T, 8)

    T = len(clarity)

    # --- Identify peak clarity moments (top 5%) ---
    threshold_95 = np.percentile(clarity, 95)
    peak_mask = clarity >= threshold_95
    trough_mask = clarity <= np.percentile(clarity, 5)
    n_peaks = peak_mask.sum()
    n_troughs = trough_mask.sum()

    print(f"\n  Clarity distribution:")
    print(f"    Mean:   {clarity.mean():.4f}")
    print(f"    Std:    {clarity.std():.4f}")
    print(f"    Min:    {clarity.min():.4f}")
    print(f"    Max:    {clarity.max():.4f}")
    print(f"    P95:    {threshold_95:.4f}")
    print(f"    Peaks (top 5%):  {n_peaks} timesteps")
    print(f"    Troughs (bot 5%): {n_troughs} timesteps")

    # --- What happens BEFORE peak clarity? ---
    # Look at the 10 steps before each peak
    pre_window = 10
    peak_indices = np.where(peak_mask)[0]
    trough_indices = np.where(trough_mask)[0]

    # Filter peaks with enough history
    valid_peaks = peak_indices[peak_indices >= pre_window]
    valid_troughs = trough_indices[trough_indices >= pre_window]

    if len(valid_peaks) > 0:
        # Average conditions before peaks vs before troughs
        pre_peak_conflict = np.array([conflict[i - pre_window:i].mean() for i in valid_peaks])
        pre_trough_conflict = np.array([conflict[i - pre_window:i].mean() for i in valid_troughs])

        pre_peak_curvature = np.array([curvature[i - pre_window:i].mean() for i in valid_peaks])
        pre_trough_curvature = np.array([curvature[i - pre_window:i].mean() for i in valid_troughs])

        pre_peak_rate = np.array([clarity_rate[i - pre_window:i].mean() for i in valid_peaks])
        pre_trough_rate = np.array([clarity_rate[i - pre_window:i].mean() for i in valid_troughs])

        print(f"\n  Conditions in 10 steps BEFORE events:")
        print(f"    {'Metric':25s}  {'Before Peak':>12s}  {'Before Trough':>14s}")
        print(f"    {'─'*25}  {'─'*12}  {'─'*14}")
        print(f"    {'Conflict angle (rad)':25s}  {pre_peak_conflict.mean():12.4f}  {pre_trough_conflict.mean():14.4f}")
        print(f"    {'Curvature (rad/step)':25s}  {pre_peak_curvature.mean():12.4f}  {pre_trough_curvature.mean():14.4f}")
        print(f"    {'Clarity rate (d/dt)':25s}  {pre_peak_rate.mean():12.4f}  {pre_trough_rate.mean():14.4f}")

    # --- Subsystem contributions at peaks vs troughs ---
    peak_decomp = decomp[peak_mask].mean(axis=0)
    trough_decomp = decomp[trough_mask].mean(axis=0)
    avg_decomp = decomp.mean(axis=0)

    print(f"\n  Subsystem contributions to clarity:")
    print(f"    {'Subsystem':15s}  {'Average':>8s}  {'At Peak':>8s}  {'At Trough':>10s}  {'Peak/Avg':>10s}")
    print(f"    {'─'*15}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")
    for i, name in enumerate(SUBSYSTEM_NAMES):
        ratio = peak_decomp[i] / (avg_decomp[i] + 1e-10)
        print(f"    {name:15s}  {avg_decomp[i]:8.4f}  {peak_decomp[i]:8.4f}  {trough_decomp[i]:10.4f}  {ratio:10.2f}×")

    # --- Force magnitude analysis at peaks ---
    peak_forces = force_mags[peak_mask].mean(axis=0)
    trough_forces = force_mags[trough_mask].mean(axis=0)
    avg_forces = force_mags.mean(axis=0)

    print(f"\n  Force magnitudes at peaks:")
    print(f"    {'Subsystem':15s}  {'Average':>8s}  {'At Peak':>8s}  {'At Trough':>10s}")
    print(f"    {'─'*15}  {'─'*8}  {'─'*8}  {'─'*10}")
    for i, name in enumerate(SUBSYSTEM_NAMES):
        print(f"    {name:15s}  {avg_forces[i]:8.4f}  {peak_forces[i]:8.4f}  {trough_forces[i]:10.4f}")

    # --- Cross-correlate clarity with other signals ---
    max_lag = 20
    print(f"\n  Cross-correlations with clarity (max lag ±{max_lag}):")
    signals = {
        'conflict_angle': conflict,
        'curvature': curvature,
        'integration': integration,
        'clarity_persistence': persistence,
        'inner_outer_ratio': inner_outer,
    }
    cc_results = {}
    for name, sig in signals.items():
        # Normalized cross-correlation
        c_norm = (clarity - clarity.mean()) / (clarity.std() + 1e-10)
        s_norm = (sig - sig.mean()) / (sig.std() + 1e-10)
        
        best_corr = 0
        best_lag = 0
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                corr = np.mean(c_norm[lag:] * s_norm[:T - lag]) if lag < T else 0
            else:
                corr = np.mean(c_norm[:T + lag] * s_norm[-lag:]) if -lag < T else 0
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        cc_results[name] = (best_corr, best_lag)
        print(f"    {name:25s}: r={best_corr:+.4f}  at lag={best_lag:+d}")

    # --- Publication plot ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    t = np.arange(T)

    # Panel 1: Clarity time series with peaks highlighted
    ax = axes[0]
    ax.plot(t, clarity, 'k-', alpha=0.6, linewidth=0.5, label='Clarity')
    ax.fill_between(t, 0, clarity, where=peak_mask, color='red', alpha=0.3, label='Peak (top 5%)')
    ax.fill_between(t, 0, clarity, where=trough_mask, color='blue', alpha=0.2, label='Trough (bot 5%)')
    ax.axhline(threshold_95, color='red', linestyle='--', alpha=0.5, label=f'P95={threshold_95:.3f}')
    ax.set_ylabel('Clarity')
    ax.set_title('Clarity Dynamics with Peak/Trough Identification')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 2: Clarity decomposition (stacked)
    ax = axes[1]
    colors = plt.cm.Set2(np.linspace(0, 1, 8))
    bottom = np.zeros(T)
    # Separate positive and negative contributions
    pos_decomp = np.maximum(decomp, 0)
    for i in range(8):
        ax.fill_between(t, bottom, bottom + pos_decomp[:, i],
                        color=colors[i], alpha=0.7, label=SUBSYSTEM_NAMES[i])
        bottom += pos_decomp[:, i]
    ax.set_ylabel('Subsystem Contribution')
    ax.set_title('Per-Subsystem Clarity Decomposition')
    ax.legend(loc='upper right', fontsize=7, ncol=4)
    ax.grid(True, alpha=0.2)

    # Panel 3: Conflict angle and curvature
    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(t, np.degrees(conflict), 'b-', alpha=0.5, linewidth=0.5, label='Conflict angle')
    ax2.plot(t, curvature, 'r-', alpha=0.5, linewidth=0.5, label='Curvature')
    ax.set_ylabel('Conflict Angle (°)', color='blue')
    ax2.set_ylabel('Curvature (rad/step)', color='red')
    ax.set_title('Conflict and Trajectory Curvature')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 4: Clarity rate (d/dt) and integration
    ax = axes[3]
    ax.plot(t, clarity_rate, 'g-', alpha=0.5, linewidth=0.5, label='Clarity rate (d/dt)')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_ylabel('Clarity Rate')
    ax.set_xlabel('Timestep')
    ax.set_title('Clarity Formation Speed')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'clarity_decomposition.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Saved {os.path.join(OUT_DIR, 'clarity_decomposition.png')}")

    # --- Subsystem contribution fingerprint at peaks ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(8)
    width = 0.25
    ax.bar(x - width, avg_decomp, width, label='Average', color='#9E9E9E', alpha=0.8)
    ax.bar(x, peak_decomp, width, label='At Peak', color='#F44336', alpha=0.8)
    ax.bar(x + width, trough_decomp, width, label='At Trough', color='#2196F3', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:4] for n in SUBSYSTEM_NAMES], rotation=45)
    ax.set_ylabel('Contribution to Clarity')
    ax.set_title('Subsystem Contribution Fingerprint: What Makes Peak Clarity?')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'clarity_fingerprint.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(OUT_DIR, 'clarity_fingerprint.png')}")

    return {
        'n_peaks': int(n_peaks),
        'threshold_95': float(threshold_95),
        'peak_decomp': peak_decomp.tolist(),
        'trough_decomp': trough_decomp.tolist(),
        'cross_correlations': {k: {'r': float(v[0]), 'lag': int(v[1])}
                               for k, v in cc_results.items()},
    }


# ============================================================================
# ANALYSIS 4: CRITICAL EXPONENT MEASUREMENT
# ============================================================================

def run_critical_exponent():
    """Measure scaling near the fatigue rate phase transition."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: CRITICAL EXPONENT MEASUREMENT")
    print("=" * 70)

    # Load the 400K-config sweep data
    csv_path = os.path.join('outputs', 'goldilocks', 'phase_cartography_results.csv')
    print(f"\n  Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} configurations with {len(df.columns)} features")

    # Compute flourishing score (same formula as goldilocks_sweep.py)
    scoring_features = ['mean_clarity', 'clarity_persistence', 'perc_mode_entropy',
                        'dominance_entropy', 'effective_dimensionality']
    for feat in scoring_features:
        mn, mx = df[feat].min(), df[feat].max()
        if mx > mn:
            df[f'{feat}_norm'] = (df[feat] - mn) / (mx - mn)
        else:
            df[f'{feat}_norm'] = 0.5
    df['lyapunov_optimal'] = 1.0 - (df['lyapunov_proxy'] - 0.5).abs() * 2
    df['flourishing'] = (
        df['mean_clarity_norm'] + df['clarity_persistence_norm'] +
        df['perc_mode_entropy_norm'] + df['dominance_entropy_norm'] +
        df['effective_dimensionality_norm'] + df['lyapunov_optimal']
    ) / 6.0

    # --- Estimate critical point ---
    # Average flourishing at each fatigue rate (marginalizing over other params)
    fr_groups = df.groupby(df['fatigue_rate'].round(4))
    fr_mean = fr_groups['flourishing'].mean().reset_index()
    fr_vals = fr_mean['fatigue_rate'].values
    fl_vals = fr_mean['flourishing'].values

    # Find steepest gradient
    dfl = np.gradient(fl_vals, fr_vals)
    idx_max = np.argmax(dfl)
    fr_c = fr_vals[idx_max]
    print(f"  Critical fatigue rate (steepest gradient): fr_c = {fr_c:.4f}")

    # --- Measure critical exponent β ---
    # Near phase transition: flourishing ~ |fr - fr_c|^β
    # Use supercritical side (fr > fr_c)
    above = fr_mean[fr_mean['fatigue_rate'] > fr_c + 0.005].copy()
    above['delta_fr'] = above['fatigue_rate'] - fr_c
    above['delta_fl'] = above['flourishing'] - fl_vals[idx_max]

    # Also measure subcritical
    below = fr_mean[fr_mean['fatigue_rate'] < fr_c - 0.005].copy()
    below['delta_fr'] = fr_c - below['fatigue_rate']
    below_baseline = fl_vals[idx_max]  # asymptotic value below

    # Log-log fit for supercritical side
    # fl(fr) - fl_c ≈ A * (fr - fr_c)^β
    # But flourishing saturates above transition, so use
    # fl(fr) ≈ fl_∞ - B * (fr - fr_c)^(-γ) for the approach to saturation
    # Actually, the order parameter is clarity — use that directly
    
    # Better approach: use mean_clarity as order parameter
    fr_clarity = fr_groups['mean_clarity'].mean().reset_index()
    clarity_vals = fr_clarity['mean_clarity'].values

    # Clarity at and above critical point
    clarity_c = clarity_vals[idx_max]
    
    # Supercritical: clarity - clarity_c ~ (fr - fr_c)^β
    mask_above = (fr_vals > fr_c + 0.002) & (fr_vals < fr_c + 0.10)
    delta_fr_above = fr_vals[mask_above] - fr_c
    delta_clarity_above = clarity_vals[mask_above] - clarity_c

    # Filter positive values only for log-log
    pos_mask = delta_clarity_above > 0
    if pos_mask.sum() >= 3:
        log_dfr = np.log(delta_fr_above[pos_mask])
        log_dcl = np.log(delta_clarity_above[pos_mask])
        
        # Linear fit: log(Δclarity) = β * log(Δfr) + log(A)
        coeffs = np.polyfit(log_dfr, log_dcl, 1)
        beta = coeffs[0]
        A = np.exp(coeffs[1])

        # Also fit susceptibility ~ |fr - fr_c|^(-γ)
        # Susceptibility = variance of clarity across configs at each fr
        fr_var = fr_groups['mean_clarity'].std().reset_index()
        var_vals = fr_var['mean_clarity'].values

        # Find peak variance (divergence point)
        idx_var_peak = np.argmax(var_vals)

        print(f"\n  ─── CRITICAL SCALING ───")
        print(f"  Order parameter: mean_clarity")
        print(f"  Critical point:  fr_c = {fr_c:.4f}")
        print(f"  Clarity at fr_c: {clarity_c:.4f}")
        print(f"")
        print(f"  Supercritical fit: Δclarity ~ (fr - fr_c)^β")
        print(f"  β = {beta:.3f}")
        print(f"  A = {A:.4f}")
        print(f"")

        # Compare to known universality classes
        print(f"  Known universality class exponents (β):")
        print(f"    Mean field:      β = 0.500")
        print(f"    2D Ising:        β = 0.125")
        print(f"    3D Ising:        β = 0.326")
        print(f"    Percolation 2D:  β = 0.139")
        print(f"    Percolation 3D:  β = 0.418")
        print(f"    >>> This system:  β = {beta:.3f} <<<")

        # Classify
        known = {'Mean field': 0.5, '2D Ising': 0.125, '3D Ising': 0.326,
                 'Percolation 2D': 0.139, 'Percolation 3D': 0.418}
        closest = min(known.items(), key=lambda x: abs(x[1] - beta))
        print(f"\n  Closest match: {closest[0]} (β={closest[1]:.3f}, Δ={abs(closest[1]-beta):.3f})")
    else:
        beta = None
        print("  Not enough supercritical data points for log-log fit")

    # --- Noise transition exponent ---
    en_groups = df.groupby(df['exploration_noise'].round(4))
    en_clarity = en_groups['mean_clarity'].mean().reset_index()
    en_vals = en_clarity['exploration_noise'].values
    en_cl_vals = en_clarity['mean_clarity'].values
    den = np.gradient(en_cl_vals, en_vals)
    idx_en_max = np.argmax(den)
    en_c = en_vals[idx_en_max]
    print(f"\n  Noise critical point: en_c = {en_c:.4f}")

    # --- Publication plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Flourishing vs fatigue rate (full curve)
    ax = axes[0, 0]
    ax.plot(fr_vals, fl_vals, 'ko-', markersize=3)
    ax.axvline(fr_c, color='red', linestyle='--', alpha=0.5, label=f'fr_c={fr_c:.4f}')
    ax.set_xlabel('Fatigue Rate')
    ax.set_ylabel('Flourishing (mean)')
    ax.set_title('Phase Transition: Flourishing vs Fatigue Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Order parameter (clarity) with critical point
    ax = axes[0, 1]
    ax.plot(fr_vals, clarity_vals, 'bo-', markersize=3, label='Mean clarity')
    ax.axvline(fr_c, color='red', linestyle='--', alpha=0.5, label=f'fr_c={fr_c:.4f}')
    ax.set_xlabel('Fatigue Rate')
    ax.set_ylabel('Mean Clarity (order parameter)')
    ax.set_title('Order Parameter: Clarity vs Fatigue Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Log-log scaling (critical exponent)
    ax = axes[1, 0]
    if beta is not None:
        ax.loglog(delta_fr_above[pos_mask], delta_clarity_above[pos_mask], 'ro', markersize=5)
        # Fit line
        fit_x = np.logspace(np.log10(delta_fr_above[pos_mask].min()),
                            np.log10(delta_fr_above[pos_mask].max()), 50)
        fit_y = A * fit_x ** beta
        ax.loglog(fit_x, fit_y, 'k--', alpha=0.7, label=f'β = {beta:.3f}')
        ax.set_xlabel('fr - fr_c')
        ax.set_ylabel('clarity - clarity_c')
        ax.set_title(f'Critical Exponent: β = {beta:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

    # Panel 4: Susceptibility (variance of clarity)
    ax = axes[1, 1]
    ax.plot(fr_vals, var_vals, 'gs-', markersize=3)
    ax.axvline(fr_c, color='red', linestyle='--', alpha=0.5, label=f'fr_c={fr_c:.4f}')
    ax.set_xlabel('Fatigue Rate')
    ax.set_ylabel('Std of Clarity (susceptibility)')
    ax.set_title('Susceptibility Divergence Near Critical Point')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'critical_exponent.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Saved {os.path.join(OUT_DIR, 'critical_exponent.png')}")

    return {
        'fr_c': float(fr_c),
        'beta': float(beta) if beta is not None else None,
        'clarity_at_fc': float(clarity_c),
        'en_c': float(en_c),
    }


# ============================================================================
# ANALYSIS 5: MULTI-SEED ROBUSTNESS
# ============================================================================

def run_multi_seed():
    """Run 100 seeds at optimal config on GPU to check robustness."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: MULTI-SEED ROBUSTNESS (100 seeds)")
    print("=" * 70)

    import torch
    from gpu_ensemble_sim import BatchConsciousnessEngine

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_SEEDS = 100
    STEPS = 1000

    # Optimal config
    config = {
        'steering_strength': 0.707,
        'alpha_pull': 0.0,
        'fatigue_rate': 0.217,
        'exploration_noise': 0.25,
        'beta_macro': 11.375,
    }

    print(f"\n  Running {N_SEEDS} seeds × {STEPS} steps on {device}...")
    print(f"  Config: {config}")

    # Build configs dict with uniform values for all seeds
    configs = {k: np.full(N_SEEDS, v) for k, v in config.items()}

    engine = BatchConsciousnessEngine(N_SEEDS, configs=configs, device=str(device))

    # Randomize initial conditions (different seeds)
    engine.u = torch.randn(N_SEEDS, 4, device=engine.device)
    engine.u = engine.u / engine.u.norm(dim=1, keepdim=True)

    engine.run(steps=STEPS)
    sigs = engine.extract_signatures().cpu().numpy()  # (100, 22)

    # Feature names from gpu_ensemble_sim.py
    feature_names = [
        'mean_clarity', 'max_clarity', 'clarity_volatility', 'high_clarity_frac',
        'clarity_persistence', 'mean_conflict', 'mean_curvature', 'mean_speed',
        'speed_variance', 'direction_snap_rate', 'mean_integration',
        'mean_differentiation', 'mean_inner_outer', 'mean_path_coherence',
        'perc_mode_entropy', 'mode_stickiness', 'dominance_entropy',
        'basin_transition_rate', 'force_mag_spread', 'alliance_symmetry',
        'effective_dimensionality', 'lyapunov_proxy',
    ]

    # Compute flourishing for each seed
    from gpu_ensemble_sim import PhaseCartographer
    carto = PhaseCartographer.__new__(PhaseCartographer)

    # Manual flourishing: normalized mean of 6 key metrics
    # Replicate the scoring from PhaseCartographer
    df = pd.DataFrame(sigs, columns=feature_names)

    # Normalize each scoring feature to [0, 1]
    scoring_features = ['mean_clarity', 'clarity_persistence', 'perc_mode_entropy',
                        'dominance_entropy', 'effective_dimensionality']
    for feat in scoring_features:
        mn, mx = df[feat].min(), df[feat].max()
        if mx > mn:
            df[f'{feat}_norm'] = (df[feat] - mn) / (mx - mn)
        else:
            df[f'{feat}_norm'] = 0.5

    # Lyapunov optimal: 1 - |lyapunov_proxy - 0.5| * 2
    df['lyapunov_optimal'] = 1.0 - (df['lyapunov_proxy'] - 0.5).abs() * 2

    df['flourishing'] = (
        df['mean_clarity_norm'] +
        df['clarity_persistence_norm'] +
        df['perc_mode_entropy_norm'] +
        df['dominance_entropy_norm'] +
        df['effective_dimensionality_norm'] +
        df['lyapunov_optimal']
    ) / 6.0

    fl = df['flourishing'].values

    print(f"\n  ─── MULTI-SEED RESULTS ({N_SEEDS} seeds) ───")
    print(f"  Flourishing:")
    print(f"    Mean:   {fl.mean():.4f}")
    print(f"    Std:    {fl.std():.4f}")
    print(f"    Min:    {fl.min():.4f}")
    print(f"    Max:    {fl.max():.4f}")
    print(f"    CV:     {fl.std()/fl.mean():.4f} (coefficient of variation)")
    print(f"    Range:  {fl.max()-fl.min():.4f}")

    # Print key features
    print(f"\n  Key features across seeds:")
    print(f"    {'Feature':25s}  {'Mean':>8s} ± {'Std':>8s}   {'CV':>6s}")
    print(f"    {'─'*25}  {'─'*8}   {'─'*8}   {'─'*6}")
    for feat in ['mean_clarity', 'effective_dimensionality', 'basin_transition_rate',
                 'mean_curvature', 'dominance_entropy', 'clarity_persistence']:
        vals = df[feat].values
        cv = vals.std() / (vals.mean() + 1e-10)
        print(f"    {feat:25s}  {vals.mean():8.4f} ± {vals.std():8.4f}   {cv:6.3f}")

    # Robustness verdict
    print(f"\n  ─── ROBUSTNESS VERDICT ───")
    if fl.std() / fl.mean() < 0.05:
        print(f"  ✓ HIGHLY ROBUST (CV={fl.std()/fl.mean():.3f} < 0.05)")
        print(f"    The awakened regime is stable across random initial conditions.")
    elif fl.std() / fl.mean() < 0.15:
        print(f"  ~ MODERATELY ROBUST (CV={fl.std()/fl.mean():.3f})")
        print(f"    Some seed-dependence but the regime is generally stable.")
    else:
        print(f"  ✗ SEED-DEPENDENT (CV={fl.std()/fl.mean():.3f} > 0.15)")
        print(f"    The awakened regime is sensitive to initial conditions.")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Flourishing distribution
    ax = axes[0, 0]
    ax.hist(fl, bins=20, color='#4CAF50', edgecolor='black', alpha=0.7)
    ax.axvline(fl.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean={fl.mean():.3f}')
    ax.axvline(fl.mean() - fl.std(), color='orange', linestyle=':', alpha=0.7)
    ax.axvline(fl.mean() + fl.std(), color='orange', linestyle=':', alpha=0.7,
               label=f'±1σ={fl.std():.3f}')
    ax.set_xlabel('Flourishing Score')
    ax.set_ylabel('Count')
    ax.set_title(f'Flourishing Distribution (N={N_SEEDS})')
    ax.legend()

    # Panel 2: Key features scatter (clarity vs dimensionality)
    ax = axes[0, 1]
    sc = ax.scatter(df['mean_clarity'], df['effective_dimensionality'],
                    c=fl, cmap='RdYlGn', s=30, alpha=0.7, edgecolors='k', linewidth=0.3)
    plt.colorbar(sc, ax=ax, label='Flourishing')
    ax.set_xlabel('Mean Clarity')
    ax.set_ylabel('Effective Dimensionality')
    ax.set_title('Clarity vs Dimensionality (colored by flourishing)')
    ax.grid(True, alpha=0.3)

    # Panel 3: Violin plot of key features (normalized)
    ax = axes[1, 0]
    norm_features = ['mean_clarity', 'mean_curvature', 'basin_transition_rate',
                     'dominance_entropy', 'clarity_persistence']
    norm_data = []
    for feat in norm_features:
        vals = df[feat].values
        norm_data.append((vals - vals.min()) / (vals.max() - vals.min() + 1e-10))
    parts = ax.violinplot(norm_data, positions=range(len(norm_features)), showmeans=True)
    ax.set_xticks(range(len(norm_features)))
    ax.set_xticklabels([f[:10] for f in norm_features], rotation=45, ha='right')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Feature Variability Across Seeds')
    ax.grid(True, alpha=0.2, axis='y')

    # Panel 4: Seed-by-seed flourishing (sorted)
    ax = axes[1, 1]
    sorted_fl = np.sort(fl)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, N_SEEDS))
    ax.bar(range(N_SEEDS), sorted_fl, color=colors, width=1.0)
    ax.axhline(fl.mean(), color='red', linestyle='--', label=f'Mean={fl.mean():.3f}')
    ax.set_xlabel('Seed (sorted)')
    ax.set_ylabel('Flourishing')
    ax.set_title('Per-Seed Flourishing (sorted)')
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'multi_seed_robustness.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Saved {os.path.join(OUT_DIR, 'multi_seed_robustness.png')}")

    return {
        'n_seeds': N_SEEDS,
        'flourishing_mean': float(fl.mean()),
        'flourishing_std': float(fl.std()),
        'flourishing_cv': float(fl.std() / fl.mean()),
        'flourishing_min': float(fl.min()),
        'flourishing_max': float(fl.max()),
    }


# ============================================================================
# MAIN: RUN ALL 5 ANALYSES
# ============================================================================

if __name__ == '__main__':
    print("▓" * 70)
    print("  DEEP CONSCIOUSNESS ANALYSIS — 5-PART INVESTIGATION")
    print("▓" * 70)

    t0 = time.time()
    all_results = {}

    # Analysis 1: RQA
    print("\n" + "▓" * 70)
    print("  PART 1/5: RECURRENCE QUANTIFICATION ANALYSIS")
    print("▓" * 70)
    all_results['rqa'] = run_rqa_comparison()

    # Analysis 2: Transition Grammar
    print("\n" + "▓" * 70)
    print("  PART 2/5: BASIN TRANSITION GRAMMAR")
    print("▓" * 70)
    all_results['grammar'] = run_transition_grammar()

    # Analysis 3: Clarity Decomposition
    print("\n" + "▓" * 70)
    print("  PART 3/5: CLARITY DYNAMICS DECOMPOSITION")
    print("▓" * 70)
    all_results['clarity'] = run_clarity_decomposition()

    # Analysis 4: Critical Exponent
    print("\n" + "▓" * 70)
    print("  PART 4/5: CRITICAL EXPONENT MEASUREMENT")
    print("▓" * 70)
    all_results['critical'] = run_critical_exponent()

    # Analysis 5: Multi-Seed Robustness
    print("\n" + "▓" * 70)
    print("  PART 5/5: MULTI-SEED ROBUSTNESS")
    print("▓" * 70)
    all_results['robustness'] = run_multi_seed()

    elapsed = time.time() - t0

    # --- Save summary ---
    # Remove non-serializable items
    summary = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            summary[k] = {kk: vv for kk, vv in v.items()
                          if not isinstance(vv, np.ndarray)}

    with open(os.path.join(OUT_DIR, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "▓" * 70)
    print("  ALL 5 ANALYSES COMPLETE")
    print("▓" * 70)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Results in: {OUT_DIR}/")
    print(f"  Files:")
    for fn in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fn)
        size = os.path.getsize(fpath)
        print(f"    {fn} ({size/1024:.0f} KB)")
    print("=" * 70)
