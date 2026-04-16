#!/usr/bin/env python3
"""
Latent Coordinate Mapping at Boundary Transitions
==================================================

Bridges the simulation core with the src/latent analysis pipeline by
mapping simulation subsystem activities to the 6-subsystem latent drive
format and running the coordinator model during boundary events.

  Part A: Drive extraction — Convert sim activities to latent drives
  Part B: Coordinator dynamics at transitions vs dwelling
  Part C: Basin similarity landscape — How does latent basin structure
          relate to macro basin transitions?
  Part D: Anomaly detection — Can the anomaly scorer detect boundaries?

OPH Credit: Framework adapted from Observer Patch Holography by FloatingPragma.
  https://github.com/FloatingPragma/observer-patch-holography

Usage:
    python latent_boundary_mapping.py [--device cuda:0] [--steps 2000] [--N 64]
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
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.latent.basins import (
    basin_similarities,
    basin_switch_event,
    initialize_basin_attractors,
)
from src.latent.coordinator import coordinator_step, run_coordinator
from src.latent.mapping import DRIVE_DIMS

from universality_test import UniversalEngine

OUT_DIR = os.path.join('outputs', 'latent_boundary')

# Map 8 simulation subsystems → 6 latent subsystems
# Sim: Motor(0), Planning(1), Attention(2), Memory(3),
#       Emotion(4), Social(5), Intuition(6), Aesthetic(7)
# Latent: perception, planning, emotion, memory, attention, executive
SIM_TO_LATENT = {
    'perception': [0, 6],      # Motor + Intuition → perception
    'planning':   [1],         # Planning → planning
    'emotion':    [4, 7],      # Emotion + Aesthetic → emotion
    'memory':     [3],         # Memory → memory
    'attention':  [2],         # Attention → attention
    'executive':  [5],         # Social → executive (decision coordination)
}


# ============================================================================
# ACTIVITY EXTRACTION FROM ENGINE
# ============================================================================

class InstrumentedEngine(UniversalEngine):
    """Engine that records per-timestep subsystem activities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hist_activities = torch.zeros(
            self.N, self.max_steps, self.n_sub, device=self.device)

    @torch.no_grad()
    def step(self):
        t = self.step_count
        u = self.u_t
        N, dev = self.N, self.device

        # Compute influences (same as parent)
        if self.manifold_type == 'flat4':
            u_dir = torch.nn.functional.normalize(u, dim=1)
            influences = torch.einsum('nd,sd->ns', u_dir, self.prefs)
        else:
            influences = torch.einsum('nd,sd->ns', u, self.prefs)
        influences = 0.5 + 0.3 * influences

        effective = influences * torch.exp(-self.fatigue)
        noise = self.exploration_noise * torch.randn(N, self.n_sub, device=dev)
        effective = (effective + noise).clamp(min=self.floor_value)
        activities = effective / (effective.sum(dim=1, keepdim=True) + 1e-8)

        # Store activities before parent step
        if t < self.max_steps:
            self.hist_activities[:, t] = activities

        # Call parent step (which will increment step_count)
        super().step()


def activities_to_drives(activities_np):
    """
    Convert (T, 8) simulation activities to list of latent drive dicts.

    Each drive dict maps subsystem_name → 4D drive vector, matching
    the format expected by src/latent/coordinator.
    """
    T = activities_np.shape[0]
    drive_sequence = []

    for t in range(T):
        act = activities_np[t]  # (8,)
        drives = {}
        for sub_name, sim_indices in SIM_TO_LATENT.items():
            # Average activity of mapped simulation subsystems
            avg_activity = np.mean([act[i] for i in sim_indices])
            # Create 4D drive vector using cosine basis (matching mapping.py)
            phase = hash(sub_name) % 360
            phase_rad = np.deg2rad(phase)
            drive = np.zeros(DRIVE_DIMS)
            for d in range(DRIVE_DIMS):
                drive[d] = avg_activity * np.cos(phase_rad + d * np.pi / DRIVE_DIMS)
            drives[sub_name] = drive
        drive_sequence.append(drives)

    return drive_sequence


# ============================================================================
# PART A: DRIVE EXTRACTION AND VISUALIZATION
# ============================================================================

def part_a_drive_extraction(device='cuda:0', steps=2000, N=64):
    """Extract drives from simulation and visualize at boundaries."""
    print('\n  ── Part A: Drive Extraction at Boundaries ──')

    engine = InstrumentedEngine(
        N=N, device=device, steps=steps,
        manifold='s3', topology='cyclic', fatigue_type='gradual',
    )
    for t in range(steps):
        engine.step()

    activities = engine.hist_activities[:, :steps].cpu().numpy()  # (N, T, 8)
    basins = engine.hist_macro_basin[:, :steps].cpu().numpy()
    clarity = engine.hist_clarity[:, :steps].cpu().numpy()

    # Sample being 0 for visualization
    b = 0
    act_b = activities[b]  # (T, 8)
    basins_b = basins[b]
    transitions = basins_b[1:] != basins_b[:-1]  # (T-1,)
    tr_idx = np.where(transitions)[0]

    # Convert to drives
    drives = activities_to_drives(act_b)

    # Extract drive magnitudes per latent subsystem
    drive_mags = {name: [] for name in SIM_TO_LATENT}
    for t in range(steps):
        for name in SIM_TO_LATENT:
            drive_mags[name].append(np.linalg.norm(drives[t][name]))

    # Compare drive magnitudes at transitions vs dwelling
    results = {}
    print(f'\n    Drive magnitude at transitions vs dwelling:')
    for name in SIM_TO_LATENT:
        mags = np.array(drive_mags[name])
        if len(tr_idx) > 0 and len(tr_idx) < len(mags) - 1:
            trans_mags = mags[tr_idx]
            dwell_mask = np.ones(steps, dtype=bool)
            dwell_mask[tr_idx] = False
            dwell_mags = mags[dwell_mask]
            if len(trans_mags) > 3 and len(dwell_mags) > 3:
                t_stat, p_val = stats.ttest_ind(trans_mags, dwell_mags)
                d = (trans_mags.mean() - dwell_mags.mean()) / np.sqrt(
                    (trans_mags.std()**2 + dwell_mags.std()**2) / 2 + 1e-15)
                results[name] = {
                    'trans_mean': float(trans_mags.mean()),
                    'dwell_mean': float(dwell_mags.mean()),
                    'cohens_d': float(d),
                    'p_value': float(p_val),
                }
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                print(f'      {name:>12s}: trans={trans_mags.mean():.4f}  '
                      f'dwell={dwell_mags.mean():.4f}  d={d:+.3f}  {sig}')

    # Plot time series with transition markers
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # Panel 1: Drive magnitudes
    for name, mags_list in drive_mags.items():
        axes[0].plot(mags_list, alpha=0.7, linewidth=0.5, label=name)
    for idx in tr_idx:
        axes[0].axvline(idx, color='red', alpha=0.15, linewidth=0.5)
    axes[0].set_ylabel('Drive Magnitude', fontsize=10)
    axes[0].legend(fontsize=7, ncol=3, loc='upper right')
    axes[0].set_title('Latent Drive Magnitudes (Being 0)', fontsize=11, fontweight='bold')

    # Panel 2: Dominant subsystem activity
    axes[1].imshow(act_b.T, aspect='auto', cmap='viridis', interpolation='nearest')
    for idx in tr_idx:
        axes[1].axvline(idx, color='red', alpha=0.3, linewidth=0.5)
    axes[1].set_ylabel('Subsystem', fontsize=10)
    axes[1].set_yticks(range(8))
    axes[1].set_yticklabels(['Motor', 'Plan', 'Attn', 'Mem',
                              'Emot', 'Soc', 'Intuit', 'Aesth'], fontsize=7)
    axes[1].set_title('Raw Subsystem Activities', fontsize=11, fontweight='bold')

    # Panel 3: Clarity + basin transitions
    axes[2].plot(clarity[b], color='#4C72B0', linewidth=0.5)
    for idx in tr_idx:
        axes[2].axvline(idx, color='red', alpha=0.3, linewidth=0.5)
    axes[2].set_ylabel('Clarity', fontsize=10)
    axes[2].set_xlabel('Timestep', fontsize=10)
    axes[2].set_title('Clarity with Transition Markers', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/part_a_drive_extraction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {OUT_DIR}/part_a_drive_extraction.png')

    return results


# ============================================================================
# PART B: COORDINATOR DYNAMICS AT TRANSITIONS
# ============================================================================

def part_b_coordinator_at_transitions(device='cuda:0', steps=2000, N=64):
    """
    Run the latent coordinator using simulation-derived drives and compare
    coordinator basin switching with simulation macro basin switching.
    """
    print('\n  ── Part B: Coordinator Basin Dynamics ──')

    engine = InstrumentedEngine(
        N=N, device=device, steps=steps,
        manifold='s3', topology='cyclic', fatigue_type='gradual',
    )
    for t in range(steps):
        engine.step()

    activities = engine.hist_activities[:, :steps].cpu().numpy()
    sim_basins = engine.hist_macro_basin[:, :steps].cpu().numpy()

    # Run coordinator for a sample of beings
    n_sample = min(N, 16)
    coord_agreement = []
    coord_switch_rates = []
    sim_switch_rates = []

    for b in range(n_sample):
        act_b = activities[b]
        drives = activities_to_drives(act_b)

        # Run coordinator
        coord_df = run_coordinator(
            drive_sequence=drives,
            n_basins=5,
            n_dims=DRIVE_DIMS,
            learning_rate=0.05,
            noise_level=0.02,
            basin_pull_strength=0.02,
            random_seed=b,
        )

        coord_basins = coord_df['chosen_basin'].values
        sim_b = sim_basins[b]

        # Compute switch rates
        coord_switches = (coord_basins[1:] != coord_basins[:-1]).mean()
        sim_switches = (sim_b[1:] != sim_b[:-1]).mean()
        coord_switch_rates.append(float(coord_switches))
        sim_switch_rates.append(float(sim_switches))

        # Compute temporal agreement: do transitions co-occur?
        coord_trans = coord_basins[1:] != coord_basins[:-1]
        sim_trans = sim_b[1:] != sim_b[:-1]

        # Cross-correlation at lag 0
        if coord_trans.std() > 0 and sim_trans.std() > 0:
            r, p = stats.pointbiserialr(
                coord_trans.astype(float), sim_trans.astype(float))
            coord_agreement.append(float(r))

    # Results
    results = {
        'coord_switch_rate_mean': float(np.mean(coord_switch_rates)),
        'sim_switch_rate_mean': float(np.mean(sim_switch_rates)),
        'agreement_r_mean': float(np.mean(coord_agreement)) if coord_agreement else 0.0,
    }

    print(f'    Coordinator switch rate: {results["coord_switch_rate_mean"]:.4f}')
    print(f'    Simulation switch rate:  {results["sim_switch_rate_mean"]:.4f}')
    print(f'    Transition agreement:    r = {results["agreement_r_mean"]:+.4f}')

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(sim_switch_rates, coord_switch_rates, alpha=0.7,
                color='#4C72B0', edgecolor='white', s=50)
    ax1.plot([0, max(sim_switch_rates)], [0, max(sim_switch_rates)],
             '--', color='gray', alpha=0.5, label='y=x')
    ax1.set_xlabel('Sim Switch Rate', fontsize=10)
    ax1.set_ylabel('Coordinator Switch Rate', fontsize=10)
    ax1.set_title('Switch Rate: Simulation vs Coordinator', fontsize=11, fontweight='bold')
    ax1.legend()

    if coord_agreement:
        ax2.hist(coord_agreement, bins=15, color='#55A868', edgecolor='white', alpha=0.7)
        ax2.axvline(np.mean(coord_agreement), color='red', ls='--',
                    label=f'μ={np.mean(coord_agreement):+.3f}')
        ax2.set_xlabel('Transition Agreement (r)', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('Temporal Agreement of Transitions', fontsize=11, fontweight='bold')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/part_b_coordinator_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {OUT_DIR}/part_b_coordinator_dynamics.png')

    return results


# ============================================================================
# PART C: BASIN SIMILARITY LANDSCAPE
# ============================================================================

def part_c_basin_landscape(device='cuda:0', steps=2000, N=64):
    """
    Analyse the latent basin similarity landscape and how it changes
    at boundary transitions.
    """
    print('\n  ── Part C: Basin Similarity Landscape ──')

    engine = InstrumentedEngine(
        N=N, device=device, steps=steps,
        manifold='s3', topology='cyclic', fatigue_type='gradual',
    )
    for t in range(steps):
        engine.step()

    activities = engine.hist_activities[:, :steps].cpu().numpy()
    sim_basins = engine.hist_macro_basin[:, :steps].cpu().numpy()

    n_latent_basins = 5
    n_dims = DRIVE_DIMS

    # Run coordinator for being 0 and extract similarity trajectories
    b = 0
    drives = activities_to_drives(activities[b])
    coord_df = run_coordinator(
        drive_sequence=drives,
        n_basins=n_latent_basins,
        n_dims=n_dims,
        random_seed=0,
    )

    # Extract similarity columns
    sim_cols = [f'basin_similarity_{i}' for i in range(n_latent_basins)]
    sims = coord_df[sim_cols].values  # (T, n_basins)

    sim_b = sim_basins[b]
    transitions = sim_b[1:] != sim_b[:-1]
    tr_idx = np.where(transitions)[0]

    # Measure similarity entropy (ambiguity) at transitions vs dwelling
    # Higher entropy = more ambiguous basin membership
    sim_positive = np.clip(sims, 1e-10, None)
    sim_normed = sim_positive / sim_positive.sum(axis=1, keepdims=True)
    sim_entropy = -np.sum(sim_normed * np.log(sim_normed + 1e-15), axis=1)

    trans_entropy = sim_entropy[tr_idx] if len(tr_idx) > 0 else np.array([])
    dwell_mask = np.ones(steps, dtype=bool)
    if len(tr_idx) > 0:
        dwell_mask[tr_idx] = False
    dwell_entropy = sim_entropy[dwell_mask]

    results = {}
    if len(trans_entropy) > 3 and len(dwell_entropy) > 3:
        t_stat, p_val = stats.ttest_ind(trans_entropy, dwell_entropy)
        d = (trans_entropy.mean() - dwell_entropy.mean()) / np.sqrt(
            (trans_entropy.std()**2 + dwell_entropy.std()**2) / 2 + 1e-15)
        results['entropy_gap'] = {
            'trans_mean': float(trans_entropy.mean()),
            'dwell_mean': float(dwell_entropy.mean()),
            'cohens_d': float(d),
            'p_value': float(p_val),
        }
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f'    Similarity entropy: trans={trans_entropy.mean():.4f}  '
              f'dwell={dwell_entropy.mean():.4f}  d={d:+.3f}  {sig}')

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: Similarity trajectories
    for i in range(n_latent_basins):
        axes[0, 0].plot(sims[:, i], alpha=0.6, linewidth=0.5,
                        label=f'Basin {i}')
    for idx in tr_idx:
        axes[0, 0].axvline(idx, color='red', alpha=0.15, linewidth=0.5)
    axes[0, 0].set_ylabel('Basin Similarity', fontsize=9)
    axes[0, 0].set_title('Latent Basin Similarity Trajectories', fontsize=10, fontweight='bold')
    axes[0, 0].legend(fontsize=7, ncol=3)

    # Panel 2: Similarity entropy over time
    axes[0, 1].plot(sim_entropy, color='#E07020', linewidth=0.5)
    for idx in tr_idx:
        axes[0, 1].axvline(idx, color='red', alpha=0.15, linewidth=0.5)
    axes[0, 1].set_ylabel('Similarity Entropy', fontsize=9)
    axes[0, 1].set_title('Basin Ambiguity Over Time', fontsize=10, fontweight='bold')

    # Panel 3: Entropy at transitions vs dwelling
    if len(trans_entropy) > 3 and len(dwell_entropy) > 3:
        parts = axes[1, 0].violinplot([trans_entropy, dwell_entropy],
                                       positions=[0, 1], showmeans=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_xticklabels(['Transition', 'Dwelling'], fontsize=9)
        axes[1, 0].set_ylabel('Similarity Entropy', fontsize=9)
        axes[1, 0].set_title('Basin Ambiguity: Boundary vs Dwelling',
                             fontsize=10, fontweight='bold')

    # Panel 4: Max similarity gap (top1 - top2) over time
    sorted_sims = np.sort(sims, axis=1)[:, ::-1]
    gap = sorted_sims[:, 0] - sorted_sims[:, 1]
    axes[1, 1].plot(gap, color='#4C72B0', linewidth=0.5)
    for idx in tr_idx:
        axes[1, 1].axvline(idx, color='red', alpha=0.15, linewidth=0.5)
    axes[1, 1].set_ylabel('Top-1 − Top-2 Similarity', fontsize=9)
    axes[1, 1].set_xlabel('Timestep', fontsize=9)
    axes[1, 1].set_title('Basin Decisiveness Gap', fontsize=10, fontweight='bold')

    plt.suptitle('Latent Basin Similarity Landscape at Boundaries (Being 0)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/part_c_basin_landscape.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {OUT_DIR}/part_c_basin_landscape.png')

    return results


# ============================================================================
# PART D: ANOMALY DETECTION AT BOUNDARIES
# ============================================================================

def part_d_anomaly_detection(device='cuda:0', steps=2000, N=64):
    """
    Use the anomaly scoring pipeline (src/anomaly) to detect boundary
    transitions as anomalous events in the drive sequence.
    """
    print('\n  ── Part D: Anomaly Detection at Boundaries ──')

    # Check if anomaly module is available
    try:
        from src.anomaly.scoring import AnomalyScorer
    except ImportError:
        print('    [SKIP] src.anomaly.scoring not importable')
        return {}

    engine = InstrumentedEngine(
        N=N, device=device, steps=steps,
        manifold='s3', topology='cyclic', fatigue_type='gradual',
    )
    for t in range(steps):
        engine.step()

    activities = engine.hist_activities[:, :steps].cpu().numpy()
    sim_basins = engine.hist_macro_basin[:, :steps].cpu().numpy()

    # Compute per-timestep anomaly scores using drive divergence
    # Simple approach: anomaly = Mahalanobis distance from running mean
    n_sample = min(N, 16)

    all_precision = []
    all_recall = []
    all_auroc = []

    for b in range(n_sample):
        act_b = activities[b]  # (T, 8)
        sim_b = sim_basins[b]
        transitions = sim_b[1:] != sim_b[:-1]

        # Compute anomaly scores: rolling z-score of activity change
        diffs = np.abs(np.diff(act_b, axis=0))  # (T-1, 8)
        diff_mag = diffs.sum(axis=1)  # (T-1,)

        # Rolling statistics
        win = 50
        scores = np.zeros(len(diff_mag))
        for t in range(win, len(diff_mag)):
            window = diff_mag[max(0, t - win):t]
            mu = window.mean()
            sigma = window.std() + 1e-8
            scores[t] = (diff_mag[t] - mu) / sigma

        # Threshold-free: use ROC-AUC
        from sklearn.metrics import roc_auc_score, precision_recall_curve

        if transitions.sum() > 5 and transitions.sum() < len(transitions) - 5:
            valid = scores[win:] != 0
            if valid.sum() > 10:
                s = scores[win:]
                t_labels = transitions[win:]
                try:
                    auroc = roc_auc_score(t_labels.astype(int), s)
                    all_auroc.append(auroc)

                    # Precision/recall at optimal threshold
                    prec, rec, thresholds = precision_recall_curve(
                        t_labels.astype(int), s)
                    f1 = 2 * prec * rec / (prec + rec + 1e-8)
                    best = np.argmax(f1)
                    all_precision.append(float(prec[best]))
                    all_recall.append(float(rec[best]))
                except ValueError:
                    pass

    results = {}
    if all_auroc:
        results = {
            'auroc_mean': float(np.mean(all_auroc)),
            'auroc_std': float(np.std(all_auroc)),
            'precision_mean': float(np.mean(all_precision)),
            'recall_mean': float(np.mean(all_recall)),
            'n_beings': len(all_auroc),
        }
        print(f'    AUROC: {results["auroc_mean"]:.3f} ± {results["auroc_std"]:.3f}')
        print(f'    Precision: {results["precision_mean"]:.3f}  '
              f'Recall: {results["recall_mean"]:.3f}')

    # Plot AUROC distribution
    if all_auroc:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_auroc, bins=15, color='#DD5555', edgecolor='white', alpha=0.7)
        ax.axvline(np.mean(all_auroc), color='black', ls='--',
                   label=f'μ={np.mean(all_auroc):.3f}')
        ax.axvline(0.5, color='gray', ls=':', label='Chance')
        ax.set_xlabel('AUROC', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Anomaly Detection of Boundary Transitions\n'
                     '(Activity Change Z-Score as Detector)',
                     fontsize=11, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/part_d_anomaly_detection.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'    Saved: {OUT_DIR}/part_d_anomaly_detection.png')

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Latent Boundary Mapping')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--N', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print('╔══════════════════════════════════════════════════════════════════════╗')
    print('║  LATENT BOUNDARY MAPPING — Connecting Sim Core to Analysis Pipeline║')
    print('║                                                                    ║')
    print('║  Bridging: activities → drives → coordinator → basins → anomalies  ║')
    print('║  OPH by FloatingPragma                                             ║')
    print('║  https://github.com/FloatingPragma/observer-patch-holography       ║')
    print('╚══════════════════════════════════════════════════════════════════════╝')

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'\nGPU: {props.name} ({props.total_memory / 1e9:.1f} GB)')

    t_start = time.time()

    results = {}
    results['part_a'] = part_a_drive_extraction(args.device, args.steps, args.N)
    results['part_b'] = part_b_coordinator_at_transitions(args.device, args.steps, args.N)
    results['part_c'] = part_c_basin_landscape(args.device, args.steps, args.N)
    results['part_d'] = part_d_anomaly_detection(args.device, args.steps, args.N)

    out_path = f'{OUT_DIR}/latent_boundary_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  All results saved to {out_path}')

    elapsed = time.time() - t_start
    print(f'  Total wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
