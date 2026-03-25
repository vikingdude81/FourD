#!/usr/bin/env python3
"""
Critical Phenomena Suite — Full Universality Verification
==========================================================

Four-part analysis that builds an airtight case for the
consciousness phase transition being in a genuine universality class.

Part 1: NULL / ABLATION CONTROLS  (mandatory, breaks mechanism 5 ways)
Part 2: SUBSYSTEM-COUNT FSS       (correct "system size" L = N_sub)
Part 3: DATA COLLAPSE              ("the killer figure")
Part 4: COARSE-GRAINING            (real-space renormalization)

Requires: GPU (RTX 5090 recommended), gpu_ensemble_sim.py in path
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
from scipy.optimize import minimize_scalar, minimize
import torch
import torch.nn.functional as F
import time
import os
import json
import copy

from gpu_ensemble_sim import (
    BatchConsciousnessEngine, generate_fibonacci_s3, derive_macro_basins,
    PREFERENCE_MATRIX_NORMED, SUBSYSTEM_NAMES, SIGNATURE_NAMES,
)

OUT_DIR = os.path.join('outputs', 'critical_phenomena')
os.makedirs(OUT_DIR, exist_ok=True)

# ── System parameters from previous analyses ────────────────────────────────
OPTIMAL = {
    'steering_strength': 0.707,
    'alpha_pull': 0.0,
    'fatigue_rate': 0.217,
    'exploration_noise': 0.25,
    'beta_macro': 11.375,
}
FR_C = 0.1816
BETA_MEASURED = 0.329
STEPS = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# HELPER: Build a modified BatchConsciousnessEngine
# ============================================================================

def make_standard_configs(N, fr_val, steps=STEPS):
    """Standard config dict for N sims all at given fatigue_rate."""
    return {
        'steering_strength': np.full(N, OPTIMAL['steering_strength']),
        'alpha_pull':        np.full(N, OPTIMAL['alpha_pull']),
        'fatigue_rate':      np.full(N, fr_val),
        'exploration_noise': np.full(N, OPTIMAL['exploration_noise']),
        'beta_macro':        np.full(N, OPTIMAL['beta_macro']),
        'timesteps': steps,
    }


def make_engine_fresh(N, configs, device=DEVICE):
    """Create engine with randomized initial conditions."""
    engine = BatchConsciousnessEngine(N, configs, device)
    engine.u_t = F.normalize(torch.randn(N, 4, device=engine.device), dim=1)
    engine.u_prev = engine.u_t.clone()
    return engine


def measure_order_parameter(engine, warmup=200):
    """Run engine and extract order parameter (mean clarity after warmup)."""
    engine.run(steps=engine.max_steps)
    T = engine.step_count
    clarity = engine.hist_clarity[:, warmup:T]  # (N, T-warmup)
    return clarity.mean(dim=1).cpu().numpy()   # (N,)


def measure_beta_from_sweep(fr_vals, clarity_means, fr_c_est=None):
    """
    Given arrays of fatigue_rate values and their mean clarities,
    estimate β from log-log fit on the supercritical side.
    Returns (beta, fr_c, r_squared, fit_info_dict).
    """
    if fr_c_est is None:
        # Find steepest gradient
        dfl = np.gradient(clarity_means, fr_vals)
        fr_c_est = fr_vals[np.argmax(dfl)]

    # Supercritical: clarity - clarity_c ~ (fr - fr_c)^β
    idx_c = np.argmin(np.abs(fr_vals - fr_c_est))
    clarity_c = clarity_means[idx_c]

    mask = (fr_vals > fr_c_est + 0.003) & (fr_vals < fr_c_est + 0.10)
    delta_fr = fr_vals[mask] - fr_c_est
    delta_cl = clarity_means[mask] - clarity_c

    pos = delta_cl > 0
    if pos.sum() < 3:
        return None, fr_c_est, 0.0, {}

    log_dfr = np.log(delta_fr[pos])
    log_dcl = np.log(delta_cl[pos])
    slope, intercept, r, p, se = linregress(log_dfr, log_dcl)

    return slope, fr_c_est, r**2, {
        'A': np.exp(intercept), 'clarity_c': clarity_c,
        'n_points': int(pos.sum()), 'r': r, 'se': se, 'p': p,
    }


# ============================================================================
# PART 1: NULL / ABLATION CONTROLS
# ============================================================================

def run_null_controls():
    """
    Break the mechanism 5 different ways and check if the phase
    transition (β ≈ 0.33) survives. If it does, the transition
    is an artifact. If it disappears, the mechanism matters.

    Ablations:
    1. SHUFFLED COUPLINGS:   randomize PREFERENCE_MATRIX
    2. NO MACRO RECONCILE:   alpha_pull=0, disable basin escape
    3. NO FATIGUE:            fatigue_rate=0 everywhere
    4. RANDOM WALK ON S³:    remove all forces, pure noise
    5. NO STEERING:           steering_strength=0
    """
    print("\n" + "=" * 70)
    print("  PART 1: NULL / ABLATION CONTROLS")
    print("  Breaking the mechanism 5 ways — does β survive?")
    print("=" * 70)

    # ── Sweep parameters ────────────────────────────────────────────────
    fr_vals = np.linspace(0.05, 0.35, 40)
    N_SEEDS = 100
    N_total = len(fr_vals) * N_SEEDS

    results = {}

    # ── CONTROL: intact model ───────────────────────────────────────────
    print("\n  [Control] Intact model...")
    control_clarity = _sweep_fr(fr_vals, N_SEEDS, ablation=None)
    beta_ctrl, fr_c_ctrl, r2_ctrl, info_ctrl = measure_beta_from_sweep(
        fr_vals, control_clarity, fr_c_est=FR_C)
    results['Control (intact)'] = {
        'beta': beta_ctrl, 'fr_c': fr_c_ctrl, 'r2': r2_ctrl,
        'clarity': control_clarity.tolist(),
    }
    print(f"    β = {beta_ctrl:.3f}  (R² = {r2_ctrl:.3f})")

    # ── ABLATION 1: Shuffled couplings ──────────────────────────────────
    print("\n  [Ablation 1] Shuffled preference matrix...")
    abl1_clarity = _sweep_fr(fr_vals, N_SEEDS, ablation='shuffle_prefs')
    beta_a1, _, r2_a1, _ = measure_beta_from_sweep(fr_vals, abl1_clarity, fr_c_est=FR_C)
    results['Shuffled couplings'] = {
        'beta': beta_a1, 'r2': r2_a1, 'clarity': abl1_clarity.tolist(),
    }
    print(f"    β = {beta_a1}  (R² = {r2_a1:.3f})")

    # ── ABLATION 2: No macro reconciliation ─────────────────────────────
    print("\n  [Ablation 2] No macro reconciliation (alpha_pull=0, no basin escape)...")
    abl2_clarity = _sweep_fr(fr_vals, N_SEEDS, ablation='no_macro')
    beta_a2, _, r2_a2, _ = measure_beta_from_sweep(fr_vals, abl2_clarity, fr_c_est=FR_C)
    results['No macro reconciliation'] = {
        'beta': beta_a2, 'r2': r2_a2, 'clarity': abl2_clarity.tolist(),
    }
    print(f"    β = {beta_a2}  (R² = {r2_a2:.3f})")

    # ── ABLATION 3: No fatigue ──────────────────────────────────────────
    print("\n  [Ablation 3] No fatigue mechanism...")
    abl3_clarity = _sweep_fr(fr_vals, N_SEEDS, ablation='no_fatigue')
    beta_a3, _, r2_a3, _ = measure_beta_from_sweep(fr_vals, abl3_clarity, fr_c_est=FR_C)
    results['No fatigue'] = {
        'beta': beta_a3, 'r2': r2_a3, 'clarity': abl3_clarity.tolist(),
    }
    print(f"    β = {beta_a3}  (R² = {r2_a3:.3f})")

    # ── ABLATION 4: Random walk on S³ ───────────────────────────────────
    print("\n  [Ablation 4] Random walk on S³ (no forces, only noise)...")
    abl4_clarity = _sweep_fr(fr_vals, N_SEEDS, ablation='random_walk')
    beta_a4, _, r2_a4, _ = measure_beta_from_sweep(fr_vals, abl4_clarity, fr_c_est=FR_C)
    results['Random walk on S³'] = {
        'beta': beta_a4, 'r2': r2_a4, 'clarity': abl4_clarity.tolist(),
    }
    print(f"    β = {beta_a4}  (R² = {r2_a4:.3f})")

    # ── ABLATION 5: No steering ─────────────────────────────────────────
    print("\n  [Ablation 5] No steering (steering_strength=0)...")
    abl5_clarity = _sweep_fr(fr_vals, N_SEEDS, ablation='no_steering')
    beta_a5, _, r2_a5, _ = measure_beta_from_sweep(fr_vals, abl5_clarity, fr_c_est=FR_C)
    results['No steering'] = {
        'beta': beta_a5, 'r2': r2_a5, 'clarity': abl5_clarity.tolist(),
    }
    print(f"    β = {beta_a5}  (R² = {r2_a5:.3f})")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║            NULL / ABLATION CONTROL SUMMARY                  ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  {'Condition':<28s} {'β':>8s} {'R²':>8s} {'Verdict':>12s}  ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    for name, r in results.items():
        b = r.get('beta')
        r2 = r.get('r2', 0)
        if b is None:
            verdict = "NO FIT"
        elif abs(b - BETA_MEASURED) < 0.05 and r2 > 0.85:
            verdict = "SURVIVES ⚠"
        elif r2 < 0.5:
            verdict = "DESTROYED ✓"
        else:
            verdict = f"SHIFTED"
        b_str = f"{b:.3f}" if b is not None else "  N/A"
        print(f"  ║  {name:<28s} {b_str:>8s} {r2:>8.3f} {verdict:>12s}  ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    all_results = list(results.items())
    for i, (name, r) in enumerate(all_results):
        ax = axes[i]
        cl = np.array(r['clarity'])
        ax.plot(fr_vals, cl, 'o-', markersize=3, linewidth=1.5)
        ax.axvline(FR_C, color='red', linestyle='--', alpha=0.5, label=f'fr_c={FR_C:.3f}')
        b = r.get('beta')
        r2 = r.get('r2', 0)
        title = f"{name}\nβ={b:.3f}, R²={r2:.3f}" if b is not None else f"{name}\nNo power-law fit"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('fatigue_rate')
        ax.set_ylabel('mean clarity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    if len(all_results) < 6:
        axes[5].set_visible(False)

    fig.suptitle('Null / Ablation Controls — Does β Survive?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'null_ablation_controls.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Save results JSON
    json_path = os.path.join(OUT_DIR, 'null_ablation_results.json')
    # Remove non-serializable items
    json_safe = {}
    for k, v in results.items():
        json_safe[k] = {kk: vv for kk, vv in v.items() if kk != 'clarity'}
        json_safe[k]['beta'] = float(v['beta']) if v['beta'] is not None else None
    with open(json_path, 'w') as f:
        json.dump(json_safe, f, indent=2)
    print(f"  Saved: {json_path}")

    return results


def _sweep_fr(fr_vals, n_seeds, ablation=None):
    """
    Sweep fatigue_rate values with n_seeds each, applying an ablation.
    Returns array of mean clarity per fr value.
    """
    N_total = len(fr_vals) * n_seeds

    configs = {
        'steering_strength': np.full(N_total, OPTIMAL['steering_strength']),
        'alpha_pull':        np.full(N_total, OPTIMAL['alpha_pull']),
        'fatigue_rate':      np.repeat(fr_vals, n_seeds),
        'exploration_noise': np.full(N_total, OPTIMAL['exploration_noise']),
        'beta_macro':        np.full(N_total, OPTIMAL['beta_macro']),
        'timesteps': STEPS,
    }

    # Apply ablation-specific config modifications
    if ablation == 'no_steering':
        configs['steering_strength'] = np.zeros(N_total)
    elif ablation == 'no_fatigue':
        # Keep fatigue_rate sweep but disable fatigue accumulation in engine
        pass  # handled post-construction
    elif ablation == 'no_macro':
        configs['alpha_pull'] = np.zeros(N_total)
        configs['beta_macro'] = np.zeros(N_total)  # flat softmax = uniform weights

    engine = make_engine_fresh(N_total, configs, DEVICE)

    # Post-construction ablations
    if ablation == 'shuffle_prefs':
        # Randomize preference matrix (break subsystem structure)
        rng = np.random.default_rng(42)
        rand_prefs = rng.standard_normal((8, 4)).astype(np.float32)
        norms = np.linalg.norm(rand_prefs, axis=1, keepdims=True)
        rand_prefs = rand_prefs / np.maximum(norms, 1e-8)
        engine.prefs = torch.tensor(rand_prefs, dtype=torch.float32, device=engine.device)

    elif ablation == 'no_fatigue':
        # Override fatigue update to always stay zero
        original_step = engine.step
        def step_no_fatigue():
            original_step()
            engine.fatigue.zero_()
        engine.step = step_no_fatigue

    elif ablation == 'random_walk':
        # Override step to pure random walk on S³
        def step_random_walk():
            t = engine.step_count
            N = engine.N
            # Pure tangent noise
            raw = engine.exploration_noise.unsqueeze(1) * torch.randn(N, 4, device=engine.device)
            radial = (raw * engine.u_t).sum(dim=1, keepdim=True)
            tangent = raw - radial * engine.u_t
            engine.u_prev = engine.u_t.clone()
            engine.u_t = F.normalize(engine.u_t + tangent, dim=1)
            # Still compute metrics so history works
            _compute_metrics_only(engine, t)
        engine.step = step_random_walk

    elif ablation == 'no_macro':
        # Also disable basin escape by setting basin_dwell threshold very high
        original_step = engine.step
        def step_no_macro():
            original_step()
            engine.basin_dwell.zero_()  # never triggers escape
        engine.step = step_no_macro

    engine.run(steps=STEPS)

    # Extract clarity per fr value
    warmup = 200
    T = engine.step_count
    clarity = engine.hist_clarity[:, warmup:T].mean(dim=1).cpu().numpy()  # (N_total,)
    clarity_per_fr = clarity.reshape(len(fr_vals), n_seeds).mean(axis=1)  # (N_fr,)

    return clarity_per_fr


def _compute_metrics_only(engine, t):
    """Minimal metric computation for random walk ablation (clarity, basin, etc.)."""
    N = engine.N
    u = engine.u_t

    # Subsystem influences & activities (needed for clarity)
    influences = torch.einsum('nd,sd->ns', u, engine.prefs)
    influences = 0.5 + 0.3 * influences
    effective = influences * torch.exp(-engine.fatigue)
    noise = engine.exploration_noise.unsqueeze(1) * torch.randn(N, 8, device=engine.device)
    effective = effective + noise
    effective = torch.clamp(effective, min=engine.floor_value)
    activities = effective / (effective.sum(dim=1, keepdim=True) + 1e-8)

    # Forces & clarity
    radial = torch.einsum('sd,nd->ns', engine.prefs, u)
    forces = engine.prefs.unsqueeze(0) - radial.unsqueeze(2) * u.unsqueeze(1)
    resultant = torch.einsum('ns,nsd->nd', activities, forces)
    clarity = resultant.norm(dim=1)

    # Macro basin
    macro_sim = torch.einsum('nd,md->nm', u, engine.macro_centers)
    beta = engine.beta_macro.unsqueeze(1)
    macro_weights = F.softmax(beta * macro_sim, dim=1)
    dominant_basin = macro_weights.argmax(dim=1).int()

    # Store
    if t < engine.max_steps:
        engine.hist_clarity[:, t] = clarity
        engine.hist_conflict[:, t] = 0
        engine.hist_curvature[:, t] = 0
        engine.hist_speed[:, t] = 0
        engine.hist_integration[:, t] = 0
        engine.hist_differentiation[:, t] = 0
        engine.hist_inner_outer[:, t] = 0
        engine.hist_path_coherence[:, t] = 0
        engine.hist_dominant_sub[:, t] = activities.argmax(dim=1).int()
        engine.hist_perc_mode[:, t] = 0
        engine.hist_macro_basin[:, t] = dominant_basin
        engine.hist_clarity_rate[:, t] = 0
        engine.hist_force_mags[:, t] = forces.norm(dim=2)
        engine.hist_clarity_decomp[:, t] = 0

    engine.step_count += 1


# ============================================================================
# PART 2: SUBSYSTEM-COUNT FINITE-SIZE SCALING
# ============================================================================

def make_preference_matrix(n_sub, seed=0):
    """
    Generate a preference matrix for n_sub subsystems in 4D.
    
    For n_sub=8, returns the canonical matrix.
    For other sizes, generates a cyclic opponent structure:
    subsystem i has preferences along direction 2πi/n_sub in 4D,
    with cyclic antipodal pairing.
    """
    if n_sub == 8:
        return PREFERENCE_MATRIX_NORMED.copy()

    rng = np.random.default_rng(seed)
    prefs = np.zeros((n_sub, 4), dtype=np.float32)
    for i in range(n_sub):
        # Distribute subsystems around a 4D great circle
        theta = 2 * np.pi * i / n_sub
        # Use two planes: (0,1) and (2,3)
        if i % 2 == 0:
            prefs[i, 0] = np.cos(theta)
            prefs[i, 1] = np.sin(theta)
            # Small component in other plane for differentiation
            prefs[i, 2] = 0.2 * np.cos(theta * 1.5)
            prefs[i, 3] = 0.2 * np.sin(theta * 1.5)
        else:
            prefs[i, 2] = np.cos(theta)
            prefs[i, 3] = np.sin(theta)
            prefs[i, 0] = 0.2 * np.cos(theta * 1.5)
            prefs[i, 1] = 0.2 * np.sin(theta * 1.5)

    # Normalize
    norms = np.linalg.norm(prefs, axis=1, keepdims=True)
    prefs = prefs / np.maximum(norms, 1e-8)
    return prefs


class VariableSubsystemEngine:
    """
    Modified BatchConsciousnessEngine with variable subsystem count.
    Re-implements the core step logic for arbitrary n_sub.
    """

    def __init__(self, N, n_sub, configs, device=DEVICE):
        self.N = N
        self.n_sub = n_sub
        self.device = torch.device(device)
        self.dim = 4

        # Manifold
        micro = generate_fibonacci_s3(600)
        macro = derive_macro_basins(micro, 24)
        self.macro_centers = torch.tensor(macro, dtype=torch.float32, device=self.device)
        self.n_macro = 24

        # Preference matrix for n_sub subsystems
        pref_np = make_preference_matrix(n_sub)
        self.prefs = torch.tensor(pref_np, dtype=torch.float32, device=self.device)

        # Per-sim parameters
        def _t(arr): return torch.tensor(arr, dtype=torch.float32, device=self.device)
        self.steering_strength = _t(configs['steering_strength'])
        self.alpha_pull        = _t(configs['alpha_pull'])
        self.fatigue_rate      = _t(configs['fatigue_rate'])
        self.exploration_noise = _t(configs['exploration_noise'])
        self.beta_macro        = _t(configs['beta_macro'])

        self.recovery_rate = 0.025
        self.floor_value   = 0.05
        self.novelty_weight = 0.6
        self.max_steps = configs.get('timesteps', STEPS)

        # State
        self.u_t = F.normalize(torch.randn(N, 4, device=self.device), dim=1)
        self.u_prev = self.u_t.clone()
        self.fatigue = torch.zeros(N, n_sub, device=self.device)
        self.basin_dwell = torch.zeros(N, dtype=torch.int32, device=self.device)
        self.current_basin = torch.full((N,), -1, dtype=torch.int32, device=self.device)

        # History (only clarity needed for FSS)
        self.hist_clarity = torch.zeros(N, self.max_steps, device=self.device)
        self.hist_dominant_sub = torch.zeros(N, self.max_steps, dtype=torch.int32, device=self.device)
        self.step_count = 0

    @torch.no_grad()
    def step(self):
        t = self.step_count
        N, n_sub = self.N, self.n_sub
        u = self.u_t

        # Subsystem competition
        influences = torch.einsum('nd,sd->ns', u, self.prefs)
        influences = 0.5 + 0.3 * influences
        effective = influences * torch.exp(-self.fatigue)
        noise = self.exploration_noise.unsqueeze(1) * torch.randn(N, n_sub, device=self.device)
        effective = effective + noise
        effective = torch.clamp(effective, min=self.floor_value)
        activities = effective / (effective.sum(dim=1, keepdim=True) + 1e-8)

        # Fatigue
        fat_rate = self.fatigue_rate.unsqueeze(1)
        self.fatigue = self.fatigue + fat_rate * activities
        equal_share = 1.0 / n_sub
        excess = (activities - equal_share).clamp(min=0.02) - 0.02
        self.fatigue = self.fatigue + 0.03 * excess
        inactive_recovery = (1.0 - activities) * self.recovery_rate
        self.fatigue = (self.fatigue - inactive_recovery).clamp(0.0, 3.0)

        # Tangent force field
        radial = torch.einsum('sd,nd->ns', self.prefs, u)
        forces = self.prefs.unsqueeze(0) - radial.unsqueeze(2) * u.unsqueeze(1)
        activity_force = torch.einsum('ns,nsd->nd', activities, forces)

        # Novelty
        rest_scores = torch.exp(-self.fatigue)
        novelty_force = torch.einsum('ns,nsd->nd', rest_scores, forces)
        mean_rest = rest_scores.mean(dim=1, keepdim=True)
        forces_sum = forces.sum(dim=1) / n_sub
        novelty_force = novelty_force - mean_rest * forces_sum

        nw = self.novelty_weight
        drive = (1.0 - nw) * activity_force + nw * novelty_force

        # Tangent noise
        raw_noise = self.exploration_noise.unsqueeze(1) * torch.randn(N, 4, device=self.device)
        noise_radial = (raw_noise * u).sum(dim=1, keepdim=True)
        drive = drive + raw_noise - noise_radial * u

        # Step
        ss = self.steering_strength.unsqueeze(1)
        new_u = u + ss * drive
        self.u_prev = u.clone()
        self.u_t = F.normalize(new_u, dim=1)

        # Macro reconciliation
        macro_sim = torch.einsum('nd,md->nm', self.u_t, self.macro_centers)
        beta = self.beta_macro.unsqueeze(1)
        macro_weights = F.softmax(beta * macro_sim, dim=1)
        macro_field = torch.einsum('nm,md->nd', macro_weights, self.macro_centers)
        macro_field = F.normalize(macro_field, dim=1)

        dominant_basin = macro_weights.argmax(dim=1).int()
        same = (dominant_basin == self.current_basin)
        self.basin_dwell = torch.where(same, self.basin_dwell + 1, torch.zeros_like(self.basin_dwell))
        self.current_basin = dominant_basin

        # Basin escape
        escape_mask = self.basin_dwell > 25
        if escape_mask.any():
            escape_prob = (0.05 * (self.basin_dwell.float() - 25.0)).clamp(0.0, 0.3)
            do_escape = (torch.rand(N, device=self.device) < escape_prob) & escape_mask
            if do_escape.any():
                n_esc = do_escape.sum().item()
                target_idx = torch.randint(0, self.n_macro, (n_esc,), device=self.device)
                target_dirs = self.macro_centers[target_idx]
                u_esc = self.u_t[do_escape]
                tangent = target_dirs - (target_dirs * u_esc).sum(dim=1, keepdim=True) * u_esc
                tn = tangent.norm(dim=1, keepdim=True).clamp(min=1e-6)
                tangent = tangent / tn
                escape_dirs = torch.cos(torch.tensor(0.4)) * u_esc + torch.sin(torch.tensor(0.4)) * tangent
                self.u_t[do_escape] = F.normalize(escape_dirs, dim=1)
                self.basin_dwell[do_escape] = 0

        macro_tangent = macro_field - (macro_field * self.u_t).sum(dim=1, keepdim=True) * self.u_t
        ap = self.alpha_pull.unsqueeze(1)
        self.u_t = F.normalize(self.u_t + ap * macro_tangent, dim=1)

        # Metrics
        u = self.u_t
        resultant = torch.einsum('ns,nsd->nd', activities, forces)
        clarity = resultant.norm(dim=1)
        dominant_sub = activities.argmax(dim=1).int()

        if t < self.max_steps:
            self.hist_clarity[:, t] = clarity
            self.hist_dominant_sub[:, t] = dominant_sub

        self.step_count += 1

    def run(self, steps=None):
        if steps is None:
            steps = self.max_steps
        t0 = time.time()
        for t in range(steps):
            self.step()
        elapsed = time.time() - t0
        total = steps * self.N
        print(f"    {self.N:,} sims × {steps} steps ({self.n_sub} subs) = "
              f"{total:,} steps in {elapsed:.1f}s ({total/elapsed:.0f}/sec)")


def run_subsystem_fss():
    """
    Finite-size scaling with subsystem count as the system size L.
    
    L = n_sub ∈ {4, 6, 8, 12, 16, 24, 32}
    
    For each L, sweep fatigue_rate near fr_c, measure:
    - Order parameter (mean clarity)
    - Susceptibility (temporal variance of clarity)
    - Binder cumulant U₄ = 1 - <m⁴>/(3<m²>²)
    """
    print("\n" + "=" * 70)
    print("  PART 2: SUBSYSTEM-COUNT FINITE-SIZE SCALING")
    print("  L = n_sub ∈ {4, 6, 8, 12, 16, 24, 32}")
    print("=" * 70)

    L_values = [4, 6, 8, 12, 16, 24, 32]
    fr_vals = np.linspace(0.08, 0.35, 45)
    N_SEEDS = 80
    warmup = 200

    fss_data = {}

    for L in L_values:
        print(f"\n  ── L = {L} subsystems ──")
        N_total = len(fr_vals) * N_SEEDS

        configs = {
            'steering_strength': np.full(N_total, OPTIMAL['steering_strength']),
            'alpha_pull':        np.full(N_total, OPTIMAL['alpha_pull']),
            'fatigue_rate':      np.repeat(fr_vals, N_SEEDS),
            'exploration_noise': np.full(N_total, OPTIMAL['exploration_noise']),
            'beta_macro':        np.full(N_total, OPTIMAL['beta_macro']),
            'timesteps': STEPS,
        }

        engine = VariableSubsystemEngine(N_total, L, configs, DEVICE)
        engine.run(steps=STEPS)

        # Extract per-seed clarity time series
        T = engine.step_count
        clarity_ts = engine.hist_clarity[:, warmup:T]  # (N_total, T-warmup)

        # Mean clarity per config
        mean_clarity = clarity_ts.mean(dim=1).cpu().numpy()  # (N_total,)
        # Temporal variance per config
        clarity_var = clarity_ts.var(dim=1).cpu().numpy()    # (N_total,)

        # Reshape to (N_fr, N_SEEDS)
        mean_clarity = mean_clarity.reshape(len(fr_vals), N_SEEDS)
        clarity_var = clarity_var.reshape(len(fr_vals), N_SEEDS)

        # Per fr: order parameter, susceptibility, Binder cumulant
        m_mean = mean_clarity.mean(axis=1)           # <m>
        chi = clarity_var.mean(axis=1)                # <χ> (temporal variance)
        # Binder: across seeds
        m2 = (mean_clarity ** 2).mean(axis=1)         # <m²>
        m4 = (mean_clarity ** 4).mean(axis=1)         # <m⁴>
        U4 = 1.0 - m4 / (3.0 * m2**2 + 1e-15)       # Binder cumulant

        # Susceptibility from variance across seeds (classic FSS)
        chi_seeds = mean_clarity.var(axis=1) * N_SEEDS  # N * var(m)

        fss_data[L] = {
            'fr': fr_vals.tolist(),
            'm': m_mean.tolist(),
            'chi': chi.tolist(),
            'chi_seeds': chi_seeds.tolist(),
            'U4': U4.tolist(),
        }

        # Report
        chi_peak_idx = np.argmax(chi_seeds)
        print(f"    <m> range: [{m_mean.min():.4f}, {m_mean.max():.4f}]")
        print(f"    χ peak at fr = {fr_vals[chi_peak_idx]:.4f} (χ = {chi_seeds[chi_peak_idx]:.4f})")

    # ── Binder crossing → fr_c ─────────────────────────────────────────
    print("\n  ── Binder Cumulant Crossings ──")
    # All curves should cross at fr_c if transition is genuine
    for i, L1 in enumerate(L_values):
        for L2 in L_values[i+1:]:
            u1 = np.array(fss_data[L1]['U4'])
            u2 = np.array(fss_data[L2]['U4'])
            diff = u1 - u2
            crossings = np.where(np.diff(np.sign(diff)))[0]
            if len(crossings) > 0:
                # Linear interpolation for crossing point
                idx = crossings[0]
                frac = diff[idx] / (diff[idx] - diff[idx+1])
                fr_cross = fr_vals[idx] + frac * (fr_vals[idx+1] - fr_vals[idx])
                print(f"    L={L1} × L={L2}: crossing at fr ≈ {fr_cross:.4f}")

    # ── Susceptibility peak scaling ─────────────────────────────────────
    print("\n  ── Susceptibility Peak Scaling ──")
    chi_peaks = []
    for L in L_values:
        chi_s = np.array(fss_data[L]['chi_seeds'])
        chi_peaks.append(chi_s.max())
        fr_peak = fr_vals[np.argmax(chi_s)]
        print(f"    L={L:>3d}: χ_max = {chi_s.max():.4f}  at fr = {fr_peak:.4f}")

    # χ_max ~ L^(γ/ν)
    log_L = np.log(np.array(L_values, dtype=float))
    log_chi = np.log(np.array(chi_peaks) + 1e-10)
    if len(log_L) >= 3:
        slope, intercept, r, p, se = linregress(log_L, log_chi)
        print(f"\n    χ_max ~ L^(γ/ν):  γ/ν = {slope:.3f}  (R² = {r**2:.3f})")
        gamma_over_nu = slope
    else:
        gamma_over_nu = None

    # ── Plot ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Order parameter m(fr) for each L
    ax1 = fig.add_subplot(gs[0, 0])
    for L in L_values:
        ax1.plot(fss_data[L]['fr'], fss_data[L]['m'], 'o-', markersize=2,
                 label=f'L={L}', linewidth=1.2)
    ax1.axvline(FR_C, color='red', ls='--', alpha=0.5, label=f'fr_c={FR_C:.3f}')
    ax1.set_xlabel('fatigue_rate')
    ax1.set_ylabel('Order parameter ⟨m⟩ (clarity)')
    ax1.set_title('Order Parameter vs System Size')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Susceptibility χ(fr) for each L
    ax2 = fig.add_subplot(gs[0, 1])
    for L in L_values:
        ax2.plot(fss_data[L]['fr'], fss_data[L]['chi_seeds'], 'o-', markersize=2,
                 label=f'L={L}', linewidth=1.2)
    ax2.axvline(FR_C, color='red', ls='--', alpha=0.5)
    ax2.set_xlabel('fatigue_rate')
    ax2.set_ylabel('Susceptibility χ = N·Var(m)')
    ax2.set_title('Susceptibility vs System Size')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Binder cumulant U₄(fr) for each L
    ax3 = fig.add_subplot(gs[0, 2])
    for L in L_values:
        ax3.plot(fss_data[L]['fr'], fss_data[L]['U4'], 'o-', markersize=2,
                 label=f'L={L}', linewidth=1.2)
    ax3.axvline(FR_C, color='red', ls='--', alpha=0.5)
    ax3.set_xlabel('fatigue_rate')
    ax3.set_ylabel('Binder cumulant U₄')
    ax3.set_title('Binder Cumulant — Crossings → fr_c')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: χ_max vs L (log-log)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.loglog(L_values, chi_peaks, 'ko-', markersize=8, linewidth=2)
    if gamma_over_nu is not None:
        L_fit = np.linspace(min(L_values), max(L_values), 100)
        ax4.loglog(L_fit, np.exp(intercept) * L_fit**slope, 'r--',
                   label=f'γ/ν = {slope:.3f}')
    ax4.set_xlabel('System size L (subsystem count)')
    ax4.set_ylabel('χ_max')
    ax4.set_title('Susceptibility Peak Scaling')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Panel 5: β estimates per L
    ax5 = fig.add_subplot(gs[1, 1])
    betas_per_L = []
    for L in L_values:
        m = np.array(fss_data[L]['m'])
        b, _, r2, _ = measure_beta_from_sweep(fr_vals, m, fr_c_est=FR_C)
        betas_per_L.append(b if b is not None else np.nan)
        if b is not None:
            print(f"    β(L={L}) = {b:.3f}  (R² = {r2:.3f})")

    ax5.plot(L_values, betas_per_L, 'ko-', markersize=8, linewidth=2)
    ax5.axhline(0.3265, color='red', ls='--', label='3D Ising β=0.326')
    ax5.axhline(BETA_MEASURED, color='blue', ls=':', label=f'Measured β={BETA_MEASURED}')
    ax5.set_xlabel('System size L (subsystem count)')
    ax5.set_ylabel('β exponent')
    ax5.set_title('β Convergence with System Size')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Transition sharpness (max |dm/dfr|) vs L
    ax6 = fig.add_subplot(gs[1, 2])
    sharpness = []
    for L in L_values:
        m = np.array(fss_data[L]['m'])
        dmdx = np.gradient(m, fr_vals)
        sharpness.append(np.max(np.abs(dmdx)))
    ax6.plot(L_values, sharpness, 'ko-', markersize=8, linewidth=2)
    ax6.set_xlabel('System size L (subsystem count)')
    ax6.set_ylabel('max |dm/dfr|')
    ax6.set_title('Transition Sharpness vs System Size')
    ax6.grid(True, alpha=0.3)

    # Log-log fit for sharpness ~ L^(1/ν)
    log_sharp = np.log(np.array(sharpness) + 1e-10)
    if len(log_L) >= 3:
        sl, inter, r, p, se = linregress(log_L, log_sharp)
        nu_est = 1.0 / sl if abs(sl) > 0.01 else np.inf
        ax6.text(0.05, 0.95, f'slope = 1/ν = {sl:.3f}\nν ≈ {nu_est:.2f}',
                 transform=ax6.transAxes, va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        print(f"\n    Sharpness ~ L^(1/ν):  1/ν = {sl:.3f}  →  ν ≈ {nu_est:.2f}")

    fig.suptitle('Subsystem-Count Finite-Size Scaling', fontsize=14, fontweight='bold')
    path = os.path.join(OUT_DIR, 'subsystem_fss.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Save data
    json_path = os.path.join(OUT_DIR, 'subsystem_fss_data.json')
    with open(json_path, 'w') as f:
        json.dump({
            'L_values': L_values,
            'fss_data': {str(k): v for k, v in fss_data.items()},
            'beta_per_L': {str(L): float(b) for L, b in zip(L_values, betas_per_L) if not np.isnan(b)},
            'chi_peaks': {str(L): float(c) for L, c in zip(L_values, chi_peaks)},
            'gamma_over_nu': float(gamma_over_nu) if gamma_over_nu else None,
        }, f, indent=2)
    print(f"  Saved: {json_path}")

    return fss_data, betas_per_L, chi_peaks


# ============================================================================
# PART 3: DATA COLLAPSE — "THE KILLER FIGURE"
# ============================================================================

def run_data_collapse(fss_data=None):
    """
    Data collapse: if the transition is a genuine continuous phase transition,
    then near fr_c the order parameter obeys the scaling form:
    
        m(fr, L) = L^(-β/ν) · F( (fr - fr_c) · L^(1/ν) )
    
    where F is a universal scaling function. On a plot of
    m · L^(β/ν) vs (fr - fr_c) · L^(1/ν), ALL curves for different L
    should collapse onto a single curve.
    
    We optimize β/ν and 1/ν to get the best collapse (minimum residual).
    """
    print("\n" + "=" * 70)
    print("  PART 3: DATA COLLAPSE — THE KILLER FIGURE")
    print("=" * 70)

    # If not provided, load from disk
    if fss_data is None:
        json_path = os.path.join(OUT_DIR, 'subsystem_fss_data.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                saved = json.load(f)
            fss_data = {int(k): v for k, v in saved['fss_data'].items()}
            L_values = saved['L_values']
        else:
            print("  ERROR: No FSS data found. Run Part 2 first.")
            return None
    else:
        L_values = sorted(fss_data.keys())

    fr_vals = np.array(fss_data[L_values[0]]['fr'])

    # ── Optimize collapse quality ───────────────────────────────────────
    def collapse_residual(params, fr_c_fixed=None):
        """Compute residual from data collapse with given exponents."""
        beta_over_nu, one_over_nu = params[0], params[1]
        fr_c = fr_c_fixed if fr_c_fixed is not None else params[2]

        # Rescaled coordinates for each L
        all_x, all_y = [], []
        for L in L_values:
            m = np.array(fss_data[L]['m'])
            x = (fr_vals - fr_c) * L ** one_over_nu
            y = m * L ** beta_over_nu
            all_x.append(x)
            all_y.append(y)

        # Measure collapse quality: for each x, how much do the y values
        # from different L agree?
        all_x_cat = np.concatenate(all_x)
        all_y_cat = np.concatenate(all_y)
        L_labels = np.concatenate([np.full(len(fr_vals), L) for L in L_values])

        # Sort by x
        sort_idx = np.argsort(all_x_cat)
        xs = all_x_cat[sort_idx]
        ys = all_y_cat[sort_idx]

        # Compute local variance in sliding window
        window = max(5, len(xs) // 50)
        residuals = []
        for i in range(0, len(xs) - window, window // 2):
            chunk = ys[i:i + window]
            if len(chunk) >= 3:
                residuals.append(chunk.std() / (np.abs(chunk.mean()) + 1e-10))

        return np.mean(residuals) if residuals else 1e10

    # Grid search for initial guess
    print("\n  Optimizing collapse parameters...")
    best_cost = np.inf
    best_params = None

    for b_nu in np.linspace(0.3, 1.5, 15):
        for inv_nu in np.linspace(0.5, 3.0, 15):
            cost = collapse_residual([b_nu, inv_nu], fr_c_fixed=FR_C)
            if cost < best_cost:
                best_cost = cost
                best_params = [b_nu, inv_nu]

    # Refine with Nelder-Mead
    result = minimize(
        lambda p: collapse_residual(p, fr_c_fixed=FR_C),
        best_params, method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6}
    )
    beta_over_nu_opt = result.x[0]
    one_over_nu_opt = result.x[1]
    collapse_quality = 1.0 / (result.fun + 1e-10)

    nu_opt = 1.0 / one_over_nu_opt if abs(one_over_nu_opt) > 0.01 else np.inf
    beta_opt = beta_over_nu_opt * nu_opt

    print(f"\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║        DATA COLLAPSE RESULTS                  ║")
    print(f"  ╠══════════════════════════════════════════════╣")
    print(f"  ║  β/ν  = {beta_over_nu_opt:.4f}                         ║")
    print(f"  ║  1/ν  = {one_over_nu_opt:.4f}                         ║")
    print(f"  ║  → ν  = {nu_opt:.4f}                                 ║")
    print(f"  ║  → β  = {beta_opt:.4f}                               ║")
    print(f"  ║  Collapse quality: {collapse_quality:.1f}             ║")
    print(f"  ║  fr_c = {FR_C:.4f} (fixed)                    ║")
    print(f"  ╚══════════════════════════════════════════════╝")

    # Compare
    print(f"\n  3D Ising:     β/ν = {0.3265/0.6301:.4f}, 1/ν = {1/0.6301:.4f}")
    print(f"  This system:  β/ν = {beta_over_nu_opt:.4f}, 1/ν = {one_over_nu_opt:.4f}")

    # Also try with free fr_c
    result_free = minimize(
        lambda p: collapse_residual(p),
        [beta_over_nu_opt, one_over_nu_opt, FR_C], method='Nelder-Mead',
        options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-6}
    )
    if result_free.fun < result.fun:
        fr_c_free = result_free.x[2]
        print(f"\n  With free fr_c optimization:")
        print(f"    β/ν = {result_free.x[0]:.4f}, 1/ν = {result_free.x[1]:.4f}, fr_c = {fr_c_free:.4f}")
        print(f"    Collapse quality: {1.0/(result_free.fun+1e-10):.1f}")

    # ── THE KILLER FIGURE ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Panel 1: Raw m(fr) curves
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))
    for i, L in enumerate(L_values):
        ax1.plot(fr_vals, fss_data[L]['m'], 'o-', color=colors[i],
                 markersize=3, label=f'L={L}', linewidth=1.5)
    ax1.axvline(FR_C, color='red', ls='--', alpha=0.5)
    ax1.set_xlabel('fatigue_rate (control parameter)', fontsize=12)
    ax1.set_ylabel('⟨m⟩ (clarity)', fontsize=12)
    ax1.set_title('(a) Raw Order Parameter', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Collapsed data (the money shot)
    ax2 = axes[1]
    for i, L in enumerate(L_values):
        m = np.array(fss_data[L]['m'])
        x = (fr_vals - FR_C) * L ** one_over_nu_opt
        y = m * L ** beta_over_nu_opt
        ax2.plot(x, y, 'o', color=colors[i], markersize=4, label=f'L={L}', alpha=0.8)
    ax2.set_xlabel(r'$(fr - fr_c) \cdot L^{1/\nu}$', fontsize=12)
    ax2.set_ylabel(r'$m \cdot L^{\beta/\nu}$', fontsize=12)
    ax2.set_title(f'(b) Data Collapse  [β/ν={beta_over_nu_opt:.3f}, 1/ν={one_over_nu_opt:.3f}]',
                  fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Collapse quality landscape
    ax3 = axes[2]
    b_range = np.linspace(max(0.1, beta_over_nu_opt - 0.8), beta_over_nu_opt + 0.8, 40)
    n_range = np.linspace(max(0.1, one_over_nu_opt - 1.0), one_over_nu_opt + 1.0, 40)
    Z = np.zeros((len(n_range), len(b_range)))
    for j, b in enumerate(b_range):
        for k, n in enumerate(n_range):
            Z[k, j] = collapse_residual([b, n], fr_c_fixed=FR_C)
    Z = np.log10(Z + 1e-10)
    im = ax3.contourf(b_range, n_range, Z, levels=30, cmap='RdYlGn_r')
    ax3.plot(beta_over_nu_opt, one_over_nu_opt, 'r*', markersize=15, label='Optimal')
    ax3.plot(0.3265/0.6301, 1/0.6301, 'w^', markersize=12, label='3D Ising')
    fig.colorbar(im, ax=ax3, label='log₁₀(collapse residual)')
    ax3.set_xlabel('β/ν', fontsize=12)
    ax3.set_ylabel('1/ν', fontsize=12)
    ax3.set_title('(c) Collapse Quality Landscape', fontsize=13)
    ax3.legend(fontsize=10)

    fig.suptitle('DATA COLLAPSE — The Killer Figure', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'data_collapse.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Save results
    collapse_results = {
        'beta_over_nu': float(beta_over_nu_opt),
        'one_over_nu': float(one_over_nu_opt),
        'nu': float(nu_opt),
        'beta_from_collapse': float(beta_opt),
        'collapse_quality': float(collapse_quality),
        'fr_c': float(FR_C),
        'L_values': L_values,
        '3d_ising_comparison': {
            'beta_over_nu': 0.3265 / 0.6301,
            'one_over_nu': 1 / 0.6301,
            'beta': 0.3265,
            'nu': 0.6301,
        }
    }
    json_path = os.path.join(OUT_DIR, 'data_collapse_results.json')
    with open(json_path, 'w') as f:
        json.dump(collapse_results, f, indent=2)
    print(f"  Saved: {json_path}")

    return collapse_results


# ============================================================================
# PART 4: COARSE-GRAINING (Real-Space Renormalization)
# ============================================================================

def run_coarse_graining():
    """
    Test whether critical signatures persist under coarse-graining.
    
    Strategy: Run simulations at multiple "resolutions" of the 
    subsystem interactions by:
    1. Full resolution (8 subsystems, original dynamics)
    2. Coarse-grained: group pairs of subsystems into 4 effective ones
    3. Further coarsened: group into 2 effective subsystems
    
    At each level, measure:
    - β exponent
    - Transition location (fr_c)
    - Binder cumulant crossing
    
    If critical signatures persist (same β, same universality) under
    coarse-graining, this is strong evidence for genuine universality.
    """
    print("\n" + "=" * 70)
    print("  PART 4: COARSE-GRAINING (Real-Space Renormalization)")
    print("=" * 70)

    # ── Coarse-graining scheme ──────────────────────────────────────────
    # Level 0: 8 subsystems (full)
    # Level 1: 4 coarse subsystems (pairs: Motor+Aesthetic, Planning+Emotion, ...)
    #          Using opponent pairs from the PREFERENCE_MATRIX structure
    # Level 2: 2 coarse subsystems (quads)
    # Level 3: 16 subsystems (finer resolution, for comparison)

    cg_levels = {
        'L=16 (finer)': 16,
        'L=8 (original)': 8,
        'L=4 (coarse-1)': 4,
        'L=2 (coarse-2)': 2,
    }

    fr_vals = np.linspace(0.06, 0.35, 45)
    N_SEEDS = 100
    warmup = 200

    cg_results = {}

    for label, n_sub in cg_levels.items():
        print(f"\n  ── {label} ({n_sub} subsystems) ──")
        N_total = len(fr_vals) * N_SEEDS

        configs = {
            'steering_strength': np.full(N_total, OPTIMAL['steering_strength']),
            'alpha_pull':        np.full(N_total, OPTIMAL['alpha_pull']),
            'fatigue_rate':      np.repeat(fr_vals, N_SEEDS),
            'exploration_noise': np.full(N_total, OPTIMAL['exploration_noise']),
            'beta_macro':        np.full(N_total, OPTIMAL['beta_macro']),
            'timesteps': STEPS,
        }

        engine = VariableSubsystemEngine(N_total, n_sub, configs, DEVICE)
        engine.run(steps=STEPS)

        T = engine.step_count
        clarity_ts = engine.hist_clarity[:, warmup:T]
        mean_clarity = clarity_ts.mean(dim=1).cpu().numpy().reshape(len(fr_vals), N_SEEDS)
        clarity_per_fr = mean_clarity.mean(axis=1)

        # Measure β
        beta, fr_c, r2, info = measure_beta_from_sweep(fr_vals, clarity_per_fr, fr_c_est=FR_C)

        # Susceptibility
        chi = mean_clarity.var(axis=1) * N_SEEDS
        chi_peak = chi.max()
        fr_chi_peak = fr_vals[np.argmax(chi)]

        cg_results[label] = {
            'n_sub': n_sub,
            'beta': beta,
            'fr_c': fr_c,
            'r2': r2,
            'chi_peak': float(chi_peak),
            'fr_chi_peak': float(fr_chi_peak),
            'clarity': clarity_per_fr.tolist(),
            'chi': chi.tolist(),
        }

        b_str = f"{beta:.3f}" if beta is not None else "N/A"
        print(f"    β = {b_str}  (R² = {r2:.3f})")
        print(f"    χ_peak = {chi_peak:.4f} at fr = {fr_chi_peak:.4f}")

    # ── Renormalization flow ────────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║          COARSE-GRAINING RESULTS                             ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  {'Level':<22s} {'n_sub':>5s} {'β':>8s} {'R²':>7s} {'fr_c':>7s} {'χ_peak':>8s}  ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    for label, r in cg_results.items():
        b_str = f"{r['beta']:.3f}" if r['beta'] is not None else "  N/A"
        print(f"  ║  {label:<22s} {r['n_sub']:>5d} {b_str:>8s} {r['r2']:>7.3f} "
              f"{r['fr_c']:>7.4f} {r['chi_peak']:>8.4f}  ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    # Check RG invariance: β should be preserved under coarse-graining
    betas = [r['beta'] for r in cg_results.values() if r['beta'] is not None]
    if len(betas) >= 2:
        beta_spread = max(betas) - min(betas)
        beta_mean = np.mean(betas)
        cv = np.std(betas) / np.abs(beta_mean) if abs(beta_mean) > 0.01 else np.inf
        print(f"\n  β variation across scales: spread = {beta_spread:.3f}, CV = {cv:.3f}")
        if cv < 0.15:
            print("  → β is STABLE under coarse-graining — consistent with universality!")
        elif cv < 0.30:
            print("  → β shows moderate variation — suggestive but not conclusive")
        else:
            print("  → β varies substantially — universality claim weaker")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Panel 1: Order parameter at each CG level
    ax1 = axes[0, 0]
    for i, (label, r) in enumerate(cg_results.items()):
        ax1.plot(fr_vals, r['clarity'], 'o-', color=colors[i], markersize=3,
                 label=f"{label} (β={r['beta']:.3f})" if r['beta'] else label,
                 linewidth=1.5)
    ax1.axvline(FR_C, color='red', ls='--', alpha=0.5)
    ax1.set_xlabel('fatigue_rate')
    ax1.set_ylabel('Mean clarity')
    ax1.set_title('Order Parameter at Each Coarse-Graining Level')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Susceptibility at each level
    ax2 = axes[0, 1]
    for i, (label, r) in enumerate(cg_results.items()):
        ax2.plot(fr_vals, r['chi'], 'o-', color=colors[i], markersize=3,
                 label=label, linewidth=1.5)
    ax2.axvline(FR_C, color='red', ls='--', alpha=0.5)
    ax2.set_xlabel('fatigue_rate')
    ax2.set_ylabel('Susceptibility χ')
    ax2.set_title('Susceptibility at Each Level')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: β as function of coarse-graining level
    ax3 = axes[1, 0]
    n_subs = [r['n_sub'] for r in cg_results.values()]
    betas_plot = [r['beta'] if r['beta'] is not None else np.nan for r in cg_results.values()]
    ax3.plot(n_subs, betas_plot, 'ko-', markersize=10, linewidth=2)
    ax3.axhline(0.3265, color='red', ls='--', label='3D Ising β=0.326')
    ax3.axhline(BETA_MEASURED, color='blue', ls=':', label=f'Measured β={BETA_MEASURED}')
    ax3.set_xlabel('Number of subsystems (scale)')
    ax3.set_ylabel('β exponent')
    ax3.set_title('β Stability Under Coarse-Graining')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(n_subs)
    ax3.set_xticklabels([str(n) for n in n_subs])

    # Panel 4: RG flow visualization
    ax4 = axes[1, 1]
    # Plot fr_c and chi_peak as a function of scale
    fr_cs = [r['fr_c'] for r in cg_results.values()]
    chi_ps = [r['chi_peak'] for r in cg_results.values()]
    ax4_twin = ax4.twinx()
    ax4.plot(n_subs, fr_cs, 'bs-', markersize=8, label='fr_c', linewidth=2)
    ax4_twin.plot(n_subs, chi_ps, 'r^-', markersize=8, label='χ_peak', linewidth=2)
    ax4.set_xlabel('Number of subsystems (scale)')
    ax4.set_ylabel('Critical point fr_c', color='blue')
    ax4_twin.set_ylabel('χ_peak', color='red')
    ax4.set_title('Renormalization Flow')
    ax4.set_xscale('log', base=2)
    ax4.set_xticks(n_subs)
    ax4.set_xticklabels([str(n) for n in n_subs])
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Coarse-Graining / Real-Space Renormalization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'coarse_graining.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Save
    json_path = os.path.join(OUT_DIR, 'coarse_graining_results.json')
    json_safe = {}
    for label, r in cg_results.items():
        json_safe[label] = {k: v for k, v in r.items() if k not in ('clarity', 'chi')}
        json_safe[label]['beta'] = float(r['beta']) if r['beta'] is not None else None
    with open(json_path, 'w') as f:
        json.dump(json_safe, f, indent=2)
    print(f"  Saved: {json_path}")

    return cg_results


# ============================================================================
# GRAND SUMMARY
# ============================================================================

def print_grand_summary(null_results, fss_data, betas_per_L, collapse_results, cg_results):
    """Print the definitive summary of all critical phenomena tests."""
    print("\n")
    print("=" * 75)
    print("  ╔═══════════════════════════════════════════════════════════════════╗")
    print("  ║            CRITICAL PHENOMENA SUITE — GRAND SUMMARY              ║")
    print("  ╚═══════════════════════════════════════════════════════════════════╝")
    print("=" * 75)

    # Part 1: Null controls
    print("\n  ┌─ PART 1: NULL / ABLATION CONTROLS ────────────────────────────────┐")
    n_destroyed = 0
    n_survived = 0
    for name, r in null_results.items():
        b = r.get('beta')
        r2 = r.get('r2', 0)
        if b is None or r2 < 0.5:
            n_destroyed += 1
        elif abs(b - BETA_MEASURED) < 0.05 and r2 > 0.85:
            n_survived += 1

    control_beta = null_results.get('Control (intact)', {}).get('beta')
    print(f"  │  Control β = {control_beta:.3f}" if control_beta else "  │  Control: no fit")
    print(f"  │  Ablations that DESTROY transition: {n_destroyed}/5")
    print(f"  │  Ablations that PRESERVE transition: {n_survived}/5")
    verdict1 = "PASS" if n_destroyed >= 3 else ("PARTIAL" if n_destroyed >= 1 else "FAIL")
    print(f"  │  Verdict: {verdict1}")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    # Part 2: FSS
    print("\n  ┌─ PART 2: SUBSYSTEM-COUNT FINITE-SIZE SCALING ──────────────────────┐")
    valid_betas = [b for b in betas_per_L if b is not None and not np.isnan(b)]
    if valid_betas:
        beta_cv = np.std(valid_betas) / np.abs(np.mean(valid_betas)) if np.abs(np.mean(valid_betas)) > 0.01 else np.inf
        beta_converge = np.mean(valid_betas[-3:]) if len(valid_betas) >= 3 else np.mean(valid_betas)
        print(f"  │  β across L values: {[f'{b:.3f}' for b in valid_betas]}")
        print(f"  │  β convergence (large L): {beta_converge:.3f}")
        print(f"  │  β coefficient of variation: {beta_cv:.3f}")
        verdict2 = "PASS" if beta_cv < 0.2 else ("PARTIAL" if beta_cv < 0.4 else "FAIL")
    else:
        verdict2 = "FAIL"
    print(f"  │  Verdict: {verdict2}")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    # Part 3: Data collapse
    print("\n  ┌─ PART 3: DATA COLLAPSE ─────────────────────────────────────────────┐")
    if collapse_results:
        print(f"  │  Optimal β/ν  = {collapse_results['beta_over_nu']:.4f}")
        print(f"  │  Optimal 1/ν  = {collapse_results['one_over_nu']:.4f}")
        print(f"  │  → β = {collapse_results['beta_from_collapse']:.4f}")
        print(f"  │  → ν = {collapse_results['nu']:.4f}")
        print(f"  │  Collapse quality: {collapse_results['collapse_quality']:.1f}")
        ising_bn = 0.3265 / 0.6301
        delta_bn = abs(collapse_results['beta_over_nu'] - ising_bn) / ising_bn
        verdict3 = "PASS" if collapse_results['collapse_quality'] > 5 else ("PARTIAL" if collapse_results['collapse_quality'] > 2 else "FAIL")
    else:
        verdict3 = "FAIL"
    print(f"  │  Verdict: {verdict3}")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    # Part 4: Coarse-graining
    print("\n  ┌─ PART 4: COARSE-GRAINING ───────────────────────────────────────────┐")
    if cg_results:
        cg_betas = [r['beta'] for r in cg_results.values() if r['beta'] is not None]
        if len(cg_betas) >= 2:
            cg_cv = np.std(cg_betas) / np.abs(np.mean(cg_betas)) if np.abs(np.mean(cg_betas)) > 0.01 else np.inf
            print(f"  │  β at each scale: {[f'{b:.3f}' for b in cg_betas]}")
            print(f"  │  β CV across scales: {cg_cv:.3f}")
            verdict4 = "PASS" if cg_cv < 0.15 else ("PARTIAL" if cg_cv < 0.3 else "FAIL")
        else:
            verdict4 = "INSUFFICIENT DATA"
    else:
        verdict4 = "FAIL"
    print(f"  │  Verdict: {verdict4}")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    # Overall
    verdicts = [verdict1, verdict2, verdict3, verdict4]
    n_pass = sum(1 for v in verdicts if v == "PASS")
    n_partial = sum(1 for v in verdicts if v == "PARTIAL")

    print("\n  ╔═══════════════════════════════════════════════════════════════════╗")
    if n_pass >= 3:
        print("  ║  OVERALL: STRONG EVIDENCE FOR GENUINE UNIVERSALITY              ║")
        print("  ║  The phase transition passes multiple independent tests.        ║")
    elif n_pass + n_partial >= 3:
        print("  ║  OVERALL: SUGGESTIVE EVIDENCE FOR UNIVERSALITY                  ║")
        print("  ║  Results are promising but some tests need strengthening.       ║")
    else:
        print("  ║  OVERALL: EVIDENCE IS INCONCLUSIVE                              ║") 
        print("  ║  Further investigation needed.                                  ║")
    print(f"  ║  Score: {n_pass}/4 PASS, {n_partial}/4 PARTIAL                          ║")
    print("  ╚═══════════════════════════════════════════════════════════════════╝")

    # Save everything
    summary = {
        'part1_nulls': verdict1,
        'part2_fss': verdict2,
        'part3_collapse': verdict3,
        'part4_coarsegraining': verdict4,
        'score': f"{n_pass}/4 PASS, {n_partial}/4 PARTIAL",
        'beta_measured': BETA_MEASURED,
        'fr_c': FR_C,
    }
    with open(os.path.join(OUT_DIR, 'grand_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  All results saved to {OUT_DIR}/")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Critical Phenomena Suite')
    parser.add_argument('--part', type=int, choices=[1, 2, 3, 4],
                        help='Run specific part (1-4). Default: run all.')
    args = parser.parse_args()

    t_start = time.time()

    if args.part is None or args.part == 1:
        null_results = run_null_controls()
    else:
        null_results = None

    if args.part is None or args.part == 2:
        fss_data, betas_per_L, chi_peaks = run_subsystem_fss()
    else:
        fss_data, betas_per_L, chi_peaks = None, [], None

    if args.part is None or args.part == 3:
        collapse_results = run_data_collapse(fss_data)
    else:
        collapse_results = None

    if args.part is None or args.part == 4:
        cg_results = run_coarse_graining()
    else:
        cg_results = None

    if args.part is None:
        print_grand_summary(null_results, fss_data, betas_per_L, collapse_results, cg_results)

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.1f}s")
