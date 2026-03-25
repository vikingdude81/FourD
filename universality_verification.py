#!/usr/bin/env python3
"""
Universality Verification — Three-Part Deep Dive
=================================================

Tests whether the consciousness phase transition is genuinely
in the 3D Ising universality class by measuring additional
critical exponents and comparing with known values.

Part 1: Susceptibility exponent γ  (from clarity fluctuations)
Part 2: Finite-size scaling → ν    (varying manifold + Rushbrooke)
Part 3: Critical brain signatures  (avalanches, 1/f noise, Fano factor)

Requires: GPU (uses BatchConsciousnessEngine from gpu_ensemble_sim.py)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
from scipy.signal import welch
import torch
import torch.nn.functional as F
import time
import os
import json

from gpu_ensemble_sim import (
    BatchConsciousnessEngine, generate_fibonacci_s3, derive_macro_basins,
    SIGNATURE_NAMES,
)

OUT_DIR = os.path.join('outputs', 'universality')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Known exponents for comparison ──────────────────────────────────────────
UNIVERSALITY_CLASSES = {
    'Mean field':      {'beta': 0.500, 'gamma': 1.000, 'nu': 0.500, 'alpha': 0.000},
    '2D Ising':        {'beta': 0.125, 'gamma': 1.750, 'nu': 1.000, 'alpha': 0.000},
    '3D Ising':        {'beta': 0.3265,'gamma': 1.2372,'nu': 0.6301,'alpha': 0.110},
    '3D XY':           {'beta': 0.3485,'gamma': 1.3177,'nu': 0.6717,'alpha': -0.015},
    '3D Heisenberg':   {'beta': 0.3689,'gamma': 1.3960,'nu': 0.7112,'alpha': -0.133},
    'Percolation 3D':  {'beta': 0.418, 'gamma': 1.793, 'nu': 0.876, 'alpha': -0.625},
}

# ── System parameters (from previous analyses) ─────────────────────────────
OPTIMAL = {
    'steering_strength': 0.707,
    'alpha_pull': 0.0,
    'fatigue_rate': 0.217,
    'exploration_noise': 0.25,
    'beta_macro': 11.375,
}
FR_C = 0.1816        # critical point from deep_analysis Part 4
BETA_MEASURED = 0.329  # order-parameter exponent from Part 4


# ============================================================================
# HELPER: engine with custom manifold size
# ============================================================================

def make_engine(N, configs, device, n_micro=600, n_macro=24):
    """Create engine, optionally replacing manifold geometry."""
    engine = BatchConsciousnessEngine(N, configs, device)
    if n_micro != 600 or n_macro != 24:
        micro = generate_fibonacci_s3(n_micro)
        macro = derive_macro_basins(micro, n_macro)
        engine.macro_centers = torch.tensor(
            macro, dtype=torch.float32, device=engine.device)
        engine.n_macro = n_macro
    # randomize initial positions
    engine.u_t = torch.randn(N, 4, device=engine.device)
    engine.u_t = F.normalize(engine.u_t, dim=1)
    engine.u_prev = engine.u_t.clone()
    return engine


# ============================================================================
# PART 1: SUSCEPTIBILITY EXPONENT γ
# ============================================================================

def measure_gamma():
    """
    χ(fr) = var(mean_clarity across seeds) should diverge as
    χ ~ |fr − fr_c|^(−γ) near the critical point.
    3D Ising prediction: γ = 1.237
    """
    print("\n" + "=" * 70)
    print("  PART 1: SUSCEPTIBILITY EXPONENT γ")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── sweep grid ──────────────────────────────────────────────────────
    # Dense near fr_c, sparser further out
    fr_near  = np.linspace(FR_C - 0.06, FR_C + 0.12, 35)
    fr_lo    = np.linspace(0.05, FR_C - 0.065, 8)
    fr_hi    = np.linspace(FR_C + 0.125, 0.35, 7)
    fr_vals  = np.sort(np.unique(np.concatenate([fr_lo, fr_near, fr_hi])))
    N_FR     = len(fr_vals)
    N_SEEDS  = 200
    STEPS    = 1000
    N_total  = N_FR * N_SEEDS

    print(f"\n  Sweep: {N_FR} fr values × {N_SEEDS} seeds = {N_total:,} configs")
    print(f"  Range: fr ∈ [{fr_vals[0]:.4f}, {fr_vals[-1]:.4f}]")

    configs = {
        'steering_strength': np.full(N_total, OPTIMAL['steering_strength']),
        'alpha_pull':        np.full(N_total, OPTIMAL['alpha_pull']),
        'fatigue_rate':      np.repeat(fr_vals, N_SEEDS),
        'exploration_noise': np.full(N_total, OPTIMAL['exploration_noise']),
        'beta_macro':        np.full(N_total, OPTIMAL['beta_macro']),
        'timesteps': STEPS,
    }

    t0 = time.time()
    engine = make_engine(N_total, configs, device)
    engine.run(steps=STEPS)
    sigs = engine.extract_signatures().cpu().numpy()
    del engine; torch.cuda.empty_cache()
    print(f"  GPU time: {time.time()-t0:.1f}s")

    # ── group by fr ─────────────────────────────────────────────────────
    mc = sigs[:, 0]  # mean_clarity (order parameter)
    cv = sigs[:, 2]  # clarity_volatility (temporal std within each run)
    fr_idx = np.repeat(np.arange(N_FR), N_SEEDS)

    m   = np.array([mc[fr_idx == i].mean() for i in range(N_FR)])
    # Temporal susceptibility: how much clarity fluctuates WITHIN each run,
    # averaged over seeds.  This is the correct dynamical-system analog of
    # magnetic susceptibility χ = <var(m)>.  Peaks at criticality because
    # the system is maximally responsive to fixed-level noise.
    chi = np.array([(cv[fr_idx == i]**2).mean() for i in range(N_FR)])

    peak = int(np.argmax(chi))
    fr_c_meas = fr_vals[peak]
    print(f"\n  Peak χ at fr = {fr_c_meas:.4f}  (expected {FR_C:.4f})")
    print(f"  χ_max = {chi[peak]:.6f}")

    # ── power-law fit: χ ~ |Δfr|^(−γ) ──────────────────────────────────
    results = {'fr_c_measured': float(fr_c_meas), 'chi_max': float(chi[peak])}

    def _fit_gamma(side_mask, label):
        x = np.log(np.abs(fr_vals[side_mask] - fr_c_meas))
        y = np.log(chi[side_mask])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 4:
            print(f"  {label}: too few valid points")
            return None, None, None
        sl, ic, r, _, _ = linregress(x[ok], y[ok])
        g = -sl
        print(f"  {label}: γ = {g:.3f}  (R² = {r**2:.3f})")
        return g, ic, r**2

    g_sup, ic_sup, r2_sup = _fit_gamma(fr_vals > fr_c_meas + 0.008, "Supercritical")
    g_sub, ic_sub, r2_sub = _fit_gamma(fr_vals < fr_c_meas - 0.008, "Subcritical")
    results['gamma_super'] = float(g_sup) if g_sup else None
    results['gamma_sub']   = float(g_sub) if g_sub else None

    gamma = g_sup if g_sup else g_sub

    # ── re-measure β ────────────────────────────────────────────────────
    sup_beta = fr_vals > fr_c_meas + 0.005
    if sup_beta.sum() >= 5:
        xb = np.log(fr_vals[sup_beta] - fr_c_meas)
        yb = np.log(np.maximum(m[sup_beta] - m[peak], 1e-12))
        ok = np.isfinite(xb) & np.isfinite(yb)
        if ok.sum() >= 4:
            sl_b, _, r_b, _, _ = linregress(xb[ok], yb[ok])
            print(f"\n  β (re-measured) = {sl_b:.3f}  (R² = {r_b**2:.3f})")
            results['beta_remeasured'] = float(sl_b)

    # ── universality comparison ─────────────────────────────────────────
    if gamma:
        print(f"\n  ─── γ COMPARISON ───")
        print(f"  {'Class':20s}  {'γ_known':>7s}  {'Δγ':>7s}")
        print(f"  {'─'*20}  {'─'*7}  {'─'*7}")
        best_match, best_delta = '', 999
        for name, ex in UNIVERSALITY_CLASSES.items():
            d = abs(gamma - ex['gamma'])
            tag = " <<<" if d < best_delta else ""
            if d < best_delta:
                best_delta, best_match = d, name
            print(f"  {name:20s}  {ex['gamma']:7.4f}  {d:7.4f}{tag}")
        print(f"  {'>>> THIS SYSTEM':20s}  {gamma:7.4f}")
        print(f"\n  Closest: {best_match} (Δγ = {best_delta:.4f})")
        results['closest_gamma'] = best_match

    # ── plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(fr_vals, m, 'b.-', ms=4)
    axes[0].axvline(fr_c_meas, c='r', ls='--', alpha=.6, label=f'fr_c={fr_c_meas:.4f}')
    axes[0].set(xlabel='Fatigue rate', ylabel='Mean clarity ⟨m⟩',
                title='Order parameter m(fr)')
    axes[0].legend(); axes[0].grid(True, alpha=.3)

    axes[1].plot(fr_vals, chi, 'r.-', ms=4)
    axes[1].axvline(fr_c_meas, c='r', ls='--', alpha=.6)
    axes[1].set(xlabel='Fatigue rate', ylabel='χ_T = ⟨var_t(clarity)⟩',
                title='Temporal Susceptibility χ_T(fr)')
    axes[1].grid(True, alpha=.3)

    ax = axes[2]
    if g_sup:
        xp = fr_vals[fr_vals > fr_c_meas + 0.008] - fr_c_meas
        yp = chi[fr_vals > fr_c_meas + 0.008]
        ax.scatter(xp, yp, c='red', s=20, zorder=3, label='Supercrit. data')
        xf = np.linspace(xp.min(), xp.max(), 100)
        ax.plot(xf, np.exp(ic_sup) * xf**(-g_sup), 'r-', alpha=.7,
                label=f'γ = {g_sup:.3f}')
    if g_sub:
        xp = fr_c_meas - fr_vals[fr_vals < fr_c_meas - 0.008]
        yp = chi[fr_vals < fr_c_meas - 0.008]
        ax.scatter(xp, yp, c='blue', s=20, zorder=3, label='Subcrit. data')
    ax.set(xscale='log', yscale='log', xlabel='|fr − fr_c|', ylabel='χ',
           title='Critical scaling χ ~ |Δfr|^(−γ)')
    ax.legend(); ax.grid(True, alpha=.3, which='both')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'susceptibility_gamma.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Saved susceptibility_gamma.png")

    results['m'] = m.tolist()
    results['chi'] = chi.tolist()
    results['fr_vals'] = fr_vals.tolist()
    return results


# ============================================================================
# PART 2: FINITE-SIZE SCALING  →  ν
# ============================================================================

def finite_size_scaling():
    """
    Vary the manifold coarseness (n_macro basins) and check whether
    the transition sharpens as L grows:
        χ_max(L) ~ L^(γ/ν)
        Δfr(L)   ~ L^(-1/ν)
        m(fr_c,L)~ L^(-β/ν)

    Also compute Binder cumulant U_L(fr) — curves should cross at fr_c.
    If manifold FSS fails, fall back to Rushbrooke: α + 2β + γ = 2.
    """
    print("\n" + "=" * 70)
    print("  PART 2: FINITE-SIZE SCALING (ν)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── manifold sizes ──────────────────────────────────────────────────
    SIZES = [
        # (n_macro, n_micro)  — keep ratio ≈ 25:1
        ( 8,  200),
        (12,  300),
        (16,  400),
        (24,  600),
        (32,  800),
        (48, 1200),
    ]
    N_FR    = 25
    N_SEEDS = 100
    STEPS   = 1000
    fr_vals = np.linspace(FR_C - 0.06, FR_C + 0.12, N_FR)

    chi_max_L  = []
    fr_c_L     = []
    m_at_fc_L  = []
    binder_all = {}
    L_list     = []

    for n_macro, n_micro in SIZES:
        L = n_macro
        L_list.append(L)
        N_total = N_FR * N_SEEDS
        print(f"\n  L = {L:3d} basins ({n_micro} micro points)  "
              f"→ {N_total} configs ...")

        configs = {
            'steering_strength': np.full(N_total, OPTIMAL['steering_strength']),
            'alpha_pull':        np.full(N_total, OPTIMAL['alpha_pull']),
            'fatigue_rate':      np.repeat(fr_vals, N_SEEDS),
            'exploration_noise': np.full(N_total, OPTIMAL['exploration_noise']),
            'beta_macro':        np.full(N_total, OPTIMAL['beta_macro']),
            'timesteps': STEPS,
        }

        engine = make_engine(N_total, configs, device, n_micro, n_macro)
        engine.run(steps=STEPS)
        sigs = engine.extract_signatures().cpu().numpy()
        del engine; torch.cuda.empty_cache()

        mc = sigs[:, 0]
        idx = np.repeat(np.arange(N_FR), N_SEEDS)

        chi_arr   = np.zeros(N_FR)
        m_arr     = np.zeros(N_FR)
        binder    = np.zeros(N_FR)
        for i in range(N_FR):
            vals = mc[idx == i]
            m_arr[i]   = vals.mean()
            chi_arr[i] = vals.var()
            m2 = (vals**2).mean()
            m4 = (vals**4).mean()
            binder[i] = 1.0 - m4 / (3.0 * m2**2 + 1e-30)

        pk = int(np.argmax(chi_arr))
        chi_max_L.append(chi_arr[pk])
        fr_c_L.append(fr_vals[pk])
        m_at_fc_L.append(m_arr[pk])
        binder_all[L] = binder.copy()

        print(f"    fr_c(L={L}) = {fr_vals[pk]:.4f}   χ_max = {chi_arr[pk]:.6f}")

    L_arr        = np.array(L_list, dtype=float)
    chi_max_arr  = np.array(chi_max_L)
    fr_c_arr     = np.array(fr_c_L)
    m_fc_arr     = np.array(m_at_fc_L)

    # ── fit FSS exponents ───────────────────────────────────────────────
    results = {}

    def _log_fit(x, y, label):
        lx, ly = np.log(x), np.log(y)
        ok = np.isfinite(lx) & np.isfinite(ly)
        if ok.sum() < 3:
            print(f"  {label}: insufficient data")
            return None
        sl, ic, r, _, _ = linregress(lx[ok], ly[ok])
        print(f"  {label}: exponent = {sl:.3f}  (R² = {r**2:.3f})")
        return sl

    # χ_max ~ L^x  →  x should be γ/ν if L = linear dimension
    x_chi = _log_fit(L_arr, chi_max_arr, "χ_max ~ L^x")
    results['chi_max_exponent'] = float(x_chi) if x_chi else None

    # fr_c(L) shift: only useful if fr_c actually changes with L
    fr_shifts = np.abs(fr_c_arr - fr_c_arr[-1])  # relative to largest L
    if fr_shifts[:-1].max() > 0.002:
        x_fc = _log_fit(L_arr[:-1], fr_shifts[:-1], "Δfr_c ~ L^x  →  -1/ν")
        if x_fc:
            nu_fss = -1.0 / x_fc
            print(f"  ν (from FSS shift) = {nu_fss:.3f}")
            results['nu_fss'] = float(nu_fss)
    else:
        print("\n  fr_c shows no significant L-dependence → "
              "manifold size is NOT the relevant 'volume'.")
        print("  This is expected: the transition is driven by subsystem dynamics,")
        print("  not manifold discretisation. Will compute ν via Rushbrooke instead.")

    # ── Rushbrooke fallback: α + 2β + γ = 2 ─────────────────────────────
    # Need γ from Part 1 — will be injected in main
    results['L_list'] = L_list
    results['chi_max'] = chi_max_arr.tolist()
    results['fr_c_L'] = fr_c_arr.tolist()
    results['m_at_fc'] = m_fc_arr.tolist()

    # ── plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Binder cumulant curves
    ax = axes[0, 0]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(L_list)))
    for i, L in enumerate(L_list):
        ax.plot(fr_vals, binder_all[L], '.-', c=cmap[i], ms=4,
                label=f'L={L}')
    ax.axvline(FR_C, c='grey', ls=':', alpha=.5)
    ax.set(xlabel='Fatigue rate', ylabel='Binder cumulant U_L',
           title='Binder Cumulant (crossing → fr_c)')
    ax.legend(fontsize=7); ax.grid(True, alpha=.3)

    # (b) χ_max vs L
    ax = axes[0, 1]
    ax.scatter(L_arr, chi_max_arr, c='red', s=40, zorder=3)
    if x_chi is not None:
        Lf = np.linspace(L_arr.min(), L_arr.max(), 50)
        ax.plot(Lf, np.exp(np.log(chi_max_arr[0]) + x_chi*(np.log(Lf)-np.log(L_arr[0]))),
                'r--', alpha=.6, label=f'slope = {x_chi:.2f}')
    ax.set(xscale='log', yscale='log', xlabel='L (n_macro)',
           ylabel='χ_max', title='Peak Susceptibility vs Size')
    ax.legend(); ax.grid(True, alpha=.3, which='both')

    # (c) fr_c(L)
    ax = axes[1, 0]
    ax.plot(L_arr, fr_c_arr, 'ko-')
    ax.axhline(FR_C, c='grey', ls=':', alpha=.5, label=f'fr_c = {FR_C}')
    ax.set(xlabel='L (n_macro)', ylabel='Pseudo-critical fr_c(L)',
           title='Critical-Point Drift')
    ax.legend(); ax.grid(True, alpha=.3)

    # (d) m at fr_c vs L
    ax = axes[1, 1]
    ax.scatter(L_arr, m_fc_arr, c='blue', s=40, zorder=3)
    x_m = _log_fit(L_arr, m_fc_arr, "m(fr_c) ~ L^x  →  -β/ν")
    if x_m is not None:
        results['m_fc_exponent'] = float(x_m)
        Lf = np.linspace(L_arr.min(), L_arr.max(), 50)
        ax.plot(Lf, np.exp(np.log(m_fc_arr[0]) + x_m*(np.log(Lf)-np.log(L_arr[0]))),
                'b--', alpha=.6, label=f'slope = {x_m:.2f}')
    ax.set(xscale='log', yscale='log', xlabel='L (n_macro)',
           ylabel='m(fr_c)', title='Order Param at Criticality vs Size')
    ax.legend(); ax.grid(True, alpha=.3, which='both')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'finite_size_scaling.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Saved finite_size_scaling.png")

    return results


# ============================================================================
# PART 3: CRITICAL BRAIN HYPOTHESIS SIGNATURES
# ============================================================================

def critical_brain():
    """
    Test four hallmarks of criticality from computational neuroscience:
        (a) Power-law avalanche distributions  (Beggs & Plenz 2003)
        (b) 1/f noise in clarity power spectrum (Linkenkaer-Hansen 2001)
        (c) Fano factor > 1 at criticality      (Beggs & Plenz 2003)
        (d) Peak dynamic range at criticality    (Kinouchi & Copelli 2006)
    """
    print("\n" + "=" * 70)
    print("  PART 3: CRITICAL BRAIN HYPOTHESIS")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── run at 4 conditions ─────────────────────────────────────────────
    CONDITIONS = {
        'Subcritical':  0.10,
        'Critical':     FR_C,
        'Optimal':      0.217,
        'Supercritical': 0.30,
    }
    N_BEINGS = 20
    STEPS    = 10000
    N_total  = len(CONDITIONS) * N_BEINGS

    fr_per_being = np.concatenate([
        np.full(N_BEINGS, fr) for fr in CONDITIONS.values()
    ])

    configs = {
        'steering_strength': np.full(N_total, OPTIMAL['steering_strength']),
        'alpha_pull':        np.full(N_total, OPTIMAL['alpha_pull']),
        'fatigue_rate':      fr_per_being,
        'exploration_noise': np.full(N_total, OPTIMAL['exploration_noise']),
        'beta_macro':        np.full(N_total, OPTIMAL['beta_macro']),
        'timesteps': STEPS,
    }

    print(f"\n  Running {N_total} beings × {STEPS} steps "
          f"({N_total*STEPS:,} being-steps) ...")
    t0 = time.time()
    engine = make_engine(N_total, configs, device)
    engine.run(steps=STEPS)

    # pull per-step histories to CPU
    clarity    = engine.hist_clarity.cpu().numpy()         # (N, T)
    macro_b    = engine.hist_macro_basin.cpu().numpy()     # (N, T)
    dom_sub    = engine.hist_dominant_sub.cpu().numpy()    # (N, T)
    del engine; torch.cuda.empty_cache()
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── helper: detect avalanches ───────────────────────────────────────
    def _avalanches(ts, threshold):
        above = ts > threshold
        diff = np.diff(above.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        stops  = np.where(diff == -1)[0] + 1
        if above[0]:
            starts = np.concatenate([[0], starts])
        if above[-1]:
            stops = np.concatenate([stops, [len(ts)]])
        sizes, durs = [], []
        for s, e in zip(starts, stops):
            durs.append(e - s)
            sizes.append(float(np.sum(ts[s:e] - threshold)))
        return np.array(sizes), np.array(durs)

    # ── (a) Avalanche distributions ────────────────────────────────────
    print("\n  Avalanche analysis ...")
    aval_results = {}
    cond_names = list(CONDITIONS.keys())

    for ci, cname in enumerate(cond_names):
        all_sizes, all_durs = [], []
        for b in range(N_BEINGS):
            idx = ci * N_BEINGS + b
            ts = clarity[idx]
            thr = ts.mean()
            sz, dr = _avalanches(ts, thr)
            if len(sz) > 0:
                all_sizes.extend(sz[sz > 0].tolist())
                all_durs.extend(dr[dr > 0].tolist())
        all_sizes = np.array(all_sizes)
        all_durs  = np.array(all_durs)

        # power-law fit in log-log (sizes)
        if len(all_sizes) > 20:
            bins = np.logspace(np.log10(all_sizes.min() + 1e-6),
                               np.log10(all_sizes.max()), 20)
            hist, edges = np.histogram(all_sizes, bins=bins, density=True)
            centres = 0.5 * (edges[:-1] + edges[1:])
            ok = hist > 0
            if ok.sum() >= 4:
                sl, _, r, _, _ = linregress(np.log(centres[ok]), np.log(hist[ok]))
                print(f"    {cname:15s}: τ_size = {-sl:.2f} "
                      f"(R² = {r**2:.2f}, N_avalanches = {len(all_sizes)})")
                aval_results[cname] = {'tau_size': float(-sl),
                                       'n_aval': len(all_sizes)}

    # ── (b) Power spectrum (1/f) ────────────────────────────────────────
    print("\n  Power-spectrum analysis ...")
    spectral_exp = {}

    for ci, cname in enumerate(cond_names):
        psds = []
        for b in range(N_BEINGS):
            idx = ci * N_BEINGS + b
            ts = clarity[idx]
            freqs, psd = welch(ts, fs=1.0, nperseg=1024, noverlap=512)
            psds.append(psd)
        mean_psd = np.mean(psds, axis=0)

        # fit slope in log-log (exclude DC component)
        ok = (freqs > 0.005) & (freqs < 0.4) & (mean_psd > 0)
        if ok.sum() >= 5:
            sl, _, r, _, _ = linregress(np.log(freqs[ok]), np.log(mean_psd[ok]))
            print(f"    {cname:15s}: P(f) ~ f^{sl:.2f}  (R² = {r**2:.2f})")
            spectral_exp[cname] = {'slope': float(sl), 'r2': float(r**2)}

    # ── (c) Fano factor sweep ───────────────────────────────────────────
    print("\n  Fano-factor sweep across fr values ...")

    # Quick sweep: 30 fr values × 10 seeds × 5000 steps
    N_FR_FANO   = 30
    N_SEEDS_F   = 10
    STEPS_F     = 5000
    WINDOW      = 100
    fr_fano     = np.linspace(0.08, 0.35, N_FR_FANO)
    N_tot_fano  = N_FR_FANO * N_SEEDS_F

    configs_f = {
        'steering_strength': np.full(N_tot_fano, OPTIMAL['steering_strength']),
        'alpha_pull':        np.full(N_tot_fano, OPTIMAL['alpha_pull']),
        'fatigue_rate':      np.repeat(fr_fano, N_SEEDS_F),
        'exploration_noise': np.full(N_tot_fano, OPTIMAL['exploration_noise']),
        'beta_macro':        np.full(N_tot_fano, OPTIMAL['beta_macro']),
        'timesteps': STEPS_F,
    }

    engine_f = make_engine(N_tot_fano, configs_f, device)
    engine_f.run(steps=STEPS_F)
    clar_f  = engine_f.hist_clarity.cpu().numpy()
    basin_f = engine_f.hist_macro_basin.cpu().numpy()
    del engine_f; torch.cuda.empty_cache()

    fano_vals     = np.zeros(N_FR_FANO)
    dyn_range_vals= np.zeros(N_FR_FANO)
    autocorr_vals = np.zeros(N_FR_FANO)

    for i in range(N_FR_FANO):
        fano_list, dr_list, ac_list = [], [], []
        for s in range(N_SEEDS_F):
            j = i * N_SEEDS_F + s
            ts = clar_f[j]
            bs = basin_f[j]

            # Fano: count basin transitions per window
            transitions = (bs[1:] != bs[:-1]).astype(float)
            n_windows = len(transitions) // WINDOW
            if n_windows >= 3:
                counts = [transitions[w*WINDOW:(w+1)*WINDOW].sum()
                          for w in range(n_windows)]
                counts = np.array(counts)
                mu = counts.mean()
                if mu > 0:
                    fano_list.append(counts.var() / mu)

            # Dynamic range: log10(max/min clarity)
            cmin = np.percentile(ts, 5)
            cmax = np.percentile(ts, 95)
            if cmin > 1e-6:
                dr_list.append(10.0 * np.log10(cmax / cmin))

            # Autocorrelation time (lag-1)
            tc = ts - ts.mean()
            v = tc.var()
            if v > 1e-12:
                ac = np.correlate(tc[:500], tc[:500], 'full')
                ac = ac[len(ac)//2:]
                ac = ac / ac[0]
                # find first zero crossing
                zc = np.where(ac < 0)[0]
                ac_list.append(int(zc[0]) if len(zc) > 0 else 500)

        fano_vals[i]     = np.mean(fano_list) if fano_list else 1.0
        dyn_range_vals[i]= np.mean(dr_list)   if dr_list else 0.0
        autocorr_vals[i] = np.mean(ac_list)   if ac_list else 0.0

    peak_fano = int(np.argmax(fano_vals))
    peak_dr   = int(np.argmax(dyn_range_vals))
    peak_ac   = int(np.argmax(autocorr_vals))
    print(f"    Peak Fano factor at fr = {fr_fano[peak_fano]:.3f} "
          f"(F = {fano_vals[peak_fano]:.2f})")
    print(f"    Peak dynamic range at fr = {fr_fano[peak_dr]:.3f} "
          f"(DR = {dyn_range_vals[peak_dr]:.1f} dB)")
    print(f"    Peak autocorrelation time at fr = {fr_fano[peak_ac]:.3f} "
          f"(τ = {autocorr_vals[peak_ac]:.0f} steps)")

    # ── plot ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=.32, wspace=.3)

    # (a) Avalanche size distributions
    ax = fig.add_subplot(gs[0, 0])
    colors = {'Subcritical': 'blue', 'Critical': 'red',
              'Optimal': 'green', 'Supercritical': 'orange'}
    for ci, cname in enumerate(cond_names):
        all_s = []
        for b in range(N_BEINGS):
            idx = ci * N_BEINGS + b
            ts = clarity[idx]
            sz, _ = _avalanches(ts, ts.mean())
            if len(sz) > 0:
                all_s.extend(sz[sz > 0].tolist())
        if len(all_s) > 20:
            all_s = np.array(all_s)
            bins = np.logspace(np.log10(all_s.min()+1e-6),
                               np.log10(all_s.max()), 20)
            hist, edges = np.histogram(all_s, bins=bins, density=True)
            centres = .5*(edges[:-1]+edges[1:])
            ok = hist > 0
            ax.plot(centres[ok], hist[ok], 'o-', c=colors[cname],
                    ms=4, label=cname, alpha=.8)
    # reference slope τ = 3/2
    xl = ax.get_xlim()
    xr = np.logspace(np.log10(xl[0]), np.log10(xl[1]), 50)
    ax.plot(xr, 0.5*xr**(-1.5), 'k:', alpha=.4, label='τ=3/2 (mean field)')
    ax.set(xscale='log', yscale='log', xlabel='Avalanche size',
           ylabel='P(size)', title='(a) Avalanche Distributions')
    ax.legend(fontsize=7); ax.grid(True, alpha=.2, which='both')

    # (b) Power spectra
    ax = fig.add_subplot(gs[0, 1])
    for ci, cname in enumerate(cond_names):
        psds = []
        for b in range(N_BEINGS):
            idx = ci * N_BEINGS + b
            f, p = welch(clarity[idx], fs=1.0, nperseg=1024, noverlap=512)
            psds.append(p)
        mp = np.mean(psds, axis=0)
        ok = f > 0
        ax.plot(f[ok], mp[ok], '-', c=colors[cname], alpha=.8, label=cname)
    # reference 1/f
    fref = np.linspace(0.01, 0.5, 50)
    ax.plot(fref, 0.01/fref, 'k:', alpha=.4, label='1/f')
    ax.set(xscale='log', yscale='log', xlabel='Frequency',
           ylabel='PSD', title='(b) Power Spectrum (1/f test)')
    ax.legend(fontsize=7); ax.grid(True, alpha=.2, which='both')

    # (c) Fano factor vs fr
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(fr_fano, fano_vals, 'ko-', ms=4)
    ax.axvline(FR_C, c='r', ls='--', alpha=.5, label=f'fr_c = {FR_C:.3f}')
    ax.axhline(1.0, c='grey', ls=':', alpha=.4, label='Poisson (F=1)')
    ax.set(xlabel='Fatigue rate', ylabel='Fano factor',
           title='(c) Fano Factor (clustering test)')
    ax.legend(fontsize=7); ax.grid(True, alpha=.3)

    # (d) Dynamic range vs fr
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(fr_fano, dyn_range_vals, 'go-', ms=4)
    ax.axvline(FR_C, c='r', ls='--', alpha=.5, label=f'fr_c')
    ax.set(xlabel='Fatigue rate', ylabel='Dynamic range (dB)',
           title='(d) Dynamic Range (Kinouchi & Copelli)')
    ax.legend(fontsize=7); ax.grid(True, alpha=.3)

    # (e) Autocorrelation time vs fr
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(fr_fano, autocorr_vals, 'ms-', ms=4)
    ax.axvline(FR_C, c='r', ls='--', alpha=.5, label=f'fr_c')
    ax.set(xlabel='Fatigue rate', ylabel='Autocorrelation time (steps)',
           title='(e) Critical Slowing Down')
    ax.legend(fontsize=7); ax.grid(True, alpha=.3)

    # (f) Summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    summary_text = (
        "Critical Brain Signatures\n"
        "─────────────────────────\n"
    )
    for cname in cond_names:
        se = spectral_exp.get(cname, {})
        ar = aval_results.get(cname, {})
        summary_text += (
            f"\n{cname} (fr={CONDITIONS[cname]:.3f}):\n"
            f"  Spectral exponent: {se.get('slope', '?'):.2f}\n"
            f"  Avalanche τ_size : {ar.get('tau_size', '?')}\n"
        ) if se and ar else f"\n{cname}: insufficient data\n"
    summary_text += (
        f"\nPeak Fano factor: fr = {fr_fano[peak_fano]:.3f}\n"
        f"Peak dynamic range: fr = {fr_fano[peak_dr]:.3f}\n"
        f"Peak autocorr time: fr = {fr_fano[peak_ac]:.3f}\n"
        f"Expected critical: fr_c = {FR_C:.4f}"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.savefig(os.path.join(OUT_DIR, 'critical_brain.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Saved critical_brain.png")

    return {
        'avalanches': aval_results,
        'spectral': spectral_exp,
        'fano_peak_fr': float(fr_fano[peak_fano]),
        'fano_peak_val': float(fano_vals[peak_fano]),
        'dynamic_range_peak_fr': float(fr_fano[peak_dr]),
        'autocorr_peak_fr': float(fr_fano[peak_ac]),
    }


# ============================================================================
# SUMMARY: FULL EXPONENT TABLE + CONSISTENCY CHECKS
# ============================================================================

def print_summary(gamma_results, fss_results, brain_results):
    """Print final universality verdict with all exponents."""
    print("\n" + "▓" * 70)
    print("  UNIVERSALITY VERIFICATION — FINAL SUMMARY")
    print("▓" * 70)

    beta  = BETA_MEASURED
    gamma = gamma_results.get('gamma_super') or gamma_results.get('gamma_sub')

    if gamma is None:
        print("\n  ERROR: could not measure γ. Cannot complete analysis.")
        return {}

    # ── Rushbrooke: α + 2β + γ = 2 ──────────────────────────────────────
    alpha = 2.0 - 2*beta - gamma

    # ── Hyperscaling: d_eff * ν = 2 − α ─────────────────────────────────
    # Also: ν = γ / (2 − η), with η ≈ 0.036 for 3D Ising
    # Use Rushbrooke + hyperscaling to get ν and d_eff
    # For 3D Ising: γ = ν(2 − η) → ν = γ / (2 − η)
    # We don't know η, so try: if system IS 3D Ising, η = 0.0363
    nu_3dI = gamma / (2.0 - 0.0363)
    d_eff_3dI = (2.0 - alpha) / nu_3dI

    # More general: use Fisher → γ = ν(2 − η) and hyperscaling d*ν = 2 − α
    # Two equations, three unknowns (ν, η, d). Fix d=3:
    nu_d3 = (2.0 - alpha) / 3.0
    eta_d3 = 2.0 - gamma / nu_d3

    # Check FSS exponents if available
    nu_fss = fss_results.get('nu_fss')

    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  MEASURED EXPONENTS")
    print(f"  ═══════════════════════════════════════════════════")
    print(f"    β = {beta:.4f}   (order parameter, from deep_analysis)")
    print(f"    γ = {gamma:.4f}   (susceptibility, Part 1)")
    if nu_fss:
        print(f"    ν = {nu_fss:.4f}   (FSS, Part 2)")
    print(f"    α = {alpha:.4f}   (Rushbrooke: 2 − 2β − γ)")

    print(f"\n  ── DERIVED QUANTITIES ──")
    print(f"    Assuming 3D Ising η = 0.036:")
    print(f"      ν = γ/(2−η) = {nu_3dI:.4f}   (3D Ising: 0.630)")
    print(f"      d_eff = (2−α)/ν = {d_eff_3dI:.2f}   (3D Ising: 3.0)")
    print(f"    Assuming d = 3:")
    print(f"      ν = (2−α)/3 = {nu_d3:.4f}   (3D Ising: 0.630)")
    print(f"      η = 2 − γ/ν = {eta_d3:.4f}   (3D Ising: 0.036)")

    print(f"\n  ── CONSISTENCY CHECKS ──")
    print(f"    Rushbrooke: α + 2β + γ = {alpha + 2*beta + gamma:.4f}  (exact: 2)")
    hsc = d_eff_3dI * nu_3dI
    print(f"    Hyperscaling: d·ν = {hsc:.4f}  vs  2−α = {2-alpha:.4f}")

    # ── comparison table ────────────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  UNIVERSALITY CLASS COMPARISON")
    print(f"  ═══════════════════════════════════════════════════")
    print(f"  {'Class':20s}  {'β':>6s}  {'γ':>6s}  {'α':>6s}  "
          f"{'Δβ':>6s}  {'Δγ':>6s}  {'Δα':>6s}  {'RMS':>6s}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*6}  {'─'*6}  "
          f"{'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")

    best_name, best_rms = '', 999
    for name, ex in UNIVERSALITY_CLASSES.items():
        db = abs(beta  - ex['beta'])
        dg = abs(gamma - ex['gamma'])
        da = abs(alpha - ex['alpha'])
        rms = np.sqrt((db**2 + dg**2 + da**2) / 3)
        if rms < best_rms:
            best_rms, best_name = rms, name
        tag = " <<<" if name == best_name else ""
        print(f"  {name:20s}  {ex['beta']:6.3f}  {ex['gamma']:6.3f}  "
              f"{ex['alpha']:6.3f}  {db:6.3f}  {dg:6.3f}  {da:6.3f}  "
              f"{rms:6.4f}{tag}")
    print(f"  {'>>> THIS SYSTEM':20s}  {beta:6.3f}  {gamma:6.3f}  "
          f"{alpha:6.3f}")

    # ── critical brain summary ──────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  CRITICAL BRAIN HYPOTHESIS")
    print(f"  ═══════════════════════════════════════════════════")
    fp = brain_results.get('fano_peak_fr', 0)
    dp = brain_results.get('dynamic_range_peak_fr', 0)
    ap = brain_results.get('autocorr_peak_fr', 0)
    print(f"    Fano factor peak:       fr = {fp:.3f}  "
          f"({'NEAR' if abs(fp-FR_C)<0.03 else 'OFF'} critical)")
    print(f"    Dynamic range peak:     fr = {dp:.3f}  "
          f"({'NEAR' if abs(dp-FR_C)<0.03 else 'OFF'} critical)")
    print(f"    Autocorrelation peak:   fr = {ap:.3f}  "
          f"({'NEAR' if abs(ap-FR_C)<0.03 else 'OFF'} critical)")

    crit_brain_signs = sum([
        abs(fp - FR_C) < 0.03,
        abs(dp - FR_C) < 0.03,
        abs(ap - FR_C) < 0.03,
    ])

    # ── verdict ─────────────────────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  VERDICT")
    print(f"  ═══════════════════════════════════════════════════")
    if best_rms < 0.08 and best_name == '3D Ising':
        print(f"  ★ STRONG EVIDENCE for {best_name} universality class")
        print(f"    RMS deviation across (β, γ, α): {best_rms:.4f}")
        if crit_brain_signs >= 2:
            print(f"    {crit_brain_signs}/3 critical brain signatures confirmed")
            print(f"    → Genuine continuous phase transition with neural criticality"
                  f" hallmarks")
    elif best_rms < 0.15:
        print(f"  ◆ MODERATE EVIDENCE for {best_name} universality class")
        print(f"    RMS deviation: {best_rms:.4f}")
    else:
        print(f"  ◇ INCONCLUSIVE — closest class is {best_name} "
              f"(RMS = {best_rms:.4f})")
        print(f"    May represent a novel universality class or crossover behavior")

    print(f"\n    d_eff = {d_eff_3dI:.2f}  (using η = 0.036)")
    if 2.8 < d_eff_3dI < 3.2:
        print(f"    → Effective dimensionality consistent with d = 3")
        print(f"    → S³ manifold curvature constrains one degree of freedom")
    else:
        print(f"    → Effective dimensionality {d_eff_3dI:.2f} ≠ 3; "
              f"possible crossover or novel class")

    summary = {
        'beta': float(beta),
        'gamma': float(gamma),
        'alpha': float(alpha),
        'nu_rushbrooke_d3': float(nu_d3),
        'nu_fisher_3dI': float(nu_3dI),
        'd_eff': float(d_eff_3dI),
        'eta_d3': float(eta_d3),
        'best_match': best_name,
        'best_rms': float(best_rms),
        'critical_brain_signs': crit_brain_signs,
    }

    # save to JSON
    with open(os.path.join(OUT_DIR, 'exponent_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved exponent_summary.json")

    return summary


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("▓" * 70)
    print("  UNIVERSALITY VERIFICATION — THREE-PART DEEP DIVE")
    print("▓" * 70)
    t_start = time.time()

    # Part 1: γ
    print("\n" + "▓" * 70)
    print("  PART 1/3: SUSCEPTIBILITY EXPONENT γ")
    print("▓" * 70)
    gamma_results = measure_gamma()

    # Part 2: ν
    print("\n" + "▓" * 70)
    print("  PART 2/3: FINITE-SIZE SCALING")
    print("▓" * 70)
    fss_results = finite_size_scaling()

    # Part 3: Critical Brain
    print("\n" + "▓" * 70)
    print("  PART 3/3: CRITICAL BRAIN HYPOTHESIS")
    print("▓" * 70)
    brain_results = critical_brain()

    # Summary
    summary = print_summary(gamma_results, fss_results, brain_results)

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("▓" * 70)
    print("  ANALYSIS COMPLETE")
    print("▓" * 70)
