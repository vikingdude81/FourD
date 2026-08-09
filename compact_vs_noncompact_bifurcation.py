#!/usr/bin/env python3
"""
Compact vs. Non-Compact Bifurcation Comparison
==================================================

mechanism_extraction.py's Part 2 bifurcation analysis only ever built a
circle (S¹, compact) deterministic skeleton -- there's no manifold axis to
"extend to flat4" within that reduced model, because the skeleton never
encoded manifold geometry to begin with (that's rather the point of
minimal_boundary_model.py: the effect doesn't need one). What the skeleton
COULD meaningfully vary is compactness itself, since that's a real structural
difference between s3 (compact, wraps around) and flat4 (non-compact,
unbounded position clipped at a boundary) that the full UniversalEngine does
encode.

This script builds a matched pair of reduced skeletons -- one compact (S¹,
essentially mechanism_extraction.py's Part 2 model, reimplemented here so
both variants share one analysis harness) and one non-compact (R¹, a bounded
line with reflecting/clamping boundaries instead of wraparound) -- and runs
the same eigenvalue-crossing bifurcation-type classification and forward/
backward hysteresis sweep on both.

Question: does compactness alone (holding everything else about the reduced
dynamics fixed) change bifurcation type or hysteresis? If so, that's a
mechanistic candidate for why s3 (compact) stays permanently locked in after
its full-engine lesion bifurcation while flat4 (non-compact) snaps back --
see criticality_sweep.py and docs/ARCHITECTURE.md's open tension.

This is a simplified proxy, not a full-engine result -- like
mechanism_extraction.py's own skeleton, exact critical fatigue_rate values
here should not be expected to match the full engine's ~0.18-0.20
numerically (see the correction in docs/ARCHITECTURE.md about FR_C). Only
the qualitative comparison (compact vs. non-compact, holding the reduction
method fixed) is the point.

Usage:
    python compact_vs_noncompact_bifurcation.py [--n_sub 3]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigvals
from scipy.optimize import fsolve

OUT_DIR = os.path.join('outputs', 'compact_vs_noncompact')

RECOVERY_RATE = 0.025
FLOOR_VALUE = 0.05
STEERING = 0.707
COMP = 0.3


def make_det_step_s1(n_sub: int):
    """Compact (S1) skeleton: state = [theta, f_0..f_{n-1}]. Opponents at
    evenly-spaced angles; alignment via cosine (naturally bounded, wraps)."""
    phi = np.array([2 * np.pi * i / n_sub for i in range(n_sub)])

    def det_step(state, fr):
        theta = state[0]
        fat = state[1:].copy()
        alignment = np.cos(theta - phi)
        effective = (0.5 + COMP * alignment) * np.exp(-fat)
        effective = np.maximum(effective, FLOOR_VALUE)
        activities = effective / (effective.sum() + 1e-8)
        tangent = np.sin(phi - theta)
        resultant = (activities * tangent).sum()
        new_theta = (theta + STEERING * resultant) % (2 * np.pi)
        new_fat = fat + fr * activities - RECOVERY_RATE * (1.0 - activities)
        new_fat = np.clip(new_fat, 0.0, 3.0)
        clarity = np.abs(resultant)
        return np.concatenate([[new_theta], new_fat]), clarity, activities

    def wrap_diff(d):
        d = d.copy()
        d[0] = (d[0] + np.pi) % (2 * np.pi) - np.pi
        return d

    return det_step, wrap_diff, phi


def make_det_step_r1(n_sub: int, bound: float = 3.0):
    """Non-compact (R1) skeleton: state = [x, f_0..f_{n-1}] on a bounded
    line (clamped at +/-bound, no wraparound -- mirrors how flat4 clips
    position norm instead of wrapping like a sphere). Opponents at evenly
    spaced positions; alignment via a Gaussian bump (bounded, smooth,
    peaks at zero separation) since cosine's periodicity has no R1 analog."""
    phi = np.linspace(-(n_sub - 1) / 2.0, (n_sub - 1) / 2.0, n_sub)
    width = 1.0

    def det_step(state, fr):
        x = state[0]
        fat = state[1:].copy()
        alignment = np.exp(-((x - phi) ** 2) / (2 * width ** 2)) * 2 - 1  # rescale to ~[-1,1]
        effective = (0.5 + COMP * alignment) * np.exp(-fat)
        effective = np.maximum(effective, FLOOR_VALUE)
        activities = effective / (effective.sum() + 1e-8)
        drive = phi - x
        resultant = (activities * drive).sum()
        new_x = x + STEERING * resultant
        new_x = np.clip(new_x, -bound, bound)
        new_fat = fat + fr * activities - RECOVERY_RATE * (1.0 - activities)
        new_fat = np.clip(new_fat, 0.0, 3.0)
        clarity = np.abs(resultant)
        return np.concatenate([[new_x], new_fat]), clarity, activities

    def no_wrap_diff(d):
        return d

    return det_step, no_wrap_diff, phi


def eigenvalue_bifurcation(det_step, wrap_diff, dim, fr_range):
    eigenvalue_mags, eigenvalue_reals, eigenvalue_imags = [], [], []
    for fr in fr_range:
        def residual(y):
            y_next, _, _ = det_step(y, fr)
            return wrap_diff(y_next - y)

        y0 = np.zeros(dim)
        y0[0] = 0.1
        try:
            ystar, info, ier, _ = fsolve(residual, y0, full_output=True)
            converged = ier == 1 and np.max(np.abs(info['fvec'])) < 1e-8
        except Exception:
            converged = False

        if converged:
            eps = 1e-6
            J = np.zeros((dim, dim))
            for j in range(dim):
                yp, yn = ystar.copy(), ystar.copy()
                yp[j] += eps
                yn[j] -= eps
                fp, _, _ = det_step(yp, fr)
                fn, _, _ = det_step(yn, fr)
                J[:, j] = wrap_diff(fp - fn) / (2 * eps)
            eigs = eigvals(J)
            eigenvalue_mags.append(np.abs(eigs))
            eigenvalue_reals.append(eigs.real)
            eigenvalue_imags.append(eigs.imag)
        else:
            eigenvalue_mags.append(np.full(dim, np.nan))
            eigenvalue_reals.append(np.full(dim, np.nan))
            eigenvalue_imags.append(np.full(dim, np.nan))

    eigenvalue_mags = np.array(eigenvalue_mags)
    eigenvalue_reals = np.array(eigenvalue_reals)
    eigenvalue_imags = np.array(eigenvalue_imags)

    max_eig = np.nanmax(eigenvalue_mags, axis=1)
    bif_candidates = np.where(np.diff(np.sign(max_eig - 1.0)))[0]
    if len(bif_candidates) > 0:
        bif_idx = bif_candidates[0]
        fr_bif = float(fr_range[bif_idx])
        largest_idx = np.nanargmax(eigenvalue_mags[bif_idx])
        is_complex = np.abs(eigenvalue_imags[bif_idx, largest_idx]) > 0.01
        if is_complex:
            bif_type = 'Neimark-Sacker (quasi-periodic onset)'
        elif eigenvalue_reals[bif_idx, largest_idx] < 0:
            bif_type = 'Period-doubling'
        else:
            bif_type = 'Saddle-node / Transcritical'
    else:
        fr_bif, bif_type = None, 'None detected'

    return dict(fr_bif=fr_bif, bif_type=bif_type,
                fr_range=fr_range.tolist(), max_eig=max_eig.tolist())


def hysteresis_check(det_step, dim, fr_range, T_per=2000, init_low=None, init_high=None):
    if init_low is None:
        init_low = np.zeros(dim)
        init_low[0] = 0.1
    if init_high is None:
        init_high = np.zeros(dim)
        init_high[0] = 0.1
        init_high[1:] = 1.5

    state_fwd = init_low.copy()
    cl_fwd = []
    for fr in fr_range:
        for _ in range(T_per):
            state_fwd, cl, _ = det_step(state_fwd, fr)
        cl_fwd.append(cl)

    state_bwd = init_high.copy()
    cl_bwd = []
    for fr in reversed(fr_range):
        for _ in range(T_per):
            state_bwd, cl, _ = det_step(state_bwd, fr)
        cl_bwd.append(cl)
    cl_bwd = list(reversed(cl_bwd))

    cl_fwd = np.array(cl_fwd)
    cl_bwd = np.array(cl_bwd)
    gap = float(np.abs(cl_fwd - cl_bwd).mean())
    return dict(fr_range=fr_range.tolist(), cl_fwd=cl_fwd.tolist(), cl_bwd=cl_bwd.tolist(),
                hysteresis_gap=gap, has_hysteresis=bool(gap > 0.01))


def run_variant(name, det_step, wrap_diff, phi, n_sub, outdir):
    dim = 1 + n_sub
    print(f'\n-- {name} (n_sub={n_sub}) --')

    fr_eig = np.linspace(0.005, 0.50, 150)
    print('  [eigenvalue bifurcation scan]')
    bif = eigenvalue_bifurcation(det_step, wrap_diff, dim, fr_eig)
    print(f"    type={bif['bif_type']}  fr_bif={bif['fr_bif']}")

    fr_hyst = np.linspace(0.005, 0.50, 100)
    print('  [hysteresis forward/backward sweep]')
    hyst = hysteresis_check(det_step, dim, fr_hyst)
    print(f"    gap={hyst['hysteresis_gap']:.4f}  has_hysteresis={hyst['has_hysteresis']}")

    return dict(name=name, n_sub=n_sub, bifurcation=bif, hysteresis=hyst)


def plot_results(results, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for i, r in enumerate(results):
        color = '#4C72B0' if 'S1' in r['name'] else '#DD5555'
        ax = axes[0][i]
        fr = np.array(r['bifurcation']['fr_range'])
        me = np.array(r['bifurcation']['max_eig'])
        ax.plot(fr, me, 'o', markersize=2, color=color)
        ax.axhline(1.0, color='black', ls='--', alpha=0.5)
        if r['bifurcation']['fr_bif']:
            ax.axvline(r['bifurcation']['fr_bif'], color='red', ls=':', alpha=0.7)
        ax.set_title(f"{r['name']}: {r['bifurcation']['bif_type']}")
        ax.set_xlabel('fatigue_rate')
        ax.set_ylabel('max |eigenvalue|')
        ax.set_ylim(0, 2.5)

        ax2 = axes[1][i]
        fr_h = np.array(r['hysteresis']['fr_range'])
        ax2.plot(fr_h, r['hysteresis']['cl_fwd'], color='#4C72B0', label='forward')
        ax2.plot(fr_h, r['hysteresis']['cl_bwd'], color='#DD5555', ls='--', label='backward')
        ax2.set_title(f"{r['name']}: hysteresis gap={r['hysteresis']['hysteresis_gap']:.4f}")
        ax2.set_xlabel('fatigue_rate')
        ax2.set_ylabel('clarity')
        ax2.legend(fontsize=8)

    fig.suptitle('Compact (S1) vs. non-compact (R1) reduced-skeleton bifurcation comparison')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'compact_vs_noncompact.png'), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sub', type=int, default=3)
    ap.add_argument('--outdir', default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print('Compact vs. Non-Compact Bifurcation Comparison')

    det_s1, wrap_s1, phi_s1 = make_det_step_s1(args.n_sub)
    det_r1, wrap_r1, phi_r1 = make_det_step_r1(args.n_sub)

    results = [
        run_variant('S1 (compact)', det_s1, wrap_s1, phi_s1, args.n_sub, args.outdir),
        run_variant('R1 (non-compact)', det_r1, wrap_r1, phi_r1, args.n_sub, args.outdir),
    ]

    with open(os.path.join(args.outdir, 'compact_vs_noncompact_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    plot_results(results, args.outdir)

    print('\nSummary:')
    for r in results:
        print(f"  {r['name']}: bif_type={r['bifurcation']['bif_type']}  "
              f"fr_bif={r['bifurcation']['fr_bif']}  "
              f"hysteresis={r['hysteresis']['has_hysteresis']} "
              f"(gap={r['hysteresis']['hysteresis_gap']:.4f})")
    print(f'\nOutputs written to {args.outdir}/')


if __name__ == '__main__':
    main()
