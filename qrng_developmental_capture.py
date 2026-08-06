#!/usr/bin/env python3
"""
Developmental Capture: Perturbation Source Comparison
========================================================

Wires a random perturbation xi_t into the bearer state b_t (see
bearer_state_competency.py's BearerEngine) as a one-shot injection at t0,
then measures how long its effect on macro-basin occupancy persists —
developmental capture DC(Delta) — under three xi_t sources:

  deterministic - fixed one-hot push toward subsystem 0 (repeatable control)
  prng          - numpy RandomState(seed), a reproducible pseudo-random source
  os_csprng     - real captured bits from data/qrng_bits.npy if present,
                   otherwise os.urandom (the OS's cryptographically-secure
                   PRNG). NOT a hardware QRNG. It is labeled os_csprng rather
                   than "qrng" deliberately: calling os.urandom output "QRNG"
                   would overstate what's actually running whenever
                   data/qrng_bits.npy is absent, which is the default state
                   of this repo. Drop real hardware-captured bits at that
                   path to test the genuine hardware-QRNG hypothesis instead.

DC(Delta) is only meaningful with the bearer state's persistence loop (see
bearer_state_competency.py): a plain UniversalEngine has no state for xi_t to
be incorporated into, so this script always runs with use_bearer=True.

This does NOT test whether randomness carries information from "outside
spacetime" -- it tests whether a given interface geometry (manifold/
topology) turns a one-time perturbation into a stable, lasting shift in
system behavior, and whether that differs measurably by perturbation source.

A non-significant ANOVA/t-test across sources is NOT proof of equivalence --
absence of evidence isn't evidence of absence. run_matrix() therefore also
runs a TOST (two one-sided t-tests) equivalence test between prng and
os_csprng specifically, against a pre-specified smallest-meaningful-effect
bound, so a null result can be reported as "differences larger than the
bound are excluded" rather than merely "not detected."

Note on the first version of this experiment: prng vs. deterministic differ
not just in "randomness quality" but in perturbation *shape* -- deterministic
concentrates its magnitude on a single subsystem while prng/os_csprng spread
it across all subsystems at matched L2 norm. A significant deterministic-vs-
distributed difference should be attributed to that shape difference, not to
anything about pseudorandomness. See perturbation_concentration_sweep.py for
a direct test of that variable.

Usage:
    python qrng_developmental_capture.py [--device cuda:0] [--steps 1800] [--N 96] [--seeds 8]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bearer_state_competency import BearerEngine, basin_occupancy, resolve_device

OUT_DIR = os.path.join('outputs', 'qrng_developmental_capture')
QRNG_BITS_PATH = os.path.join('data', 'qrng_bits.npy')

_warned_fallback = False


def load_qrng_bits(n: int) -> np.ndarray:
    """Return n bits in {0,1}. Prefers real captured QRNG data; falls back to
    the OS CSPRNG (os.urandom) with a one-time warning if none is present."""
    global _warned_fallback
    if os.path.exists(QRNG_BITS_PATH):
        arr = np.load(QRNG_BITS_PATH)
        if len(arr) >= n:
            start = int(np.frombuffer(os.urandom(8), dtype=np.uint64)[0] % max(1, len(arr) - n))
            return arr[start:start + n].astype(np.float32)
    if not _warned_fallback:
        print(f'  [qrng] no usable bitstream at {QRNG_BITS_PATH}; '
              f'falling back to os.urandom (OS CSPRNG) as QRNG stand-in.')
        _warned_fallback = True
    raw = os.urandom((n + 7) // 8)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n]
    return bits.astype(np.float32)


def make_xi(source: str, N: int, n_sub: int, device: str, seed: int, strength: float = 3.0):
    if source == 'deterministic':
        xi = torch.zeros(N, n_sub, device=device)
        xi[:, 0] = strength
        return xi
    if source == 'prng':
        rng = np.random.RandomState(seed)
        vals = rng.standard_normal((N, n_sub)).astype(np.float32)
        vals = vals / (np.linalg.norm(vals, axis=1, keepdims=True) + 1e-8)
        return torch.tensor(vals, device=device) * strength
    if source == 'os_csprng':
        bits = load_qrng_bits(N * n_sub)
        vals = (bits.reshape(N, n_sub) * 2 - 1).astype(np.float32)
        vals = vals / (np.linalg.norm(vals, axis=1, keepdims=True) + 1e-8)
        return torch.tensor(vals, device=device) * strength
    raise ValueError(f'Unknown xi source: {source}')


def build_pair(manifold, topology, device, steps, N, seed):
    # eng_pert and eng_ctrl each get their own RNG generator, both seeded
    # identically from `seed`. Without private generators, interleaving
    # eng_pert.step() and eng_ctrl.step() in the same loop would make them
    # alternate draws from Torch's one global RNG stream, so "control" would
    # silently stop being a matched replay of "perturbed" -- see BearerEngine
    # docstring / rng_seed. With private generators the only difference
    # between the two trajectories is the injected xi_t.
    torch.manual_seed(seed)
    np.random.seed(seed)
    eng_pert = BearerEngine(N=N, device=device, steps=steps, manifold=manifold,
                             topology=topology, fatigue_type='gradual', use_bearer=True,
                             rng_seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    eng_ctrl = BearerEngine(N=N, device=device, steps=steps, manifold=manifold,
                             topology=topology, fatigue_type='gradual', use_bearer=True,
                             rng_seed=seed)
    return eng_pert, eng_ctrl


def divergence_curve(eng_pert, eng_ctrl, t0, steps, window=20, max_delta=500):
    basins_p = eng_pert.hist_macro_basin[:, :steps].cpu().numpy()
    basins_c = eng_ctrl.hist_macro_basin[:, :steps].cpu().numpy()
    n_macro = eng_pert.n_macro

    deltas, divs = [], []
    n_win = min(max_delta // window, (steps - t0 - window) // window)
    for w in range(max(1, n_win)):
        t1 = t0 + w * window
        t2 = t1 + window
        if t2 > steps:
            break
        p_p = basin_occupancy(basins_p[:, t1:t2], n_macro)
        p_c = basin_occupancy(basins_c[:, t1:t2], n_macro)
        deltas.append(w * window)
        divs.append(float(np.sum(np.abs(p_p - p_c))))
    return np.array(deltas), np.array(divs)


def horizon_from_curve(deltas, divs):
    if len(divs) == 0 or divs.max() < 1e-6:
        return 0.0
    peak_idx = int(np.argmax(divs))
    threshold = divs[peak_idx] / np.e
    for i in range(peak_idx, len(divs)):
        if divs[i] < threshold:
            return float(deltas[i])
    return float(deltas[-1])


def run_condition(manifold, topology, source, device, steps, N, seed, t0, strength):
    eng_pert, eng_ctrl = build_pair(manifold, topology, device, steps, N, seed)
    xi = make_xi(source, N, eng_pert.n_sub, device, seed=seed * 1000 + hash(source) % 997, strength=strength)
    eng_pert.schedule_perturbation(t0, xi)
    for _ in range(steps):
        eng_pert.step()
        eng_ctrl.step()
    deltas, divs = divergence_curve(eng_pert, eng_ctrl, t0, steps)
    horizon = horizon_from_curve(deltas, divs)
    checkpoints = {}
    for cp in (40, 120, 300):
        idx = np.searchsorted(deltas, cp)
        checkpoints[f'dc_at_{cp}'] = float(divs[idx]) if idx < len(divs) else float('nan')
    return dict(peak_divergence=float(divs.max()) if len(divs) else 0.0,
                horizon=horizon, **checkpoints), (deltas, divs)


def tost_equivalence(x: np.ndarray, y: np.ndarray, bound: float):
    """Two one-sided t-tests (Welch) for equivalence of means within +/-bound.
    Returns (tost_p, mean_diff). tost_p < 0.05 supports "the true difference
    is smaller than bound" -- a non-significant ordinary t-test alone cannot
    support that claim, it can only fail to reject "no difference detected"."""
    from scipy import stats as sstats
    if len(x) < 2 or len(y) < 2:
        return float('nan'), float('nan')
    _, p_lower = sstats.ttest_ind(x, y - bound, equal_var=False, alternative='greater')
    _, p_upper = sstats.ttest_ind(x, y + bound, equal_var=False, alternative='less')
    tost_p = max(float(p_lower), float(p_upper))
    return tost_p, float(np.mean(x) - np.mean(y))


def run_matrix(device, steps, N, seeds, outdir, t0=None, strength=3.0):
    os.makedirs(outdir, exist_ok=True)
    device = resolve_device(device)
    if t0 is None:
        t0 = steps // 2

    manifolds = ['s3', 'flat4']
    sources = ['deterministic', 'prng', 'os_csprng']
    rows = []
    curves = {}
    total = len(manifolds) * len(sources) * seeds
    done = 0
    t_start = time.time()

    for manifold in manifolds:
        for source in sources:
            for seed in range(seeds):
                metrics, (deltas, divs) = run_condition(
                    manifold, 'cyclic', source, device, steps, N, seed, t0, strength)
                rows.append(dict(manifold=manifold, source=source, seed=seed, **metrics))
                curves[(manifold, source, seed)] = (deltas.tolist(), divs.tolist())
                done += 1
                elapsed = time.time() - t_start
                print(f'  [{done}/{total}] {manifold:6s} {source:12s} seed={seed}  '
                      f'horizon={metrics["horizon"]:.0f}  ({elapsed:.0f}s elapsed)')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'dc_per_seed.csv'), index=False)

    summary = (df.groupby(['manifold', 'source'])
               [['peak_divergence', 'horizon', 'dc_at_40', 'dc_at_120', 'dc_at_300']]
               .agg(['mean', 'std']))
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(outdir, 'dc_summary.csv'), index=False)

    # Source-comparison ANOVA-lite: does horizon differ across sources within each manifold?
    from scipy import stats as sstats
    source_tests = []
    equivalence_tests = []
    # Smallest meaningful effect for equivalence testing: half the horizon
    # measurement's own resolution (divergence_curve's window=20), so we
    # only claim equivalence at a granularity we can actually resolve.
    equivalence_bound = 10.0
    for manifold in manifolds:
        groups = [df[(df.manifold == manifold) & (df.source == s)]['horizon'].values
                  for s in sources]
        if all(len(g) > 1 for g in groups):
            f_stat, p_val = sstats.f_oneway(*groups)
            source_tests.append(dict(manifold=manifold, f_stat=float(f_stat), p_value=float(p_val)))

        prng_vals = df[(df.manifold == manifold) & (df.source == 'prng')]['horizon'].values
        csprng_vals = df[(df.manifold == manifold) & (df.source == 'os_csprng')]['horizon'].values
        tost_p, mean_diff = tost_equivalence(prng_vals, csprng_vals, equivalence_bound)
        equivalence_tests.append(dict(manifold=manifold, comparison='prng_vs_os_csprng',
                                       bound=equivalence_bound, mean_diff=mean_diff,
                                       tost_p_value=tost_p,
                                       equivalent=bool(tost_p < 0.05) if not np.isnan(tost_p) else None))

    with open(os.path.join(outdir, 'source_comparison.json'), 'w') as f:
        json.dump({
            't0': t0, 'strength': strength, 'steps': steps, 'N': N, 'seeds': seeds,
            'os_csprng_bits_source': 'data/qrng_bits.npy' if os.path.exists(QRNG_BITS_PATH) else 'os.urandom fallback (NOT real QRNG)',
            'source_anova_by_manifold': source_tests,
            'prng_vs_os_csprng_equivalence_tests': equivalence_tests,
        }, f, indent=2)

    with open(os.path.join(outdir, 'divergence_curves.json'), 'w') as f:
        json.dump({f'{m}|{s}|{seed}': {'deltas': d, 'divs': v}
                    for (m, s, seed), (d, v) in curves.items()}, f)

    plot_results(df, curves, manifolds, sources, seeds, outdir)
    return df, source_tests, equivalence_tests


def plot_results(df, curves, manifolds, sources, seeds, outdir):
    fig, axes = plt.subplots(1, len(manifolds), figsize=(6 * len(manifolds), 4.5), sharey=True)
    if len(manifolds) == 1:
        axes = [axes]
    colors = {'deterministic': '#8172B3', 'prng': '#4C72B0', 'os_csprng': '#DD5555'}

    for ax, manifold in zip(axes, manifolds):
        for source in sources:
            all_divs = [np.array(curves[(manifold, source, seed)][1]) for seed in range(seeds)]
            deltas = np.array(curves[(manifold, source, 0)][0])
            min_len = min(len(d) for d in all_divs)
            stacked = np.stack([d[:min_len] for d in all_divs])
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            ax.plot(deltas[:min_len], mean, label=source, color=colors[source])
            ax.fill_between(deltas[:min_len], mean - std, mean + std, alpha=0.15, color=colors[source])
        ax.set_title(f'manifold={manifold}')
        ax.set_xlabel('steps since perturbation (Delta)')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('basin-occupancy divergence (perturbed vs control)')

    fig.suptitle('Developmental capture DC(Delta) by xi_t source')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'dc_curves.png'), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--steps', type=int, default=1800)
    ap.add_argument('--N', type=int, default=96)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--strength', type=float, default=3.0)
    ap.add_argument('--outdir', default=OUT_DIR)
    args = ap.parse_args()

    print('Developmental Capture: Perturbation Source Comparison')
    print(f'  device={args.device}  steps={args.steps}  N={args.N}  seeds={args.seeds}  '
          f'strength={args.strength}')
    if not os.path.exists(QRNG_BITS_PATH):
        print(f'  NOTE: {QRNG_BITS_PATH} not found — os_csprng condition uses os.urandom, '
              f'NOT a real hardware QRNG. Drop real captured bits there to test the genuine hypothesis.')

    df, source_tests, equivalence_tests = run_matrix(
        args.device, args.steps, args.N, args.seeds, args.outdir, strength=args.strength)

    print('\nMean DC(Delta) horizon by manifold x source:')
    print(df.groupby(['manifold', 'source'])['horizon'].mean().round(1).to_string())
    print('\nOne-way ANOVA across sources (per manifold, on horizon):')
    for t in source_tests:
        print(f"  {t['manifold']}: F={t['f_stat']:.3f}  p={t['p_value']:.4f}")
    print('\nprng vs. os_csprng equivalence (TOST, bound=+/-10 steps):')
    for t in equivalence_tests:
        verdict = 'EQUIVALENT' if t['equivalent'] else ('NOT equivalent' if t['equivalent'] is False else 'inconclusive')
        print(f"  {t['manifold']}: mean_diff={t['mean_diff']:.2f}  tost_p={t['tost_p_value']:.4f}  -> {verdict}")
    print(f'\nOutputs written to {args.outdir}/')


if __name__ == '__main__':
    main()
