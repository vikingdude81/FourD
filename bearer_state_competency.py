#!/usr/bin/env python3
"""
Bearer State + Competency Vector
=================================

Adds a persistent "bearer" state b_t to the UniversalEngine and asks whether
it is *constitutive* rather than merely observational: does incorporated
history change what the system can subsequently do?

    influences_t   = base_influences_t + bearer_weight * b_t      (perception, biased by b_t)
    activities_t    = softmax-like normalize(influences_t * fatigue decay + noise)   (decision)
    b_{t+1}         = (1 - decay) * b_t + gain * activities_t + xi_t                (incorporation)

b_t feeds forward into the next step's influences, so anything incorporated
into it keeps shaping perception/decision until it decays — this is the
minimal loop needed for "developmental capture" rather than a passive log.

Four competencies are measured per (manifold, topology, bearer on/off, seed):

  memory_horizon     - steps until a one-time bearer perturbation's effect on
                        macro-basin occupancy decays below 1/e of its peak
  lesion_recovery     - steps for clarity to return to within 10% of its
                        pre-lesion baseline after a subsystem is zeroed out
  adaptation_speed    - steps for basin-occupancy entropy to re-stabilize
                        after the macro-basin centers are rotated (env shift)
  self_maintenance    - inverse coefficient of variation of clarity over a
                        long undisturbed run (higher = more stable)

developmental_capture() additionally reproduces the DC(Delta) measurement
from the interface-competency write-up directly: inject a perturbation into
b_t at t0, compare perturbed vs. control bearer trajectories at t0+Delta.
The perturbation source is pluggable (deterministic / PRNG here) so a QRNG
bitstream can be substituted later without changing the measurement.

Usage:
    python bearer_state_competency.py [--device cuda:0] [--steps 1800] [--N 96] [--seeds 5]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from universality_test import UniversalEngine, make_macro_centers

OUT_DIR = os.path.join('outputs', 'bearer_state_competency')


def resolve_device(device: str) -> str:
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA requested but unavailable in this environment; using CPU.')
        return 'cpu'
    return device


# ============================================================================
# BEARER-STATE ENGINE
# ============================================================================

class BearerEngine(UniversalEngine):
    """UniversalEngine + persistent constitutive bearer state b_t.

    Reimplements step() (rather than patching the parent) because the bearer
    bias must enter the influence computation *before* activities/forces are
    derived, and the incorporation update must happen *after* — both points
    are inside the parent's single step() body.
    """

    def __init__(self, *args, use_bearer: bool = True, bearer_decay: float = 0.04,
                 bearer_gain: float = 0.35, bearer_weight: float = 0.6,
                 rng_seed: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Private RNG stream. Without this, two engines stepped in an
        # interleaved loop (as every perturbed/control comparison in this
        # codebase does) draw from Torch's single global RNG in alternation,
        # so "control" silently stops being a matched no-perturbation replay
        # of "perturbed" after the first step -- the two trajectories diverge
        # for a reason that has nothing to do with any injected signal. Each
        # engine must own its own generator, seeded independently, so
        # call-order/interleaving cannot contaminate the comparison.
        self.generator = None
        if rng_seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(rng_seed)
        self.use_bearer = use_bearer
        self.bearer_decay = bearer_decay
        self.bearer_gain = bearer_gain
        self.bearer_weight = bearer_weight
        self.b_t = torch.zeros(self.N, self.n_sub, device=self.device)
        # pending one-shot perturbations: list of (step_idx, tensor (N, n_sub))
        self._pending_xi: dict[int, torch.Tensor] = {}
        # pending lesions: {step_idx: sub_idx}
        self._pending_lesion: dict[int, int] = {}
        self._lesioned = torch.zeros(self.n_sub, dtype=torch.bool, device=self.device)
        # pending macro-center rotation: {step_idx: True}
        self._pending_rotation: dict[int, bool] = {}
        self.hist_bearer_norm = torch.zeros(self.N, self.max_steps, device=self.device)

    def schedule_perturbation(self, step_idx: int, xi: torch.Tensor):
        self._pending_xi[step_idx] = xi

    def schedule_lesion(self, step_idx: int, sub_idx: int):
        self._pending_lesion[step_idx] = sub_idx

    def schedule_rotation(self, step_idx: int):
        self._pending_rotation[step_idx] = True

    @torch.no_grad()
    def step(self):
        t = self.step_count
        N, dev = self.N, self.device
        u = self.u_t

        if t in self._pending_lesion:
            self._lesioned[self._pending_lesion[t]] = True

        if t in self._pending_rotation:
            # Environment shift: apply a random orthogonal rotation to the macro
            # basin centers' *positions*. A plain index permutation would leave
            # the occupancy distribution's entropy invariant by construction
            # (same multiset of probabilities, different labels) and would not
            # actually test adaptation to anything.
            dim = self.macro_centers.shape[1]
            raw = torch.randn(dim, dim, device=dev, generator=self.generator)
            q, r = torch.linalg.qr(raw)
            sign = torch.sign(torch.diagonal(r))
            q = q * sign.unsqueeze(0)
            self.macro_centers = self.macro_centers @ q.T

        prefs = self.prefs.clone()
        if self._lesioned.any():
            prefs[self._lesioned] = 0.0

        if self.manifold_type == 'flat4':
            u_dir = F.normalize(u, dim=1)
            influences = torch.einsum('nd,sd->ns', u_dir, prefs)
        else:
            influences = torch.einsum('nd,sd->ns', u, prefs)
        influences = 0.5 + 0.3 * influences

        if self.use_bearer:
            influences = influences + self.bearer_weight * self.b_t

        effective = influences * torch.exp(-self.fatigue)
        noise = self.exploration_noise * torch.randn(N, self.n_sub, device=dev, generator=self.generator)
        effective = (effective + noise).clamp(min=self.floor_value)
        activities = effective / (effective.sum(dim=1, keepdim=True) + 1e-8)

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
        elif self.fatigue_type == 'stochastic':
            spike = (torch.rand(N, self.n_sub, device=dev, generator=self.generator) < 0.3 * activities).float()
            self.fatigue = self.fatigue + self.fatigue_rate * spike
            self.fatigue = (self.fatigue - self.recovery_rate).clamp(0, 3)

        if self.manifold_type in ('s3', 's2'):
            radial = torch.einsum('sd,nd->ns', prefs, u)
            forces = prefs.unsqueeze(0) - radial.unsqueeze(2) * u.unsqueeze(1)
        else:
            forces = prefs.unsqueeze(0) - u.unsqueeze(1)

        activity_force = torch.einsum('ns,nsd->nd', activities, forces)
        rest_scores = torch.exp(-self.fatigue)
        novelty_force = torch.einsum('ns,nsd->nd', rest_scores, forces)
        mean_rest = rest_scores.mean(dim=1, keepdim=True)
        forces_mean = forces.mean(dim=1)
        novelty_force = novelty_force - mean_rest * forces_mean

        nw = self.novelty_weight
        drive = (1 - nw) * activity_force + nw * novelty_force

        raw_noise = self.exploration_noise * torch.randn(N, self.dim, device=dev, generator=self.generator)
        if self.manifold_type in ('s3', 's2'):
            noise_rad = (raw_noise * u).sum(dim=1, keepdim=True)
            drive = drive + raw_noise - noise_rad * u
        else:
            drive = drive + raw_noise

        new_u = u + self.steering_strength * drive
        if self.manifold_type in ('s3', 's2'):
            self.u_t = F.normalize(new_u, dim=1)
        else:
            norm = new_u.norm(dim=1, keepdim=True)
            scale = torch.where(norm > 2.0, 2.0 / norm, torch.ones_like(norm))
            self.u_t = new_u * scale

        if self.manifold_type in ('s3', 's2'):
            macro_sim = torch.einsum('nd,md->nm', self.u_t, self.macro_centers)
        else:
            dists = ((self.u_t.unsqueeze(1) - self.macro_centers.unsqueeze(0)) ** 2).sum(-1)
            macro_sim = -dists

        macro_weights = F.softmax(self.beta_macro * macro_sim, dim=1)
        dominant_basin = macro_weights.argmax(dim=1).int()

        resultant = torch.einsum('ns,nsd->nd', activities, forces)
        clarity = resultant.norm(dim=1)
        dominant_sub = activities.argmax(dim=1).int()

        # === INCORPORATION: activities become part of the persistent bearer ===
        if self.use_bearer:
            self.b_t = (1 - self.bearer_decay) * self.b_t + self.bearer_gain * activities
            if t in self._pending_xi:
                self.b_t = self.b_t + self._pending_xi[t]

        if t < self.max_steps:
            self.hist_clarity[:, t] = clarity
            self.hist_macro_basin[:, t] = dominant_basin
            self.hist_dominant_sub[:, t] = dominant_sub
            self.hist_bearer_norm[:, t] = self.b_t.norm(dim=1)

        self.step_count += 1


# ============================================================================
# COMPETENCY MEASUREMENTS
# ============================================================================

def build_engine(manifold, topology, use_bearer, device, steps, N, seed, rng_seed=None,
                  fatigue_rate=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    engine_kwargs = {}
    if fatigue_rate is not None:
        engine_kwargs['fatigue_rate'] = fatigue_rate
    return BearerEngine(
        N=N, device=device, steps=steps,
        manifold=manifold, topology=topology, fatigue_type='gradual',
        use_bearer=use_bearer,
        rng_seed=rng_seed if rng_seed is not None else seed,
        **engine_kwargs,
    )


def basin_occupancy(basins_slice: np.ndarray, n_macro: int) -> np.ndarray:
    counts = np.bincount(basins_slice.ravel(), minlength=n_macro).astype(np.float64)
    return counts / (counts.sum() + 1e-15)


def entropy(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log2(p + 1e-15)))


def memory_horizon(manifold, topology, use_bearer, device, steps, N, seed,
                    warmup=200, window=40, max_horizon=400) -> float:
    """Steps until a one-shot bearer perturbation's effect on basin occupancy
    decays below 1/e of its peak divergence from a matched control run."""
    xi = torch.zeros(N, 8, device=resolve_device(device))
    xi[:, 0] = 3.0  # push every being toward subsystem 0 for one step

    eng_pert = build_engine(manifold, topology, use_bearer, device, steps, N, seed)
    eng_ctrl = build_engine(manifold, topology, use_bearer, device, steps, N, seed)
    eng_pert.schedule_perturbation(warmup, xi)

    for _ in range(steps):
        eng_pert.step()
        eng_ctrl.step()

    basins_p = eng_pert.hist_macro_basin[:, :steps].cpu().numpy()
    basins_c = eng_ctrl.hist_macro_basin[:, :steps].cpu().numpy()
    n_macro = eng_pert.n_macro

    divergences = []
    n_win = (steps - warmup - window) // window
    for w in range(max(1, n_win)):
        t0 = warmup + w * window
        t1 = t0 + window
        if t1 > steps:
            break
        p_p = basin_occupancy(basins_p[:, t0:t1], n_macro)
        p_c = basin_occupancy(basins_c[:, t0:t1], n_macro)
        divergences.append(float(np.sum(np.abs(p_p - p_c))))

    if not divergences or max(divergences) < 1e-6:
        return 0.0
    peak = max(divergences)
    threshold = peak / np.e
    horizon_windows = len(divergences)
    for i, d in enumerate(divergences):
        if d < threshold and i > int(np.argmax(divergences)):
            horizon_windows = i
            break
    return float(horizon_windows * window)


def lesion_recovery(manifold, topology, use_bearer, device, steps, N, seed,
                     lesion_at=None, baseline_window=150, tol=0.10, fatigue_rate=None) -> float:
    if lesion_at is None:
        lesion_at = steps // 2
    eng = build_engine(manifold, topology, use_bearer, device, steps, N, seed, fatigue_rate=fatigue_rate)
    eng.schedule_lesion(lesion_at, sub_idx=0)
    for _ in range(steps):
        eng.step()

    clarity = eng.hist_clarity[:, :steps].cpu().numpy().mean(axis=0)
    pre = clarity[max(0, lesion_at - baseline_window):lesion_at]
    if len(pre) == 0:
        return dict(d_immediate=float('nan'), c_lesion=float('nan'), t_recovery=float('nan'))
    target = pre.mean()
    post = clarity[lesion_at:]

    # Immediate deficit: how much clarity dropped right after lesioning.
    immediate_window = min(20, len(post))
    d_immediate = float(target - post[:immediate_window].mean())

    # Cumulative deficit ("post-lesion regret"): sums the *shortfall* only,
    # so overshoot above baseline doesn't cancel out a real dip.
    cap = min(len(post), 600)
    c_lesion = float(np.sum(np.clip(target - post[:cap], 0, None)))

    # Recovery time is only a meaningful quantity if something was actually
    # disrupted -- otherwise "0 steps to recover" is indistinguishable from
    # "the lesion had no effect", and those are opposite findings.
    no_effect_threshold = 0.03 * (abs(target) + 1e-8)
    if d_immediate < no_effect_threshold:
        t_recovery = float('nan')
    else:
        within = np.abs(post - target) <= tol * (abs(target) + 1e-8)
        t_recovery = float(steps - lesion_at)
        for i, ok in enumerate(within):
            if ok and np.all(within[i:i + 20]):
                t_recovery = float(i)
                break

    return dict(d_immediate=d_immediate, c_lesion=c_lesion, t_recovery=t_recovery)


def adaptation_speed(manifold, topology, use_bearer, device, steps, N, seed,
                      shift_at=None, window=40, tol=0.05, fatigue_rate=None) -> float:
    if shift_at is None:
        shift_at = steps // 2
    eng = build_engine(manifold, topology, use_bearer, device, steps, N, seed, fatigue_rate=fatigue_rate)
    eng.schedule_rotation(shift_at)
    for _ in range(steps):
        eng.step()

    basins = eng.hist_macro_basin[:, :steps].cpu().numpy()
    n_macro = eng.n_macro
    pre_ent = entropy(basin_occupancy(basins[:, max(0, shift_at - window):shift_at], n_macro))

    n_win = (steps - shift_at - window) // window
    ents = []
    for w in range(max(1, n_win)):
        t0 = shift_at + w * window
        t1 = t0 + window
        ents.append(entropy(basin_occupancy(basins[:, t0:t1], n_macro)))
    ents = np.array(ents) if ents else np.array([pre_ent])

    # Immediate displacement: entropy jump in the first post-shift window.
    d_immediate = float(abs(ents[0] - pre_ent))
    c_adapt = float(np.sum(np.abs(ents - pre_ent)))

    no_effect_threshold = 0.05 * (abs(pre_ent) + 1e-8)
    if d_immediate < no_effect_threshold:
        t_adapt = float('nan')
    else:
        t_adapt = float(steps - shift_at)
        for w, ent in enumerate(ents):
            if abs(ent - pre_ent) <= tol * (abs(pre_ent) + 1e-8):
                t_adapt = float(w * window)
                break

    return dict(d_immediate=d_immediate, c_adapt=c_adapt, t_adapt=t_adapt)


def self_maintenance(manifold, topology, use_bearer, device, steps, N, seed) -> float:
    eng = build_engine(manifold, topology, use_bearer, device, steps, N, seed)
    for _ in range(steps):
        eng.step()
    clarity = eng.hist_clarity[:, :steps].cpu().numpy().mean(axis=0)
    tail = clarity[steps // 4:]
    mean, std = tail.mean(), tail.std()
    if mean < 1e-8:
        return 0.0
    cv = std / mean
    return float(1.0 / (cv + 1e-8))


def competency_vector(manifold, topology, use_bearer, device, steps, N, seed) -> dict:
    lesion = lesion_recovery(manifold, topology, use_bearer, device, steps, N, seed)
    adapt = adaptation_speed(manifold, topology, use_bearer, device, steps, N, seed)
    return {
        # memory_horizon is dominated by the fixed bearer_decay hyperparameter
        # (~1/e decay time of b_t itself) rather than being purely emergent
        # from geometry -- treat it as a sanity check on the decay constant,
        # not as evidence of geometrically-enabled memory on its own.
        'memory_horizon': memory_horizon(manifold, topology, use_bearer, device, steps, N, seed),
        'lesion_d_immediate': lesion['d_immediate'],
        'lesion_c_lesion': lesion['c_lesion'],
        'lesion_t_recovery': lesion['t_recovery'],
        'adapt_d_immediate': adapt['d_immediate'],
        'adapt_c_adapt': adapt['c_adapt'],
        'adapt_t_adapt': adapt['t_adapt'],
        'self_maintenance': self_maintenance(manifold, topology, use_bearer, device, steps, N, seed),
    }


# ============================================================================
# RUN MATRIX
# ============================================================================

MANIFOLDS = ['s3', 's2', 'flat4']
TOPOLOGIES = ['cyclic', 'random']


def run_matrix(device, steps, N, seeds, outdir):
    os.makedirs(outdir, exist_ok=True)
    device = resolve_device(device)
    rows = []
    total = len(MANIFOLDS) * len(TOPOLOGIES) * 2 * seeds
    done = 0
    t_start = time.time()

    for manifold in MANIFOLDS:
        for topology in TOPOLOGIES:
            for use_bearer in (False, True):
                for seed in range(seeds):
                    cv = competency_vector(manifold, topology, use_bearer, device, steps, N, seed)
                    row = dict(manifold=manifold, topology=topology,
                               use_bearer=use_bearer, seed=seed, **cv)
                    rows.append(row)
                    done += 1
                    elapsed = time.time() - t_start
                    print(f'  [{done}/{total}] {manifold:6s} {topology:7s} '
                          f'bearer={use_bearer!s:5s} seed={seed}  '
                          f'({elapsed:.0f}s elapsed)')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'competency_per_seed.csv'), index=False)

    metrics = ['memory_horizon',
               'lesion_d_immediate', 'lesion_c_lesion', 'lesion_t_recovery',
               'adapt_d_immediate', 'adapt_c_adapt', 'adapt_t_adapt',
               'self_maintenance']

    summary = (df.groupby(['manifold', 'topology', 'use_bearer'])[metrics]
               .agg(['mean', 'std']))
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(outdir, 'competency_summary.csv'), index=False)

    # bearer_on minus bearer_off, paired by (manifold, topology, seed)
    piv_on = df[df.use_bearer].set_index(['manifold', 'topology', 'seed'])
    piv_off = df[~df.use_bearer].set_index(['manifold', 'topology', 'seed'])
    delta = (piv_on[metrics] - piv_off[metrics]).reset_index()
    delta.to_csv(os.path.join(outdir, 'bearer_delta_per_seed.csv'), index=False)

    delta_summary = (delta.groupby(['manifold', 'topology'])[metrics]
                      .agg(['mean', 'std']))
    delta_summary.columns = ['_'.join(c) for c in delta_summary.columns]
    delta_summary = delta_summary.reset_index()

    with open(os.path.join(outdir, 'run_summary.json'), 'w') as f:
        json.dump({
            'n_configs': total,
            'steps': steps,
            'N': N,
            'seeds': seeds,
            'wall_time_s': time.time() - t_start,
            'delta_summary': json.loads(delta_summary.to_json(orient='records')),
        }, f, indent=2)

    plot_results(df, delta, metrics, outdir)
    return df, delta


def plot_results(df, delta, metrics, outdir):
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    configs = df[['manifold', 'topology']].drop_duplicates().values.tolist()
    labels = [f'{m}/{t}' for m, t in configs]
    x = np.arange(len(configs))
    width = 0.35

    for ax, metric in zip(axes, metrics):
        off_means, on_means, off_stds, on_stds = [], [], [], []
        for m, t in configs:
            sub_off = df[(df.manifold == m) & (df.topology == t) & (~df.use_bearer)][metric]
            sub_on = df[(df.manifold == m) & (df.topology == t) & (df.use_bearer)][metric]
            off_means.append(sub_off.mean())
            off_stds.append(sub_off.std())
            on_means.append(sub_on.mean())
            on_stds.append(sub_on.std())
        ax.bar(x - width / 2, off_means, width, yerr=off_stds, label='bearer off',
               color='#8172B3', capsize=3)
        ax.bar(x + width / 2, on_means, width, yerr=on_stds, label='bearer on',
               color='#4C72B0', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(metric)
        ax.legend(fontsize=8)

    fig.suptitle('Competency vector by interface geometry (bearer state ablation)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'competency_by_geometry.png'), dpi=140)
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--steps', type=int, default=1800)
    ap.add_argument('--N', type=int, default=96)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--outdir', default=OUT_DIR)
    args = ap.parse_args()

    print('Bearer State + Competency Vector')
    print(f'  device={args.device}  steps={args.steps}  N={args.N}  seeds={args.seeds}')
    df, delta = run_matrix(args.device, args.steps, args.N, args.seeds, args.outdir)

    print('\nMean bearer-on minus bearer-off delta by geometry:')
    metrics = ['memory_horizon',
               'lesion_d_immediate', 'lesion_c_lesion', 'lesion_t_recovery',
               'adapt_d_immediate', 'adapt_c_adapt', 'adapt_t_adapt',
               'self_maintenance']
    print(delta.groupby(['manifold', 'topology'])[metrics].mean().round(2).to_string())
    print(f'\nOutputs written to {args.outdir}/')


if __name__ == '__main__':
    main()
