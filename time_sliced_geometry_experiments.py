#!/usr/bin/env python3
"""
Time-Sliced Geometry Experiments for Boundary Negotiation
=========================================================

Implements three extensions motivated by time-slicing quantum geometry ideas:

1) Static vs time-sliced manifold schedules (S3/R4 switching)
2) Discrete angular preference family (Z_n-inspired)
3) Coupling sweep (novelty_weight, beta_macro)

All experiments reuse the existing boundary_metrics() so results are directly
comparable to prior FourD analyses.

Usage:
    python time_sliced_geometry_experiments.py [--device cuda:0] [--steps 1200] [--N 96]
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from universality_test import UniversalEngine, boundary_metrics, make_macro_centers

OUT_DIR = os.path.join('outputs', 'time_sliced_geometry')


def resolve_device(device: str) -> str:
    """Fallback to CPU if a CUDA device is requested but unavailable."""
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA requested but unavailable in this Python environment; falling back to CPU.')
        return 'cpu'
    return device


def build_schedule(name: str, steps: int, slice_len: int = 120) -> List[str]:
    """Build a per-step manifold schedule using only 4D-compatible manifolds."""
    if name == 'static_s3':
        return ['s3'] * steps
    if name == 'static_r4':
        return ['flat4'] * steps
    if name == 's3_to_r4':
        pivot = steps // 2
        return ['s3'] * pivot + ['flat4'] * (steps - pivot)
    if name == 'r4_to_s3':
        pivot = steps // 2
        return ['flat4'] * pivot + ['s3'] * (steps - pivot)
    if name == 'alternating':
        sched = []
        cur = 's3'
        for _ in range(steps):
            if len(sched) % slice_len == 0 and len(sched) > 0:
                cur = 'flat4' if cur == 's3' else 's3'
            sched.append(cur)
        return sched
    raise ValueError(f'Unknown schedule: {name}')


def zn_preferences(n_sub: int = 8, n_sectors: int = 8) -> np.ndarray:
    """
    Build a Z_n-inspired angular opponent family in R4.

    Four base directions are sampled on the first 2 coordinates with angular
    spacing 2pi/n, then mirrored to form opponent pairs.
    """
    if n_sub != 8:
        raise ValueError('zn_preferences currently expects n_sub=8')

    angle_step = 2.0 * np.pi / float(n_sectors)
    prefs = np.zeros((n_sub, 4), dtype=np.float32)

    for i in range(4):
        theta = i * angle_step
        # Add a small second-harmonic component in dims 3-4 to avoid planar collapse.
        v = np.array([
            np.cos(theta),
            np.sin(theta),
            0.35 * np.cos(2.0 * theta),
            0.35 * np.sin(2.0 * theta),
        ], dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        prefs[i] = v
        prefs[i + 4] = -v

    return prefs


def run_engine_with_schedule(
    schedule: List[str],
    device: str,
    n_beings: int,
    preferences: np.ndarray | None = None,
    seed: int = 0,
    novelty_weight: float = 0.6,
    beta_macro: float = 11.375,
    fatigue_rate: float = 0.217,
    steering_strength: float = 0.707,
) -> Dict[str, float]:
    """Run one simulation with optional manifold switching and custom preferences."""
    if len(schedule) < 10:
        raise ValueError('Schedule must have at least 10 steps')

    torch.manual_seed(seed)
    np.random.seed(seed)

    steps = len(schedule)
    first = schedule[0]
    if first not in ('s3', 'flat4'):
        raise ValueError('This script supports only s3 and flat4 schedules')

    engine = UniversalEngine(
        N=n_beings,
        device=device,
        steps=steps,
        manifold=first,
        topology='cyclic',
        fatigue_type='gradual',
        fatigue_rate=float(fatigue_rate),
        steering_strength=float(steering_strength),
    )

    engine.novelty_weight = float(novelty_weight)
    engine.beta_macro = float(beta_macro)

    if preferences is not None:
        pref_t = torch.tensor(preferences, dtype=torch.float32, device=engine.device)
        pref_t = F.normalize(pref_t, dim=1)
        engine.prefs = pref_t

    centers = {
        's3': make_macro_centers('s3', n_macro=24, device=device),
        'flat4': make_macro_centers('flat4', n_macro=24, device=device),
    }

    active = first
    engine.manifold_type = active
    engine.macro_centers = centers[active]

    for t in range(steps):
        manifold = schedule[t]
        if manifold != active:
            active = manifold
            engine.manifold_type = active
            engine.macro_centers = centers[active]
        engine.step()

    metrics = boundary_metrics(engine, n_shuffles=40)

    warmup = steps // 4
    clarity = float(engine.hist_clarity[:, warmup:steps].mean().item())
    metrics['mean_clarity'] = clarity
    return metrics


def experiment_a_static_vs_sliced(device: str, steps: int, n_beings: int) -> Dict[str, Dict[str, float]]:
    """Compare baseline static manifolds against time-sliced schedules."""
    print('\n  -- Experiment A: Static vs Time-Sliced --')
    schedules = {
        'static_s3': build_schedule('static_s3', steps),
        'static_r4': build_schedule('static_r4', steps),
        's3_to_r4': build_schedule('s3_to_r4', steps),
        'r4_to_s3': build_schedule('r4_to_s3', steps),
        'alternating': build_schedule('alternating', steps, slice_len=max(40, steps // 10)),
    }

    out: Dict[str, Dict[str, float]] = {}
    for name, sched in schedules.items():
        print(f'    {name:>11s}:', end='', flush=True)
        m = run_engine_with_schedule(sched, device=device, n_beings=n_beings, seed=0)
        out[name] = m
        print(f"  r={m['edge_clarity_r']:+.3f} d={m['cohens_d']:+.3f} z={m['null_z']:+.1f} TR={m['transition_rate']:.3f}")

    labels = list(out.keys())
    x = np.arange(len(labels))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x - 1.5 * width, [out[k]['edge_clarity_r'] for k in labels], width, label='edge_clarity_r')
    axes[0].bar(x - 0.5 * width, [out[k]['cohens_d'] for k in labels], width, label='cohens_d')
    axes[0].bar(x + 0.5 * width, [out[k]['transition_rate'] for k in labels], width, label='transition_rate')
    axes[0].bar(x + 1.5 * width, [out[k]['mean_clarity'] for k in labels], width, label='mean_clarity')
    axes[0].axhline(0, color='k', linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].set_title('Core Metrics by Schedule', fontweight='bold')
    axes[0].legend(fontsize=8)

    axes[1].bar(labels, [out[k]['null_z'] for k in labels], color='#4C72B0')
    axes[1].axhline(0, color='k', linewidth=0.4)
    axes[1].set_title('Null-Model z by Schedule', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=20)

    plt.suptitle('Experiment A: Static vs Time-Sliced Geometry', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp_a_static_vs_sliced.png'), dpi=150, bbox_inches='tight')
    plt.close()
    return out


def experiment_b_zn_family(device: str, steps: int, n_beings: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate Z_n-inspired angular preference families on both manifolds."""
    print('\n  -- Experiment B: Z_n Angular Family --')
    sectors = [3, 4, 6, 8, 12]
    manifolds = ['s3', 'flat4']

    result: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in manifolds}
    for manifold in manifolds:
        print(f'    {manifold}:', end='', flush=True)
        schedule = build_schedule('static_s3' if manifold == 's3' else 'static_r4', steps)
        for n in sectors:
            prefs = zn_preferences(n_sub=8, n_sectors=n)
            m = run_engine_with_schedule(
                schedule,
                device=device,
                n_beings=n_beings,
                preferences=prefs,
                seed=7 + n,
            )
            result[manifold][f'Z{n}'] = m
            print('.', end='', flush=True)
        print(' done')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.array(sectors)

    for manifold, color in [('s3', '#4C72B0'), ('flat4', '#55A868')]:
        edge_r = [result[manifold][f'Z{n}']['edge_clarity_r'] for n in sectors]
        cohens_d = [result[manifold][f'Z{n}']['cohens_d'] for n in sectors]
        axes[0].plot(x, edge_r, 'o-', color=color, label=manifold)
        axes[1].plot(x, cohens_d, 'o-', color=color, label=manifold)

    axes[0].set_xlabel('n in Z_n')
    axes[0].set_ylabel('edge_clarity_r')
    axes[0].set_title('Boundary Coupling vs Angular Resolution', fontweight='bold')
    axes[0].axhline(0, color='k', linewidth=0.4)

    axes[1].set_xlabel('n in Z_n')
    axes[1].set_ylabel("Cohen's d")
    axes[1].set_title('Transition Clarity Boost vs Angular Resolution', fontweight='bold')
    axes[1].axhline(0, color='k', linewidth=0.4)

    for ax in axes:
        ax.legend()

    plt.suptitle('Experiment B: Z_n Angular Preferences', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp_b_zn_family.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return result


def experiment_c_coupling_sweep(device: str, steps: int, n_beings: int) -> Dict[str, Dict[str, object]]:
    """Sweep novelty and macro-softmax sharpness for static and sliced schedules."""
    print('\n  -- Experiment C: Coupling Sweep --')
    novelty_vals = np.linspace(0.0, 1.0, 6)
    beta_vals = np.array([6.0, 9.0, 11.375, 14.0, 17.0])

    schedules = {
        'static_r4': build_schedule('static_r4', steps),
        's3_to_r4': build_schedule('s3_to_r4', steps),
    }

    out: Dict[str, Dict[str, object]] = {}

    for name, sched in schedules.items():
        print(f'    {name}:', end='', flush=True)
        grid = np.zeros((len(novelty_vals), len(beta_vals)), dtype=np.float32)

        for i, nw in enumerate(novelty_vals):
            for j, bm in enumerate(beta_vals):
                m = run_engine_with_schedule(
                    sched,
                    device=device,
                    n_beings=n_beings,
                    novelty_weight=float(nw),
                    beta_macro=float(bm),
                    seed=100 + i * 10 + j,
                )
                grid[i, j] = m['edge_clarity_r']
            print('.', end='', flush=True)
        print(' done')

        out[name] = {
            'novelty_vals': novelty_vals.tolist(),
            'beta_vals': beta_vals.tolist(),
            'edge_clarity_grid': grid.tolist(),
            'present_fraction': float((grid > 0.3).sum() / grid.size),
            'strong_fraction': float((grid > 0.5).sum() / grid.size),
        }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, name in zip(axes, ['static_r4', 's3_to_r4']):
        grid = np.array(out[name]['edge_clarity_grid'])
        im = ax.imshow(
            grid,
            origin='lower',
            aspect='auto',
            vmin=-0.5,
            vmax=1.0,
            cmap='RdYlGn',
            extent=[beta_vals[0], beta_vals[-1], novelty_vals[0], novelty_vals[-1]],
        )
        ax.set_xlabel('beta_macro')
        ax.set_ylabel('novelty_weight')
        ax.set_title(
            f"{name} | present={out[name]['present_fraction']:.0%}, strong={out[name]['strong_fraction']:.0%}",
            fontweight='bold',
        )
        plt.colorbar(im, ax=ax, label='edge_clarity_r')

    plt.suptitle('Experiment C: Coupling Sweep Sensitivity', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp_c_coupling_sweep.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Time-sliced geometry experiments')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--N', type=int, default=96)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    resolved_device = resolve_device(args.device)

    print('╔═══════════════════════════════════════════════════════════════════╗')
    print('║  TIME-SLICED GEOMETRY EXPERIMENTS                                ║')
    print('║  Static vs sliced manifolds, Z_n angular family, coupling sweep  ║')
    print('║  Compatible with FourD boundary_metrics                          ║')
    print('╚═══════════════════════════════════════════════════════════════════╝')

    if torch.cuda.is_available() and resolved_device.startswith('cuda'):
        props = torch.cuda.get_device_properties(0)
        print(f'\nGPU: {props.name} ({props.total_memory / 1e9:.1f} GB)')

    t0 = time.time()

    all_results: Dict[str, object] = {}
    all_results['experiment_a'] = experiment_a_static_vs_sliced(resolved_device, args.steps, args.N)
    all_results['experiment_b'] = experiment_b_zn_family(resolved_device, args.steps, args.N)
    all_results['experiment_c'] = experiment_c_coupling_sweep(
        resolved_device,
        steps=min(args.steps, 1000),
        n_beings=min(args.N, 64),
    )

    out_path = os.path.join(OUT_DIR, 'time_sliced_geometry_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f'\nSaved results to: {out_path}')
    print(f'Total wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
