#!/usr/bin/env python3
"""
Controller Ablation (Multi-Seed)
================================

Evaluates and ranks controller variants with confidence intervals:
- static_s3
- static_r4
- staged_no_guard
- staged_guard

Outputs:
- outputs/controller_ablation/per_seed_metrics.csv
- outputs/controller_ablation/variant_summary.csv
- outputs/controller_ablation/ablation_summary.json
- outputs/controller_ablation/ablation_rankings.png

Usage:
    python controller_ablation_multiseed.py [--device cuda:0] [--steps 1600] [--N 96] [--seeds 8]
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from universality_test import UniversalEngine
from staged_gateway_control import (
    ControlConfig,
    compute_window_gateway_metrics,
    run_staged_policy,
    resolve_device,
)

OUT_DIR = os.path.join('outputs', 'controller_ablation')

SCORE_PROFILES = {
    'diversity_first': {
        'mean_clarity': +0.75,
        'effective_basins': +1.00,
        'transition_rate': +0.25,
        'concentration_hhi': -1.00,
        'funnelness_mean': -0.90,
        'mean_dwell': -0.15,
    },
    'clarity_first': {
        'mean_clarity': +1.25,
        'effective_basins': +0.25,
        'transition_rate': +0.15,
        'concentration_hhi': -0.55,
        'funnelness_mean': -0.55,
        'mean_dwell': -0.10,
    },
}


@dataclass
class StaticConfig:
    centrality_window: int = 260
    centrality_stride: int = 20


def effective_basins(basin_slice: np.ndarray, n_basins: int) -> float:
    counts = np.bincount(basin_slice.ravel(), minlength=n_basins).astype(np.float64)
    p = counts / (counts.sum() + 1e-15)
    h = -np.sum(p * np.log2(p + 1e-15))
    return float(2 ** h)


def concentration_hhi(basin_slice: np.ndarray, n_basins: int) -> float:
    counts = np.bincount(basin_slice.ravel(), minlength=n_basins).astype(np.float64)
    p = counts / (counts.sum() + 1e-15)
    return float(np.sum(p ** 2))


def mean_dwell_length(basin_slice: np.ndarray) -> float:
    vals = []
    for row in basin_slice:
        cur = row[0]
        run = 1
        for v in row[1:]:
            if v == cur:
                run += 1
            else:
                vals.append(run)
                cur = v
                run = 1
        vals.append(run)
    return float(np.mean(vals)) if vals else 0.0


def run_static_variant(manifold: str, device: str, steps: int, n_beings: int, seed: int, cfg: StaticConfig):
    torch.manual_seed(seed)
    np.random.seed(seed)

    engine = UniversalEngine(
        N=n_beings,
        device=device,
        steps=steps,
        manifold=manifold,
        topology='cyclic',
        fatigue_type='gradual',
    )

    for _ in range(steps):
        engine.step()

    basins = engine.hist_macro_basin[:, :steps].detach().cpu().numpy().astype(np.int32)
    clarity = engine.hist_clarity[:, :steps].detach().cpu().numpy().astype(np.float32)
    n_basins = int(basins.max()) + 1

    transition_rate = float((basins[:, 1:] != basins[:, :-1]).mean())

    f_vals = []
    for t in range(cfg.centrality_window, steps, cfg.centrality_stride):
        t0 = t - cfg.centrality_window + 1
        t1 = t + 1
        b_win = basins[:, t0:t1]
        c_win = clarity[:, t0:t1]
        _, _, f, _ = compute_window_gateway_metrics(b_win, c_win, n_basins, 0, b_win.shape[1])
        f_vals.append(float(f))

    return {
        'mean_clarity': float(clarity[:, steps // 4:].mean()),
        'effective_basins': effective_basins(basins[:, steps // 4:], n_basins),
        'concentration_hhi': concentration_hhi(basins[:, steps // 4:], n_basins),
        'mean_dwell': mean_dwell_length(basins[:, steps // 4:]),
        'transition_rate': transition_rate,
        'funnelness_mean': float(np.mean(f_vals)) if f_vals else 0.0,
        'funnelness_max': float(np.max(f_vals)) if f_vals else 0.0,
        'switch_step': -1,
        'route_entropy_at_switch': np.nan,
    }


def ci95(series: pd.Series) -> tuple[float, float, float]:
    x = series.astype(float).values
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(max(1, len(x)))
    return m, m - half, m + half


def rank_variants(summary_df: pd.DataFrame, weights: dict[str, float], profile_name: str) -> pd.DataFrame:
    """Rank variants using a named score profile."""
    metrics = dict(weights)

    z = {}
    for k in metrics.keys():
        vals = summary_df[k].values.astype(float)
        mu = vals.mean()
        sd = vals.std() + 1e-12
        z[k] = (vals - mu) / sd

    score = np.zeros(len(summary_df), dtype=np.float64)
    for k, w in metrics.items():
        score += w * z[k]

    out = summary_df.copy()
    out['profile'] = profile_name
    out['composite_score'] = score
    out = out.sort_values('composite_score', ascending=False).reset_index(drop=True)
    out['rank'] = np.arange(1, len(out) + 1)
    return out


def plot_rankings(summary_ranked: pd.DataFrame, out_path: str, profile_name: str):
    variants = summary_ranked['variant'].tolist()
    x = np.arange(len(variants))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Composite score panel
    axes[0, 0].bar(variants, summary_ranked['composite_score'], color='#4C72B0')
    axes[0, 0].axhline(0, color='k', linewidth=0.4)
    axes[0, 0].set_title(f'Composite Controller Score [{profile_name}]', fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=20)

    # Clarity CI
    axes[0, 1].errorbar(
        x,
        summary_ranked['mean_clarity'],
        yerr=[summary_ranked['mean_clarity'] - summary_ranked['mean_clarity_ci_lo'],
              summary_ranked['mean_clarity_ci_hi'] - summary_ranked['mean_clarity']],
        fmt='o',
        capsize=4,
        color='#55A868',
    )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(variants, rotation=20)
    axes[0, 1].set_title('Mean Clarity (95% CI)', fontweight='bold')

    # Diversity/Concentration panel
    width = 0.35
    axes[1, 0].bar(x - width / 2, summary_ranked['effective_basins'], width, label='effective_basins', color='#8172B3')
    axes[1, 0].bar(x + width / 2, summary_ranked['concentration_hhi'], width, label='concentration_hhi', color='#C44E52')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(variants, rotation=20)
    axes[1, 0].set_title('Diversity vs Concentration', fontweight='bold')
    axes[1, 0].legend(fontsize=8)

    # Funnelness panel
    axes[1, 1].bar(x - width / 2, summary_ranked['funnelness_mean'], width, label='funnelness_mean', color='#64B5CD')
    axes[1, 1].bar(x + width / 2, summary_ranked['funnelness_max'], width, label='funnelness_max', color='#4878A8')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(variants, rotation=20)
    axes[1, 1].set_title('Funnelness', fontweight='bold')
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Multi-seed controller ablation runner')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--N', type=int, default=96)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--entropy-threshold', type=float, default=1.9)
    parser.add_argument('--route-window', type=int, default=240)
    parser.add_argument('--profiles', type=str, default='diversity_first,clarity_first')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = resolve_device(args.device)

    staged_cfg = ControlConfig(
        route_window=args.route_window,
        entropy_threshold=args.entropy_threshold,
    )
    static_cfg = StaticConfig()

    print('==============================================================')
    print('Controller Ablation Multi-Seed')
    print('Variants: static_s3, static_r4, staged_no_guard, staged_guard')
    print('==============================================================')

    t0 = time.time()

    rows = []

    for seed in range(args.seeds):
        print(f'\nSeed {seed + 1}/{args.seeds}')

        s3 = run_static_variant('s3', device, args.steps, args.N, seed, static_cfg)
        s3.update({'variant': 'static_s3', 'seed': seed})
        rows.append(s3)
        print(f"  static_s3       clarity={s3['mean_clarity']:.3f} funnel={s3['funnelness_mean']:.3f}")

        r4 = run_static_variant('flat4', device, args.steps, args.N, seed, static_cfg)
        r4.update({'variant': 'static_r4', 'seed': seed})
        rows.append(r4)
        print(f"  static_r4       clarity={r4['mean_clarity']:.3f} funnel={r4['funnelness_mean']:.3f}")

        s_ng, _, _ = run_staged_policy(
            device=device,
            steps=args.steps,
            n_beings=args.N,
            cfg=staged_cfg,
            seed=seed,
            anti_lock=False,
        )
        ng_row = {
            'variant': 'staged_no_guard',
            'seed': seed,
            'mean_clarity': s_ng['mean_clarity_post'],
            'effective_basins': s_ng['post_diversity_effective_basins'],
            'concentration_hhi': s_ng['post_concentration_hhi'],
            'mean_dwell': s_ng['post_mean_dwell'],
            'transition_rate': s_ng['transition_rate_post'],
            'funnelness_mean': s_ng['funnelness_mean'],
            'funnelness_max': s_ng['funnelness_max'],
            'switch_step': s_ng['switch_step'],
            'route_entropy_at_switch': s_ng['route_entropy_at_switch'],
        }
        rows.append(ng_row)
        print(f"  staged_no_guard clarity={ng_row['mean_clarity']:.3f} funnel={ng_row['funnelness_mean']:.3f}")

        s_g, _, _ = run_staged_policy(
            device=device,
            steps=args.steps,
            n_beings=args.N,
            cfg=staged_cfg,
            seed=seed,
            anti_lock=True,
        )
        g_row = {
            'variant': 'staged_guard',
            'seed': seed,
            'mean_clarity': s_g['mean_clarity_post'],
            'effective_basins': s_g['post_diversity_effective_basins'],
            'concentration_hhi': s_g['post_concentration_hhi'],
            'mean_dwell': s_g['post_mean_dwell'],
            'transition_rate': s_g['transition_rate_post'],
            'funnelness_mean': s_g['funnelness_mean'],
            'funnelness_max': s_g['funnelness_max'],
            'switch_step': s_g['switch_step'],
            'route_entropy_at_switch': s_g['route_entropy_at_switch'],
        }
        rows.append(g_row)
        print(f"  staged_guard    clarity={g_row['mean_clarity']:.3f} funnel={g_row['funnelness_mean']:.3f}")

    per_seed = pd.DataFrame(rows)
    per_seed_path = os.path.join(OUT_DIR, 'per_seed_metrics.csv')
    per_seed.to_csv(per_seed_path, index=False)

    summary_rows = []
    for variant, g in per_seed.groupby('variant'):
        row = {'variant': variant, 'n_seeds': int(len(g))}
        for metric in [
            'mean_clarity', 'effective_basins', 'concentration_hhi',
            'mean_dwell', 'transition_rate', 'funnelness_mean', 'funnelness_max',
        ]:
            m, lo, hi = ci95(g[metric])
            row[metric] = m
            row[f'{metric}_ci_lo'] = lo
            row[f'{metric}_ci_hi'] = hi

        # switch metrics for staged variants only (NaN for static).
        if g['switch_step'].notna().any() and (g['switch_step'] >= 0).any():
            valid = g[g['switch_step'] >= 0]
            row['switch_step_mean'] = float(valid['switch_step'].mean())
            row['route_entropy_at_switch_mean'] = float(valid['route_entropy_at_switch'].mean())
        else:
            row['switch_step_mean'] = np.nan
            row['route_entropy_at_switch_mean'] = np.nan

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    profile_names = [p.strip() for p in args.profiles.split(',') if p.strip()]
    ranked_by_profile = {}

    for profile_name in profile_names:
        if profile_name not in SCORE_PROFILES:
            raise ValueError(f'Unknown profile: {profile_name}. Available: {sorted(SCORE_PROFILES)}')

        ranked = rank_variants(summary_df, SCORE_PROFILES[profile_name], profile_name)
        ranked_by_profile[profile_name] = ranked

        summary_path = os.path.join(OUT_DIR, f'variant_summary_{profile_name}.csv')
        ranked.to_csv(summary_path, index=False)

        fig_path = os.path.join(OUT_DIR, f'ablation_rankings_{profile_name}.png')
        plot_rankings(ranked, fig_path, profile_name)

    # Backward-compatible default artifacts use the first requested profile.
    default_profile = profile_names[0]
    ranked = ranked_by_profile[default_profile]
    summary_path = os.path.join(OUT_DIR, 'variant_summary.csv')
    ranked.to_csv(summary_path, index=False)

    fig_path = os.path.join(OUT_DIR, 'ablation_rankings.png')
    plot_rankings(ranked, fig_path, default_profile)

    json_path = os.path.join(OUT_DIR, 'ablation_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'config': {
                    'steps': args.steps,
                    'N': args.N,
                    'seeds': args.seeds,
                    'entropy_threshold': args.entropy_threshold,
                    'route_window': args.route_window,
                    'profiles': profile_names,
                },
                'rankings_by_profile': {
                    profile: table.to_dict(orient='records')
                    for profile, table in ranked_by_profile.items()
                },
            },
            f,
            indent=2,
        )

    elapsed = time.time() - t0
    print('\nSaved artifacts:')
    print(f'  {per_seed_path}')
    print(f'  {summary_path}')
    print(f'  {json_path}')
    print(f'  {fig_path}')
    for profile_name in profile_names[1:]:
        print(f"  {os.path.join(OUT_DIR, f'variant_summary_{profile_name}.csv')}")
        print(f"  {os.path.join(OUT_DIR, f'ablation_rankings_{profile_name}.png')}")
    print(f'Total wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
