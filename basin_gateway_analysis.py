#!/usr/bin/env python3
"""
Basin Gateway Analysis
======================

Identifies basin-to-basin "gateway" transitions that carry the strongest
clarity signal and post-transition persistence.

Outputs:
- outputs/basin_gateway/{manifold}_gateway_edges.csv
- outputs/basin_gateway/gateway_summary.json
- outputs/basin_gateway/gateway_heatmaps.png
- outputs/basin_gateway/top_gateway_edges.png

Usage:
    python basin_gateway_analysis.py [--device cuda:0] [--steps 2000] [--N 128]
"""

from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from universality_test import UniversalEngine

OUT_DIR = os.path.join('outputs', 'basin_gateway')


def run_engine(manifold: str, device: str, steps: int, n_beings: int):
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
    basins = engine.hist_macro_basin[:, :steps].cpu().numpy().astype(np.int32)
    clarity = engine.hist_clarity[:, :steps].cpu().numpy().astype(np.float32)
    return basins, clarity


def post_transition_dwell(seq: np.ndarray, idx: int) -> int:
    """
    Given a transition at idx -> idx+1, return how long sequence remains in seq[idx+1].
    Includes arrival timestep idx+1 in the dwell count.
    """
    target = seq[idx + 1]
    j = idx + 1
    length = 0
    while j < len(seq) and seq[j] == target:
        length += 1
        j += 1
    return length


def gateway_edge_table(basins: np.ndarray, clarity: np.ndarray) -> tuple[pd.DataFrame, dict]:
    n_beings, steps = basins.shape
    n_basins = int(basins.max()) + 1

    transitions = basins[:, 1:] != basins[:, :-1]
    dwell_clarity_global = float(clarity[:, 1:][~transitions].mean())

    edge_records = {}

    for b in range(n_beings):
        seq = basins[b]
        c = clarity[b]
        for t in range(steps - 1):
            src = int(seq[t])
            dst = int(seq[t + 1])
            if src == dst:
                continue

            key = (src, dst)
            rec = edge_records.setdefault(key, {
                'count': 0,
                'clarity_transition_sum': 0.0,
                'clarity_pre_sum': 0.0,
                'clarity_post_sum': 0.0,
                'dwell_after_sum': 0.0,
            })

            rec['count'] += 1
            rec['clarity_transition_sum'] += float(c[t + 1])
            rec['clarity_pre_sum'] += float(c[t])
            if t + 2 < steps:
                rec['clarity_post_sum'] += float(c[t + 2])
            else:
                rec['clarity_post_sum'] += float(c[t + 1])
            rec['dwell_after_sum'] += float(post_transition_dwell(seq, t))

    rows = []
    total_transitions = int(transitions.sum())

    for (src, dst), rec in edge_records.items():
        count = rec['count']
        mean_t = rec['clarity_transition_sum'] / count
        mean_pre = rec['clarity_pre_sum'] / count
        mean_post = rec['clarity_post_sum'] / count
        mean_dwell_after = rec['dwell_after_sum'] / count

        clarity_boost_vs_global = mean_t - dwell_clarity_global
        clarity_jump = mean_t - mean_pre
        persistence_gain = mean_dwell_after

        # Composite score favors frequent, high-clarity, sticky gateways.
        edge_score = (count / max(1, total_transitions)) * (clarity_boost_vs_global + 1e-6) * persistence_gain

        rows.append({
            'src_basin': src,
            'dst_basin': dst,
            'count': count,
            'transition_fraction': count / max(1, total_transitions),
            'mean_clarity_transition': mean_t,
            'mean_clarity_pre': mean_pre,
            'mean_clarity_post': mean_post,
            'clarity_boost_vs_global_dwell': clarity_boost_vs_global,
            'clarity_jump_pre_to_transition': clarity_jump,
            'mean_dwell_after_transition': mean_dwell_after,
            'gateway_score': edge_score,
        })

    df = pd.DataFrame(rows).sort_values(['gateway_score', 'count'], ascending=False)

    # Build heatmaps for counts and clarity boost.
    count_mat = np.zeros((n_basins, n_basins), dtype=np.float64)
    boost_mat = np.zeros((n_basins, n_basins), dtype=np.float64)
    for _, r in df.iterrows():
        i = int(r['src_basin'])
        j = int(r['dst_basin'])
        count_mat[i, j] = float(r['count'])
        boost_mat[i, j] = float(r['clarity_boost_vs_global_dwell'])

    meta = {
        'n_basins': n_basins,
        'total_transitions': total_transitions,
        'global_dwell_clarity': dwell_clarity_global,
        'count_matrix': count_mat,
        'boost_matrix': boost_mat,
    }
    return df, meta


def plot_heatmaps(meta_by_manifold: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, manifold in enumerate(['s3', 'flat4']):
        meta = meta_by_manifold[manifold]
        cmat = np.log1p(meta['count_matrix'])
        bmat = meta['boost_matrix']

        im1 = axes[0, col].imshow(cmat, cmap='viridis', aspect='auto')
        axes[0, col].set_title(f'{manifold} log(1+count)', fontweight='bold')
        axes[0, col].set_xlabel('to basin')
        axes[0, col].set_ylabel('from basin')
        plt.colorbar(im1, ax=axes[0, col], fraction=0.046, pad=0.04)

        im2 = axes[1, col].imshow(bmat, cmap='coolwarm', aspect='auto')
        axes[1, col].set_title(f'{manifold} clarity boost vs global dwell', fontweight='bold')
        axes[1, col].set_xlabel('to basin')
        axes[1, col].set_ylabel('from basin')
        plt.colorbar(im2, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'gateway_heatmaps.png'), dpi=160, bbox_inches='tight')
    plt.close()


def plot_top_edges(top_by_manifold: dict):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    for ax, manifold, color in zip(axes, ['s3', 'flat4'], ['#4C72B0', '#55A868']):
        top = top_by_manifold[manifold]
        labels = [f"{int(r['src_basin'])}->{int(r['dst_basin'])}" for _, r in top.iterrows()]
        vals = top['gateway_score'].values
        ax.barh(np.arange(len(vals)), vals, color=color)
        ax.set_yticks(np.arange(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f'{manifold} top gateway edges', fontweight='bold')
        ax.set_xlabel('gateway_score')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'top_gateway_edges.png'), dpi=160, bbox_inches='tight')
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Basin gateway edge analysis')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--N', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print('╔══════════════════════════════════════════════════════════════════╗')
    print('║ BASIN GATEWAY ANALYSIS                                          ║')
    print('║ ranked transition edges by clarity gain and persistence         ║')
    print('╚══════════════════════════════════════════════════════════════════╝')

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA requested but unavailable; falling back to CPU.')
        args.device = 'cpu'

    t0 = time.time()

    summary = {}
    meta_by_manifold = {}
    top_by_manifold = {}

    for manifold in ['s3', 'flat4']:
        print(f'\n-- Running {manifold} --')
        basins, clarity = run_engine(manifold, device=args.device, steps=args.steps, n_beings=args.N)
        df, meta = gateway_edge_table(basins, clarity)

        csv_path = os.path.join(OUT_DIR, f'{manifold}_gateway_edges.csv')
        df.to_csv(csv_path, index=False)

        top = df.head(12).copy()
        top_by_manifold[manifold] = top
        meta_by_manifold[manifold] = meta

        summary[manifold] = {
            'total_transitions': meta['total_transitions'],
            'global_dwell_clarity': meta['global_dwell_clarity'],
            'top_edges': top.to_dict(orient='records'),
            'mean_top_gateway_score': float(top['gateway_score'].mean()) if len(top) else 0.0,
            'mean_top_clarity_boost': float(top['clarity_boost_vs_global_dwell'].mean()) if len(top) else 0.0,
            'mean_top_dwell_after': float(top['mean_dwell_after_transition'].mean()) if len(top) else 0.0,
        }

        print(f'  saved: {csv_path}')
        print(f"  top edge score mean: {summary[manifold]['mean_top_gateway_score']:.5f}")
        print(f"  top clarity boost mean: {summary[manifold]['mean_top_clarity_boost']:.5f}")
        print(f"  top dwell-after mean: {summary[manifold]['mean_top_dwell_after']:.3f}")

    plot_heatmaps(meta_by_manifold)
    plot_top_edges(top_by_manifold)

    out_json = os.path.join(OUT_DIR, 'gateway_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    print(f'\nSaved: {out_json}')
    print(f'Saved: {os.path.join(OUT_DIR, "gateway_heatmaps.png")}')
    print(f'Saved: {os.path.join(OUT_DIR, "top_gateway_edges.png")}')
    print(f'Total wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
