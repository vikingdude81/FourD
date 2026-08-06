#!/usr/bin/env python3
"""
Staged Gateway Control + Funnelness Analysis
===========================================

Implements requested experiments:
1) Gateway-centrality metric over time (inbound weighted by gateway_score)
2) Staged S3 -> R4 control policy triggered by route-entropy threshold
3) Anti-lock safeguard in R4 penalizing repeated entry to dominant destination basin

Outputs:
- outputs/staged_gateway_control/control_summary.json
- outputs/staged_gateway_control/centrality_timeseries.csv
- outputs/staged_gateway_control/inbound_centrality.csv
- outputs/staged_gateway_control/funnelness_entropy_timeseries.png
- outputs/staged_gateway_control/inbound_centrality_top_basins.png
- outputs/staged_gateway_control/pre_post_comparison.png

Usage:
    python staged_gateway_control.py [--device cuda:0] [--steps 2400] [--N 128]
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter, deque
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from universality_test import UniversalEngine, make_macro_centers

OUT_DIR = os.path.join('outputs', 'staged_gateway_control')


@dataclass
class ControlConfig:
    route_window: int = 240
    min_switch_step: int = 500
    entropy_threshold: float = 0.95
    centrality_window: int = 260
    centrality_stride: int = 20
    dominance_window: int = 160
    dominance_threshold: float = 0.42
    anti_lock_strength: float = 0.22
    anti_lock_noise_boost: float = 1.20


def resolve_device(device: str) -> str:
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA requested but unavailable; falling back to CPU.')
        return 'cpu'
    return device


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    p = p / (p.sum() + 1e-15)
    return float(-np.sum(p * np.log2(p + 1e-15)))


def transition_entropy(basins_window: np.ndarray, n_basins: int) -> float:
    """Entropy rate from aggregated transition matrix in a window."""
    trans = np.zeros((n_basins, n_basins), dtype=np.float64)
    src = basins_window[:, :-1].ravel()
    dst = basins_window[:, 1:].ravel()
    np.add.at(trans, (src, dst), 1)

    row_sum = trans.sum(axis=1, keepdims=True)
    P = trans / (row_sum + 1e-15)
    occ = np.bincount(basins_window.ravel(), minlength=n_basins).astype(np.float64)
    pi = occ / (occ.sum() + 1e-15)
    row_h = -np.sum(P * np.log2(P + 1e-15), axis=1)
    return float(np.sum(pi * row_h))


def compute_window_gateway_metrics(
    basins: np.ndarray,
    clarity: np.ndarray,
    n_basins: int,
    t0: int,
    t1: int,
):
    """
    Gateway score in [t0, t1):
      score(i->j) = frac(i->j) * max(0, clarity_boost) * (1 + persistence)
    where persistence = P(stay in j at t+2 | transition to j at t+1).
    """
    if t1 - t0 < 4:
        return np.zeros((n_basins, n_basins)), np.zeros(n_basins), 0.0, -1

    b = basins[:, t0:t1]
    c = clarity[:, t0:t1]
    trans_mask = b[:, 1:] != b[:, :-1]

    dwell_vals = c[:, 1:][~trans_mask]
    global_dwell = float(dwell_vals.mean()) if dwell_vals.size > 0 else float(c.mean())

    total_transitions = int(trans_mask.sum())
    if total_transitions == 0:
        return np.zeros((n_basins, n_basins)), np.zeros(n_basins), 0.0, -1

    count = np.zeros((n_basins, n_basins), dtype=np.float64)
    clarity_sum = np.zeros((n_basins, n_basins), dtype=np.float64)
    persist_sum = np.zeros((n_basins, n_basins), dtype=np.float64)

    n_beings, w_steps = b.shape
    for i in range(n_beings):
        for t in range(w_steps - 1):
            src = int(b[i, t])
            dst = int(b[i, t + 1])
            if src == dst:
                continue
            count[src, dst] += 1
            clarity_sum[src, dst] += float(c[i, t + 1])
            if t + 2 < w_steps:
                persist_sum[src, dst] += float(b[i, t + 2] == dst)
            else:
                persist_sum[src, dst] += 1.0

    frac = count / float(total_transitions)
    mean_clarity = clarity_sum / (count + 1e-15)
    persistence = persist_sum / (count + 1e-15)
    clarity_boost = np.maximum(0.0, mean_clarity - global_dwell)

    edge_score = frac * clarity_boost * (1.0 + persistence)
    inbound = edge_score.sum(axis=0)
    pos_sum = float(np.sum(inbound))
    if pos_sum > 0:
        funnelness = float(np.max(inbound) / pos_sum)
        top_basin = int(np.argmax(inbound))
    else:
        funnelness = 0.0
        top_basin = -1

    return edge_score, inbound, funnelness, top_basin


def effective_basins(basin_slice: np.ndarray, n_basins: int) -> float:
    counts = np.bincount(basin_slice.ravel(), minlength=n_basins)
    h = shannon_entropy_from_counts(counts)
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


def apply_anti_lock_repulsion(
    engine: UniversalEngine,
    curr_basins: np.ndarray,
    prev_basins: np.ndarray,
    dominant_dst: int,
    strength: float,
):
    """
    Penalize repeated entry into dominant basin by repelling newly-entered states
    from that basin center in flat R4 mode.
    """
    entry_mask = (curr_basins == dominant_dst) & (prev_basins != dominant_dst)
    if not np.any(entry_mask):
        return

    idx = np.where(entry_mask)[0]
    idx_t = torch.tensor(idx, device=engine.device, dtype=torch.long)
    center = engine.macro_centers[dominant_dst]

    # Euclidean repulsion: move away from center and keep within existing cap behavior.
    u = engine.u_t[idx_t]
    delta = u - center.unsqueeze(0)
    engine.u_t[idx_t] = u + strength * delta

    norm = engine.u_t[idx_t].norm(dim=1, keepdim=True)
    scale = torch.where(norm > 2.0, 2.0 / norm, torch.ones_like(norm))
    engine.u_t[idx_t] = engine.u_t[idx_t] * scale


def run_staged_policy(
    device: str,
    steps: int,
    n_beings: int,
    cfg: ControlConfig,
    seed: int,
    anti_lock: bool,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    engine = UniversalEngine(
        N=n_beings,
        device=device,
        steps=steps,
        manifold='s3',
        topology='cyclic',
        fatigue_type='gradual',
    )

    centers = {
        's3': make_macro_centers('s3', n_macro=24, device=device),
        'flat4': make_macro_centers('flat4', n_macro=24, device=device),
    }

    engine.manifold_type = 's3'
    engine.macro_centers = centers['s3']

    switched = False
    switch_step = None

    n_basins = 24

    entropy_series = np.full(steps, np.nan, dtype=np.float64)
    mode_series = np.zeros(steps, dtype=np.int32)  # 0=S3,1=R4

    # For anti-lock destination dominance tracking.
    recent_entries = deque(maxlen=cfg.dominance_window)

    # Capture rolling funnelness/centrality.
    t_points = []
    funnelness = []
    top_basin = []
    top_basin_share = []
    inbound_rows = []

    prev_basins = None

    base_noise = float(engine.exploration_noise)

    for t in range(steps):
        # Update manifold mode before step.
        mode_series[t] = 1 if switched else 0
        if switched:
            engine.manifold_type = 'flat4'
            engine.macro_centers = centers['flat4']
        else:
            engine.manifold_type = 's3'
            engine.macro_centers = centers['s3']

        engine.step()

        curr = engine.hist_macro_basin[:, t].detach().cpu().numpy().astype(np.int32)

        # Route entropy for trigger (computed after enough history).
        if t >= cfg.route_window:
            w = engine.hist_macro_basin[:, t - cfg.route_window + 1:t + 1].detach().cpu().numpy().astype(np.int32)
            h = transition_entropy(w, n_basins=n_basins)
            entropy_series[t] = h

            if (not switched) and (t >= cfg.min_switch_step) and (h <= cfg.entropy_threshold):
                switched = True
                switch_step = t

        # Anti-lock safeguard in R4.
        if switched and anti_lock and prev_basins is not None:
            entered = curr[prev_basins != curr]
            for d in entered.tolist():
                recent_entries.append(int(d))

            if len(recent_entries) >= max(20, cfg.dominance_window // 3):
                c = Counter(recent_entries)
                dom_basin, dom_count = c.most_common(1)[0]
                dom_share = dom_count / max(1, len(recent_entries))

                if dom_share >= cfg.dominance_threshold:
                    apply_anti_lock_repulsion(
                        engine,
                        curr_basins=curr,
                        prev_basins=prev_basins,
                        dominant_dst=dom_basin,
                        strength=cfg.anti_lock_strength,
                    )
                    # Slightly increase exploration while dominance is high.
                    engine.exploration_noise = base_noise * cfg.anti_lock_noise_boost
                else:
                    engine.exploration_noise = base_noise
            else:
                engine.exploration_noise = base_noise

        # Rolling centrality/funnelness snapshots.
        if (t >= cfg.centrality_window) and (t % cfg.centrality_stride == 0):
            t0 = t - cfg.centrality_window + 1
            t1 = t + 1
            b_win = engine.hist_macro_basin[:, t0:t1].detach().cpu().numpy().astype(np.int32)
            c_win = engine.hist_clarity[:, t0:t1].detach().cpu().numpy().astype(np.float32)
            _, inbound, f, top = compute_window_gateway_metrics(b_win, c_win, n_basins, 0, b_win.shape[1])

            t_points.append(t)
            funnelness.append(f)
            top_basin.append(top)
            share = float(inbound[top] / (np.sum(inbound) + 1e-15)) if top >= 0 else 0.0
            top_basin_share.append(share)

            for b_id in range(n_basins):
                inbound_rows.append({
                    'mode': 'guard' if anti_lock else 'no_guard',
                    't': t,
                    'basin': b_id,
                    'inbound_centrality': float(inbound[b_id]),
                })

        prev_basins = curr

    basins = engine.hist_macro_basin[:, :steps].detach().cpu().numpy().astype(np.int32)
    clarity = engine.hist_clarity[:, :steps].detach().cpu().numpy().astype(np.float32)

    # If never switched, set switch step to end for consistent slicing.
    if switch_step is None:
        switch_step = steps // 2

    pre0 = max(0, switch_step - 400)
    pre1 = switch_step
    post0 = min(steps - 1, switch_step + 1)
    post1 = min(steps, switch_step + 401)

    pre_slice = basins[:, pre0:pre1] if pre1 > pre0 else basins[:, :steps // 2]
    post_slice = basins[:, post0:post1] if post1 > post0 else basins[:, steps // 2:]

    summary = {
        'mode': 'guard' if anti_lock else 'no_guard',
        'switch_step': int(switch_step),
        'switched_fraction': float(np.mean(mode_series)),
        'route_entropy_at_switch': float(entropy_series[switch_step]) if np.isfinite(entropy_series[switch_step]) else None,
        'pre_diversity_effective_basins': effective_basins(pre_slice, n_basins),
        'post_diversity_effective_basins': effective_basins(post_slice, n_basins),
        'pre_concentration_hhi': concentration_hhi(pre_slice, n_basins),
        'post_concentration_hhi': concentration_hhi(post_slice, n_basins),
        'pre_mean_dwell': mean_dwell_length(pre_slice),
        'post_mean_dwell': mean_dwell_length(post_slice),
        'mean_clarity_pre': float(clarity[:, pre0:pre1].mean()) if pre1 > pre0 else float(clarity.mean()),
        'mean_clarity_post': float(clarity[:, post0:post1].mean()) if post1 > post0 else float(clarity.mean()),
        'transition_rate_pre': float((pre_slice[:, 1:] != pre_slice[:, :-1]).mean()) if pre_slice.shape[1] > 1 else 0.0,
        'transition_rate_post': float((post_slice[:, 1:] != post_slice[:, :-1]).mean()) if post_slice.shape[1] > 1 else 0.0,
        'funnelness_mean': float(np.mean(funnelness)) if funnelness else 0.0,
        'funnelness_max': float(np.max(funnelness)) if funnelness else 0.0,
        'top_basin_share_mean': float(np.mean(top_basin_share)) if top_basin_share else 0.0,
    }

    ts_rows = []
    for i, t in enumerate(t_points):
        ts_rows.append({
            'mode': summary['mode'],
            't': int(t),
            'funnelness': float(funnelness[i]),
            'top_basin': int(top_basin[i]) if top_basin[i] >= 0 else -1,
            'top_basin_share': float(top_basin_share[i]),
            'route_entropy': float(entropy_series[t]) if np.isfinite(entropy_series[t]) else np.nan,
            'active_mode': 'flat4' if mode_series[t] == 1 else 's3',
            'switch_step': int(switch_step),
        })

    return summary, ts_rows, inbound_rows


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(ts_rows: list[dict], inbound_rows: list[dict], summary: dict):
    df_ts = pd_from_rows(ts_rows)
    df_in = pd_from_rows(inbound_rows)

    # Plot 1: funnelness + entropy over time.
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for mode, color in [('no_guard', '#4C72B0'), ('guard', '#DD8452')]:
        d = df_ts[df_ts['mode'] == mode]
        if d.empty:
            continue
        axes[0].plot(d['t'], d['funnelness'], label=mode, color=color, linewidth=1.8)
        axes[1].plot(d['t'], d['route_entropy'], label=mode, color=color, linewidth=1.8)
        sw = int(d['switch_step'].iloc[0])
        axes[0].axvline(sw, color=color, linestyle='--', alpha=0.45)
        axes[1].axvline(sw, color=color, linestyle='--', alpha=0.45)

    axes[0].set_ylabel('Funnelness')
    axes[0].set_title('Inbound Gateway Funnelness Over Time', fontweight='bold')
    axes[0].legend()
    axes[1].set_ylabel('Route Entropy (bits/step)')
    axes[1].set_xlabel('Timestep')
    axes[1].set_title('Route Entropy and Switch Points', fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'funnelness_entropy_timeseries.png'), dpi=160, bbox_inches='tight')
    plt.close()

    # Plot 2: inbound centrality of top basins over time (per mode).
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, mode, color in zip(axes, ['no_guard', 'guard'], ['#4C72B0', '#DD8452']):
        d = df_in[df_in['mode'] == mode]
        if d.empty:
            ax.set_title(f'{mode} (no data)')
            continue
        by_basin = d.groupby('basin')['inbound_centrality'].mean().sort_values(ascending=False)
        top_basins = by_basin.head(4).index.tolist()
        for b in top_basins:
            line = d[d['basin'] == b].sort_values('t')
            ax.plot(line['t'], line['inbound_centrality'], label=f'basin {b}')
        ax.set_title(f'{mode}: top inbound basins', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Inbound centrality')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'inbound_centrality_top_basins.png'), dpi=160, bbox_inches='tight')
    plt.close()

    # Plot 3: pre/post comparison bars.
    metrics = [
        ('pre_diversity_effective_basins', 'post_diversity_effective_basins', 'Effective Basins'),
        ('pre_concentration_hhi', 'post_concentration_hhi', 'Concentration HHI'),
        ('pre_mean_dwell', 'post_mean_dwell', 'Mean Dwell'),
        ('transition_rate_pre', 'transition_rate_post', 'Transition Rate'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    modes = ['no_guard', 'guard']
    colors = ['#4C72B0', '#DD8452']

    for ax, (k_pre, k_post, title) in zip(axes, metrics):
        x = np.arange(2)
        width = 0.35
        pre_vals = [summary[m][k_pre] for m in modes]
        post_vals = [summary[m][k_post] for m in modes]
        ax.bar(x - width / 2, pre_vals, width, label='pre', color='#AAB7CF')
        ax.bar(x + width / 2, post_vals, width, label='post', color='#6078A8')
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=15)
        ax.set_title(title, fontweight='bold')
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pre_post_comparison.png'), dpi=160, bbox_inches='tight')
    plt.close()


def pd_from_rows(rows: list[dict]):
    # Lazy local import so pandas remains optional until needed.
    import pandas as pd
    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Staged gateway control experiment')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--entropy-threshold', type=float, default=0.95)
    parser.add_argument('--route-window', type=int, default=240)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = ControlConfig(
        entropy_threshold=args.entropy_threshold,
        route_window=args.route_window,
    )

    device = resolve_device(args.device)

    print('╔══════════════════════════════════════════════════════════════════╗')
    print('║ STAGED GATEWAY CONTROL                                           ║')
    print('║ centrality/funnelness + entropy trigger + anti-lock safeguard    ║')
    print('╚══════════════════════════════════════════════════════════════════╝')

    t0 = time.time()

    summary = {}
    all_ts = []
    all_in = []

    for anti_lock in [False, True]:
        mode = 'guard' if anti_lock else 'no_guard'
        print(f'\n-- Running {mode} --')
        s, ts_rows, in_rows = run_staged_policy(
            device=device,
            steps=args.steps,
            n_beings=args.N,
            cfg=cfg,
            seed=args.seed,
            anti_lock=anti_lock,
        )
        summary[mode] = s
        all_ts.extend(ts_rows)
        all_in.extend(in_rows)

        print(f"  switch_step={s['switch_step']}, entropy_at_switch={s['route_entropy_at_switch']}")
        print(f"  pre->post effective basins: {s['pre_diversity_effective_basins']:.2f} -> {s['post_diversity_effective_basins']:.2f}")
        print(f"  pre->post mean dwell: {s['pre_mean_dwell']:.2f} -> {s['post_mean_dwell']:.2f}")
        print(f"  funnelness mean/max: {s['funnelness_mean']:.3f} / {s['funnelness_max']:.3f}")

    write_csv(os.path.join(OUT_DIR, 'centrality_timeseries.csv'), all_ts)
    write_csv(os.path.join(OUT_DIR, 'inbound_centrality.csv'), all_in)

    out_json = os.path.join(OUT_DIR, 'control_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    make_plots(all_ts, all_in, summary)

    elapsed = time.time() - t0
    print(f'\nSaved: {out_json}')
    print(f'Saved: {os.path.join(OUT_DIR, "centrality_timeseries.csv")}')
    print(f'Saved: {os.path.join(OUT_DIR, "inbound_centrality.csv")}')
    print(f'Saved plots in: {OUT_DIR}')
    print(f'Total wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
