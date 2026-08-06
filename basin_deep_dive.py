#!/usr/bin/env python3
"""
Basin Deep Dive Analysis
========================

Focused exploration of macro-basin dynamics:

1) Basin occupancy concentration and effective basin usage
2) Dwell-time survival and hazard shape
3) Transition network structure (entropy rate, asymmetry, spectral gap)
4) Basin sequence motifs (top trigrams and cycle score)

Compares S3 and flat R4 using the existing UniversalEngine to stay compatible
with prior FourD analyses.

Usage:
    python basin_deep_dive.py [--device cuda:0] [--steps 2000] [--N 128]
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from universality_test import UniversalEngine

OUT_DIR = os.path.join('outputs', 'basin_deep_dive')


def resolve_device(device: str) -> str:
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA requested but unavailable in this environment; using CPU.')
        return 'cpu'
    return device


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


def gini(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    s = x.sum()
    if s <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * s + 1e-15))


def occupancy_metrics(basins: np.ndarray, n_basins: int):
    counts = np.bincount(basins.ravel(), minlength=n_basins).astype(np.float64)
    probs = counts / (counts.sum() + 1e-15)
    entropy = -np.sum(probs * np.log2(probs + 1e-15))
    effective = 2 ** entropy
    concentration = np.sum(probs ** 2)
    return {
        'counts': counts,
        'probs': probs,
        'entropy_bits': float(entropy),
        'effective_basins': float(effective),
        'concentration_hhi': float(concentration),
        'gini': gini(counts),
    }


def dwell_durations(seq: np.ndarray):
    if len(seq) == 0:
        return []
    runs = []
    cur = seq[0]
    length = 1
    for v in seq[1:]:
        if v == cur:
            length += 1
        else:
            runs.append((int(cur), int(length)))
            cur = v
            length = 1
    runs.append((int(cur), int(length)))
    return runs


def dwell_metrics(basins: np.ndarray):
    all_durations = []
    per_basin = {}
    for i in range(basins.shape[0]):
        runs = dwell_durations(basins[i])
        for b, d in runs:
            all_durations.append(d)
            per_basin.setdefault(b, []).append(d)

    d = np.array(all_durations, dtype=np.int32)
    if len(d) == 0:
        return {
            'mean_dwell': 0.0,
            'median_dwell': 0.0,
            'p90_dwell': 0.0,
            'max_dwell': 0,
            'tail_alpha': None,
            'durations': d,
            'per_basin': per_basin,
        }

    # Simple log-log tail fit on survival S(t) for t >= 3.
    t_vals = np.arange(1, int(d.max()) + 1)
    surv = np.array([(d >= t).mean() for t in t_vals])
    mask = (t_vals >= 3) & (surv > 0)
    tail_alpha = None
    if mask.sum() >= 4:
        x = np.log(t_vals[mask])
        y = np.log(surv[mask])
        slope, _ = np.polyfit(x, y, 1)
        tail_alpha = float(-slope)

    return {
        'mean_dwell': float(d.mean()),
        'median_dwell': float(np.median(d)),
        'p90_dwell': float(np.percentile(d, 90)),
        'max_dwell': int(d.max()),
        'tail_alpha': tail_alpha,
        'durations': d,
        'per_basin': per_basin,
    }


def transition_metrics(basins: np.ndarray, n_basins: int):
    trans = np.zeros((n_basins, n_basins), dtype=np.float64)

    for i in range(basins.shape[0]):
        src = basins[i, :-1]
        dst = basins[i, 1:]
        for a, b in zip(src, dst):
            trans[a, b] += 1

    row_sum = trans.sum(axis=1, keepdims=True)
    P = trans / (row_sum + 1e-15)

    occ = np.bincount(basins.ravel(), minlength=n_basins).astype(np.float64)
    pi = occ / (occ.sum() + 1e-15)

    # Entropy rate H = sum_i pi_i H(P_i.)
    row_entropy = -np.sum(P * np.log2(P + 1e-15), axis=1)
    entropy_rate = float(np.sum(pi * row_entropy))

    # Transition asymmetry index
    asym = np.abs(trans - trans.T).sum() / (trans.sum() + 1e-15)

    # Spectral gap from eigenvalues of P^T
    eigvals = np.linalg.eigvals(P.T)
    eig_abs = np.sort(np.abs(eigvals))[::-1]
    lambda2 = float(eig_abs[1]) if len(eig_abs) > 1 else 0.0
    spectral_gap = float(1.0 - lambda2)

    self_loop = float(np.trace(trans) / (trans.sum() + 1e-15))

    return {
        'transition_matrix': trans,
        'P': P,
        'stationary_pi': pi,
        'entropy_rate_bits': entropy_rate,
        'asymmetry_index': float(asym),
        'spectral_gap': spectral_gap,
        'self_loop_fraction': self_loop,
    }


def motif_metrics(basins: np.ndarray):
    bigram_counter = Counter()
    trigram_counter = Counter()
    cycle_score_num = 0
    cycle_score_den = 0

    for i in range(basins.shape[0]):
        s = basins[i].tolist()
        for t in range(len(s) - 1):
            bigram_counter[(s[t], s[t + 1])] += 1
        for t in range(len(s) - 2):
            tri = (s[t], s[t + 1], s[t + 2])
            trigram_counter[tri] += 1
            cycle_score_den += 1
            if s[t] == s[t + 2] and s[t] != s[t + 1]:
                cycle_score_num += 1

    top_bigrams = bigram_counter.most_common(10)
    top_trigrams = trigram_counter.most_common(10)
    cycle_score = float(cycle_score_num / (cycle_score_den + 1e-15))

    return {
        'top_bigrams': [
            {'pattern': list(k), 'count': int(v)} for k, v in top_bigrams
        ],
        'top_trigrams': [
            {'pattern': list(k), 'count': int(v)} for k, v in top_trigrams
        ],
        'cycle_score': cycle_score,
    }


def make_plots(results: dict):
    # 1) Occupancy bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, manifold, color in zip(axes, ['s3', 'flat4'], ['#4C72B0', '#55A868']):
        p = np.array(results[manifold]['occupancy']['probs'])
        ax.bar(np.arange(len(p)), p, color=color)
        ax.set_title(f'{manifold} basin occupancy', fontweight='bold')
        ax.set_xlabel('Basin ID')
        ax.set_ylabel('Probability')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'occupancy_comparison.png'), dpi=160, bbox_inches='tight')
    plt.close()

    # 2) Dwell survival plot
    plt.figure(figsize=(7, 5))
    for manifold, color in [('s3', '#4C72B0'), ('flat4', '#55A868')]:
        d = np.array(results[manifold]['dwell']['durations'])
        if len(d) == 0:
            continue
        t_vals = np.arange(1, int(d.max()) + 1)
        surv = np.array([(d >= t).mean() for t in t_vals])
        mask = surv > 0
        plt.loglog(t_vals[mask], surv[mask], 'o-', markersize=3, linewidth=1.2, label=manifold, color=color)
    plt.xlabel('Dwell length t')
    plt.ylabel('Survival P(T >= t)')
    plt.title('Dwell-Time Survival', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'dwell_survival.png'), dpi=160, bbox_inches='tight')
    plt.close()

    # 3) Transition heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, manifold in zip(axes, ['s3', 'flat4']):
        P = np.array(results[manifold]['transition']['P'])
        im = ax.imshow(P, aspect='auto', cmap='magma')
        ax.set_title(f'{manifold} transition matrix P', fontweight='bold')
        ax.set_xlabel('to basin')
        ax.set_ylabel('from basin')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'transition_matrices.png'), dpi=160, bbox_inches='tight')
    plt.close()

    # 4) Top trigram counts side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, manifold, color in zip(axes, ['s3', 'flat4'], ['#4C72B0', '#55A868']):
        top = results[manifold]['motifs']['top_trigrams'][:8]
        labels = ['-'.join(map(str, d['pattern'])) for d in top]
        vals = [d['count'] for d in top]
        ax.barh(np.arange(len(vals)), vals, color=color)
        ax.set_yticks(np.arange(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f'{manifold} top trigram motifs', fontweight='bold')
        ax.set_xlabel('count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'motif_trigrams.png'), dpi=160, bbox_inches='tight')
    plt.close()


def write_csv(summary: dict, path: str):
    rows = []
    for manifold in ['s3', 'flat4']:
        occ = summary[manifold]['occupancy']
        dw = summary[manifold]['dwell']
        tr = summary[manifold]['transition']
        mo = summary[manifold]['motifs']
        rows.append({
            'manifold': manifold,
            'entropy_bits': occ['entropy_bits'],
            'effective_basins': occ['effective_basins'],
            'concentration_hhi': occ['concentration_hhi'],
            'gini': occ['gini'],
            'mean_dwell': dw['mean_dwell'],
            'median_dwell': dw['median_dwell'],
            'p90_dwell': dw['p90_dwell'],
            'tail_alpha': dw['tail_alpha'] if dw['tail_alpha'] is not None else '',
            'entropy_rate_bits': tr['entropy_rate_bits'],
            'asymmetry_index': tr['asymmetry_index'],
            'spectral_gap': tr['spectral_gap'],
            'self_loop_fraction': tr['self_loop_fraction'],
            'cycle_score': mo['cycle_score'],
        })

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sanitize_for_json(results: dict):
    clean = {}
    for manifold in ['s3', 'flat4']:
        clean[manifold] = {
            'occupancy': {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in results[manifold]['occupancy'].items()
            },
            'dwell': {
                'mean_dwell': results[manifold]['dwell']['mean_dwell'],
                'median_dwell': results[manifold]['dwell']['median_dwell'],
                'p90_dwell': results[manifold]['dwell']['p90_dwell'],
                'max_dwell': results[manifold]['dwell']['max_dwell'],
                'tail_alpha': results[manifold]['dwell']['tail_alpha'],
                'durations': results[manifold]['dwell']['durations'].tolist(),
            },
            'transition': {
                'entropy_rate_bits': results[manifold]['transition']['entropy_rate_bits'],
                'asymmetry_index': results[manifold]['transition']['asymmetry_index'],
                'spectral_gap': results[manifold]['transition']['spectral_gap'],
                'self_loop_fraction': results[manifold]['transition']['self_loop_fraction'],
                'stationary_pi': results[manifold]['transition']['stationary_pi'].tolist(),
                'P': results[manifold]['transition']['P'].tolist(),
            },
            'motifs': results[manifold]['motifs'],
        }
    return clean


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Basin deep-dive analysis')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--N', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    device = resolve_device(args.device)

    print('╔══════════════════════════════════════════════════════════╗')
    print('║ BASIN DEEP DIVE                                         ║')
    print('║ occupancy, dwell survival, transition network, motifs   ║')
    print('╚══════════════════════════════════════════════════════════╝')

    t0 = time.time()
    results = {}

    for manifold in ['s3', 'flat4']:
        print(f'\n-- Running {manifold} --')
        basins, clarity = run_engine(manifold, device=device, steps=args.steps, n_beings=args.N)
        n_basins = int(basins.max()) + 1

        occ = occupancy_metrics(basins, n_basins)
        dw = dwell_metrics(basins)
        tr = transition_metrics(basins, n_basins)
        mo = motif_metrics(basins)

        results[manifold] = {
            'occupancy': occ,
            'dwell': dw,
            'transition': tr,
            'motifs': mo,
            'mean_clarity': float(clarity[:, args.steps // 4:].mean()),
            'transition_rate': float((basins[:, 1:] != basins[:, :-1]).mean()),
        }

        print(f"  effective basins: {occ['effective_basins']:.2f} / {n_basins}")
        print(f"  concentration HHI: {occ['concentration_hhi']:.3f} | gini: {occ['gini']:.3f}")
        print(f"  mean dwell: {dw['mean_dwell']:.2f} | p90 dwell: {dw['p90_dwell']:.2f}")
        print(f"  entropy rate: {tr['entropy_rate_bits']:.3f} bits/step")
        print(f"  asymmetry: {tr['asymmetry_index']:.3f} | spectral gap: {tr['spectral_gap']:.3f}")
        print(f"  cycle score: {mo['cycle_score']:.3f}")

    make_plots(results)

    clean = sanitize_for_json(results)
    out_json = os.path.join(OUT_DIR, 'basin_deep_dive_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2)

    out_csv = os.path.join(OUT_DIR, 'basin_deep_dive_summary.csv')
    write_csv(clean, out_csv)

    elapsed = time.time() - t0
    print(f'\nSaved: {out_json}')
    print(f'Saved: {out_csv}')
    print(f'Saved plots in: {OUT_DIR}')
    print(f'Total wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
