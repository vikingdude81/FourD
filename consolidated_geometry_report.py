#!/usr/bin/env python3
"""
Consolidated Geometry Report
============================

Merges:
- outputs/geometry_comparison/geometry_comparison_results.json
- outputs/time_sliced_geometry/time_sliced_geometry_results.json

Produces:
- outputs/consolidated_geometry/consolidated_summary.json
- outputs/consolidated_geometry/consolidated_summary.csv
- outputs/consolidated_geometry/consolidated_report.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_GEOMETRY_JSON = os.path.join(
    'outputs', 'geometry_comparison', 'geometry_comparison_results.json'
)
DEFAULT_TS_JSON = os.path.join(
    'outputs', 'time_sliced_geometry', 'time_sliced_geometry_results.json'
)
DEFAULT_OUT_DIR = os.path.join('outputs', 'consolidated_geometry')


def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_rows(geometry: Dict, ts: Dict) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    # Geometry comparison key rows
    s3_slope = geometry['part_a']['s3']['max_slope']
    r4_slope = geometry['part_a']['flat4']['max_slope']
    rows.append({
        'group': 'geometry_part_a',
        'metric': 'max_slope',
        'entity': 's3',
        'value': s3_slope,
    })
    rows.append({
        'group': 'geometry_part_a',
        'metric': 'max_slope',
        'entity': 'flat4',
        'value': r4_slope,
    })

    for manifold in ['s3', 'flat4']:
        for metric in ['unigram_entropy', 'bigram_entropy', 'trigram_entropy', 'unique_basins']:
            rows.append({
                'group': 'geometry_part_b',
                'metric': metric,
                'entity': manifold,
                'value': geometry['part_b'][manifold][metric],
            })

    for manifold in ['s3', 'flat4']:
        rows.append({
            'group': 'geometry_part_d',
            'metric': 'goldilocks_present_fraction',
            'entity': manifold,
            'value': geometry['part_d'][manifold]['fraction_present'],
        })
        rows.append({
            'group': 'geometry_part_d',
            'metric': 'goldilocks_strong_fraction',
            'entity': manifold,
            'value': geometry['part_d'][manifold]['fraction_strong'],
        })

    # Time-sliced experiment A
    for schedule, metrics in ts['experiment_a'].items():
        for metric_name in ['edge_clarity_r', 'cohens_d', 'null_z', 'transition_rate', 'mean_clarity']:
            rows.append({
                'group': 'time_sliced_exp_a',
                'metric': metric_name,
                'entity': schedule,
                'value': metrics[metric_name],
            })

    # Time-sliced experiment B (Zn family)
    for manifold in ['s3', 'flat4']:
        for zn_label, metrics in ts['experiment_b'][manifold].items():
            n = int(zn_label.replace('Z', ''))
            rows.append({
                'group': 'time_sliced_exp_b',
                'metric': 'edge_clarity_r',
                'entity': f'{manifold}_{zn_label}',
                'n': n,
                'value': metrics['edge_clarity_r'],
            })
            rows.append({
                'group': 'time_sliced_exp_b',
                'metric': 'cohens_d',
                'entity': f'{manifold}_{zn_label}',
                'n': n,
                'value': metrics['cohens_d'],
            })

    # Time-sliced experiment C
    for schedule_name, metrics in ts['experiment_c'].items():
        rows.append({
            'group': 'time_sliced_exp_c',
            'metric': 'present_fraction',
            'entity': schedule_name,
            'value': metrics['present_fraction'],
        })
        rows.append({
            'group': 'time_sliced_exp_c',
            'metric': 'strong_fraction',
            'entity': schedule_name,
            'value': metrics['strong_fraction'],
        })

    return rows


def build_summary(geometry: Dict, ts: Dict) -> Dict[str, object]:
    summary: Dict[str, object] = {}

    summary['phase_transition'] = {
        's3_max_slope': geometry['part_a']['s3']['max_slope'],
        'r4_max_slope': geometry['part_a']['flat4']['max_slope'],
        'r4_over_s3_slope_ratio': (
            geometry['part_a']['flat4']['max_slope'] / (geometry['part_a']['s3']['max_slope'] + 1e-15)
        ),
    }

    summary['basin_grammar'] = {
        's3_trigram_entropy': geometry['part_b']['s3']['trigram_entropy'],
        'r4_trigram_entropy': geometry['part_b']['flat4']['trigram_entropy'],
        's3_over_r4_trigram_ratio': (
            geometry['part_b']['s3']['trigram_entropy'] /
            (geometry['part_b']['flat4']['trigram_entropy'] + 1e-15)
        ),
    }

    summary['goldilocks_static'] = {
        's3_present_fraction': geometry['part_d']['s3']['fraction_present'],
        'r4_present_fraction': geometry['part_d']['flat4']['fraction_present'],
        's3_strong_fraction': geometry['part_d']['s3']['fraction_strong'],
        'r4_strong_fraction': geometry['part_d']['flat4']['fraction_strong'],
    }

    exp_a = ts['experiment_a']
    summary['time_sliced_schedule_rank_edge_r'] = sorted(
        [{'schedule': k, 'edge_clarity_r': v['edge_clarity_r']} for k, v in exp_a.items()],
        key=lambda x: x['edge_clarity_r'],
        reverse=True,
    )

    exp_b = ts['experiment_b']
    def avg_metric(manifold: str, metric: str) -> float:
        vals = [m[metric] for m in exp_b[manifold].values()]
        return float(np.mean(vals))

    summary['zn_family_means'] = {
        's3_avg_edge_r': avg_metric('s3', 'edge_clarity_r'),
        'r4_avg_edge_r': avg_metric('flat4', 'edge_clarity_r'),
        's3_avg_cohens_d': avg_metric('s3', 'cohens_d'),
        'r4_avg_cohens_d': avg_metric('flat4', 'cohens_d'),
    }

    summary['coupling_sensitivity'] = {
        'static_r4_present_fraction': ts['experiment_c']['static_r4']['present_fraction'],
        's3_to_r4_present_fraction': ts['experiment_c']['s3_to_r4']['present_fraction'],
        'static_r4_strong_fraction': ts['experiment_c']['static_r4']['strong_fraction'],
        's3_to_r4_strong_fraction': ts['experiment_c']['s3_to_r4']['strong_fraction'],
    }

    return summary


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    # Normalize sparse keys.
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_report(geometry: Dict, ts: Dict, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: static vs sliced schedules (edge_r and cohens_d)
    exp_a = ts['experiment_a']
    sched = list(exp_a.keys())
    x = np.arange(len(sched))
    width = 0.35
    axes[0, 0].bar(x - width / 2, [exp_a[k]['edge_clarity_r'] for k in sched], width, label='edge_clarity_r')
    axes[0, 0].bar(x + width / 2, [exp_a[k]['cohens_d'] for k in sched], width, label='cohens_d')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(sched, rotation=20, ha='right')
    axes[0, 0].set_title('Time-Sliced Schedules (Exp A)', fontweight='bold')
    axes[0, 0].axhline(0, color='k', linewidth=0.4)
    axes[0, 0].legend(fontsize=8)

    # Panel 2: Zn edge_r curves
    sectors = [int(k.replace('Z', '')) for k in ts['experiment_b']['s3'].keys()]
    sectors_sorted = sorted(sectors)
    s3_edge = [ts['experiment_b']['s3'][f'Z{n}']['edge_clarity_r'] for n in sectors_sorted]
    r4_edge = [ts['experiment_b']['flat4'][f'Z{n}']['edge_clarity_r'] for n in sectors_sorted]
    axes[0, 1].plot(sectors_sorted, s3_edge, 'o-', label='s3')
    axes[0, 1].plot(sectors_sorted, r4_edge, 'o-', label='flat4')
    axes[0, 1].set_title('Z_n Family Edge Coupling (Exp B)', fontweight='bold')
    axes[0, 1].set_xlabel('n in Z_n')
    axes[0, 1].set_ylabel('edge_clarity_r')
    axes[0, 1].axhline(0, color='k', linewidth=0.4)
    axes[0, 1].legend()

    # Panel 3: Goldilocks fractions
    g_present = [
        geometry['part_d']['s3']['fraction_present'],
        geometry['part_d']['flat4']['fraction_present'],
        ts['experiment_c']['static_r4']['present_fraction'],
        ts['experiment_c']['s3_to_r4']['present_fraction'],
    ]
    g_labels = ['static_s3', 'static_r4', 'coupling_static_r4', 'coupling_s3_to_r4']
    axes[1, 0].bar(g_labels, g_present, color=['#4C72B0', '#55A868', '#55A868', '#8172B3'])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel('present fraction')
    axes[1, 0].set_title('Goldilocks / Present Fraction', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=20)

    # Panel 4: headline summary text
    slope_ratio = geometry['part_a']['flat4']['max_slope'] / (geometry['part_a']['s3']['max_slope'] + 1e-15)
    trigram_ratio = geometry['part_b']['s3']['trigram_entropy'] / (
        geometry['part_b']['flat4']['trigram_entropy'] + 1e-15
    )
    text = (
        f"Phase sharpness R4/S3: {slope_ratio:.2f}x\n"
        f"Basin grammar S3/R4 (trigram): {trigram_ratio:.2f}x\n"
        f"Static present fraction S3: {geometry['part_d']['s3']['fraction_present']:.0%}\n"
        f"Static present fraction R4: {geometry['part_d']['flat4']['fraction_present']:.0%}\n"
        f"Coupling present fraction (static_r4): {ts['experiment_c']['static_r4']['present_fraction']:.0%}\n"
        f"Coupling present fraction (s3_to_r4): {ts['experiment_c']['s3_to_r4']['present_fraction']:.0%}"
    )
    axes[1, 1].axis('off')
    axes[1, 1].text(0.02, 0.95, text, va='top', ha='left', fontsize=11)
    axes[1, 1].set_title('Headlines', fontweight='bold')

    plt.suptitle('Consolidated Geometry + Time-Sliced Report', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Consolidate geometry analysis outputs')
    parser.add_argument('--geometry-json', default=DEFAULT_GEOMETRY_JSON)
    parser.add_argument('--timesliced-json', default=DEFAULT_TS_JSON)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    geometry = load_json(args.geometry_json)
    ts = load_json(args.timesliced_json)

    rows = extract_rows(geometry, ts)
    summary = build_summary(geometry, ts)

    summary_path = os.path.join(args.out_dir, 'consolidated_summary.json')
    csv_path = os.path.join(args.out_dir, 'consolidated_summary.csv')
    fig_path = os.path.join(args.out_dir, 'consolidated_report.png')

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    write_csv(rows, csv_path)
    plot_report(geometry, ts, fig_path)

    print(f'Saved: {summary_path}')
    print(f'Saved: {csv_path}')
    print(f'Saved: {fig_path}')


if __name__ == '__main__':
    main()
