#!/usr/bin/env python3
"""
Positive Geometry Readout (a.k.a. Combinatorial Gateway Readout)
====================================================================

Naming note: this script does NOT implement positive-Grassmannian or
amplituhedron machinery. What it implements is the weighted spanning-tree
polytope of a graph (Kirchhoff/Matrix-Tree theorem, Foster's theorem) --
a genuinely rigorous positive-geometry object, but a much simpler one than
what "positive geometry" usually evokes in the Arkani-Hamed/Hoffman sense.
If you want the more precise name for what's actually running here, think
"Combinatorial Gateway Readout": a test of whether graph topology alone
predicts empirically-measured transition significance. The theoretical
motivation (below) stays; the implementation should not be read as more
than what it is.

Tests whether a quantity derived purely from graph combinatorics -- with no
access to the simulation's clarity/dynamics measurements -- predicts which
basin-transition edges the simulation independently marks as meaningful
"gateways" (basin_gateway_analysis.py's gateway_score).

Why this is the honest, buildable version of the Levin/Hoffman claim
----------------------------------------------------------------------
Hoffman's proposed bridge from agent dynamics to the positive Grassmannian /
amplituhedron is not a constructed mathematical result -- it is an open
conjecture. What IS a rigorous, well-established positive-geometry object is
the weighted spanning-tree polytope of a graph: its canonical form assigns to
each edge e a residue equal to

    p_span(e) = w_e * R_e

where w_e is the edge weight and R_e is the effective resistance across e in
the whole graph (via the Moore-Penrose pseudoinverse of the graph Laplacian).
This is Foster's theorem: p_span(e) is exactly the probability edge e appears
in a weighted-uniform random spanning tree. It is "math deriving information
by itself" in a precise sense -- R_e depends on the *entire* graph's
structure, not just e's own weight, so p_span(e) is not reducible to a local
count.

The question this script asks: does p_span(e), computed from nothing but the
basin-transition graph's topology, predict basin_gateway_analysis.py's
gateway_score(e), which was computed from clarity dynamics the graph
construction never saw? A positive answer says global combinatorial
structure alone carries predictive information about which transitions the
dynamics treat as significant -- the falsifiable core of the "interface
already contains derivable structure" claim, with no metaphysics attached.

Because gateway_score's own formula includes a factor of edge count, a raw
correlation between p_span and gateway_score would partly just be "high
traffic predicts high traffic." This script therefore also reports the
partial correlation after regressing out log(count+1) from both sides --
the part of the story that traffic volume alone cannot explain.

Also computes, per manifold, three purely-topological invariants that
require no simulation to interpret: weighted spanning-tree count (Kirchhoff
determinant), Kirchhoff index (sum of all-pairs effective resistance), and
algebraic connectivity (second-smallest Laplacian eigenvalue) -- and prints
a small symbolic (sympy) spanning-tree generating polynomial for the top-6
busiest basins per manifold, as a literal instance of the math "by itself."

Reads:  outputs/basin_gateway/{manifold}_gateway_edges.csv
        (produced by basin_gateway_analysis.py)

Usage:
    python positive_geometry_readout.py [--indir outputs/basin_gateway] [--outdir outputs/positive_geometry]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

IN_DIR = os.path.join('outputs', 'basin_gateway')
OUT_DIR = os.path.join('outputs', 'positive_geometry')


def discover_manifolds(indir: str) -> list[str]:
    paths = glob.glob(os.path.join(indir, '*_gateway_edges.csv'))
    manifolds = []
    for p in paths:
        m = re.match(r'(.+)_gateway_edges\.csv$', os.path.basename(p))
        if m:
            manifolds.append(m.group(1))
    return sorted(manifolds)


def build_weight_matrix(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    n_basins = int(max(df['src_basin'].max(), df['dst_basin'].max())) + 1
    directed = np.zeros((n_basins, n_basins), dtype=np.float64)
    for _, r in df.iterrows():
        directed[int(r['src_basin']), int(r['dst_basin'])] = float(r['count'])
    # Symmetrize: effective-resistance / spanning-tree theory is defined on
    # undirected weighted graphs. Two-way traffic between a pair of basins
    # is combined into a single undirected edge weight.
    W = directed + directed.T
    np.fill_diagonal(W, 0.0)
    return W, n_basins


def laplacian_invariants(W: np.ndarray) -> dict:
    n = W.shape[0]
    deg = W.sum(axis=1)
    L = np.diag(deg) - W

    eigvals = np.linalg.eigvalsh(L)
    eigvals_sorted = np.sort(eigvals)
    algebraic_connectivity = float(eigvals_sorted[1]) if n > 1 else 0.0

    # Weighted spanning-tree count via Kirchhoff's theorem: any cofactor of L.
    L_reduced = L[1:, 1:]
    sign, logdet = np.linalg.slogdet(L_reduced)
    spanning_tree_count = float(sign * np.exp(logdet)) if sign > 0 else 0.0

    L_pinv = np.linalg.pinv(L)
    diag = np.diag(L_pinv)
    R = diag[:, None] + diag[None, :] - 2 * L_pinv
    np.fill_diagonal(R, 0.0)

    kirchhoff_index = float(np.sum(R) / 2)

    return dict(
        n_basins=n,
        algebraic_connectivity=algebraic_connectivity,
        spanning_tree_count=spanning_tree_count,
        kirchhoff_index=kirchhoff_index,
        L=L, R=R,
    )


def spanning_tree_edge_probabilities(df: pd.DataFrame, W: np.ndarray, R: np.ndarray) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        i, j = int(r['src_basin']), int(r['dst_basin'])
        w_edge = W[i, j]  # symmetrized weight
        r_eff = R[i, j]
        p_span = w_edge * r_eff  # Foster's theorem: P(edge in random spanning tree)
        rows.append(dict(src_basin=i, dst_basin=j, sym_weight=w_edge,
                          effective_resistance=r_eff, p_span=p_span))
    return pd.DataFrame(rows)


def partial_correlation_controlling_for_count(gateway_score, p_span, count):
    log_count = np.log1p(count)

    def residualize(y, x):
        x = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        return y - x @ coef

    gs_resid = residualize(gateway_score, log_count)
    ps_resid = residualize(p_span, log_count)
    if np.std(gs_resid) < 1e-12 or np.std(ps_resid) < 1e-12:
        return 0.0, 1.0
    r, p = stats.pearsonr(gs_resid, ps_resid)
    return float(r), float(p)


def symbolic_spanning_tree_polynomial(df: pd.DataFrame, top_k: int = 6):
    """Print the literal weighted spanning-tree generating polynomial (via
    Matrix-Tree cofactor determinant, symbolic weights) for the top_k busiest
    basins -- a direct, non-metaphorical instance of math deriving structure
    with no simulation involved past the initial edge-count input."""
    import sympy as sp

    traffic = {}
    for _, r in df.iterrows():
        i, j = int(r['src_basin']), int(r['dst_basin'])
        traffic[i] = traffic.get(i, 0) + r['count']
        traffic[j] = traffic.get(j, 0) + r['count']
    top_nodes = sorted(traffic, key=traffic.get, reverse=True)[:top_k]
    node_index = {n: k for k, n in enumerate(top_nodes)}
    k = len(top_nodes)

    w_syms = {}
    L = sp.zeros(k, k)
    seen = set()
    for _, r in df.iterrows():
        i, j = int(r['src_basin']), int(r['dst_basin'])
        if i not in node_index or j not in node_index:
            continue
        a, b = node_index[i], node_index[j]
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        # No `positive=True`: sympy's default det() invokes assumption
        # resolution on every simplification step when symbols carry
        # assumptions, which is catastrophically slow (>60s hang observed
        # on a 6-node subgraph with 15 symbols vs. 0.06s without).
        sym = sp.symbols(f'w_{key[0]}_{key[1]}')
        w_syms[key] = sym
        L[a, a] += sym
        L[b, b] += sym
        L[a, b] -= sym
        L[b, a] -= sym

    if k < 2:
        return top_nodes, sp.Integer(0)

    reduced = L[1:, 1:]
    poly = sp.expand(reduced.det(method='berkowitz'))
    return top_nodes, poly


def process_manifold(manifold: str, indir: str, outdir: str) -> dict:
    csv_path = os.path.join(indir, f'{manifold}_gateway_edges.csv')
    df = pd.read_csv(csv_path)
    W, n_basins = build_weight_matrix(df)

    inv = laplacian_invariants(W)
    span_df = spanning_tree_edge_probabilities(df, W, inv['R'])

    merged = df.merge(span_df, on=['src_basin', 'dst_basin'])
    merged.to_csv(os.path.join(outdir, f'{manifold}_positive_geometry_edges.csv'), index=False)

    gs = merged['gateway_score'].values
    ps = merged['p_span'].values
    count = merged['count'].values

    raw_pearson_r, raw_pearson_p = stats.pearsonr(gs, ps)
    raw_spearman_r, raw_spearman_p = stats.spearmanr(gs, ps)
    baseline_r, baseline_p = stats.pearsonr(gs, np.log1p(count))
    partial_r, partial_p = partial_correlation_controlling_for_count(gs, ps, count)

    top_nodes, poly = symbolic_spanning_tree_polynomial(df)

    result = dict(
        manifold=manifold,
        n_basins=inv['n_basins'],
        n_edges=len(merged),
        algebraic_connectivity=inv['algebraic_connectivity'],
        spanning_tree_count=inv['spanning_tree_count'],
        kirchhoff_index=inv['kirchhoff_index'],
        raw_pearson_r=float(raw_pearson_r), raw_pearson_p=float(raw_pearson_p),
        raw_spearman_r=float(raw_spearman_r), raw_spearman_p=float(raw_spearman_p),
        baseline_count_pearson_r=float(baseline_r), baseline_count_pearson_p=float(baseline_p),
        partial_r_controlling_for_count=partial_r, partial_p_controlling_for_count=partial_p,
        symbolic_top_nodes=top_nodes,
        symbolic_spanning_tree_polynomial=str(poly),
    )
    return result, merged


def plot_results(results: dict, merged_by_manifold: dict, outdir: str):
    manifolds = list(results.keys())
    fig, axes = plt.subplots(1, len(manifolds), figsize=(6 * len(manifolds), 5))
    if len(manifolds) == 1:
        axes = [axes]
    for ax, manifold in zip(axes, manifolds):
        m = merged_by_manifold[manifold]
        ax.scatter(m['p_span'], m['gateway_score'], s=18, alpha=0.6, c='#4C72B0')
        r = results[manifold]['raw_pearson_r']
        pr = results[manifold]['partial_r_controlling_for_count']
        ax.set_xlabel('p_span(e)  [pure graph combinatorics]')
        ax.set_ylabel('gateway_score(e)  [empirical clarity dynamics]')
        ax.set_title(f'{manifold}: r={r:.3f}, partial r (ctrl. count)={pr:.3f}')
    fig.suptitle('Does purely-combinatorial spanning-tree structure predict empirical gateway importance?')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'positive_geometry_correlation.png'), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', default=IN_DIR)
    ap.add_argument('--outdir', default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    manifolds = discover_manifolds(args.indir)
    if not manifolds:
        raise SystemExit(f'No *_gateway_edges.csv found in {args.indir}. '
                          f'Run basin_gateway_analysis.py first.')

    print('Positive Geometry Readout')
    print(f'  manifolds found: {manifolds}')

    results = {}
    merged_by_manifold = {}
    for manifold in manifolds:
        print(f'\n-- {manifold} --')
        result, merged = process_manifold(manifold, args.indir, args.outdir)
        results[manifold] = result
        merged_by_manifold[manifold] = merged
        print(f"  spanning_tree_count      = {result['spanning_tree_count']:.4g}")
        print(f"  algebraic_connectivity   = {result['algebraic_connectivity']:.4g}")
        print(f"  kirchhoff_index          = {result['kirchhoff_index']:.4g}")
        print(f"  raw corr(p_span, gateway_score)      r={result['raw_pearson_r']:.3f} "
              f"p={result['raw_pearson_p']:.4g}")
        print(f"  baseline corr(count, gateway_score)  r={result['baseline_count_pearson_r']:.3f}")
        print(f"  partial r controlling for count       = {result['partial_r_controlling_for_count']:.3f} "
              f"p={result['partial_p_controlling_for_count']:.4g}")
        print(f"  symbolic spanning-tree polynomial (top-"
              f"{len(result['symbolic_top_nodes'])} busiest basins {result['symbolic_top_nodes']}):")
        print(f"    {result['symbolic_spanning_tree_polynomial']}")

    with open(os.path.join(args.outdir, 'positive_geometry_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    plot_results(results, merged_by_manifold, args.outdir)
    print(f'\nOutputs written to {args.outdir}/')


if __name__ == '__main__':
    main()
