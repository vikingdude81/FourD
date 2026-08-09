#!/usr/bin/env python3
"""
Emergent Geometry Readout
============================

Tests whether "space" is recoverable purely from the basin-transition
graph's combinatorics -- with zero access to the manifold coordinates that
actually generated the dynamics. This is the honest, bounded version of
"recovering geometry from combinatorics": not a claim about physics or
consciousness, but a specific, falsifiable question about this system.

positive_geometry_readout.py already computes effective resistance R_ij
between every pair of basins from the transition graph's weighted Laplacian
(Moore-Penrose pseudoinverse) -- a quantity derived purely from "which
basins transition to which, how often," with no coordinates involved.
Effective resistance has a known Euclidean interpretation: there exists an
embedding (via the Laplacian pseudoinverse's eigenvectors, scaled by
sqrt(eigenvalue)) in which R_ij is EXACTLY the squared Euclidean distance
between nodes i and j -- the "resistive embedding." That means R can be fed
through classical multidimensional scaling (MDS, via double-centering) to
recover:

  1. An emergent low-dimensional embedding of the basins, purely from
     traffic patterns -- no coordinates used as input.
  2. An effective dimensionality: how many eigenvalues of the double-
     centered R matrix are non-negligible. Does the graph "know" it's
     roughly 4-dimensional (both s3 and flat4 embed in R^4)?
  3. Whether pairwise graph-derived distance correlates with the TRUE
     geometric distance between basin centers on the manifold that
     generated the dynamics -- recomputed deterministically via
     universality_test.make_macro_centers, which the graph construction
     never saw.

A strong positive result says the interface's causal/traffic structure
alone encodes real spatial information. A null result says it doesn't, and
spatial information here genuinely required privileged access to
coordinates -- an equally informative, non-negotiable finding either way.

Reads:  outputs/basin_gateway/{manifold}_gateway_edges.csv
        (produced by basin_gateway_analysis.py)

Usage:
    python emergent_geometry_readout.py [--indir outputs/basin_gateway] [--outdir outputs/emergent_geometry]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from positive_geometry_readout import discover_manifolds, build_weight_matrix, laplacian_invariants
from universality_test import make_macro_centers

IN_DIR = os.path.join('outputs', 'basin_gateway')
OUT_DIR = os.path.join('outputs', 'emergent_geometry')


def true_distance_matrix(manifold: str, n_macro: int) -> np.ndarray:
    """Geodesic (s3/s2) or Euclidean (flat4) distance between the true macro
    centers, recomputed deterministically -- never fed to the graph."""
    centers = make_macro_centers(manifold, n_macro, device='cpu').numpy()
    if manifold in ('s3', 's2'):
        dots = np.clip(centers @ centers.T, -1.0, 1.0)
        D = np.arccos(dots)
    else:
        diff = centers[:, None, :] - centers[None, :, :]
        D = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(D, 0.0)
    return D


def classical_mds(R: np.ndarray, k: int = 8):
    """Double-centering MDS. Treats R as a squared-distance matrix (exact
    for effective resistance under the resistive-embedding interpretation)
    and returns a k-dim coordinate embedding plus the full eigenvalue
    spectrum (for judging effective dimensionality)."""
    n = R.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ R @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals_pos = np.clip(eigvals, 0, None)
    coords = eigvecs[:, :k] * np.sqrt(eigvals_pos[:k])
    return coords, eigvals


def effective_dimensionality(eigvals: np.ndarray, frac_variance: float = 0.90) -> int:
    pos = np.clip(eigvals, 0, None)
    total = pos.sum()
    if total < 1e-12:
        return 0
    cumfrac = np.cumsum(pos) / total
    return int(np.searchsorted(cumfrac, frac_variance) + 1)


def process_manifold(manifold: str, indir: str, outdir: str) -> dict:
    df = pd.read_csv(os.path.join(indir, f'{manifold}_gateway_edges.csv'))
    W, n_basins = build_weight_matrix(df)
    inv = laplacian_invariants(W)
    R = inv['R']  # effective resistance -- purely combinatorial

    D_true = true_distance_matrix(manifold, n_basins)
    iu = np.triu_indices(n_basins, k=1)

    # R interpreted as squared distance vs. true squared distance.
    r_sq, p_sq = stats.pearsonr(R[iu], D_true[iu] ** 2)
    rs_sq, ps_sq = stats.spearmanr(R[iu], D_true[iu] ** 2)

    # sqrt(R) as a proper distance vs. true distance (raw, not squared).
    sqrtR = np.sqrt(np.clip(R, 0, None))
    r_lin, p_lin = stats.pearsonr(sqrtR[iu], D_true[iu])
    rs_lin, ps_lin = stats.spearmanr(sqrtR[iu], D_true[iu])

    coords, eigvals = classical_mds(R, k=8)
    eff_dim = effective_dimensionality(eigvals, 0.90)

    # Does the MDS-reconstructed embedding (using only the top manifold-dim
    # components) recover distances that track the true manifold distance?
    manifold_dim = {'s3': 4, 's2': 3, 'flat4': 4}[manifold]
    recon_coords = coords[:, :manifold_dim]
    recon_D = squareform(pdist(recon_coords))
    r_recon, p_recon = stats.pearsonr(recon_D[iu], D_true[iu])

    return dict(
        manifold=manifold, n_basins=n_basins, true_manifold_dim=manifold_dim,
        effective_dim_90pct=eff_dim,
        eigval_spectrum_top10=[float(v) for v in eigvals[:10]],
        pearson_R_vs_true_sq=float(r_sq), pearson_p=float(p_sq),
        spearman_R_vs_true_sq=float(rs_sq), spearman_p=float(ps_sq),
        pearson_sqrtR_vs_true=float(r_lin), pearson_sqrtR_p=float(p_lin),
        pearson_mds_recon_vs_true=float(r_recon), pearson_mds_recon_p=float(p_recon),
    ), R, D_true, coords, eigvals


def plot_results(results_by_manifold: dict, data_by_manifold: dict, outdir: str):
    manifolds = list(results_by_manifold.keys())
    fig, axes = plt.subplots(2, len(manifolds), figsize=(6 * len(manifolds), 9), squeeze=False)

    for col, manifold in enumerate(manifolds):
        R, D_true, coords, eigvals = data_by_manifold[manifold]
        n = R.shape[0]
        iu = np.triu_indices(n, k=1)

        ax1 = axes[0][col]
        ax1.scatter(D_true[iu] ** 2, R[iu], s=14, alpha=0.5, c='#4C72B0')
        r = results_by_manifold[manifold]['pearson_R_vs_true_sq']
        ax1.set_xlabel('true squared geometric distance')
        ax1.set_ylabel('effective resistance R (graph-only)')
        ax1.set_title(f'{manifold}: r={r:.3f}')

        ax2 = axes[1][col]
        pos_eig = np.clip(eigvals[:12], 0, None)
        total = np.clip(eigvals, 0, None).sum() + 1e-12
        ax2.bar(range(len(pos_eig)), pos_eig / total, color='#55A868')
        eff_dim = results_by_manifold[manifold]['effective_dim_90pct']
        true_dim = results_by_manifold[manifold]['true_manifold_dim']
        ax2.axvline(true_dim - 0.5, color='red', ls='--', alpha=0.6,
                    label=f'true manifold dim={true_dim}')
        ax2.set_xlabel('MDS component')
        ax2.set_ylabel('fraction of variance')
        ax2.set_title(f'{manifold}: effective_dim(90%)={eff_dim}')
        ax2.legend(fontsize=8)

    fig.suptitle('Emergent geometry: does graph combinatorics alone recover manifold distance/dimension?')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'emergent_geometry.png'), dpi=140)
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

    print('Emergent Geometry Readout')
    print(f'  manifolds found: {manifolds}')

    results = {}
    data = {}
    for manifold in manifolds:
        print(f'\n-- {manifold} --')
        result, R, D_true, coords, eigvals = process_manifold(manifold, args.indir, args.outdir)
        results[manifold] = result
        data[manifold] = (R, D_true, coords, eigvals)
        print(f"  true manifold dim = {result['true_manifold_dim']}, "
              f"effective_dim(90% var) = {result['effective_dim_90pct']}")
        print(f"  corr(R, true_dist^2):      pearson r={result['pearson_R_vs_true_sq']:.3f} "
              f"p={result['pearson_p']:.4g}  spearman r={result['spearman_R_vs_true_sq']:.3f}")
        print(f"  corr(sqrt(R), true_dist):  pearson r={result['pearson_sqrtR_vs_true']:.3f} "
              f"p={result['pearson_sqrtR_p']:.4g}")
        print(f"  corr(MDS-recon, true_dist) [top-{result['true_manifold_dim']} dims]: "
              f"r={result['pearson_mds_recon_vs_true']:.3f} p={result['pearson_mds_recon_p']:.4g}")

    with open(os.path.join(args.outdir, 'emergent_geometry_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    plot_results(results, data, args.outdir)
    print(f'\nOutputs written to {args.outdir}/')


if __name__ == '__main__':
    main()
