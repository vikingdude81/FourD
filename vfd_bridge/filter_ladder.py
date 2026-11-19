"""S1 — Spectral Selection Scaling: the VFD filter across graph families.

The VFD trick is a *designed null*: a linear filter H(lambda) = 7 - lambda/2
whose planted root (lambda* = 14) coincides with an eigenvalue family of the
600-cell, deleting 36/120 = 30% of the mode space with one scalar.

Question for the Scaling Law of Symmetry program: how does the EFFICIENCY of
such a cancellation scale as graph symmetry/complexity degrades from the
600-cell (maximal symmetry) down through lattices to Barabasi-Albert
scale-free graphs?

Metrics per graph:
  n_families        # distinct Laplacian eigenvalues (spectral granularity)
  max_mult_frac     # largest family as fraction of n  (symmetry signature)
  H_spec            # Shannon entropy (nats) over mode measure p_r = m_r/n
  H(14)=0 exact     # fraction of modes exactly cancelled by VFD's raw root
  root_frac@1e-3    # fraction within |lambda - lambda*| < 1e-3 (robustness)
  adaptive_root     # same with root re-planted at 7*mean_degree/6
  dH_exact          # entropy reduction from the exact cancellation

Hypothesis: designed-null efficiency tracks spectral concentration.
600-cell: 30% deleted by one planted root.  BA: spectrum nearly continuous,
exact-root cancellation ~0 modes, dH ~ 0.  The curve of dH_exact vs
max_mult_frac across the ladder is the candidate scaling law.

Run:  python filter_ladder.py            (prints table, writes CSV)
"""

import csv
import os

import numpy as np

from graphs import (LADDER_SIZES, barabasi_albert, cycle, erdos_renyi, grid,
                    palindromagon, six00_cell, two4_cell)

TOL_FAMILY = 1e-6
EPS_ROOT = 1e-3


def laplacian(A):
    deg = A.sum(1)
    return np.diag(deg) - A


def families(ev, tol=TOL_FAMILY):
    out = []
    for x in np.sort(ev):
        if not out or abs(out[-1][0] - x) > tol:
            out.append([float(x), 1])
        else:
            out[-1][1] += 1
    return out


def spec_entropy(mults):
    p = np.asarray(mults, dtype=float)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def run_filter(fams, root):
    """Cancel modes at `root` (exact + within EPS_ROOT). Return stats."""
    mults = [m for _, m in fams]
    n = sum(mults)
    killed_exact = sum(m for lam, m in fams if abs(lam - root) < TOL_FAMILY)
    killed_eps = sum(m for lam, m in fams if abs(lam - root) < EPS_ROOT)
    h_before = spec_entropy(mults)
    kept = [m for lam, m in fams if abs(lam - root) >= TOL_FAMILY
            and lam > TOL_FAMILY]  # also drop uniform mode (mean removal)
    removed_uniform = sum(m for lam, m in fams if lam < TOL_FAMILY)
    h_after = spec_entropy(kept) if kept else 0.0
    # entropy of mean-removal-only baseline for fair dH (uniform mode out)
    h_mean_only = spec_entropy([m for lam, m in fams if lam > TOL_FAMILY])
    return {
        "n": n,
        "n_families": len(fams),
        "max_mult_frac": max(mults) / n,
        "H_spec": spec_entropy(mults),
        "killed_exact_frac": killed_exact / n,
        "killed_eps_frac": killed_eps / n,
        "dH_exact": h_mean_only - h_after,
        "removed_uniform": removed_uniform,
    }


def main():
    rows = []

    def add(name, A, extra=""):
        fams = families(np.linalg.eigvalsh(laplacian(A)))
        mean_deg = float(np.asarray(A.sum(1)).mean())
        r_fixed = run_filter(fams, 14.0)
        r_adapt = run_filter(fams, 7 * mean_deg / 6)
        rows.append({
            "graph": name, "n": r_fixed["n"],
            "mean_degree": round(mean_deg, 3),
            "n_families": r_fixed["n_families"],
            "max_mult_frac": round(r_fixed["max_mult_frac"], 4),
            "H_spec": round(r_fixed["H_spec"], 4),
            "VFDroot14_killed_frac": round(r_fixed["killed_exact_frac"], 4),
            "VFDroot14_killed_eps": round(r_fixed["killed_eps_frac"], 4),
            "VFDroot14_dH": round(r_fixed["dH_exact"], 4),
            "adaptroot_killed_frac": round(r_adapt["killed_exact_frac"], 4),
            "adaptroot_dH": round(r_adapt["dH_exact"], 4),
            "note": extra,
        })

    A, _ = six00_cell()
    add("600cell", A, "maximal symmetry (VFD case)")
    A, _ = two4_cell()
    add("24cell", A, "maximal symmetry (lower)")
    A, _ = palindromagon(9)
    add("palindromagon9", A, "placeholder defn — see graphs.py")
    for n in (9, 16, 36, 64, 144):
        A, _ = cycle(n)
        add(f"cycle_{n}", A)
    for s in (3, 4, 6, 8, 12, 16):
        A, _ = grid(s, s)
        add(f"grid_{s}x{s}", A)
    for n in LADDER_SIZES:
        A, _ = barabasi_albert(n, m=2, seed=7)
        add(f"ba_{n}", A, "scale-free")
    for n in LADDER_SIZES:
        A, _ = erdos_renyi(n, p=4.0 / n, seed=7)
        add(f"er_{n}", A, "mean deg ~4")

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "outputs", "spectral_selection_scaling.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = (f"{'graph':>16} {'n':>5} {'deg':>5} {'fam':>5} {'maxmF':>7} "
           f"{'H_spec':>7} {'kill@14':>8} {'dH@14':>7} {'kill_ad':>8} "
           f"{'dH_ad':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['graph']:>16} {r['n']:>5} {r['mean_degree']:>5} "
              f"{r['n_families']:>5} {r['max_mult_frac']:>7} {r['H_spec']:>7} "
              f"{r['VFDroot14_killed_frac']:>8} {r['VFDroot14_dH']:>7} "
              f"{r['adaptroot_killed_frac']:>8} {r['adaptroot_dH']:>7}")
    print(f"\nCSV written: {out}")


if __name__ == "__main__":
    main()
