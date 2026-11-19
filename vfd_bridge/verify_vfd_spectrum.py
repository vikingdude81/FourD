"""Verify the VFD (vibrational field dynamics) claim table independently.

VFD claims: on the 600-cell 1-skeleton, L = 12I - A has exactly 9 eigenvalue
families, and the linear filter H(lambda) = 7 - lambda/2 cancels the
lambda = 14 family (multiplicity 36) EXACTLY, leaving 7 families spanning
dim 120 - 1 - 36 = 83.  Two coupled oscillator branches per family,
omega^2_{+,r} = kappa*lambda_r + {0, 2g}, give up to 2*7 = 14 resonant
branches.

Run:  python verify_vfd_spectrum.py
Exit 0 = claims verified, 1 = mismatch.
"""

import sys

import numpy as np

from graphs import six00_cell

# The table as published in the VFD infographic (4 dp).
CLAIMED = [
    (0.0000, 1, "REMOVED (uniform mode)"),
    (2.2918, 4, "PASSED"),
    (5.5279, 9, "PASSED"),
    (9.0000, 16, "PASSED"),
    (12.0000, 25, "PASSED"),
    (14.0000, 36, "CANCELLED by H(14)=0"),
    (14.4721, 9, "PASSED"),
    (15.0000, 16, "PASSED"),
    (15.7082, 4, "PASSED"),
]


def families(ev, tol=1e-6):
    out = []
    for x in np.sort(ev):
        if not out or abs(out[-1][0] - x) > tol:
            out.append([float(x), 1])
        else:
            out[-1][1] += 1
    return out


def main():
    A, meta = six00_cell()
    n = meta["n"]
    n_edges = int(A.sum() // 2)
    print(f"600-cell skeleton: {n} vertices, {n_edges} edges, "
          f"degree set {sorted(set(A.sum(1)))}")
    assert n == 120 and n_edges == 720, "skeleton construction failed"

    L = 12 * np.eye(n) - A
    fams = families(np.linalg.eigvalsh(L))

    ok = len(fams) == len(CLAIMED)
    print(f"\n{'lambda':>9} {'mult':>5} {'H=7-l/2':>9}  claimed  match")
    for (lam, m), (clam, cmult, verdict) in zip(fams, CLAIMED):
        match = abs(lam - clam) < 5e-4 and m == cmult
        ok = ok and match
        print(f"{lam:9.4f} {m:5d} {7 - lam / 2:9.4f}  {clam:7.4f}  "
              f"{'OK' if match else 'MISMATCH ' + str((round(lam,4), m))}")

    mult_sum = sum(m for _, m in fams)
    kept_dim = mult_sum - 1 - 36  # remove uniform mode + cancelled family
    n_families_kept = len(fams) - 2
    print(f"\nmultiplicities sum: {mult_sum} (expect 120)")
    print(f"retained dim after mean-removal + H(14) cancellation: {kept_dim} "
          f"(VFD claims 83)")
    print(f"retained spectral families: {n_families_kept} (VFD claims 7)")
    print(f"oscillator branches omega^2 = kappa*lambda_r + {{0, 2g}}: "
          f"at most {2 * n_families_kept} distinct (VFD claims up to 14)")

    ok = ok and kept_dim == 83 and n_families_kept == 7
    print("\nVERDICT:", "VFD TABLE VERIFIED" if ok else "MISMATCH — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
