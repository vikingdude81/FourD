"""Graph builders for the VFD bridge track.

All builders return (A, meta) where A is a symmetric 0/1 adjacency matrix
(numpy int array) and meta is a small dict describing the family.

No networkx dependency — construction is done directly with numpy.
"""

import itertools

import numpy as np


# ── Regular polytope skeletons ────────────────────────────────────────────

def _unit_vertex_graph(vertices, edge_len, tol=1e-6, name=""):
    """Connect vertex pairs at the given (minimal) distance."""
    P = np.asarray(vertices, dtype=float)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    A = (np.abs(d - edge_len) < tol).astype(int)
    return A, {"name": name, "n": len(P), "degree": int(A.sum(1)[0]),
               "vertices": P}


def six00_cell():
    """1-skeleton of the 600-cell (H4 roots, unit circumradius).

    120 vertices, 12-regular, 720 edges, edge length phi^-1.
    """
    phi = (1 + 5 ** 0.5) / 2
    pts = []
    # 24 part: (±1,0,0,0) perms + (±1/2,±1/2,±1/2,±1/2)
    for i in range(4):
        for s in (1, -1):
            v = [0.0] * 4
            v[i] = s
            pts.append(v)
    for signs in itertools.product((1, -1), repeat=4):
        pts.append([s * 0.5 for s in signs])
    # 96 part: even permutations and sign choices of (±1/2, ±1/(2phi), 0, ±phi/2)
    base = [0.5, 1 / (2 * phi), 0.0, phi / 2]

    def perm_sign(p):
        inv = 0
        for i in range(4):
            for j in range(i + 1, 4):
                if p[i] > p[j]:
                    inv += 1
        return inv % 2

    seen = set()
    for p in (q for q in itertools.permutations(range(4)) if perm_sign(q) == 0):
        for signs in itertools.product((1, -1), repeat=3):
            w, si = [0.0] * 4, 0
            for k in range(4):
                if base[p[k]] != 0:
                    w[k] = base[p[k]] * signs[si]
                    si += 1
            t = tuple(round(x, 9) for x in w)
            if t not in seen:
                seen.add(t)
                pts.append(list(t))
    return _unit_vertex_graph(pts, 1 / phi, name="600cell")


def two4_cell():
    """1-skeleton of the 24-cell: 24 vertices, 8-regular, 96 edges."""
    pts = []
    for i in range(4):
        for s in (1, -1):
            v = [0.0] * 4
            v[i] = s
            pts.append(v)
    for signs in itertools.product((1, -1), repeat=4):
        pts.append([s * 0.5 for s in signs])
    return _unit_vertex_graph(pts, 1.0, name="24cell")


# ── Classic families ──────────────────────────────────────────────────────

def palindromagon(n=9):
    """PLACEHOLDER for the 9-node Palindromagon.

    The original construction was not found in FourD/consciousness-sim repos.
    This placeholder is the natural 'palindromic path constraint' graph on n
    nodes: the n-cycle plus the chords pairing each vertex with its mirror
    image under every reflection axis of the regular n-gon (dihedral D_n
    symmetry, i.e. all paths read the same forwards/backwards under the
    reflection group). TODO: replace with the original Palindromagon
    adjacency once located in the source chat archive.
    """
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    for axis in range(n):  # n reflection axes; mirror map i -> (2*axis - i) % n
        for i in range(n):
            j = (2 * axis - i) % n
            if i != j:
                A[i, j] = A[j, i] = 1
    return A, {"name": f"palindromagon_{n}", "n": n, "degree": int(A.sum(1)[0])}


def cycle(n):
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    return A, {"name": f"cycle_{n}", "n": n, "degree": 2}


def grid(nx, ny):
    n = nx * ny
    A = np.zeros((n, n), dtype=int)
    for x in range(nx):
        for y in range(ny):
            i = x * ny + y
            if x + 1 < nx:
                A[i, i + ny] = A[i + ny, i] = 1
            if y + 1 < ny:
                A[i, i + 1] = A[i + 1, i] = 1
    return A, {"name": f"grid_{nx}x{ny}", "n": n, "degree": None}


def erdos_renyi(n, p, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < p).astype(int)
    A = np.triu(A, 1)
    A = A + A.T
    return A, {"name": f"er_{n}_p{p}", "n": n, "degree": None}


def barabasi_albert(n, m=2, seed=0):
    rng = np.random.default_rng(seed)
    # seed: complete graph on m+1 nodes
    A = np.zeros((n, n), dtype=int)
    A[: m + 1, : m + 1] = 1
    np.fill_diagonal(A[: m + 1, : m + 1], 0)
    for new in range(m + 1, n):
        probs = A[:new, :new].sum(0) / A[:new, :new].sum()
        chosen, tries = [], 0
        while len(chosen) < m and tries < 10 * n:
            k = int(rng.choice(new, p=probs))
            if k not in chosen:
                chosen.append(k)
            tries += 1
        for k in chosen:
            A[new, k] = A[k, new] = 1
    return A, {"name": f"ba_{n}_m{m}", "n": n, "degree": None}


LADDER_SIZES = [9, 16, 25, 36, 49, 64, 100, 144, 196, 256]
