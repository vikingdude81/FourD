"""T6 — hostability certificate on real LPF market structure.

Substrate = daily cross-sectional return direction vectors (R^6:
SPY, SMH, IWM, MU, NVDA, AMD) from the local parquet cache.

Steps:
  1. Certificate: antipodal closure fraction, M2 isotropy dev,
     top principal-direction concentration.
  2. Engine test: 60-day subsample as UniversalEngine preference
     substrate (3 seeds) -> null_z (C1 hostability).
  3. Control: close the same sample antipodally (P + -P) -> predicted dead.

Prediction (from T1/T2/D1): markets are empirically non-antipodal and
anisotropic => certificate passes => engine survives. If it does, the
market's preference geometry is a hospitable structure for
boundary-negotiation dynamics -> LPF basin dynamics connect to the
Tegmark hostability program.
"""
import json, os, sys
import numpy as np
import pandas as pd
import torch

ROOT = r"C:\Users\akbon\OneDrive\Documents\GitHub\FourD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tegmark_bridge"))
from universality_test import UniversalEngine, boundary_metrics
from d1_autopsy import make_engine

HIST = r"E:\lpf_data\history"
TICKERS = ["SPY", "SMH", "IWM", "MU"]   # R^4 basket to match s3 engine dim
WIN = 504  # ~2y of daily bars, matching residual_ou basin window

def load_returns():
    cols = {}
    for t in TICKERS:
        df = pd.read_parquet(os.path.join(HIST, f"{t}.parquet"))
        s = df["close"].astype(float)
        if len(df) and s.iloc[-1] / s.iloc[-2] > 1.8:  # split guard
            pass
        cols[t] = np.log(s).diff()
    R = pd.DataFrame(cols).dropna()
    return R.tail(WIN)

def certs(P, name):
    n = len(P)
    closed = 0
    G = P @ P.T
    for i in range(n):
        g = G[i].copy(); g[i] = 1.0
        if (g < -0.999).any():
            closed += 1
    M2 = P.T @ P
    dev2 = np.linalg.norm(M2 - np.trace(M2) / P.shape[1] * np.eye(P.shape[1])) / np.linalg.norm(M2)
    sv = np.linalg.svd(P, compute_uv=False)
    conc = (sv ** 2 / (sv ** 2).sum()).max()
    out = dict(name=name, n=n, dim=P.shape[1], antiFrac=round(closed / n, 4),
               dev2=round(float(dev2), 4), top_pc_conc=round(float(conc), 4))
    print(f"{name:22s} n={n:4d} antiFrac={out['antiFrac']:.4f} dev2={out['dev2']:.4f} conc={out['conc'] if False else out['top_pc_conc']:.4f}")
    return out

def engine_z(P, seeds=(2026, 2027, 2028)):
    zs, trs = [], []
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        eng = make_engine("cuda:0", 128, 1000, P, "s3")
        for _ in range(1000):
            eng.step()
        mm = boundary_metrics(eng, n_shuffles=50)
        zs.append(mm["null_z"]); trs.append(mm["transition_rate"])
    return float(np.mean(zs)), float(np.std(zs)), float(np.mean(trs))

def main():
    R = load_returns()
    V = R.values
    Vc = V - V.mean(0)                       # center: directions relative to mean drift
    P = Vc / np.linalg.norm(Vc, axis=1, keepdims=True)
    rng = np.random.RandomState(11)

    print(f"window: {R.index[0]} -> {R.index[-1]}  ({len(P)} bars, dim {P.shape[1]})")
    report = {}
    report['cert_market_full'] = certs(P, "market_full")

    idx = rng.choice(len(P), 60, replace=False)
    S = P[idx]
    report['cert_market_60'] = certs(S, "market_60")
    Sc = np.vstack([S, -S])                  # antipodal closure control
    report['cert_closed_ctrl'] = certs(Sc, "market_60_closed")

    rnd = rng.randn(60, 4); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    report['cert_random60'] = certs(rnd, "random_60_ctrl")

    # structural-residual variant: strip per-ticker beta vs SPY (col 0),
    # i.e. the LPF "epsilon" — directions of unexplained co-movement only
    y = V[:, 0:1]
    beta = (V * y).sum(0, keepdims=True) / (y * y).sum(keepdims=True)
    Vr = V - beta * y
    Vr = Vr / np.linalg.norm(Vr, axis=1, keepdims=True)
    Sr = Vr[idx]
    report['cert_market_resid60'] = certs(Sr, "market_resid_60")

    # whitened market: same data, anisotropy removed (M2^{-1/2})
    M2 = S.T @ S / len(S)
    w, Q = np.linalg.eigh(M2)
    Wm = Q @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-6, None))) @ Q.T
    Sw = S @ Wm
    Sw = Sw / np.linalg.norm(Sw, axis=1, keepdims=True)
    report['cert_market_white60'] = certs(Sw, "market_white_60")

    # regime prototypes: cluster the full 504 directions (LPF regime_discovery
    # analogue) -> well-separated prototype substrate instead of raw days
    from scipy.cluster.vq import kmeans2
    protos = {}
    for k in (8, 16, 24):
        cent, _ = kmeans2(P, k, seed=3, minit='++')
        cent = cent / np.linalg.norm(cent, axis=1, keepdims=True)
        protos[k] = cent
        report[f'cert_protos{k}'] = certs(cent, f"protos_k{k}")

    print("\nengine (3 seeds x 1000 steps, N=128, s3):")
    for key, Psub in [("market_60", S), ("market_60_closed", Sc),
                      ("market_resid_60", Sr), ("market_white_60", Sw),
                      ("protos_k8", protos[8]), ("protos_k16", protos[16]),
                      ("protos_k24", protos[24]),
                      ("random_60_ctrl", rnd)]:
        z, zsd, tr = engine_z(Psub)
        state = "HOSTS" if (z > 3 and tr < 0.9) else ("BOILS" if tr >= 0.9 else "DEAD")
        report[key + '_engine'] = dict(z=round(z, 1), z_sd=round(zsd, 1), tr=round(tr, 3),
                                       state=state)
        print(f"  {key:18s} z={z:+7.1f} ±{zsd:4.1f}  tr={tr:.3f}  -> {state}")

    out = os.path.join(ROOT, "tegmark_bridge", "outputs")
    os.makedirs(out, exist_ok=True)
    json.dump(report, open(os.path.join(out, "t6_market_certificate.json"), "w"), indent=1)
    print("\nWROTE outputs/t6_market_certificate.json")

if __name__ == "__main__":
    main()
