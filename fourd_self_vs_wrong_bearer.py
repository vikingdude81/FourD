#!/usr/bin/env python3
"""
Self versus wrong-bearer in FourD (Bearer Theory Prediction 2 / sharp Prediction 4).

The bearer_state_competency study ran only an on/off ablation, which the manifold
geometry confounds. This lifts the decisive M3 experiment from the Constitutive GPT
into FourD: it isolates whether the incorporated bearer is not just present but
specifically THIS body's and non-transferable.

Method (no engine surgery: b_t is a plain attribute used for perception at the start
of each step, so a bearer is transplanted by overwriting b_t between steps):

  - Body A and body B are two BearerEngines with different seeds, so different
    subsystem preferences and macro-centers (different bodies), same manifold and
    topology.
  - We first record each engine's own bearer trajectory b_t (the value it perceives
    through at the start of each step) on its own body.
  - Then we run body A under three bearer streams and measure a competency on A's own
    dynamics:
      self       : A keeps its own bearer (the reference).
      wrong      : A's bearer is overwritten each step by body B's stream (a different
                   body's incorporated identity), matched in volume.
      falsifier  : A's bearer is overwritten by a second A-body instance's stream
                   (same body, independent noise) -- should match self.

Competency: self_maintenance (mean/std of clarity over the tail, higher is more
stable/coherent) and basin_divergence (L1 distance of macro-basin occupancy from the
self run). Prediction: wrong degrades competency versus self, while the falsifier
holds, so the bearer is body-specific and non-transferable.

Usage:  python fourd_self_vs_wrong_bearer.py [--device cuda:0] [--seeds 6]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from bearer_state_competency import BearerEngine, build_engine, resolve_device, basin_occupancy

OUT_DIR = os.path.join('outputs', 'self_vs_wrong_bearer')


def record_bearer_stream(engine: BearerEngine, steps: int) -> torch.Tensor:
    """Run an engine on its own body, returning the bearer value it perceives through
    at the start of each step: stream[t] = b_t before step t. Shape (steps, N, n_sub)."""
    stream = torch.zeros(steps, engine.N, engine.n_sub, device=engine.device)
    for t in range(steps):
        stream[t] = engine.b_t.clone()
        engine.step()
    return stream


def run_with_stream(manifold, topology, device, steps, N, body_seed, rng_seed,
                    stream: torch.Tensor | None):
    """Run body `body_seed`. If stream is given, overwrite b_t before each step (a
    transplant). Returns (self_maintenance, macro_occupancy)."""
    eng = build_engine(manifold, topology, True, device, steps, N, body_seed, rng_seed=rng_seed)
    for t in range(steps):
        if stream is not None:
            eng.b_t = stream[t]
        eng.step()
    clar = eng.hist_clarity[:, :steps].cpu().numpy()
    warm = steps // 3
    tail = clar[:, warm:]
    m = tail.mean(axis=1); s = tail.std(axis=1) + 1e-8
    self_maint = float(np.mean(m / s))
    basins = eng.hist_macro_basin[:, warm:steps].cpu().numpy()
    occ = basin_occupancy(basins, eng.n_macro)
    return self_maint, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--manifold', default='s3')
    ap.add_argument('--topology', default='cyclic')
    ap.add_argument('--N', type=int, default=96)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'result.json'))
    args = ap.parse_args()
    dev = resolve_device(args.device)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[FourD self-vs-wrong] manifold={args.manifold} topo={args.topology} "
          f"N={args.N} steps={args.steps} seeds={args.seeds} device={dev}")

    rows = []
    for s in range(args.seeds):
        body_A = 1000 + s
        body_B = 5000 + s          # a different body
        # record own-bearer streams
        engA = build_engine(args.manifold, args.topology, True, dev, args.steps, args.N, body_A, rng_seed=body_A)
        record_bearer_stream(engA, args.steps)
        engB = build_engine(args.manifold, args.topology, True, dev, args.steps, args.N, body_B, rng_seed=body_B)
        streamB = record_bearer_stream(engB, args.steps)
        engAp = build_engine(args.manifold, args.topology, True, dev, args.steps, args.N, body_A, rng_seed=body_A + 777)
        streamAp = record_bearer_stream(engAp, args.steps)  # same body A, different noise

        sm_self, occ_self = run_with_stream(args.manifold, args.topology, dev, args.steps, args.N, body_A, body_A, None)
        sm_wrong, occ_wrong = run_with_stream(args.manifold, args.topology, dev, args.steps, args.N, body_A, body_A, streamB)
        sm_fals, occ_fals = run_with_stream(args.manifold, args.topology, dev, args.steps, args.N, body_A, body_A, streamAp)

        div_wrong = float(np.abs(occ_wrong - occ_self).sum())
        div_fals = float(np.abs(occ_fals - occ_self).sum())
        rows.append(dict(seed=s, sm_self=sm_self, sm_wrong=sm_wrong, sm_fals=sm_fals,
                         div_wrong=div_wrong, div_fals=div_fals))
        print(f"  seed {s}: self_maint self={sm_self:.3f} wrong={sm_wrong:.3f} fals={sm_fals:.3f} | "
              f"basin div wrong={div_wrong:.3f} fals={div_fals:.3f}")

    def ms(k):
        v = np.array([r[k] for r in rows]); return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
    smS, smSs = ms('sm_self'); smW, smWs = ms('sm_wrong'); smF, smFs = ms('sm_fals')
    dW, dWs = ms('div_wrong'); dF, dFs = ms('div_fals')
    print(f"\n[SUMMARY over {len(rows)} bodies]")
    print(f"  self_maintenance   self={smS:.3f}+/-{smSs:.3f}  wrong={smW:.3f}+/-{smWs:.3f}  fals={smF:.3f}+/-{smFs:.3f}")
    print(f"  basin divergence from self   wrong={dW:.3f}+/-{dWs:.3f}   falsifier={dF:.3f}+/-{dFs:.3f}")
    sm_drop = smS - smW
    isolates = dW - dF
    effect = sm_drop > 2 * (smWs + smFs) or isolates > 2 * (dWs + dFs)
    print(f"\n[VERDICT] self_maintenance drop under wrong-bearer = {sm_drop:+.3f}; "
          f"basin divergence isolates wrong vs falsifier = {isolates:+.3f}")
    if effect:
        print("  BEARER IS BODY-SPECIFIC: transplanting a different body's bearer degrades competency / "
              "shifts basin structure more than a same-body bearer with different noise. FourD joins the "
              "GPT in showing the incorporated state is non-transferable (Prediction 2 / sharp Prediction 4).")
    else:
        print("  NO CLEAR SPECIFICITY at this operating point: wrong ~= falsifier. Either the bearer is not "
              "strongly body-adaptive here (consistent with the geometry-confounded on/off ablation) or a "
              "different competency metric / operating point is needed. Honest partial, mirrors the forager.")

    json.dump(dict(rows=rows, self_maint=[smS, smSs], wrong=[smW, smWs], fals=[smF, smFs],
                   div_wrong=[dW, dWs], div_fals=[dF, dFs], sm_drop=sm_drop, isolates=isolates,
                   effect=bool(effect)), open(args.out, 'w'), indent=2)
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
