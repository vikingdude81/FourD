#!/usr/bin/env python3
"""
FourD foreclosure: make the incorporated bearer the ONLY body identity, then re-run
self vs wrong-bearer.

The plain self-vs-wrong test gave a null because the body is its fixed subsystem
preferences and b_t is a weak add-on (the Generic-Policy Escape in a dynamical system).
This forecloses that escape:

  - all bodies share the SAME physical engine (same seed -> identical prefs and
    macro-centers), so the fixed preferences cannot carry body identity;
  - the preference channel is weakened (prefs scaled down) and the bearer channel is
    strengthened (high bearer_weight), so behavior must route through b_t;
  - each body's identity is a distinct vector seeded into b_t (b_0 = id_i) and then
    sustained only by the incorporation loop, not re-injected -- so it is genuinely
    incorporated, not a constant record.

Body identity now lives only in the incorporated state. If that state is load-bearing,
transplanting a different body's grown bearer (wrong) should drive body A toward that
body's behavior, diverging from A's self baseline, while the falsifier (a second
same-identity instance with different noise) holds. That is non-transferability of the
incorporated identity, the FourD analogue of Constitutive GPT M3.

Usage:  python fourd_foreclosure.py [--device cuda:0] [--pref-scale 0.15]
                                    [--bearer-weight 2.5] [--seeds 6]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from bearer_state_competency import BearerEngine, build_engine, resolve_device, basin_occupancy

OUT_DIR = os.path.join('outputs', 'foreclosure')


def make_body(manifold, topology, device, steps, N, phys_seed, rng_seed,
              pref_scale, bearer_weight, id_vec):
    """Physically identical engine (phys_seed), weakened prefs, strong bearer, and a
    body-identity vector seeded into b_0."""
    eng = build_engine(manifold, topology, True, device, steps, N, phys_seed, rng_seed=rng_seed)
    eng.prefs = eng.prefs * pref_scale
    eng.bearer_weight = bearer_weight
    eng.b_t = id_vec.clone()
    return eng


def record_stream(eng, steps):
    stream = torch.zeros(steps, eng.N, eng.n_sub, device=eng.device)
    for t in range(steps):
        stream[t] = eng.b_t.clone()
        eng.step()
    return stream


def run_measure(manifold, topology, device, steps, N, phys_seed, rng_seed,
                pref_scale, bearer_weight, id_vec, stream):
    eng = make_body(manifold, topology, device, steps, N, phys_seed, rng_seed,
                    pref_scale, bearer_weight, id_vec)
    for t in range(steps):
        if stream is not None:
            eng.b_t = stream[t]
        eng.step()
    warm = steps // 3
    clar = eng.hist_clarity[:, warm:steps].cpu().numpy()
    m = clar.mean(axis=1); s = clar.std(axis=1) + 1e-8
    self_maint = float(np.mean(m / s))
    basins = eng.hist_macro_basin[:, warm:steps].cpu().numpy()
    return self_maint, basin_occupancy(basins, eng.n_macro)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--manifold', default='s3')
    ap.add_argument('--topology', default='cyclic')
    ap.add_argument('--N', type=int, default=96)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--pref-scale', type=float, default=0.15, help='<1 weakens fixed prefs (foreclose escape)')
    ap.add_argument('--bearer-weight', type=float, default=2.5, help='>0.6 strengthens the bearer channel')
    ap.add_argument('--id-strength', type=float, default=1.5)
    ap.add_argument('--phys-seed', type=int, default=42, help='shared physical body (identical prefs)')
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'result.json'))
    args = ap.parse_args()
    dev = resolve_device(args.device)
    os.makedirs(OUT_DIR, exist_ok=True)
    n_sub = 8
    print(f"[FourD foreclosure] pref_scale={args.pref_scale} bearer_weight={args.bearer_weight} "
          f"id_strength={args.id_strength} N={args.N} steps={args.steps} seeds={args.seeds} dev={dev}")

    g = torch.Generator(device=dev).manual_seed(31)
    rows = []
    for s in range(args.seeds):
        # distinct identities for body A and body B; A' shares A's identity
        idA = (torch.randn(n_sub, generator=g, device=dev) * args.id_strength).expand(args.N, n_sub).contiguous()
        idB = (torch.randn(n_sub, generator=g, device=dev) * args.id_strength).expand(args.N, n_sub).contiguous()

        common = dict(manifold=args.manifold, topology=args.topology, device=dev,
                      steps=args.steps, N=args.N, phys_seed=args.phys_seed,
                      pref_scale=args.pref_scale, bearer_weight=args.bearer_weight)
        # grow each identity's bearer stream (own dynamics, no overwrite)
        streamB = record_stream(make_body(rng_seed=5000 + s, id_vec=idB, **common), args.steps)
        streamAp = record_stream(make_body(rng_seed=1000 + s + 777, id_vec=idA, **common), args.steps)

        sm_self, occ_self = run_measure(rng_seed=1000 + s, id_vec=idA, stream=None, **common)
        sm_wrong, occ_wrong = run_measure(rng_seed=1000 + s, id_vec=idA, stream=streamB, **common)
        sm_fals, occ_fals = run_measure(rng_seed=1000 + s, id_vec=idA, stream=streamAp, **common)

        div_wrong = float(np.abs(occ_wrong - occ_self).sum())
        div_fals = float(np.abs(occ_fals - occ_self).sum())
        rows.append(dict(seed=s, sm_self=sm_self, sm_wrong=sm_wrong, sm_fals=sm_fals,
                         div_wrong=div_wrong, div_fals=div_fals))
        print(f"  body {s}: self_maint self={sm_self:.3f} wrong={sm_wrong:.3f} fals={sm_fals:.3f} | "
              f"basin div  wrong={div_wrong:.3f}  fals={div_fals:.3f}")

    def ms(k):
        v = np.array([r[k] for r in rows]); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
    dW, dWs = ms('div_wrong'); dF, dFs = ms('div_fals')
    smS, _ = ms('sm_self'); smW, _ = ms('sm_wrong'); smF, _ = ms('sm_fals')
    print(f"\n[SUMMARY over {len(rows)} bodies]")
    print(f"  self_maintenance  self={smS:.3f} wrong={smW:.3f} fals={smF:.3f}")
    print(f"  basin divergence from self:  wrong={dW:.3f}+/-{dWs:.3f}   falsifier={dF:.3f}+/-{dFs:.3f}")
    isolate = dW - dF
    effect = isolate > 2 * (dWs + dFs) and dW > 0.05
    print(f"\n[VERDICT] wrong-vs-falsifier basin divergence gap = {isolate:+.3f}")
    if effect:
        print("  FORECLOSURE WORKS: with identity carried only by the incorporated bearer, transplanting a "
              "different body's grown bearer drives A away from its self baseline, while the same-identity "
              "falsifier holds. The incorporated state is load-bearing and non-transferable in FourD. Fourth "
              "positive cell: the design law's foreclosure direction confirmed in a dynamical system.")
    else:
        print("  STILL NULL / difficulty valley: weakening prefs did not make the bearer carry a "
              "transferable-then-non-transferable identity at this operating point (wrong ~= falsifier, or "
              "dynamics washed the identity out). Honest banked attempt; a task-driven adaptive bearer, not "
              "just a seeded one, may be required -- the forager's obligate arc ported to FourD.")

    json.dump(dict(rows=rows, div_wrong=[dW, dWs], div_fals=[dF, dFs], isolate=isolate,
                   pref_scale=args.pref_scale, bearer_weight=args.bearer_weight, effect=bool(effect)),
              open(args.out, 'w'), indent=2)
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
