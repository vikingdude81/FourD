"""E3 — does the immortal entity self-maintain under perturbation? (agency test)
Borrowed from Hamon et al. 2402.10236: an "individual" is judged by robustness of
its spatiotemporal unity to perturbation. Target: dodeca19 immortal entity (basin 15).
Kick at t=1500; recovery = leader 15 re-established for >=100 steps AND share15
back to >=95% of home share.
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = r"C:\Users\akbon\OneDrive\Documents\GitHub\FourD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tegmark_bridge"))
from d1_autopsy import make_engine

phi = (1 + 5 ** 0.5) / 2
dodeca = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)] +
    [[0, s1/phi, s2*phi] for s1 in (-1, 1) for s2 in (-1, 1)] +
    [[s1/phi, s2*phi, 0] for s1 in (-1, 1) for s2 in (-1, 1)] +
    [[s1*phi, 0, s2/phi] for s1 in (-1, 1) for s2 in (-1, 1)], float)
dodeca = np.c_[dodeca, np.zeros(20)]; dodeca /= np.linalg.norm(dodeca, axis=1)[:, None]
P = dodeca[:-1]

STEPS, T_KICK, ENT_BASIN = 4000, 1500, 15
HOLD = 100  # recrowned = leader for >= HOLD consecutive steps

def run(cond, seed):
    torch.manual_seed(seed); np.random.seed(seed % 2**31)
    eng = make_engine("cuda:0", 128, STEPS, P, "s3")
    dev = eng.device
    for t in range(STEPS):
        if t == T_KICK and cond != "control":
            memb = (eng.hist_macro_basin[:, t-1] == ENT_BASIN)
            idx = torch.nonzero(memb).squeeze(1)
            if cond.startswith("teleport"):
                frac = {"teleport25": 0.25, "teleport50": 0.5, "teleport100": 1.0}[cond]
                k = int(round(frac * idx.numel()))
                sel = idx[torch.randperm(idx.numel(), device=dev)[:k]]
                v = F.normalize(torch.randn(sel.numel(), eng.dim, device=dev), dim=1)
                eng.u_t[sel] = v
            elif cond == "wind":
                # displace the whole crowd 80% of the way to a far basin center
                tgt = eng.macro_centers[(ENT_BASIN + 9) % 24].unsqueeze(0)
                u = eng.u_t[idx]
                u_new = F.normalize(u + 4.0 * (tgt - u), dim=1)
                eng.u_t[idx] = u_new
            elif cond == "noise_burst":
                eng._saved_noise = eng.exploration_noise
                eng.exploration_noise = eng._saved_noise * 4.0
        if cond == "noise_burst" and t == T_KICK + 50 and hasattr(eng, "_saved_noise"):
            eng.exploration_noise = eng._saved_noise
        eng.step()

    B = eng.hist_macro_basin[:, :STEPS].cpu().numpy()
    nb = int(B.max()) + 1
    occ = np.array([np.bincount(B[:, t], minlength=nb) for t in range(STEPS)]).T.astype(float)
    share15 = occ[ENT_BASIN, :] / eng.N
    leader = occ.argmax(0)
    home = share15[1000:T_KICK].mean()
    post = share15[T_KICK:]
    min_share = post[:200].min()
    # recrown: first t where leader==15 and holds for HOLD
    recrown = None
    for t in range(T_KICK, STEPS - HOLD):
        if (leader[t:t+HOLD] == ENT_BASIN).all():
            recrown = t; break
    # recovery of home share after recrown
    recover = None
    if recrown is not None:
        tgt = 0.95 * home
        after = np.where(share15[recrown:] >= tgt)[0]
        if after.size: recover = recrown + int(after[0])
    post_mean = share15[T_KICK+200:].mean()
    lead15 = (leader[T_KICK:] == ENT_BASIN).mean()
    return dict(cond=cond, seed=seed, home_share=round(float(home), 3),
                min_share=round(float(min_share), 3),
                recrown_step=(None if recrown is None else int(recrown - T_KICK)),
                recover_step=(None if recover is None else int(recover - T_KICK)),
                post_mean=round(float(post_mean), 3), lead15_frac=round(float(lead15), 3))

CONDS = ["control", "teleport25", "teleport50", "teleport100", "wind", "noise_burst"]
out = []
for cond in CONDS:
    for sd in (9201, 9202, 9203):
        r = run(cond, sd); out.append(r); print(cond, r, flush=True)
json.dump(out, open(os.path.join(ROOT, "tegmark_bridge", "outputs", "e3_agency.json"), "w"), indent=1)
print("DONE")
