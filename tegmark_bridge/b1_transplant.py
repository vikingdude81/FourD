"""B1 — bearer transplant: WHERE does an entity live? (bearer-theory bridge)

The bearer theory wall says 'the bearer is not the thing changed'. Made
executable on the immortal entity (dodeca19, basin 15, 18/18 in E2):

  A control:   dodeca19 untouched, 4000 steps.
  B bearer-carried: at t=1500 replace prefs with 19 RANDOM points
               (different structure, same cardinality), keep u_t AND
               fatigue (fatigue channels now label the wrong preferences:
               the bearer carries its history into a foreign structure).
               At t=2500 restore dodeca19.
  C bearer-wiped:   same swaps, but reset u_t and fatigue at each swap
               (structure persists, bearer continuity broken).

If entity lives in the STRUCTURE: B's entity re-crowns instantly on
restoration and C also recrowns (structure guarantees identity).
If entity lives in the BEARER: B survives the away-phase (carried
fatigue holds the entity together in foreign structure), C loses it.
If entity lives in the RELATION (bearer x structure): both die away,
B recrowns faster/stronger than C on restoration.
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
P_HOME = dodeca[:-1]
STEPS = 4000; T1 = 1500; T2 = 2500

def phase_stats(eng, a, b):
    B = eng.hist_macro_basin[:, a:b].cpu().numpy()
    share15 = (B == 15).mean(0)
    occ = np.zeros((24, b - a))
    for t in range(b - a):
        for bb in range(24):
            occ[bb, t] = (B[:, t] == bb).sum()
    leader = occ.argmax(0)
    lead15 = float((leader == 15).mean())
    return dict(share15=round(float(share15.mean()), 3),
                lead15_frac=round(lead15, 3),
                top_leader=int(np.bincount(leader).argmax()))

def run(mode, seed):
    rng = np.random.RandomState(seed + 500)
    P_AWAY = rng.randn(19, 4); P_AWAY /= np.linalg.norm(P_AWAY, axis=1, keepdims=True)
    torch.manual_seed(seed); np.random.seed(seed)
    eng = make_engine("cuda:0", 128, STEPS, P_HOME, "s3")

    def set_prefs(P):
        Pt = torch.tensor(P, dtype=torch.float32, device=eng.device)
        Pt = Pt / Pt.norm(dim=1, keepdims=True)
        eng.prefs = Pt; eng.n_sub = Pt.shape[0]

    def wipe_bearer():
        eng.fatigue = torch.zeros(eng.N, eng.n_sub, device=eng.device)
        eng.u_t = F.normalize(torch.randn(eng.N, 4, device=eng.device), dim=1)

    t = 0
    while t < STEPS:
        if t == T1:
            set_prefs(P_AWAY)
            if mode == "C": wipe_bearer()
        if t == T2:
            set_prefs(P_HOME)
            if mode == "C": wipe_bearer()
        eng.step(); t += 1

    out = dict(mode=mode, seed=seed,
               home=phase_stats(eng, 0, T1),
               away=phase_stats(eng, T1 + 100, T2),
               back=phase_stats(eng, T2 + 100, STEPS))
    print(out, flush=True)
    return out

res = []
for sd in (9101, 9102, 9103):
    for mode in ("A", "B", "C"):
        if mode == "A" and sd != 9101: pass  # control same across seeds anyway; keep all
        res.append(run(mode, sd))
json.dump(res, open(os.path.join(ROOT, "tegmark_bridge", "outputs", "b1_transplant.json"), "w"), indent=1)
print("WROTE b1_transplant.json")
