"""B3 — co-rotating-frame transport test: what stayed the same across
distinct realizations of the form?

Protocol (suggested by external frontier-model audit, 2026-11-20):
Rotate dodeca19 prefs by theta at t=1500 (B2 mechanism, same plane/seed),
LEAVE the measuring grid fixed, but record the agents' CONTINUOUS directions
(eng.u_t) during the away window BEFORE assigning basin labels. Undo the
imposed rotation on those recorded directions (apply R(-theta)) and only THEN
assign them to the fixed 24-center grid (replicating the engine's own rule:
argmax <u, center>; validated against hist_macro_basin on untransported
snapshots). Compare transported away-window occupancy vs control runs
(no rotation) at the same seed; the baseline for "same distribution" is
ordinary run-to-run variation (pairwise TV across control seeds).

If TV(transported vs control same-seed) falls inside the run-to-run TV band,
the form — the population distribution in the co-rotating frame — is preserved
across distinct realizations even when the fixed-grid leader changes address.
"""
import sys, os, json
import numpy as np
import torch

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
STEPS = 4000; T1 = 1500; W0, W1 = 1600, 2600  # away window (after 100-step transient)
NB = 24

def givens(theta, plane, dim=4):
    R = np.eye(dim)
    i, j = plane
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c; R[j, j] = c; R[i, j] = -s; R[j, i] = s
    return R

def occupancy_from_u(u, C):
    """Replicate engine rule: argmax <u, center> (softmax is monotone)."""
    return (u @ C.T).argmax(1)

def run(theta_deg, seed, rotate):
    rng = np.random.RandomState(4242)
    plane = tuple(rng.choice(4, 2, replace=False))
    th = np.radians(theta_deg) if rotate else 0.0
    Rm = givens(th, plane)
    Rinv = givens(-th, plane)
    torch.manual_seed(seed); np.random.seed(seed)
    eng = make_engine("cuda:0", 128, STEPS, P_HOME, "s3")
    P_rot = P_HOME @ Rm.T
    occ_raw = np.zeros((NB, 0)); occ_tr = np.zeros((NB, 0))
    match_hits = match_tot = 0
    C = eng.macro_centers.cpu().numpy()
    for t in range(STEPS):
        if t == T1 and rotate:
            Pt = torch.tensor(P_rot, dtype=torch.float32, device=eng.device)
            Pt = Pt / Pt.norm(dim=1, keepdim=True)
            eng.prefs = Pt; eng.n_sub = Pt.shape[0]
        eng.step()
        if W0 <= t < W1:
            u = eng.u_t.cpu().numpy()
            lab_raw = occupancy_from_u(u, C)
            lab_tr = occupancy_from_u(u @ Rinv.T, C)
            # sanity: replica rule vs engine's own label at same step
            hist = eng.hist_macro_basin[:, t].cpu().numpy()
            match_hits += int((lab_raw == hist).sum()); match_tot += len(hist)
            occ_raw = np.c_[occ_raw, np.bincount(lab_raw, minlength=NB)]
            occ_tr = np.c_[occ_tr, np.bincount(lab_tr, minlength=NB)]
    p_raw = occ_raw.sum(1) / occ_raw.sum()
    p_tr = occ_tr.sum(1) / occ_tr.sum()
    share_raw, share_tr = p_raw[15], p_tr[15]
    return dict(theta=theta_deg, seed=seed, rotate=rotate,
                rule_match=round(match_hits / match_tot, 4),
                raw_leader=int(p_raw.argmax()), raw_share15=round(float(share_raw), 3),
                tr_leader=int(p_tr.argmax()), tr_share15=round(float(share_tr), 3),
                p=p_tr.tolist()), (np.array(p_raw), np.array(p_tr))

def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())

res = []; dist_rows = []
# control arm first (rotate=False, theta recorded as 0)
ctrl = {}
for sd in (9201, 9202, 9203):
    rec, (p_raw, p_tr) = run(0, sd, rotate=False)
    ctrl[sd] = p_raw; res.append(rec)
base_tv = [round(tv(ctrl[a], ctrl[b]), 4) for a, b in
           [(9201, 9202), (9202, 9203), (9201, 9203)]]
print("baseline run-to-run TV (control seeds):", base_tv, flush=True)

for th in (15, 30, 45, 60, 90):
    for sd in (9201, 9202, 9203):
        rec, (p_raw, p_tr) = run(th, sd, rotate=True)
        d_raw_ctrl = tv(p_raw, ctrl[sd])   # raw away vs control (B2-style loss)
        d_tr_ctrl = tv(p_tr, ctrl[sd])     # transported vs control (form test)
        rec.update(tv_raw_vs_ctrl=round(d_raw_ctrl, 4),
                   tv_transport_vs_ctrl=round(d_tr_ctrl, 4),
                   baseline_tv_max=max(base_tv),
                   preserved=bool(d_tr_ctrl <= max(base_tv)))
        res.append(rec)
        dist_rows.append((th, sd, rec["raw_leader"], rec["tr_leader"],
                          rec["tr_share15"], d_raw_ctrl, d_tr_ctrl, rec["preserved"]))
        print(rec, flush=True)

json.dump(dict(baseline_tv=base_tv, rows=res), open(
    os.path.join(ROOT, "tegmark_bridge", "outputs", "b3_transport.json"), "w"), indent=1)
print("\n=== transport summary (theta, seed, raw_leader, transported_leader, "
      "tr_share15, TV_raw, TV_transport, preserved) ===")
for r in dist_rows:
    print(*r)
print("WROTE b3_transport.json")
