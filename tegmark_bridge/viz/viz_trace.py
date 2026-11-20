"""V1 — trace visualizer: record agent navigation for four key scenes and
emit a self-contained HTML scrubber (S^3 projected to 3D: preferences live
in the z=0 hyperplane, so agent depth above/below it is visible).
Scenes: A dodeca19 crystallization, B gap w=0.70 diluted pinned peak,
C near-vertex displaced 45deg moving peak (rival form), D dodeca20 boiling.
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
dodeca = np.c_[dodeca, np.zeros(20)]
dodeca /= np.linalg.norm(dodeca, axis=1)[:, None]
P19, V20 = dodeca[:-1], dodeca[-1]
STEPS = 4000; STRIDE = 12

def scene(tag, P_raw, seed=9201):
    torch.manual_seed(seed); np.random.seed(seed)
    eng = make_engine("cuda:0", 128, STEPS, P19, "s3")
    Pt = torch.tensor(np.asarray(P_raw), dtype=torch.float32, device=eng.device)
    eng.prefs = Pt; eng.n_sub = Pt.shape[0]
    eng.fatigue = torch.zeros(eng.N, eng.n_sub, device=eng.device)
    C = eng.macro_centers.cpu().numpy()
    frames = []; shares = []; leaders = []
    for t in range(STEPS):
        eng.step()
        if t % STRIDE == 0:
            u = eng.u_t.cpu().numpy()
            lab = (u @ C.T).argmax(1)
            cnt = np.bincount(lab, minlength=24)
            frames.append((np.round(u, 3).ravel().tolist(), lab.tolist()))
            shares.append(round(cnt[15] / 128, 3))
            leaders.append(int(cnt.argmax()))
    meta = dict(
        tag=tag, nframes=len(frames), stride=STRIDE,
        share15=shares, leaders=leaders,
        prefs=np.round(P_raw, 3).tolist(),
        centers=np.round(C, 3).tolist())
    print(tag, "frames", len(frames), "share15 first/mid/last:",
          shares[0], shares[len(shares)//2], shares[-1], flush=True)
    return meta, frames

scenes = []
S = []
m, f = scene("A_dodeca19_crystallize", P19); S.append((m, f))
m, f = scene("B_gap_w0.70_diluted", np.r_[P19, [[0.7 * x for x in V20]]]); S.append((m, f))
# near vertex: max cos to c15 under native grid
torch.manual_seed(9201); np.random.seed(9201)
_e = make_engine("cuda:0", 128, 10, P19, "s3")
c15 = _e.macro_centers.cpu().numpy()[15]
vi = int((P19 @ c15).argmax())
v = P19[vi]; e2 = c15 - (c15 @ v) * v; e2 /= np.linalg.norm(e2)
a = np.radians(45); Pd = P19.copy(); Pd[vi] = np.cos(a) * v + np.sin(a) * e2
m, f = scene("C_near45deg_moving_peak", Pd); S.append((m, f))
m, f = scene("D_dodeca20_boiling", dodeca); S.append((m, f))

# center coords (4D) for display
C4 = np.round(_e.macro_centers.cpu().numpy(), 3).tolist()
payload = {"scenes": [{"meta": m, "frames": [
    {"u": fr[0], "lab": fr[1]} for fr in f]} for m, f in S], "C4": C4}
json.dump(payload, open(os.path.join(ROOT, "tegmark_bridge", "viz",
                                    "traces.json"), "w"))
print("WROTE traces.json")
