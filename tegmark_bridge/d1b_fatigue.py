import sys, os, numpy as np, torch
ROOT = r"C:\Users\akbon\OneDrive\Documents\GitHub\FourD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "vfd_bridge"))
import tegmark_bridge.d1_autopsy as d1

def fatigue_profile(P, manifold='s3'):
    torch.manual_seed(4000); np.random.seed(4000)
    if P is None:
        eng = d1.make_engine('cuda:0', 128, 1000, None, manifold, topology='cyclic')
    else:
        eng = d1.make_engine('cuda:0', 128, 1000, P, manifold)
    for _ in range(1000):
        eng.step()
    F = eng.fatigue.mean(0).cpu().numpy()
    Pn = eng.prefs.cpu().numpy()
    mism = []
    for i in range(len(Pn)):
        j = int(np.argmin(Pn @ (-Pn[i])))
        if Pn[i] @ Pn[j] < -0.999:
            mism.append(abs(F[i] - F[j]))
    mm = float(np.mean(mism)) if mism else float('nan')
    return F.max(), F.std(), mm

_, m6 = d1.six00_cell()
V = m6['vertices']
rng = np.random.RandomState(5)
base = rng.randn(12, 4); base /= np.linalg.norm(base, axis=1, keepdims=True)
anti12 = np.vstack([base, -base])
s8 = V[rng.choice(len(V), 8, replace=False)]
for name, P in [('600cell_full', V), ('antipodal12', anti12),
                ('600cell_s8', s8), ('cyclic8', None)]:
    mx, sd, mm = fatigue_profile(P)
    print(f"{name:<15s} maxFatigue={mx:6.2f}  spread(std)={sd:6.3f}  mean|F(v)-F(-v)|={mm:.4f}")
