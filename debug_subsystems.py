import numpy as np

CONFIG = {'manifold_dim': 4}

print("Subsystems preference directions:")
for i in range(8):
    seed = 1000 + i * 7919
    np.random.seed(seed)
    pref = np.random.normal(size=CONFIG['manifold_dim'])
    norm = np.linalg.norm(pref)
    normalized = [round(x/norm, 3) for x in pref]
    print(f"  Subsystem {i}: {normalized}")

print("\nInitial u_t (all equal):", [1/np.sqrt(4)] * 4)

# Compute dot products with initial state
print("\nDot products with initial uniform direction:")
for i in range(8):
    seed = 1000 + i * 7919
    np.random.seed(seed)
    pref = np.random.normal(size=CONFIG['manifold_dim'])
    norm = np.linalg.norm(pref)
    normalized = pref / norm
    dot = sum([normalized[j] * (1/np.sqrt(4)) for j in range(4)])
    influence = 0.3 * dot + 0.7
    print(f"  Subsystem {i}: dot={dot:.4f}, influence={influence:.4f}")