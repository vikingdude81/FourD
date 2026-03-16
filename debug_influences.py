import numpy as np

CONFIG = {
    'manifold_dim': 4,
    'n_subsystems': 8,
}

# Simulate the subsystem behavior
np.random.seed(42)

SUBSYSTEM_NAMES = [
    'Motor Control', 'Planning', 'Attention', 'Memory',
    'Emotion', 'Social', 'Intuition', 'Aesthetic'
]

def get_preference_direction(subsystem_idx):
    """Get preferred direction for a subsystem."""
    seed = 1000 + subsystem_idx * 7919
    np.random.seed(seed)
    pref = np.random.normal(size=CONFIG['manifold_dim'])
    norm = np.linalg.norm(pref)
    return pref / norm

# Initial state (uniform direction)
u_t = np.ones(CONFIG['manifold_dim']) / np.sqrt(CONFIG['manifold_dim'])
print(f"Initial u_t: {[round(x,4) for x in u_t]}")

# Compute influences for each timestep and track dominant subsystem
dominant_counts = {name: 0 for name in SUBSYSTEM_NAMES}

for step in range(10):
    # Compute raw influences
    influences = np.zeros(CONFIG['n_subsystems'])
    for i in range(CONFIG['n_subsystems']):
        pref_dir = get_preference_direction(i)
        dot_product = float(np.dot(u_t, pref_dir))
        influences[i] = 0.5 * dot_product + 0.7
    
    # Apply competition (softmax-like normalization)
    effective = np.maximum(influences, 0.05)  # Floor value
    activities = effective / np.sum(effective)
    
    dominant_idx = np.argmax(activities)
    dominant_counts[SUBSYSTEM_NAMES[dominant_idx]] += 1
    
    print(f"Step {step}:")
    print(f"  Influences: {[round(x,3) for x in influences]}")
    print(f"  Activities: {[round(x,3) for x in activities]}")
    print(f"  Dominant: {SUBSYSTEM_NAMES[dominant_idx]} (idx={dominant_idx})")
    
    # Perturb u_t randomly and re-normalize
    perturbation = np.random.normal(0, 0.1, CONFIG['manifold_dim'])
    u_new = u_t + 0.2 * perturbation
    u_t = u_new / np.linalg.norm(u_new)

print(f"\nDominant counts: {dominant_counts}")