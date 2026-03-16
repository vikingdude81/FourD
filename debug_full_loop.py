import numpy as np
from sklearn.cluster import KMeans

CONFIG = {
    'manifold_dim': 4,
    'n_subsystems': 8,
    'n_macro': 120,
    'beta_macro': 8.0,
    'alpha_pull': 0.15,
}

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

# Generate macro centers (simulating the manifold's pre-computed geometry)
np.random.seed(42)
micro_pts = np.random.normal(size=(600, CONFIG['manifold_dim']))
km = KMeans(n_clusters=CONFIG['n_macro'], random_state=42, n_init=10)
km.fit(micro_pts)
macro_centers = km.cluster_centers_.copy()
macro_norms = np.linalg.norm(macro_centers, axis=1, keepdims=True) + 1e-8
macro_centers = macro_centers / macro_norms

# Initial state (uniform direction)
u_t = np.ones(CONFIG['manifold_dim']) / np.sqrt(CONFIG['manifold_dim'])

def compute_macro_field(u):
    """Compute weighted macro field."""
    # Normalize input first
    if isinstance(u, np.ndarray):
        norm_u = np.linalg.norm(u) + 1e-8
        u = u / norm_u
    
    similarities = np.dot(macro_centers, u)
    exp_sim = np.exp(CONFIG['beta_macro'] * similarities)
    weights = exp_sim / (np.sum(exp_sim) + 1e-8)
    field = np.dot(weights, macro_centers)
    norm_field = np.linalg.norm(field) + 1e-8
    return field / norm_field

# Run simulation loop with full dynamics
dominant_counts = {name: 0 for name in SUBSYSTEM_NAMES}

for step in range(50):
    # STAGE 1: Compute subsystem influences
    raw_influences = np.zeros(CONFIG['n_subsystems'])
    for i in range(CONFIG['n_subsystems']):
        pref_dir = get_preference_direction(i)
        dot_product = float(np.dot(u_t, pref_dir))
        raw_influences[i] = 0.5 * dot_product + 0.7
    
    # Apply competition and fatigue (simplified - no actual fatigue tracking for now)
    effective = np.maximum(raw_influences, 0.05)
    activities = effective / np.sum(effective)
    
    dominant_idx = np.argmax(activities)
    dominant_counts[SUBSYSTEM_NAMES[dominant_idx]] += 1
    
    # Perturb state
    perturbation = np.random.normal(0, 0.1, CONFIG['manifold_dim'])
    u_perturbed = u_t + 0.2 * perturbation
    u_perturbed = u_perturbed / np.linalg.norm(u_perturbed)
    
    # STAGE 2: Macro field pull
    macro_field = compute_macro_field(u_perturbed)
    pull_direction = (1 - CONFIG['alpha_pull']) * u_perturbed + CONFIG['alpha_pull'] * macro_field
    u_t = pull_direction / np.linalg.norm(pull_direction)
    
    print(f"Step {step}: dominant={SUBSYSTEM_NAMES[dominant_idx]}, u_t={[round(x,3) for x in u_t]}")

print(f"\nDominant counts: {dominant_counts}")