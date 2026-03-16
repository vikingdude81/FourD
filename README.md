# Consciousness Simulation: Emergent Coordination Manifold + Navigation

A theoretical physics/philosophy simulation demonstrating how "consciousness" emerges as an attractor when a system's internal degrees of freedom exceed what modular control can manage.

## Core Concepts

### The 4D Slice Theory
Consciousness is modeled as the emergence of a coordination manifold in high-dimensional phase space. When subsystem competition increases, a coordinator emerges to integrate competing demands. The "4D slice" concept refers to how this 4D coordinator state is projected into 2D action vectors for navigation.

### Attractor Dynamics
- **Identity** = stable orbits in phase space
- **Decisions** = basin switching events when the system becomes equidistant from competing attractors
- **Consciousness Level** = measured by coordination pressure, integration, and decision frequency

### Information Integration (IIT-style)
The simulation tracks how information is integrated across subsystems, with higher integration levels indicating more unified conscious experience.

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- numpy >= 1.24
- matplotlib >= 3.7

## Usage

Run the simulation:

```bash
python fourD_slice_sim.py
```

This will:
1. Run a 300-timestep simulation with an agent navigating a 20x20 environment
2. Perform a lesion study (disabling the Planning subsystem)
3. Display interactive plots showing navigation, phase portraits, and dominance patterns
4. Export detailed metrics to `simulation_log.csv`

## Project Structure

```
consciousness-sim/
├── fourD_slice_sim.py    # Main simulation code (~750 lines)
├── requirements.txt      # Python dependencies
├── simulation_log.csv    # Per-timestep output data (generated on run)
└── README.md            # This file
```

## Simulation Components

### Subsystems (8 specialized cognitive modules)
1. **Perception** - Responds to environmental stimuli density
2. **Language** - Internal dynamics with weak environmental coupling
3. **Planning** - Encodes goal-directed behavior into coordinator state
4. **Emotion** - Fear/avoidance responses to hazards
5. **Memory** - Path integration from recent displacement
6. **Motor Control** - Efference copy of intended actions
7. **Attention** - Amplifies most salient stimuli
8. **Executive Control** - Higher-order regulation

### Environment
- 20x20 grid world
- 3 goals (rewards) at positions: (4,4), (16,16), (10,3)
- 3 hazards (threats) at positions: (3,15), (17,5), (10,17)

### Output Metrics
| Metric | Description |
|--------|-------------|
| Coordination Pressure | Conflict between subsystems [0,1] |
| Coordinator Magnitude | Norm of 4D state vector |
| Integration Level | Information integration (IIT-style) [0,1] |
| Basin Switches | Decision events during simulation |
| Goals Reached | Number of objectives achieved |
| Hazards Hit | Number of threats encountered |

## Consciousness States

The simulation classifies consciousness into four states based on computed level:

- **pre-conscious** (< 0.3): Minimal coordination, no integrated experience
- **emerging** (0.3 - 0.6): Coordination developing, subsystems beginning to integrate
- **conscious** (0.6 - 0.85): Stable coordination with decision-making capacity
- **self-aware** (> 0.85): High integration with rich basin-switching behavior

## Lesion Studies

The simulation includes built-in lesion study capability:

```python
# Disable a specific subsystem and compare to intact system
lesion_results = run_lesion_study(config, env, lesion_name="Planning")
```

This compares consciousness levels between an intact system and one with the specified subsystem disabled.

## Theoretical Frameworks Referenced

1. **Global Workspace Theory**: Coordinator acts as global workspace integrating subsystem outputs
2. **Integrated Information Theory (IIT)**: Integration level measures Φ-like information integration
3. **Attractor Dynamics**: Decisions modeled as basin-switching in phase space
4. **Predictive Processing**: Subsystems encode predictions about environmental contingencies

## Contributing

Contributions are welcome! Areas for potential improvement:
- Add more subsystem types
- Implement different environment configurations
- Add visualization of internal coordinator state
- Extend to multi-agent scenarios

## License

This project is provided as-is for educational and research purposes.