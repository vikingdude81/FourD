# FourD

Higher-dimensional latent-state simulation and QRNG analysis pipeline.

## What this repo contains

This project now has two connected parts:

1. **`fourD_slice_sim.py`**  
   A 4D latent coordination / navigation simulator where a hidden 4D coordinator
   is projected into 2D action.

2. **`src/` pipeline**  
   A modular QRNG analysis stack that:
   - windows bitstreams,
   - extracts local features,
   - computes anomaly scores,
   - maps feature windows into latent subsystem drives,
   - reconstructs latent trajectories,
   - runs recurrence analysis,
   - and visualizes results in a dashboard.

## Structure

```text
src/
  features/
  anomaly/
  latent/
  recurrence/
  viz/
  pipelines/
```

## Install

```bash
pip install -r requirements.txt
```

## Run the simulator

```bash
python3 fourD_slice_sim.py
```

This will:
- run the 4D latent navigation simulation,
- export `simulation_log.csv`,
- run a lesion study,
- show plots for navigation, phase portrait, dominance, and lesion comparison.

## Run the QRNG pipeline

Place one or more bitstream text files in a directory. Each file should contain `0` and `1` characters.

Example:

```bash
python3 -m src.pipelines.run_qrng_pipeline data/bitstreams --output outputs/qrng_pipeline
```

Outputs per stream:
- `window_features.csv`
- `latent_trajectory.csv`
- `summary.json`

## Open the dashboard

```bash
python3 -m src.viz.dashboard \
  --features outputs/qrng_pipeline/<stream_id>/window_features.csv \
  --latent outputs/qrng_pipeline/<stream_id>/latent_trajectory.csv
```

## Current workflow

1. Window QRNG bitstreams
2. Extract local complexity / entropy / autocorrelation / change features
3. Build anomaly scores
4. Map feature rows into latent subsystem drives
5. Run latent coordinator / basin dynamics
6. Analyze recurrence of anomaly behavior
7. Inspect with dashboard

## Notes

- The simulator is a conceptual latent-state model, not evidence of exotic influence.
- Null-model comparison is still essential for serious analysis.
- The current subsystem mapping is hand-designed and should eventually be compared against alternative mappings.

## Next recommended expansions

- null / surrogate generators
- dashboard animation
- direct QRNG-to-simulator integration
- cross-stream motif clustering