"""
End-to-end QRNG analysis pipeline.

Reads one or more bitstream text files from a directory and runs the full
pipeline for each:

1. Extract features  (windows → feature CSV)
2. Score windows     (feature CSV → scored CSV with anomaly scores)
3. Latent analysis   (scored CSV → latent trajectory CSV)
4. Write summary JSON

Usage::

    python3 -m src.pipelines.run_qrng_pipeline data/bitstreams \\
        --output outputs/qrng_pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.pipelines.extract_features import extract_features_from_stream, load_bitstream
from src.pipelines.run_latent_analysis import run_latent_analysis
from src.pipelines.score_windows import score_feature_dataframe


def run_pipeline_for_stream(
    bitstream_path: Path,
    output_dir: Path,
    window_size: int = 4096,
    overlap: float = 0.5,
    n_basins: int = 5,
    n_dims: int = 4,
    random_seed: int | None = 42,
) -> Dict[str, Any]:
    """
    Run the full pipeline for a single bitstream file.

    Parameters
    ----------
    bitstream_path:
        Path to the bitstream text file.
    output_dir:
        Directory to write outputs.  Will be created if it does not exist.
    window_size:
        Sliding-window size in bits.
    overlap:
        Window overlap fraction.
    n_basins:
        Number of latent basin attractors.
    n_dims:
        Coordinator dimensionality.
    random_seed:
        Seed for reproducibility.

    Returns
    -------
    Summary dict with key statistics.
    """
    stream_id = bitstream_path.stem
    stream_out = output_dir / stream_id
    stream_out.mkdir(parents=True, exist_ok=True)

    # Step 1: load bitstream
    bits = load_bitstream(bitstream_path)
    print(f"  [{stream_id}] Loaded {len(bits):,} bits")

    # Step 2: extract features
    features_df = extract_features_from_stream(
        bits, stream_id=stream_id, window_size=window_size, overlap=overlap
    )
    features_path = stream_out / "window_features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"  [{stream_id}] Features: {len(features_df)} windows → {features_path}")

    # Step 3: score windows
    scored_df = score_feature_dataframe(features_df)
    # (we keep scored results merged into the features file for convenience)

    # Step 4: latent analysis
    latent_df = run_latent_analysis(
        scored_df, n_basins=n_basins, n_dims=n_dims, random_seed=random_seed
    )
    latent_path = stream_out / "latent_trajectory.csv"
    latent_df.to_csv(latent_path, index=False)
    print(f"  [{stream_id}] Latent trajectory: {len(latent_df)} steps → {latent_path}")

    # Step 5: summary
    summary: Dict[str, Any] = {
        "stream_id": stream_id,
        "n_bits": int(len(bits)),
        "n_windows": int(len(features_df)),
        "window_size": window_size,
        "overlap": overlap,
        "mean_anomaly_score": float(scored_df["anomaly_score"].mean()),
        "max_anomaly_score": float(scored_df["anomaly_score"].max()),
        "n_basin_switches": int(latent_df["chosen_basin"].diff().ne(0).sum()) if "chosen_basin" in latent_df.columns else 0,
        "outputs": {
            "window_features": str(features_path),
            "latent_trajectory": str(latent_path),
        },
    }
    summary_path = stream_out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [{stream_id}] Summary → {summary_path}")

    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m src.pipelines.run_qrng_pipeline",
        description="Run the full FourD QRNG analysis pipeline on a directory of bitstreams.",
    )
    p.add_argument("bitstreams_dir", help="Directory containing bitstream .txt files")
    p.add_argument("--output", default="outputs/qrng_pipeline", help="Output directory")
    p.add_argument("--window-size", type=int, default=4096, dest="window_size")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--n-basins", type=int, default=5, dest="n_basins")
    p.add_argument("--n-dims", type=int, default=4, dest="n_dims")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    bitstreams_dir = Path(args.bitstreams_dir)
    if not bitstreams_dir.is_dir():
        sys.exit(f"Not a directory: {bitstreams_dir}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    bitstream_files = sorted(bitstreams_dir.glob("*.txt"))
    if not bitstream_files:
        sys.exit(f"No .txt files found in {bitstreams_dir}")

    print(f"Found {len(bitstream_files)} bitstream(s) in {bitstreams_dir}")
    all_summaries = []
    for bf in bitstream_files:
        print(f"\nProcessing: {bf.name}")
        summary = run_pipeline_for_stream(
            bf,
            output_dir,
            window_size=args.window_size,
            overlap=args.overlap,
            n_basins=args.n_basins,
            n_dims=args.n_dims,
            random_seed=args.seed,
        )
        all_summaries.append(summary)

    summaries_path = output_dir / "summaries.json"
    with open(summaries_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll summaries → {summaries_path}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
