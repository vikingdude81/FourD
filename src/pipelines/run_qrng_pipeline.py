"""
QRNG Analysis Pipeline.

End-to-end pipeline that:
1. Loads QRNG bitstream data
2. Slides windows across the stream
3. Extracts features from each window
4. Maps features to subsystem drives
5. Runs the latent coordinator simulation
6. Outputs results and visualizations

Usage:
    python -m src.pipelines.run_qrng_pipeline --input data/qrng_bits.npy --output outputs/qrng_analysis/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import feature extraction modules
from src.features.windows import sliding_windows, window_metadata
from src.features import extract_all_features
from src.latent.mapping import feature_row_to_drives, DEFAULT_SUBSYSTEM_WEIGHTS
from src.latent.coordinator import run_coordinator


def load_bitstream(
    input_path: str,
    format: str = "npy"
) -> np.ndarray:
    """
    Load a bitstream from file.
    
    Parameters
    ----------
    input_path : str
        Path to the input file.
    format : str
        File format ('npy', 'csv', 'txt').
        
    Returns
    -------
    np.ndarray
        Array of binary values (0 or 1).
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if format == "npy":
        bits = np.load(input_path)
    elif format == "csv":
        bits = np.loadtxt(input_path, delimiter=",")
    elif format == "txt":
        with open(input_path, 'r') as f:
            content = f.read().strip()
            # Handle different formats
            if all(c in '01\n\r' for c in content):
                bits = np.array([int(c) for c in content if c in '01'])
            else:
                bits = np.loadtxt(input_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Ensure binary values
    bits = (bits >= 0.5).astype(np.int64)
    
    return bits


def extract_features_from_stream(
    bits: np.ndarray,
    window_size: int = 4096,
    overlap: float = 0.5,
    block_size: int = 8,
    m: int = 2,
    max_lag: int = 20,
    window_size_cp: int = 64,
) -> Tuple[List[dict], List[dict]]:
    """
    Extract features from a bitstream using sliding windows.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    window_size : int
        Number of bits per analysis window.
    overlap : float
        Fraction of window that overlaps with previous window.
    block_size : int
        Block size for entropy/min-entropy computation.
    m : int
        Template length for sample/approximate entropy.
    max_lag : int
        Maximum lag for autocorrelation tests.
    window_size_cp : int
        Window size for change-point detection.
        
    Returns
    -------
    Tuple of (feature_list, metadata_list)
    """
    features_list = []
    metadata_list = []
    
    # Get sliding windows with indices
    windows = list(sliding_windows(bits, window_size=window_size, overlap=overlap))
    
    for idx, (start_idx, window_bits) in enumerate(windows):
        # Extract all features from this window
        features = extract_all_features(
            window_bits,
            block_size=block_size,
            m=m,
            max_lag=max_lag,
            window_size=window_size_cp,
        )
        
        # Add metadata
        meta = {
            "window_index": idx,
            "start_bit": int(start_idx),
            "end_bit": int(start_idx + len(window_bits)),
            "window_size": len(window_bits),
        }
        
        features_list.append(features)
        metadata_list.append(meta)
    
    return features_list, metadata_list


def run_qrng_analysis(
    bits: np.ndarray,
    window_size: int = 4096,
    overlap: float = 0.5,
    n_basins: int = 6,
    n_dims: int = 4,
    learning_rate: float = 0.05,
    noise_level: float = 0.02,
    basin_pull_strength: float = 0.02,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Run the full QRNG analysis pipeline.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    window_size : int
        Number of bits per analysis window.
    overlap : float
        Fraction of window that overlaps with previous window.
    n_basins : int
        Number of latent basin attractors.
    n_dims : int
        Dimensionality of the coordinator state.
    learning_rate : float
        Step size toward weighted drive input.
    noise_level : float
        Standard deviation of additive noise.
    basin_pull_strength : float
        Step size toward basin attractor.
    random_seed : Optional[int]
        Random seed for reproducibility.
        
    Returns
    -------
    dict
        Pipeline results including features, drives, and coordinator trajectory.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    print(f"  Loading {len(bits)} bits...")
    print(f"  Window size: {window_size}, Overlap: {overlap}")
    
    # Step 1: Extract features from bitstream
    print("  Extracting features...")
    features_list, metadata_list = extract_features_from_stream(
        bits,
        window_size=window_size,
        overlap=overlap,
    )
    print(f"    Extracted {len(features_list)} windows")
    
    # Step 2: Convert features to subsystem drives
    print("  Converting features to subsystem drives...")
    drive_sequence = []
    for features in features_list:
        drives = feature_row_to_drives(features)
        drive_sequence.append(drives)
    
    # Step 3: Run coordinator simulation
    print("  Running latent coordinator simulation...")
    trajectory_df = run_coordinator(
        drive_sequence=drive_sequence,
        n_basins=n_basins,
        n_dims=n_dims,
        learning_rate=learning_rate,
        noise_level=noise_level,
        basin_pull_strength=basin_pull_strength,
        random_seed=random_seed,
    )
    
    # Compile results
    results = {
        "metadata": metadata_list,
        "features": features_list,
        "drive_sequence": drive_sequence,
        "trajectory": trajectory_df,
        "config": {
            "window_size": window_size,
            "overlap": overlap,
            "n_basins": n_basins,
            "n_dims": n_dims,
            "learning_rate": learning_rate,
            "noise_level": noise_level,
            "basin_pull_strength": basin_pull_strength,
            "random_seed": random_seed,
        },
    }
    
    return results


def export_results(
    results: Dict,
    output_dir: str,
) -> None:
    """
    Export pipeline results to files.
    
    Parameters
    ----------
    results : dict
        Pipeline results dictionary.
    output_dir : str
        Output directory path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export trajectory as CSV
    traj_df = results["trajectory"]
    traj_csv = output_path / "coordinator_trajectory.csv"
    traj_df.to_csv(traj_csv, index=False)
    print(f"  Saved: {traj_csv}")
    
    # Export features as CSV (one row per window)
    features_df = pd.DataFrame(results["features"])
    features_csv = output_path / "extracted_features.csv"
    features_df.to_csv(features_csv, index=False)
    print(f"  Saved: {features_csv}")
    
    # Export metadata
    meta_json = output_path / "window_metadata.json"
    with open(meta_json, 'w') as f:
        json.dump(results["metadata"], f, indent=2)
    print(f"  Saved: {meta_json}")
    
    # Export configuration
    config_json = output_path / "pipeline_config.json"
    with open(config_json, 'w') as f:
        json.dump(results["config"], f, indent=2)
    print(f"  Saved: {config_json}")
    
    # Compute and export summary statistics
    summary = compute_summary_statistics(results)
    summary_json = output_path / "summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_json}")


def compute_summary_statistics(results: Dict) -> Dict:
    """
    Compute summary statistics from pipeline results.
    
    Parameters
    ----------
    results : dict
        Pipeline results dictionary.
        
    Returns
    -------
    dict
        Summary statistics.
    """
    trajectory = results["trajectory"]
    features_list = results["features"]
    
    # Basin analysis
    basin_counts = trajectory["chosen_basin"].value_counts().to_dict()
    n_switches = (trajectory["chosen_basin"].diff().fillna(0) != 0).sum()
    
    # Feature statistics
    feature_stats = {}
    for col in features_df.columns:
        if col in features_df.select_dtypes(include=[np.number]).columns:
            feature_stats[col] = {
                "mean": float(features_df[col].mean()),
                "std": float(features_df[col].std()),
                "min": float(features_df[col].min()),
                "max": float(features_df[col].max()),
            }
    
    # Coordinator statistics
    coord_stats = {}
    for col in trajectory.columns:
        if col.startswith("coord_"):
            coord_stats[col] = {
                "mean": float(trajectory[col].mean()),
                "std": float(trajectory[col].std()),
                "min": float(trajectory[col].min()),
                "max": float(trajectory[col].max()),
            }
    
    return {
        "n_windows": len(features_list),
        "n_basin_switches": int(n_switches),
        "basin_distribution": basin_counts,
        "feature_statistics": feature_stats,
        "coordinator_statistics": coord_stats,
    }


def main(argv: Optional[List[str]] = None):
    """Main entry point for the QRNG analysis pipeline."""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipelines.run_qrng_pipeline",
        description="QRNG Analysis Pipeline"
    )
    
    # Input/Output
    parser.add_argument("--input", "-i", required=True, help="Input bitstream file (.npy, .csv, or .txt)")
    parser.add_argument("--output", "-o", default="outputs/qrng_analysis/", help="Output directory")
    parser.add_argument("--format", choices=["npy", "csv", "txt"], default="npy", help="Input file format")
    
    # Window parameters
    parser.add_argument("--window-size", type=int, default=4096, help="Window size in bits")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction (0-1)")
    
    # Coordinator parameters
    parser.add_argument("--n-basins", type=int, default=6, help="Number of basin attractors")
    parser.add_argument("--n-dims", type=int, default=4, help="Coordinator dimensionality")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--noise-level", type=float, default=0.02, help="Noise level")
    parser.add_argument("--basin-pull-strength", type=float, default=0.02, help="Basin pull strength")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args(argv)
    
    print(f"\n{'=' * 60}")
    print("  QRNG ANALYSIS PIPELINE")
    print(f"{'=' * 60}\n")
    
    # Load bitstream
    bits = load_bitstream(args.input, format=args.format)
    
    # Run analysis
    results = run_qrng_analysis(
        bits=bits,
        window_size=args.window_size,
        overlap=args.overlap,
        n_basins=args.n_basins,
        n_dims=args.n_dims,
        learning_rate=args.learning_rate,
        noise_level=args.noise_level,
        basin_pull_strength=args.basin_pull_strength,
        random_seed=args.seed,
    )
    
    # Export results
    export_results(results, args.output)
    
    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()