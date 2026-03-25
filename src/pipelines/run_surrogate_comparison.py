"""
Surrogate Data Comparison Pipeline.

Compares real QRNG bitstreams against:
1. PRNG (Pseudo-Random Number Generator) controls
2. Shuffled surrogates (preserving statistics but destroying structure)
3. Phase-randomized surrogates (preserving power spectrum but randomizing phases)

This allows detection of anomalies and structural differences between real
and synthetic data sources.

Usage:
    python -m src.pipelines.run_surrogate_comparison --input data/qrng_bits.npy --n-surrogates 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import feature extraction modules
from src.features.windows import sliding_windows
from src.features import extract_all_features
from src.anomaly.scoring import AnomalyScorer, NullDistribution


def generate_prng_bits(
    n_bits: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate pseudo-random bits using numpy's PRNG.
    
    Parameters
    ----------
    n_bits : int
        Number of bits to generate.
    seed : Optional[int]
        Random seed for reproducibility.
        
    Returns
    -------
    np.ndarray
        Array of binary values (0 or 1).
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.randint(0, 2, size=n_bits)


def create_shuffled_surrogate(bits: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Create a shuffled surrogate that preserves bit frequencies but destroys structure.
    
    Parameters
    ----------
    bits : np.ndarray
        Original bitstream.
    seed : Optional[int]
        Random seed for reproducibility.
        
    Returns
    -------
    np.ndarray
        Shuffled bitstream.
    """
    if seed is not None:
        np.random.seed(seed)
    
    shuffled = bits.copy()
    np.random.shuffle(shuffled)
    
    return shuffled


def create_phase_randomized_surrogate(
    bits: np.ndarray,
    n_surrogates: int = 1,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Create phase-randomized surrogates preserving power spectrum.
    
    This method preserves the amplitude spectrum but randomizes phases,
    which maintains linear correlations while destroying nonlinear structure.
    
    Parameters
    ----------
    bits : np.ndarray
        Original bitstream (converted to bipolar).
    n_surrogates : int
        Number of surrogates to generate.
    seed : Optional[int]
        Base random seed.
        
    Returns
    -------
    List[np.ndarray]
        List of phase-randomized surrogate bitstreams.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to bipolar (-1, 1) for FFT analysis
    signal = 2 * bits.astype(float) - 1
    
    n = len(signal)
    surrogates = []
    
    for i in range(n_surrogates):
        if seed is not None:
            np.random.seed(seed + i)
        
        # Compute FFT
        fft_result = np.fft.rfft(signal)
        
        # Get amplitude and phase
        amplitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        
        # Randomize phases (except DC and Nyquist components)
        random_phase = np.random.uniform(0, 2 * np.pi, len(phase))
        
        # Preserve symmetry for real signal reconstruction
        if n % 2 == 0:
            random_phase[1:-1] = -random_phase[1:-1]
        else:
            random_phase[1:] = -random_phase[1:]
        
        # Reconstruct with randomized phases
        fft_surrogate = amplitude * np.exp(1j * random_phase)
        
        # Inverse FFT and threshold to binary
        surrogate_signal = np.fft.irfft(fft_surrogate, n=n)
        surrogate_bits = (surrogate_signal > 0).astype(np.int64)
        
        surrogates.append(surrogate_bits)
    
    return surrogates


def run_surrogate_comparison(
    real_bits: np.ndarray,
    n_prng: int = 100,
    n_shuffled: int = 50,
    n_phase_randomized: int = 20,
    window_size: int = 4096,
    overlap: float = 0.5,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Run comprehensive surrogate comparison analysis.
    
    Parameters
    ----------
    real_bits : np.ndarray
        Real QRNG bitstream to analyze.
    n_prng : int
        Number of PRNG samples for null distribution.
    n_shuffled : int
        Number of shuffled surrogates.
    n_phase_randomized : int
        Number of phase-randomized surrogates.
    window_size : int
        Window size for feature extraction.
    overlap : float
        Overlap fraction between windows.
    random_seed : Optional[int]
        Random seed for reproducibility.
        
    Returns
    -------
    dict
        Comparison results including statistics and anomaly scores.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    print(f"  Real data: {len(real_bits)} bits")
    print(f"  PRNG samples: {n_prng}")
    print(f"  Shuffled surrogates: {n_shuffled}")
    print(f"  Phase-randomized surrogates: {n_phase_randomized}")
    
    # Step 1: Generate null distributions from PRNG
    print("\n  Generating PRNG null distribution...")
    prng_scorer = AnomalyScorer()
    prng_null = prng_scorer.add_null_distribution("prng")
    
    for i in range(n_prng):
        prng_bits = generate_prng_bits(len(real_bits), seed=random_seed + i if random_seed else None)
        
        # Extract features from a few windows to build null distribution
        windows = list(sliding_windows(prng_bits, window_size=window_size, overlap=overlap))[:10]
        for _, window in windows:
            features = extract_all_features(window)
            prng_null.update(features)
    
    prng_null.finalize()
    print(f"    Built null distribution from {n_prng * 10} PRNG windows")
    
    # Step 2: Analyze shuffled surrogates
    print("\n  Analyzing shuffled surrogates...")
    shuffled_features = []
    
    for i in range(n_shuffled):
        if random_seed is not None:
            np.random.seed(random_seed + 1000 + i)
        
        shuffled = create_shuffled_surrogate(real_bits, seed=random_seed + 2000 + i if random_seed else None)
        
        windows = list(sliding_windows(shuffled, window_size=window_size, overlap=overlap))[:5]
        for _, window in windows:
            features = extract_all_features(window)
            shuffled_features.append(features)
    
    # Step 3: Analyze phase-randomized surrogates
    print("  Analyzing phase-randomized surrogates...")
    phase_rand_surrogates = create_phase_randomized_surrogate(
        real_bits, n_surrogates=n_phase_randomized, seed=random_seed + 5000 if random_seed else None
    )
    
    phase_rand_features = []
    for i, surrogate in enumerate(phase_rand_surrogates):
        windows = list(sliding_windows(surrogate, window_size=window_size, overlap=overlap))[:5]
        for _, window in windows:
            features = extract_all_features(window)
            phase_rand_features.append(features)
    
    # Step 4: Analyze real data
    print("  Analyzing real data...")
    real_features = []
    
    windows = list(sliding_windows(real_bits, window_size=window_size, overlap=overlap))
    for _, window in windows:
        features = extract_all_features(window)
        real_features.append(features)
    
    # Step 5: Compute anomaly scores
    print("\n  Computing anomaly scores...")
    
    def compute_mean_anomaly_score(features_list: List[dict], scorer: AnomalyScorer) -> float:
        """Compute mean anomaly score for a list of feature windows."""
        scores = []
        for features in features_list[:10]:  # Use first 10 windows for efficiency
            try:
                score = scorer.compute_composite_anomaly_score(features, control_type="prng")
                scores.append(score)
            except Exception as e:
                pass
        return np.mean(scores) if scores else 0.0
    
    real_mean_score = compute_mean_anomaly_score(real_features, prng_scorer)
    shuffled_mean_score = compute_mean_anomaly_score(shuffled_features, prng_scorer)
    phase_rand_mean_score = compute_mean_anomaly_score(phase_rand_features, prng_scorer)
    
    # Step 6: Compute feature-level statistics
    print("  Computing feature-level statistics...")
    
    def compute_feature_stats(features_list: List[dict]) -> Dict[str, dict]:
        """Compute mean and std for each feature across all windows."""
        if not features_list:
            return {}
        
        # Get all feature names
        feature_names = list(features_list[0].keys())
        
        stats = {}
        for feat_name in feature_names:
            values = [f.get(feat_name, 0) for f in features_list]
            
            # Filter finite values
            finite_values = [v for v in values if np.isfinite(v)]
            
            if finite_values:
                stats[feat_name] = {
                    "mean": float(np.mean(finite_values)),
                    "std": float(np.std(finite_values)),
                    "min": float(np.min(finite_values)),
                    "max": float(np.max(finite_values)),
                }
        
        return stats
    
    real_feature_stats = compute_feature_stats(real_features)
    shuffled_feature_stats = compute_feature_stats(shuffled_features)
    phase_rand_feature_stats = compute_feature_stats(phase_rand_features)
    
    # Compile results
    results = {
        "config": {
            "n_prng": n_prng,
            "n_shuffled": n_shuffled,
            "n_phase_randomized": n_phase_randomized,
            "window_size": window_size,
            "overlap": overlap,
            "random_seed": random_seed,
        },
        "anomaly_scores": {
            "real_data_mean_score": float(real_mean_score),
            "shuffled_surrogates_mean_score": float(shuffled_mean_score),
            "phase_randomized_surrogates_mean_score": float(phase_rand_mean_score),
        },
        "feature_statistics": {
            "real": real_feature_stats,
            "shuffled": shuffled_feature_stats,
            "phase_randomized": phase_rand_feature_stats,
        },
    }
    
    return results


def export_comparison_results(
    results: Dict,
    output_dir: str,
) -> None:
    """Export surrogate comparison results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export configuration
    config_json = output_path / "comparison_config.json"
    with open(config_json, 'w') as f:
        json.dump(results["config"], f, indent=2)
    print(f"  Saved: {config_json}")
    
    # Export anomaly scores
    scores_json = output_path / "anomaly_scores.json"
    with open(scores_json, 'w') as f:
        json.dump(results["anomaly_scores"], f, indent=2)
    print(f"  Saved: {scores_json}")
    
    # Export feature statistics
    stats_json = output_path / "feature_statistics.json"
    with open(stats_json, 'w') as f:
        json.dump(results["feature_statistics"], f, indent=2)
    print(f"  Saved: {stats_json}")


def main(argv: Optional[List[str]] = None):
    """Main entry point for surrogate comparison pipeline."""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipelines.run_surrogate_comparison",
        description="Surrogate Data Comparison Pipeline"
    )
    
    # Input
    parser.add_argument("--input", "-i", required=True, help="Input bitstream file (.npy)")
    parser.add_argument("--format", choices=["npy", "csv"], default="npy", help="Input file format")
    
    # Surrogate counts
    parser.add_argument("--n-prng", type=int, default=100, help="Number of PRNG samples")
    parser.add_argument("--n-shuffled", type=int, default=50, help="Number of shuffled surrogates")
    parser.add_argument("--n-phase-randomized", type=int, default=20, help="Number of phase-randomized surrogates")
    
    # Window parameters
    parser.add_argument("--window-size", type=int, default=4096, help="Window size in bits")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction (0-1)")
    
    # Output
    parser.add_argument("--output", "-o", default="outputs/surrogate_comparison/", help="Output directory")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args(argv)
    
    print(f"\n{'=' * 60}")
    print("  SURROGATE COMPARISON PIPELINE")
    print(f"{'=' * 60}\n")
    
    # Load real data
    if args.format == "npy":
        real_bits = np.load(args.input)
    else:
        real_bits = np.loadtxt(args.input, delimiter=",")
    
    real_bits = (real_bits >= 0.5).astype(np.int64)
    
    # Run comparison
    results = run_surrogate_comparison(
        real_bits=real_bits,
        n_prng=args.n_prng,
        n_shuffled=args.n_shuffled,
        n_phase_randomized=args.n_phase_randomized,
        window_size=args.window_size,
        overlap=args.overlap,
        random_seed=args.seed,
    )
    
    # Export results
    export_comparison_results(results, args.output)
    
    print(f"\n{'=' * 60}")
    print("  COMPARISON COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()