"""
Pipeline step: run latent coordinator analysis on scored windows.

Usage::

    python3 -m src.pipelines.run_latent_analysis \\
        scored_features.csv --output latent_trajectory.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.latent.coordinator import run_coordinator
from src.latent.mapping import dataframe_to_drives


def run_latent_analysis(
    scored_df: pd.DataFrame,
    n_basins: int = 5,
    n_dims: int = 4,
    random_seed: int | None = 42,
) -> pd.DataFrame:
    """
    Map scored feature windows into latent drives and run the coordinator.

    Parameters
    ----------
    scored_df:
        Feature DataFrame with an ``anomaly_score`` column.
    n_basins:
        Number of latent basin attractors.
    n_dims:
        Coordinator dimensionality.
    random_seed:
        Seed for reproducibility.

    Returns
    -------
    DataFrame with the latent trajectory (one row per window).
    """
    drive_sequence = dataframe_to_drives(scored_df, dims=n_dims)
    return run_coordinator(
        drive_sequence,
        n_basins=n_basins,
        n_dims=n_dims,
        random_seed=random_seed,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m src.pipelines.run_latent_analysis",
        description="Run latent coordinator analysis on scored windows.",
    )
    p.add_argument("scored_features", help="Path to the scored feature CSV")
    p.add_argument("--output", default=None, help="Path to save latent trajectory CSV")
    p.add_argument("--n-basins", type=int, default=5, dest="n_basins")
    p.add_argument("--n-dims", type=int, default=4, dest="n_dims")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    path = Path(args.scored_features)
    if not path.exists():
        sys.exit(f"Scored features file not found: {path}")

    df = pd.read_csv(path)
    latent_df = run_latent_analysis(df, n_basins=args.n_basins, n_dims=args.n_dims, random_seed=args.seed)

    output_path = Path(args.output) if args.output else path.with_suffix(".latent.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latent_df.to_csv(output_path, index=False)
    print(f"Latent trajectory saved to {output_path}  ({len(latent_df)} steps)")


if __name__ == "__main__":
    main()
