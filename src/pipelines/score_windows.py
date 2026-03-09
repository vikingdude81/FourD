"""
Pipeline step: compute anomaly scores for feature windows.

Usage::

    python3 -m src.pipelines.score_windows \\
        features.csv --output scored.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.anomaly.scoring import score_windows
from src.anomaly.standardize import zscore_dataframe

# Feature columns used for anomaly scoring
FEATURE_COLUMNS = [
    "bias",
    "runs_count",
    "longest_run",
    "autocorr_lag1",
    "spectral_entropy",
    "permutation_entropy",
    "lz_complexity",
    "sample_entropy",
]


def score_feature_dataframe(
    df: pd.DataFrame,
    method: str = "zscore",
) -> pd.DataFrame:
    """
    Standardize features and compute anomaly scores.

    Parameters
    ----------
    df:
        Feature DataFrame produced by :mod:`src.pipelines.extract_features`.
    method:
        Scoring method passed to :func:`~src.anomaly.scoring.score_windows`.

    Returns
    -------
    DataFrame with standardized feature columns and an ``anomaly_score`` column.
    """
    # Keep only columns that are present in this DataFrame
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    standardized = zscore_dataframe(df, columns=cols)
    scored = score_windows(standardized, method=method, columns=cols)
    return scored


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m src.pipelines.score_windows",
        description="Compute anomaly scores from a feature CSV.",
    )
    p.add_argument("features", help="Path to the feature CSV file")
    p.add_argument("--output", default=None, help="Path to save scored CSV")
    p.add_argument("--method", default="zscore", choices=["zscore", "mahalanobis"])
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    path = Path(args.features)
    if not path.exists():
        sys.exit(f"Feature file not found: {path}")

    df = pd.read_csv(path)
    scored = score_feature_dataframe(df, method=args.method)

    output_path = Path(args.output) if args.output else path.with_suffix(".scored.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"Scored features saved to {output_path}")


if __name__ == "__main__":
    main()
