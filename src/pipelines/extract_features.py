"""
Pipeline step: extract features from all windows of a bitstream.

Usage::

    python3 -m src.pipelines.extract_features \\
        <bitstream.txt> --output features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.autocorr import compute_autocorr_features
from src.features.basic_stats import compute_basic_stats
from src.features.changepoint import compute_changepoint_features
from src.features.complexity import compute_complexity_features
from src.features.entropy import compute_entropy_features
from src.features.windows import sliding_windows


def extract_features_from_stream(
    bits: np.ndarray,
    stream_id: str = "",
    window_size: int = 4096,
    overlap: float = 0.5,
) -> pd.DataFrame:
    """
    Extract all features from every window of a bitstream.

    Parameters
    ----------
    bits:
        1-D numpy array of integer bits (0 or 1).
    stream_id:
        Identifier label included as a column in the output DataFrame.
    window_size:
        Number of bits per sliding window.
    overlap:
        Fractional overlap between consecutive windows.

    Returns
    -------
    DataFrame with one row per window and all feature columns.
    """
    records = []
    for window_idx, (start, window) in enumerate(
        sliding_windows(bits, window_size, overlap)
    ):
        row: dict = {
            "stream_id": stream_id,
            "window_index": window_idx,
            "start_bit": start,
            "end_bit": start + window_size,
            "window_size": window_size,
            "overlap": overlap,
        }
        row.update(compute_basic_stats(window))
        row.update(compute_entropy_features(window))
        row.update(compute_complexity_features(window))
        row.update(compute_autocorr_features(window))
        row.update(compute_changepoint_features(window))
        records.append(row)

    return pd.DataFrame(records)


def load_bitstream(path: Path) -> np.ndarray:
    """
    Load a bitstream from a text file.

    The file may contain ``0`` and ``1`` characters with any whitespace
    or newlines between them.
    """
    text = path.read_text()
    bits = np.array([int(c) for c in text if c in ("0", "1")], dtype=np.uint8)
    return bits


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m src.pipelines.extract_features",
        description="Extract features from a bitstream file.",
    )
    p.add_argument("bitstream", help="Path to the bitstream text file")
    p.add_argument("--output", default=None, help="Path to save the CSV output")
    p.add_argument("--window-size", type=int, default=4096, dest="window_size")
    p.add_argument("--overlap", type=float, default=0.5)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    path = Path(args.bitstream)
    if not path.exists():
        sys.exit(f"Bitstream file not found: {path}")

    bits = load_bitstream(path)
    stream_id = path.stem
    df = extract_features_from_stream(
        bits,
        stream_id=stream_id,
        window_size=args.window_size,
        overlap=args.overlap,
    )

    output_path = Path(args.output) if args.output else path.with_suffix(".features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}  ({len(df)} windows)")


if __name__ == "__main__":
    main()
