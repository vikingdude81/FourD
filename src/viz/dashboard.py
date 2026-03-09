"""
Interactive dashboard for exploring QRNG pipeline outputs.

Usage (command line)::

    python3 -m src.viz.dashboard \\
        --features outputs/qrng_pipeline/<stream_id>/window_features.csv \\
        --latent  outputs/qrng_pipeline/<stream_id>/latent_trajectory.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_anomaly_scores(ax: plt.Axes, features_df: pd.DataFrame) -> None:
    """Line chart of anomaly scores over window index."""
    if "anomaly_score" not in features_df.columns:
        ax.set_title("Anomaly Scores (not available)")
        return
    ax.plot(features_df.index, features_df["anomaly_score"], color="steelblue", lw=1.5)
    ax.axhline(features_df["anomaly_score"].mean(), color="orange", ls="--", lw=1, label="mean")
    ax.set_title("Anomaly Scores per Window")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Anomaly score")
    ax.legend()


def plot_feature_heatmap(ax: plt.Axes, features_df: pd.DataFrame) -> None:
    """Heatmap of normalized feature values across windows."""
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    if "anomaly_score" in numeric_cols:
        numeric_cols.remove("anomaly_score")
    if not numeric_cols:
        ax.set_title("Features (no numeric columns)")
        return

    data = features_df[numeric_cols].to_numpy(dtype=float)
    col_std = data.std(axis=0)
    col_std[col_std < 1e-8] = 1.0
    data_norm = (data - data.mean(axis=0)) / col_std

    im = ax.imshow(data_norm.T, aspect="auto", cmap="RdYlBu_r", origin="lower")
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols, fontsize=7)
    ax.set_xlabel("Window index")
    ax.set_title("Normalized Feature Heatmap")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_latent_trajectory(ax: plt.Axes, latent_df: pd.DataFrame) -> None:
    """2-D scatter of the first two coordinator coordinates, coloured by time."""
    coord_cols = [c for c in latent_df.columns if c.startswith("coord_")]
    if len(coord_cols) < 2:
        ax.set_title("Latent trajectory (insufficient coords)")
        return

    x = latent_df[coord_cols[0]].to_numpy()
    y = latent_df[coord_cols[1]].to_numpy()
    t = np.arange(len(x))

    sc = ax.scatter(x, y, c=t, cmap="plasma", s=12, alpha=0.7)
    ax.plot(x, y, lw=0.5, color="grey", alpha=0.4)
    plt.colorbar(sc, ax=ax, label="Time step")
    ax.set_xlabel(coord_cols[0])
    ax.set_ylabel(coord_cols[1])
    ax.set_title("Latent Coordinator Trajectory")


def plot_basin_switches(ax: plt.Axes, latent_df: pd.DataFrame) -> None:
    """Bar chart / scatter of chosen basin over time."""
    if "chosen_basin" not in latent_df.columns:
        ax.set_title("Basin switches (not available)")
        return
    ax.step(latent_df.index, latent_df["chosen_basin"], where="post", color="teal", lw=1.5)
    ax.set_title("Chosen Basin over Time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Basin index")


def build_dashboard(
    features_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    stream_id: str = "",
) -> plt.Figure:
    """
    Build a 2×2 dashboard figure.

    Parameters
    ----------
    features_df:
        Window-features CSV as a DataFrame.
    latent_df:
        Latent-trajectory CSV as a DataFrame.
    stream_id:
        Label shown in the figure title.

    Returns
    -------
    Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"FourD Pipeline Dashboard  |  {stream_id}", fontsize=13)

    plot_anomaly_scores(axes[0, 0], features_df)
    plot_feature_heatmap(axes[0, 1], features_df)
    plot_latent_trajectory(axes[1, 0], latent_df)
    plot_basin_switches(axes[1, 1], latent_df)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m src.viz.dashboard",
        description="Display a FourD pipeline dashboard.",
    )
    p.add_argument("--features", required=True, help="Path to window_features.csv")
    p.add_argument("--latent", required=True, help="Path to latent_trajectory.csv")
    p.add_argument("--save", default=None, help="Save figure to this path instead of showing it")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    features_path = Path(args.features)
    latent_path = Path(args.latent)

    if not features_path.exists():
        sys.exit(f"Features file not found: {features_path}")
    if not latent_path.exists():
        sys.exit(f"Latent file not found: {latent_path}")

    features_df = pd.read_csv(features_path)
    latent_df = pd.read_csv(latent_path)
    stream_id = features_path.parent.name

    fig = build_dashboard(features_df, latent_df, stream_id)

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Dashboard saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
