from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(frame_df: pd.DataFrame, sequence_df: pd.DataFrame, overall_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trackers = list(overall_df["tracker"])

    x = np.arange(len(trackers))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.bar(x - width, overall_df["mean_iou"], width, label="Mean IoU")
    ax.bar(x, overall_df["success_auc"], width, label="Success AUC")
    ax.bar(x + width, overall_df["precision_20"], width, label="Precision@20")
    ax.set_xticks(x, trackers, rotation=22, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall tracking accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "overall_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(trackers, overall_df["fps"])
    ax.tick_params(axis="x", rotation=22)
    ax.set_yscale("log")
    ax.set_ylabel("FPS (log scale)")
    ax.set_title("Median update speed on the current machine")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fps.png", dpi=180)
    plt.close(fig)

    thresholds = np.linspace(0.0, 1.0, 101)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for tracker in trackers:
        values = frame_df.loc[frame_df["tracker"] == tracker, "iou"].to_numpy(dtype=float)
        curve = [(values >= threshold).mean() for threshold in thresholds]
        ax.plot(thresholds, curve, label=tracker)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Success rate")
    ax.set_title("Success plot")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "success_plot.png", dpi=180)
    plt.close(fig)

    center_thresholds = np.arange(0, 51, 1)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for tracker in trackers:
        values = frame_df.loc[frame_df["tracker"] == tracker, "center_error"].to_numpy(dtype=float)
        curve = [(values <= threshold).mean() for threshold in center_thresholds]
        ax.plot(center_thresholds, curve, label=tracker)
    ax.set_xlabel("Center-error threshold (pixels)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision plot")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_plot.png", dpi=180)
    plt.close(fig)

    pivot = sequence_df.pivot(index="sequence", columns="tracker", values="mean_iou")
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, f"{pivot.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Mean IoU by scenario")
    fig.colorbar(image, ax=ax, label="Mean IoU")
    fig.tight_layout()
    fig.savefig(output_dir / "per_sequence_iou.png", dpi=180)
    plt.close(fig)


