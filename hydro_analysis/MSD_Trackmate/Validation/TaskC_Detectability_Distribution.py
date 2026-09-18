"""
Distribution of detection quality across all sampled 35 nm detections, with
the three selected examples (Task C composite figure) marked.

Pure consumer: reads cache/detectability_35nm.pkl (TaskC_Detectability_Compute.py).
Never touches raw TIFF/XML. Uses the same _detectability_selection.select_examples()
helper as TaskC_Detectability_Figure.py, so the marked points are identical
in both figures.

SNR histogram (particle detections only, background excluded) and a
LoG-peak-vs-SNR scatter across the full sampled population, confirming the
three chosen examples sit within the real experimental distribution rather
than being isolated hand-picked cases.

Run TaskC_Detectability_Compute.py first (or after any raw-data change) to
refresh the cache.

Shows the figure first (plt.show(), blocking); only after the window is
closed does it save into its own Auswertungsbilder\\Detectability_Distribution\\
subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation._detectability_selection import select_examples

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent.parent / "cache" / "detectability_35nm.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "Detectability_Distribution"

COLOR_DETECTION = "#3B6E8C"
MARKER_STYLE = {
    "good": dict(marker="*", color="#2ca02c", markersize=16, label="Very good (selected)"),
    "challenging": dict(marker="D", color="#d62728", markersize=11, label="Challenging (selected)"),
    "no_particle": dict(marker="X", color="#7f7f7f", markersize=11, label="No particle (selected)"),
}


def load_df() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_FILE}\nBitte zuerst TaskC_Detectability_Compute.py ausführen."
        )
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def plot_snr_histogram(detections: pd.DataFrame, examples: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    ax.hist(detections["snr"].dropna(), bins=40, color=COLOR_DETECTION, edgecolor="#2A4F66", alpha=0.85)
    for key in ("good", "challenging"):
        style = MARKER_STYLE[key]
        ax.axvline(examples[key]["snr"], color=style["color"], linewidth=1.8, linestyle="--",
                   label=f"{style['label']} (SNR={examples[key]['snr']:.2g})")
    ax.set_xlabel("SNR = (local max - background) / background SD")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    return fig


def plot_log_vs_snr(df: pd.DataFrame, examples: dict) -> plt.Figure:
    detections = df[~df["is_background"]]
    background = df[df["is_background"]]

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    ax.scatter(detections["snr"], detections["log_snr"], s=14, alpha=0.35,
               facecolor=COLOR_DETECTION, edgecolor="none", label="detections")
    ax.scatter(background["snr"], background["log_snr"], s=14, alpha=0.35,
               facecolor="#999999", edgecolor="none", label="background")

    for key, style in MARKER_STYLE.items():
        row = examples[key]
        ax.scatter([row["snr"]], [row["log_snr"]], marker=style["marker"], s=style["markersize"] ** 2,
                  color=style["color"], edgecolor="black", linewidth=0.8, zorder=5, label=style["label"])

    ax.set_xlabel("Raw-intensity SNR")
    ax.set_ylabel("LoG-response SNR")
    ax.legend(loc="best", fontsize=8, frameon=False)
    return fig


def main() -> None:
    df = load_df()
    print(f"Cache geladen: {CACHE_FILE} ({len(df)} Crops)")
    examples = select_examples(df)

    detections = df[~df["is_background"]]

    with plt.rc_context(_RC):
        fig1 = plot_snr_histogram(detections, examples)
        plt.show()
        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            fig1.savefig(SAVE_PATH / "snr_distribution.png", dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {SAVE_PATH / 'snr_distribution.png'}")

        fig2 = plot_log_vs_snr(df, examples)
        plt.show()
        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            fig2.savefig(SAVE_PATH / "log_snr_vs_snr.png", dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {SAVE_PATH / 'log_snr_vs_snr.png'}")


if __name__ == "__main__":
    main()
