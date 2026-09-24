"""Refined SNR-by-size boxplots with DLS labels and condition colors.

Consumer script: reads only cache/particle_size_snr.pkl (written by
Particle_Size_SNR_Compute.py --mode all), table "tracks" -- one SNR value per
track (median over its detections), all tracked files per condition and
size, including every 50 nm movie. SNR = (peak within R - annulus mean) /
annulus std, as in SNR_Analysis.py.

Two variants per figure: with and without immobilized particles; the "with"
variant is only written when the pickle contains an immobilized condition.
The former paired raw/DoG example comparison has been removed.

Output (own folder, PNG, 600 dpi): snr_overview_by_size_with_immobilized.png,
snr_overview_by_size_without_immobilized.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.snr_signal_profile import CONDITION_COLORS
from hydro_analysis.MSD_Trackmate.Validation_Claude.SNR_Analysis import PSS_CACHE, load_track_snr

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Figures_Refined" / "SNR_Analysis"


def plot_snr_overview(df: pd.DataFrame, dls_labels: dict[float, int], conditions: list[str]) -> plt.Figure:
    sizes = sorted(df["particle_size_nm"].unique())
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    positions = np.arange(len(sizes))
    width = 0.8 / max(len(conditions), 1)
    for i, condition in enumerate(conditions):
        color = CONDITION_COLORS.get(condition, "#808080")
        offsets = positions + (i - (len(conditions) - 1) / 2) * width
        box_data = [df.loc[(df["particle_size_nm"] == s) & (df["condition"] == condition), "snr"].to_numpy()
                    for s in sizes]
        bp = ax.boxplot(box_data, positions=offsets, widths=width * 0.85, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor(color)
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color(color)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{dls_labels.get(s, int(s))} nm" for s in sizes])
    ax.set_xlabel("Particle Size (nm)")
    ax.set_ylabel("Signal-to-Noise Ratio")
    handles = [Patch(facecolor=CONDITION_COLORS.get(c, "#808080"), alpha=0.5,
                      label=c.capitalize()) for c in conditions]
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)
    return fig


def main(pss_cache=PSS_CACHE, output_dir=SAVE_PATH) -> None:
    df = load_track_snr(Path(pss_cache))
    dls_labels = get_dls_labels()
    all_conditions = sorted(df["condition"].unique())

    variants = [("without_immobilized", [c for c in all_conditions if c != "immobilized"])]
    if "immobilized" in all_conditions:
        variants.insert(0, ("with_immobilized", all_conditions))

    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        for suffix, conditions in variants:
            fig = plot_snr_overview(df[df["condition"].isin(conditions)], dls_labels, conditions)
            path = output_dir / f"snr_overview_by_size_{suffix}.png"
            safe_savefig(fig, path, dpi=600, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot gespeichert: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PSS_CACHE)
    parser.add_argument("--output-dir", type=Path, default=SAVE_PATH)
    args = parser.parse_args()
    main(args.input, args.output_dir)
