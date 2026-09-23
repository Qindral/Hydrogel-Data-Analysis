"""Refined SNR-by-size boxplots with DLS labels and condition colors.

Retains the original raw-distribution figures and adds paired raw/DoG
comparisons plus percentage-change boxplots, with/without immobilized.
New plots consume the paired measurements from snr_signal_profile.py;
they never compare the old raw CSV against a different filtered sample.
DoG is a SciPy reanalysis without median prefiltering, not exact Fiji replay.
The paired example sample is small and is labelled with group counts.
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

# ── Configuration ──────────────────────────────────────────────────────────────
SNR_CSV = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "SNR_Analysis" / "snr_values_by_size.csv"

SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Figures_Refined" / "SNR_Analysis"


PAIRED_CSV = Path(__file__).resolve().parents[4] / "analysis_results" / "snr_profiles_dog" / "snr_dog_particle_metrics.csv"


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


def load_paired_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"condition", "particle_size_nm", "snr", "dog_snr", "frame", "spot_id"}
    if missing := required - set(df.columns):
        raise ValueError(f"Paired SNR input missing columns: {sorted(missing)}")
    keys = ["condition", "particle_size_nm", "frame", "spot_id"]
    if df.duplicated(keys).any():
        raise ValueError("Paired SNR input contains duplicate particle/frame measurements")
    valid = np.isfinite(df["snr"]) & np.isfinite(df["dog_snr"])
    if not valid.all():
        print(f"Excluded {(~valid).sum()} non-finite SNR pairs")
    df = df.loc[valid].copy()
    if df.empty:
        raise ValueError("No finite paired SNR measurements")
    df["snr_change_percent"] = np.where(df["snr"] > 0, 100 * (df["dog_snr"] / df["snr"] - 1), np.nan)
    return df


def plot_paired_overview(df, dls_labels, conditions, *, change=False):
    """Condition-grouped boxes; adjacent raw/DoG boxes share paired samples."""
    sizes = sorted(df["particle_size_nm"].unique())
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    positions = np.arange(len(sizes))
    width = 0.8 / max(len(conditions), 1)
    stages = [("snr_change_percent", "Change")] if change else [("snr", "Raw"), ("dog_snr", "DoG")]
    for i, condition in enumerate(conditions):
        color = CONDITION_COLORS.get(condition, "#808080")
        for j, size in enumerate(sizes):
            group = df[(df["particle_size_nm"] == size) & (df["condition"] == condition)]
            center = positions[j] + (i - (len(conditions) - 1) / 2) * width
            counts = []
            for stage_idx, (column, _) in enumerate(stages):
                values = group[column].to_numpy()
                values = values[np.isfinite(values)]
                counts.append(len(values))
                if not len(values):
                    continue
                position = center + (stage_idx - (len(stages) - 1) / 2) * width / len(stages)
                bp = ax.boxplot([values], positions=[position], widths=width * 0.8 / len(stages),
                                patch_artist=True, showfliers=False)
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_edgecolor(color)
                    patch.set_alpha(0.5 if change or stage_idx == 0 else 0.22)
                    if not change and stage_idx == 1:
                        patch.set_hatch("////")
                for element in ("whiskers", "caps", "medians"):
                    for line in bp[element]:
                        line.set_color(color)
                if len(values) == 1:
                    ax.scatter([position], values, color=color, s=12, zorder=4)
            ax.text(center, 0.98, str(max(counts)), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7, color=color)
    ax.margins(y=0.15)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{dls_labels.get(s, int(s))} nm" for s in sizes])
    ax.set_xlabel("Particle Size (nm)")
    ax.set_ylabel("SNR Change after DoG (%)" if change else "Signal-to-Noise Ratio")
    ax.set_title("Paired particles | numbers above groups = n", fontsize=9)
    handles = [Patch(facecolor=CONDITION_COLORS.get(c, "#808080"), alpha=0.5,
                     label=c.capitalize()) for c in conditions]
    if not change:
        handles += [Patch(facecolor="gray", alpha=0.5, label="Raw"),
                    Patch(facecolor="gray", alpha=0.22, hatch="////", label="DoG")]
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)
    return fig


def main(snr_csv=SNR_CSV, paired_csv=PAIRED_CSV, output_dir=SAVE_PATH) -> None:
    if not snr_csv.exists():
        raise FileNotFoundError(f"Nicht gefunden: {snr_csv}\nBitte zuerst SNR_Analysis.py ausführen.")
    df = pd.read_csv(snr_csv)
    paired = load_paired_data(Path(paired_csv))
    dls_labels = get_dls_labels()

    all_conditions = sorted(df["condition"].unique())
    non_immob_conditions = [c for c in all_conditions if c != "immobilized"]

    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        fig_with = plot_snr_overview(df, dls_labels, all_conditions)
        path_with = output_dir / "snr_overview_by_size_with_immobilized.png"
        safe_savefig(fig_with, path_with, dpi=600, bbox_inches="tight")
        plt.close(fig_with)
        print(f"Plot gespeichert: {path_with}")

        fig_without = plot_snr_overview(df[df["condition"] != "immobilized"], dls_labels, non_immob_conditions)
        path_without = output_dir / "snr_overview_by_size_without_immobilized.png"
        safe_savefig(fig_without, path_without, dpi=600, bbox_inches="tight")
        plt.close(fig_without)
        print(f"Plot gespeichert: {path_without}")

        for include_immobilized in (True, False):
            subset = paired if include_immobilized else paired[paired["condition"] != "immobilized"]
            if subset.empty:
                continue
            conditions = sorted(subset["condition"].unique())
            suffix = "with_immobilized" if include_immobilized else "without_immobilized"
            for change, name in ((False, "snr_raw_vs_dog_by_size"), (True, "snr_dog_change_by_size")):
                fig = plot_paired_overview(subset, dls_labels, conditions, change=change)
                path = output_dir / f"{name}_{suffix}.png"
                safe_savefig(fig, path, dpi=600, bbox_inches="tight")
                plt.close(fig)
                print(f"Plot gespeichert: {path}")
    paired.to_csv(output_dir / "snr_paired_values_by_size.csv", index=False)
    summary = paired.groupby(["condition", "particle_size_nm"]).agg(
        n=("snr", "size"), raw_snr_median=("snr", "median"), dog_snr_median=("dog_snr", "median"),
        change_n=("snr_change_percent", "count"), change_percent_median=("snr_change_percent", "median"))
    summary.to_csv(output_dir / "snr_paired_summary_by_size.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snr-csv", type=Path, default=SNR_CSV)
    parser.add_argument("--paired-csv", type=Path, default=PAIRED_CSV)
    parser.add_argument("--output-dir", type=Path, default=SAVE_PATH)
    args = parser.parse_args()
    main(args.snr_csv, args.paired_csv, args.output_dir)
