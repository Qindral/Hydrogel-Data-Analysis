"""
Refined re-plot of Fittingpoints_robustness.py's D/n-vs-fit-points robustness
check, per item 6 of the refined-figures spec: one separate 6-panel figure
each for D and n (panels = particle size, not lines-in-one-axes), one
version using all available fit-point counts (4-8) and one using only
4/5/6, linear y-axes for both D and n, no dashed/dotted lines for the two
hydrogel conditions -- markers distinguish condition instead (solid lines
throughout). Size colors use the same orange (small) -> blue (large)
palette already established for size in Correlations.py. English labels,
DLS size labels, shared axes with outer-only labels, bolder panel titles.

Does NOT modify Fittingpoints_robustness.py or any cache/CSV -- pure
consumer, reads the existing fittingpoints_robustness.csv produced by that
script (median/IQR of D and n per particle size x condition x fit-point
count, already computed from cache/msd_d0_results.pkl and
cache/msd_20mg_files.pkl via core.analysis.fit_powerlaw_with_errors).

Outputs (own folder, PNG, 600 dpi): fittingpoints_D_by_size_all_points.png,
fittingpoints_D_by_size_456_only.png, fittingpoints_n_by_size_all_points.png,
fittingpoints_n_by_size_456_only.png.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Fittingpoints_Robustness" / "fittingpoints_robustness.csv"

SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Figures_Refined" / "Fittingpoints_Robustness"

CONDITION_MARKERS = {
    "Wasser": "o",
    "Hydrogel surface loading": "s",
    "Hydrogel injection": "^",
    "Hydrogel": "D",
}
CONDITION_LABELS_EN = {
    "Wasser": "Water",
    "Hydrogel surface loading": "Hydrogel surface loading",
    "Hydrogel injection": "Hydrogel injection",
    "Hydrogel": "Hydrogel",
}


def plot_metric_by_size(df: pd.DataFrame, value_col: str, iqr_low_col: str, iqr_high_col: str,
                         ylabel: str, dls_labels: dict[float, int], fit_points: list[int],
                         sizes_filter: set[float] | None = None,
                         y_range: tuple[float, float] | None = None) -> plt.Figure:
    sizes = [s for s in sorted(df["partikelgröße_nm"].unique())
             if s in SIZE_COLORS and (sizes_filter is None or s in sizes_filter)]
    n = len(sizes)
    ncols = 3 if n > 3 else max(n, 1)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.15 / 3, nrows * 5.00 / 2.2),
                              constrained_layout=True, squeeze=False, sharex=True, sharey=True)
    ax_flat = axes.flatten()
    for ax in ax_flat[n:]:
        ax.set_visible(False)

    legend_conditions: dict[str, str] = {}
    for ax, size_nm in zip(ax_flat, sizes):
        face, edge = SIZE_COLORS[size_nm]
        sub_size = df[df["partikelgröße_nm"] == size_nm]
        for condition, g in sub_size.groupby("bedingung"):
            g = g.sort_values("fittingpoints")
            x = g["fittingpoints"].to_numpy()
            y = g[value_col].to_numpy()
            marker = CONDITION_MARKERS.get(condition, "o")
            ax.plot(x, y, color=edge, linewidth=1.4, linestyle="-", marker=marker,
                     markersize=4.5, markerfacecolor=face, markeredgecolor=edge, markeredgewidth=0.6)
            ax.fill_between(x, g[iqr_low_col], g[iqr_high_col], color=face, alpha=0.15, linewidth=0)
            legend_conditions.setdefault(condition, marker)
        ax.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm", fontsize=9.5, fontweight="semibold")
        ax.set_xticks(fit_points)
        if y_range is not None:
            ax.set_ylim(*y_range)
        ax.label_outer()

    for ax in ax_flat[:n]:
        ax.set_xlabel("Number of fit points")
        ax.set_ylabel(ylabel)
    for ax in ax_flat[:n]:
        ax.label_outer()

    handles = [Line2D([0], [0], color="black", marker=marker, linestyle="-", markersize=5,
                       markerfacecolor="white", markeredgecolor="black",
                       label=CONDITION_LABELS_EN.get(cond, cond))
               for cond, marker in legend_conditions.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=max(len(handles), 1),
               frameon=False, fontsize=8)
    return fig


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Nicht gefunden: {CSV_PATH}\nBitte zuerst Fittingpoints_robustness.py ausführen.")
    df = pd.read_csv(CSV_PATH)
    dls_labels = get_dls_labels()

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    variants = [
        (sorted(df["fittingpoints"].unique()), "all_points"),
        ([p for p in (4, 5, 6) if p in df["fittingpoints"].unique()], "456_only"),
    ]
    with plt.rc_context(_RC):
        for fit_points, suffix in variants:
            df_variant = df[df["fittingpoints"].isin(fit_points)]

            fig_d = plot_metric_by_size(df_variant, "D_median", "D_IQR_low", "D_IQR_high",
                                         r"$D$ ($\mathrm{\mu m^2/s}$)", dls_labels, fit_points)
            path_d = SAVE_PATH / f"fittingpoints_D_by_size_{suffix}.png"
            safe_savefig(fig_d, path_d, dpi=600, bbox_inches="tight")
            plt.close(fig_d)
            print(f"Plot gespeichert: {path_d}")

            fig_n = plot_metric_by_size(df_variant, "n_median", "n_IQR_low", "n_IQR_high",
                                         r"Anomalous exponent $n$", dls_labels, fit_points)
            path_n = SAVE_PATH / f"fittingpoints_n_by_size_{suffix}.png"
            safe_savefig(fig_n, path_n, dpi=600, bbox_inches="tight")
            plt.close(fig_n)
            print(f"Plot gespeichert: {path_n}")

        # Follow-up feedback: for the 456_only D variant specifically, the
        # 200/500/1000 nm (nominal) sizes are hard to read on the shared
        # scale set by the much larger 20/50/100 nm D values -- give them
        # their own 3-panel figure with a fixed [0, 4] um^2/s scale.
        large_sizes = {200.0, 500.0, 1000.0}
        fit_points_456 = [p for p in (4, 5, 6) if p in df["fittingpoints"].unique()]
        df_456 = df[df["fittingpoints"].isin(fit_points_456)]
        fig_d_large = plot_metric_by_size(df_456, "D_median", "D_IQR_low", "D_IQR_high",
                                           r"$D$ ($\mathrm{\mu m^2/s}$)", dls_labels, fit_points_456,
                                           sizes_filter=large_sizes, y_range=(0.0, 4.0))
        path_d_large = SAVE_PATH / "fittingpoints_D_by_size_456_only_200plus_0to4.png"
        safe_savefig(fig_d_large, path_d_large, dpi=600, bbox_inches="tight")
        plt.close(fig_d_large)
        print(f"Plot gespeichert: {path_d_large}")


if __name__ == "__main__":
    main()
