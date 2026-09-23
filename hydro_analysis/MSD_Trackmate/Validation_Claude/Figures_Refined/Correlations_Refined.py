"""
Refined re-plots of three of Correlations.py's figures, per items 7, 8, 9
of the refined-figures spec: D-vs-n scatter (per file), track-length-bin
comparison, and loc_err correction deviation. Does NOT modify
Correlations.py or any cache/CSV -- pure consumer, reuses that script's own
functions/constants by import (SIZE_COLORS, STYLE_WATER, STYLE_HYDROGEL,
LENGTH_BIN_LABELS, build_file_standard_table, _representative_sigma_loc_nm,
correct_and_refit) and reads its own already-computed length_bin_comparison
.csv. The one exception (see build_correction_table_all_sizes below) is new
code, not a modification of Correlations.py -- it applies the SAME
correction formula (correct_and_refit, imported unchanged) to every
particle size instead of the two hard-coded TARGET_SIZES_NM, explicitly
marking sizes without a sigma_loc reference as missing rather than
inventing a substitute (exactly what item 9 asks for).

Item 7 (D vs n scatter), updated per follow-up feedback: LINEAR D-axis
[0, 15] um^2/s (confirmed unit: fit_results_MSD stores D_um2_per_s in
um^2/s throughout the codebase) with n-axis [0, 1.3], plus an additional
zoomed variant with D-axis [0, 4] um^2/s for examining the hydrogel
cluster (D values generally much smaller than water's). Values above the
upper bound are clipped to the edge and marked with a triangle, not
silently dropped.

Item 8 (length-bin comparison), redesigned per follow-up feedback: the
original single water/hydrogel grid was too crowded and mis-centered
hydrogel's boxes (it used the global 6-size count for the offset
calculation even though hydrogel only has 2 sizes). Now: one separate
figure per water particle size (single box per length bin, that size's own
SIZE_COLORS), and one separate hydrogel figure whose box offsets are
computed from the sizes hydrogel actually has, so they center correctly.
All of these share the same D-scale (log) so they stay comparable; n-scale
unchanged at [-0.3, 1.2]. No overall title; bolder panel titles.

Item 9 (correction deviation): Before/After BOXPLOTS instead of
scatter+median-line (no median line/marker drawn inside any box, not even
the default one -- medianprops set invisible). D and n, and Water/Hydrogel,
stay visually distinguishable via panel row and box color. Sizes without a
valid sigma_loc reference show only the Before box; the missing After box
is explicitly annotated, never substituted.

Outputs (own folder, PNG, 600 dpi): C_D_vs_n_scatter_refined.png,
B_length_bin_comparison_refined.png, A_correction_deviation_refined.png.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, NullFormatter, ScalarFormatter

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import (
    CACHE_D0, CACHE_20MG, CACHE_LOCERR, SAVE_PATH as ORIG_SAVE_PATH,
    SIZE_COLORS, STYLE_WATER, STYLE_HYDROGEL, LENGTH_BIN_LABELS,
    build_file_standard_table, _representative_sigma_loc_nm, correct_and_refit,
    _load_pickle,
)

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Figures_Refined" / "Correlations"

LENGTH_BIN_CSV = ORIG_SAVE_PATH / "length_bin_comparison.csv"

N_AXIS_RANGE = (-0.3, 1.2)          # item 8/9 n-range (unchanged)
N_AXIS_RANGE_D_VS_N = (0.0, 1.3)    # item 7 n-range, per follow-up feedback
D_AXIS_UPPER_MAIN = 15.0            # item 7 main D-range (linear, per follow-up feedback)
D_AXIS_UPPER_ZOOM = 4.0             # item 7 zoomed variant, for the hydrogel cluster


# ── Item 7: D vs n scatter (linear D-axis, per follow-up feedback) ──────────────

def plot_d_vs_n_refined(file_standard: pd.DataFrame, dls_labels: dict[float, int],
                         d_upper: float, n_range: tuple[float, float]) -> tuple[plt.Figure, dict]:
    """Linear D-axis [0, d_upper], linear n-axis n_range. Values above
    d_upper are clipped to the edge and marked with a triangle, not
    silently dropped -- same convention as the original clipped-log
    version, just on a linear scale now."""
    sizes = sorted(file_standard["particle_size_nm"].dropna().unique())
    n_panels = len(sizes)
    ncols = 3 if n_panels > 3 else n_panels
    nrows = int(np.ceil(n_panels / ncols))

    D_all = file_standard["D_um2_per_s"].to_numpy(dtype=float)
    D_finite = D_all[np.isfinite(D_all)]
    n_out_of_range = int((D_finite > d_upper).sum())

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.15 / 3, nrows * 5.00 / 2.2),
                              constrained_layout=True, squeeze=False, sharex=True, sharey=True)
    ax_flat = axes.flatten()
    for ax in ax_flat[n_panels:]:
        ax.set_visible(False)

    for ax, size_nm in zip(ax_flat, sizes):
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle=_DASH_THEORY, zorder=2)
        sub = file_standard[file_standard["particle_size_nm"] == size_nm]
        for condition, style in (("water", STYLE_WATER), ("hydrogel", STYLE_HYDROGEL)):
            g = sub[sub["condition"] == condition]
            if g.empty:
                continue
            d_vals = g["D_um2_per_s"].to_numpy(dtype=float)
            is_clipped = d_vals > d_upper
            d_clipped = np.clip(d_vals, None, d_upper)
            ax.scatter(d_clipped[~is_clipped], g["exponent"].to_numpy(dtype=float)[~is_clipped], s=30, alpha=0.7,
                       facecolor=style["face"], edgecolor=style["edge"], linewidth=0.6, zorder=3)
            if is_clipped.any():
                ax.scatter(np.full(is_clipped.sum(), d_upper),
                           g["exponent"].to_numpy(dtype=float)[is_clipped], s=42, alpha=0.9,
                           facecolor=style["face"], edgecolor="black", linewidth=0.8, marker=">", zorder=4)
        ax.set_xlim(0.0, d_upper)
        ax.set_ylim(*n_range)
        ax.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm", fontsize=9.5, fontweight="semibold")
        ax.label_outer()

    for ax in ax_flat[:n_panels]:
        ax.set_xlabel(r"$D$ ($\mathrm{\mu m^2/s}$)")
        ax.set_ylabel(r"Anomalous exponent $n$")
    for ax in ax_flat[:n_panels]:
        ax.label_outer()

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.0, linestyle=_DASH_THEORY, label="n = 1"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STYLE_WATER["face"],
               markeredgecolor=STYLE_WATER["edge"], markersize=7, label="Water"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STYLE_HYDROGEL["face"],
               markeredgecolor=STYLE_HYDROGEL["edge"], markersize=7, label="Hydrogel"),
        Line2D([0], [0], marker=">", color="w", markerfacecolor="#888888", markeredgecolor="black",
               markersize=7, label=f"$D$ > {d_upper:g} µm²/s (clipped)"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    diag = {"d_upper_um2_s": d_upper, "n_out_of_range": n_out_of_range}
    return fig, diag


# ── Item 8: length-bin comparison -- water split per size, hydrogel separate ────
# Follow-up refinement: the original single water/hydrogel-side-by-side grid
# grouped all 6 water sizes into one crowded panel and centered hydrogel's
# boxes using the GLOBAL size count (6) instead of the 2 sizes hydrogel
# actually has, so its boxes sat off to one side of each bin group instead
# of centered. Now: one separate figure per water size (single box per bin,
# that size's own SIZE_COLORS), and one hydrogel figure whose box offsets
# are computed from the sizes hydrogel actually has, so they center properly.
# Both D and n share axis ranges across ALL of these outputs (d_ylim/n-range
# computed once from the full table) so they stay directly comparable.

def _shared_length_bin_ylims(df: pd.DataFrame) -> tuple[float, float]:
    d_all = df["D_um2_per_s"].to_numpy(dtype=float)
    d_all = d_all[np.isfinite(d_all) & (d_all > 0)]
    return ((10.0 ** np.floor(np.log10(d_all.min())), 10.0 ** np.ceil(np.log10(d_all.max())))
            if d_all.size else (1e-3, 1e2))


def _length_bin_panel(ax, box_groups: list[tuple[float, np.ndarray, str, str]], value_col_is_D: bool,
                       d_ylim: tuple[float, float]) -> None:
    """box_groups: list of (x_position, values, face_color, edge_color)."""
    for pos, vals, face, edge in box_groups:
        if vals.size == 0:
            continue
        bp = ax.boxplot([vals], positions=[pos], widths=0.5, showfliers=False,
                         patch_artist=True, medianprops=dict(color=edge, linewidth=1.2))
        for patch in bp["boxes"]:
            patch.set_facecolor(face)
            patch.set_alpha(0.6)
            patch.set_edgecolor(edge)
        for element in ("whiskers", "caps"):
            for line in bp[element]:
                line.set_color(edge)
    if value_col_is_D:
        ax.set_yscale("log")
        ax.set_ylim(*d_ylim)
    else:
        ax.set_ylim(*N_AXIS_RANGE)


def plot_length_bins_one_size(df: pd.DataFrame, size_nm: float, condition: str,
                               dls_labels: dict[float, int], d_ylim: tuple[float, float]) -> plt.Figure:
    face, edge = SIZE_COLORS.get(size_nm, ("#999999", "#555555"))
    sub = df[(df["condition"] == condition) & (df["particle_size_nm"] == size_nm)]
    bin_positions = np.arange(len(LENGTH_BIN_LABELS))

    fig, (ax_d, ax_n) = plt.subplots(2, 1, figsize=(4.2, 6.2), constrained_layout=True, sharex=True)
    for ax, value_col, ylabel, is_d in ((ax_d, "D_um2_per_s", r"$D$ ($\mathrm{\mu m^2/s}$)", True),
                                          (ax_n, "exponent", r"Anomalous exponent $n$", False)):
        groups = [(bin_positions[j], sub.loc[sub["length_bin"] == label, value_col].dropna().to_numpy(), face, edge)
                  for j, label in enumerate(LENGTH_BIN_LABELS)]
        _length_bin_panel(ax, groups, is_d, d_ylim)
        ax.set_ylabel(ylabel)
    ax_d.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm", fontsize=10, fontweight="semibold")
    ax_n.set_xticks(bin_positions)
    ax_n.set_xticklabels(LENGTH_BIN_LABELS)
    ax_n.set_xlabel("Track length bin (frames)")
    return fig


def plot_length_bins_hydrogel(df: pd.DataFrame, dls_labels: dict[float, int],
                               d_ylim: tuple[float, float]) -> plt.Figure:
    sub_cond = df[df["condition"] == "hydrogel"]
    sizes = sorted(s for s in sub_cond["particle_size_nm"].dropna().unique() if s in SIZE_COLORS)
    n_sizes = max(len(sizes), 1)   # centering uses the sizes hydrogel ACTUALLY has, not the global count
    bin_positions = np.arange(len(LENGTH_BIN_LABELS))
    group_width = 0.7
    box_width = group_width / n_sizes

    fig, (ax_d, ax_n) = plt.subplots(2, 1, figsize=(5.5, 6.2), constrained_layout=True, sharex=True)
    for ax, value_col, ylabel, is_d in ((ax_d, "D_um2_per_s", r"$D$ ($\mathrm{\mu m^2/s}$)", True),
                                          (ax_n, "exponent", r"Anomalous exponent $n$", False)):
        groups = []
        for i, size_nm in enumerate(sizes):
            face, edge = SIZE_COLORS[size_nm]
            offset = (i - (n_sizes - 1) / 2) * box_width
            for j, label in enumerate(LENGTH_BIN_LABELS):
                vals = sub_cond.loc[(sub_cond["length_bin"] == label) & (sub_cond["particle_size_nm"] == size_nm),
                                     value_col].dropna().to_numpy()
                groups.append((bin_positions[j] + offset, vals, face, edge))
        _length_bin_panel(ax, groups, is_d, d_ylim)
        ax.set_ylabel(ylabel)
    ax_d.set_title("Hydrogel", fontsize=10, fontweight="semibold")
    ax_n.set_xticks(bin_positions)
    ax_n.set_xticklabels(LENGTH_BIN_LABELS)
    ax_n.set_xlabel("Track length bin (frames)")

    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=face, markeredgecolor=edge, markersize=9,
                       label=f"{dls_labels.get(s, int(s))} nm")
               for s, (face, edge) in sorted(SIZE_COLORS.items()) if s in sizes]
    fig.legend(handles=handles, loc="outside lower center", ncol=max(len(handles), 1), frameon=False, fontsize=8)
    return fig


# ── Item 9: correction deviation, Before/After boxplots, all sizes ──────────────

def build_correction_table_all_sizes(d0_results: dict, hydrogel_results: dict, locerr_df: pd.DataFrame) -> pd.DataFrame:
    """Same correction formula as Correlations.py::build_file_correction_table
    (correct_and_refit, imported unchanged), applied to every particle size
    present in the data instead of the hard-coded TARGET_SIZES_NM=(20,50).
    Sizes without a finite sigma_loc reference get D_after/n_after = NaN
    (has_reference=False) -- shown as missing, never substituted."""
    rows = []
    for results, condition in ((d0_results, "water"), (hydrogel_results, "hydrogel")):
        for r in results.values():
            size_nm = r.get("particle_size_nm")
            fit = r.get("fit_results_MSD")
            if size_nm is None or fit is None or fit.get("emsd") is None:
                continue
            D_before, n_before = fit.get("D_um2_per_s"), fit.get("exponent")
            if D_before is None or n_before is None:
                continue
            sigma_loc_nm = _representative_sigma_loc_nm(locerr_df, float(size_nm))
            has_reference = bool(np.isfinite(sigma_loc_nm))
            D_after = n_after = np.nan
            if has_reference:
                correction = correct_and_refit(fit, sigma_loc_nm / 1000.0)
                corrected_fit = correction["corrected_fit"]
                if corrected_fit is not None:
                    D_after = float(corrected_fit["A"][0] / 4.0)
                    n_after = float(corrected_fit["n"][0])
            rows.append({
                "xml_path": r.get("xml_path"), "base_name": r.get("base_name"),
                "particle_size_nm": float(size_nm), "condition": condition,
                "has_reference": has_reference,
                "D_before": float(D_before), "D_after": D_after,
                "n_before": float(n_before), "n_after": n_after,
            })
    return pd.DataFrame(rows)


def plot_correction_before_after(table: pd.DataFrame, dls_labels: dict[float, int]) -> plt.Figure:
    sizes = sorted(table["particle_size_nm"].dropna().unique())
    fig, axes = plt.subplots(2, len(sizes), figsize=(3.6 * len(sizes), 6.2), constrained_layout=True, squeeze=False)
    hidden_median = dict(color="none", linewidth=0)

    for col, size_nm in enumerate(sizes):
        sub = table[table["particle_size_nm"] == size_nm]
        missing_after = bool(sub["has_reference"].eq(False).any()) or sub["D_after"].isna().all()
        for row, (before_col, after_col, ylabel) in enumerate((
                ("D_before", "D_after", r"$D$ ($\mathrm{\mu m^2/s}$)"),
                ("n_before", "n_after", r"Anomalous exponent $n$"))):
            ax = axes[row][col]
            positions, box_data, colors, alphas = [], [], [], []
            xtick_pos, xtick_labels = [], []
            pos = 0.0
            for condition, style in (("water", STYLE_WATER), ("hydrogel", STYLE_HYDROGEL)):
                g = sub[sub["condition"] == condition]
                if g.empty:
                    continue
                before_vals = g[before_col].dropna().to_numpy()
                after_vals = g[after_col].dropna().to_numpy()
                if before_vals.size:
                    box_data.append(before_vals); positions.append(pos); colors.append(style["face"]); alphas.append(0.35)
                    xtick_pos.append(pos); xtick_labels.append("Before")
                pos += 1.0
                if after_vals.size:
                    box_data.append(after_vals); positions.append(pos); colors.append(style["face"]); alphas.append(0.85)
                    xtick_pos.append(pos); xtick_labels.append("After")
                elif before_vals.size:
                    ax.text(pos, 0.5, "no σ_loc\nreference", transform=ax.get_xaxis_transform(),
                            ha="center", va="center", fontsize=6.5, color="#888888", rotation=90)
                    xtick_pos.append(pos); xtick_labels.append("After")
                pos += 1.3
            if box_data:
                bp = ax.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True, showfliers=False,
                                 medianprops=hidden_median)
                for patch, color, alpha in zip(bp["boxes"], colors, alphas):
                    patch.set_facecolor(color)
                    patch.set_alpha(alpha)
                    patch.set_edgecolor(color)
                for element in ("whiskers", "caps"):
                    for line in bp[element]:
                        line.set_alpha(0.8)
            ax.set_xticks(xtick_pos)
            ax.set_xticklabels(xtick_labels, fontsize=7.5)
            if row == 0:
                ax.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm", fontsize=10, fontweight="semibold")
            if col == 0:
                ax.set_ylabel(ylabel)

    handles = [Patch(facecolor=STYLE_WATER["face"], alpha=0.6, label="Water"),
               Patch(facecolor=STYLE_HYDROGEL["face"], alpha=0.6, label="Hydrogel"),
               Patch(facecolor="#888888", alpha=0.35, label="Without log-error correction"),
               Patch(facecolor="#888888", alpha=0.85, label="After correction")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    return fig


def main() -> None:
    d0_results = _load_pickle(CACHE_D0, "MSD_FromTrackmate_D0.py")
    hydrogel_results = _load_pickle(CACHE_20MG, "MSD_FromTrackmate_20mg.py")
    locerr_df = _load_pickle(CACHE_LOCERR, "Loc_Error_Analyse_immob_particle.py")
    dls_labels = get_dls_labels()
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_RC):
        # Item 7: main (0-15 um^2/s) + zoomed (0-4 um^2/s, for the hydrogel cluster)
        file_standard = build_file_standard_table(d0_results, hydrogel_results)
        for d_upper, suffix in ((D_AXIS_UPPER_MAIN, ""), (D_AXIS_UPPER_ZOOM, "_zoom_0to4")):
            fig7, diag7 = plot_d_vs_n_refined(file_standard, dls_labels, d_upper, N_AXIS_RANGE_D_VS_N)
            print(f"D vs n{suffix}: linear D-range = (0, {diag7['d_upper_um2_s']:g}) um^2/s, "
                  f"n-range = {N_AXIS_RANGE_D_VS_N}. {diag7['n_out_of_range']} file(s) above the upper "
                  "bound -- marked with a triangle, not dropped.")
            path7 = SAVE_PATH / f"C_D_vs_n_scatter_refined{suffix}.png"
            safe_savefig(fig7, path7, dpi=600, bbox_inches="tight")
            plt.close(fig7)
            print(f"Plot gespeichert: {path7}")

        # Item 8: water split into one plot per size, hydrogel as its own
        # correctly-centered plot -- both share the same D-scale.
        if not LENGTH_BIN_CSV.exists():
            raise FileNotFoundError(f"Nicht gefunden: {LENGTH_BIN_CSV}\nBitte zuerst Correlations.py ausführen.")
        length_bin_table = pd.read_csv(LENGTH_BIN_CSV)
        d_ylim = _shared_length_bin_ylims(length_bin_table)
        water_sizes = sorted(s for s in length_bin_table.loc[length_bin_table["condition"] == "water",
                                                               "particle_size_nm"].dropna().unique()
                              if s in SIZE_COLORS)
        for size_nm in water_sizes:
            fig8 = plot_length_bins_one_size(length_bin_table, size_nm, "water", dls_labels, d_ylim)
            path8 = SAVE_PATH / f"B_length_bin_water_{dls_labels.get(size_nm, int(size_nm))}nm.png"
            safe_savefig(fig8, path8, dpi=600, bbox_inches="tight")
            plt.close(fig8)
            print(f"Plot gespeichert: {path8}")
        fig8_gel = plot_length_bins_hydrogel(length_bin_table, dls_labels, d_ylim)
        path8_gel = SAVE_PATH / "B_length_bin_hydrogel.png"
        safe_savefig(fig8_gel, path8_gel, dpi=600, bbox_inches="tight")
        plt.close(fig8_gel)
        print(f"Plot gespeichert: {path8_gel}")

        # Item 9
        correction_table = build_correction_table_all_sizes(d0_results, hydrogel_results, locerr_df)
        n_missing = int(correction_table.groupby("particle_size_nm")["has_reference"].apply(lambda s: not s.any()).sum())
        print(f"Correction deviation: {correction_table['particle_size_nm'].nunique()} particle size(s) considered, "
              f"{n_missing} without a sigma_loc reference (Before shown, After explicitly marked missing).")
        fig9 = plot_correction_before_after(correction_table, dls_labels)
        path9 = SAVE_PATH / "A_correction_deviation_refined.png"
        safe_savefig(fig9, path9, dpi=600, bbox_inches="tight")
        plt.close(fig9)
        print(f"Plot gespeichert: {path9}")


if __name__ == "__main__":
    main()
