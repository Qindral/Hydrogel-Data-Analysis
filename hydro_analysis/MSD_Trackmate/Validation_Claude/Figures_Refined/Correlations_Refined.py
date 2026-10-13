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

Item 9 axis ranges (user, 2026-10-12): D 0-15 um^2/s for DLS 35 and 50 nm, 0-6 um^2/s for
DLS 100-1370 nm; n 0.5-1.3 for all sizes. Values outside a range stay in the box statistics and
are not marked in the figure (user, 2026-10-12); their number per box is written to
A_correction_deviation_refined_values_outside_axes.csv and printed.

Style: figure style guide v2 (thesis.mplstyle, width classes full/half, water = category A,
hydrogel = category B colours, no bbox_inches="tight").
Outputs (own folder, each as PDF and PNG 600 dpi): C_D_vs_n_scatter_refined(_zoom_0to4),
B_length_bin_water_<size>nm, B_length_bin_hydrogel, A_correction_deviation_refined.
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
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _DASH_THEORY, _add_log_minor_ticks
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import (
    CACHE_D0, CACHE_20MG, CACHE_LOCERR, SAVE_PATH as ORIG_SAVE_PATH,
    SIZE_COLORS, LENGTH_BIN_LABELS,
    build_file_standard_table, _representative_sigma_loc_nm, correct_and_refit,
    _load_pickle,
)

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Figures_Refined" / "Correlations"

LENGTH_BIN_CSV = ORIG_SAVE_PATH / "length_bin_comparison.csv"

# Figure style v2 (Styleguide_Figures_Dissertation.md): rcParams, width classes, colours
STYLE_FILE = Path(__file__).resolve().parents[3] / "thesis.mplstyle"
WIDTH_IN = {"full": 6.30, "narrow": 4.72, "half": 3.07, "third": 2.01}
# Style guide section 11: water/buffer = category A, hydrogel = category B (local override of the
# older colours in Correlations.py, which is not modified)
STYLE_WATER = dict(face="#3B8C8C", edge="#2A6666", label="Water")
STYLE_HYDROGEL = dict(face="#D98C3D", edge="#A6672D", label="Hydrogel")
D_LABEL = r"Diffusion coefficient $D$ (µm$^2$ s$^{-1}$)"
N_LABEL = r"Anomalous exponent $n$"

N_AXIS_RANGE = (-0.3, 1.2)          # item 8 n-range (unchanged)
N_AXIS_RANGE_D_VS_N = (0.0, 1.3)    # item 7 n-range, per follow-up feedback
D_AXIS_UPPER_MAIN = 15.0            # item 7 main D-range (linear, per follow-up feedback)
D_AXIS_UPPER_ZOOM = 4.0             # item 7 zoomed variant, for the hydrogel cluster

# Item 9 axis ranges, decided by user, 2026-10-12
CORR_SMALL_SIZES_NM = (20.0, 50.0)  # nominal sizes of DLS 35 and 50 nm
CORR_D_RANGE_SMALL = (0.0, 15.0)    # D axis, DLS 35 and 50 nm
CORR_D_RANGE_LARGE = (0.0, 6.0)     # D axis, DLS 100 to 1370 nm
CORR_N_RANGE = (0.5, 1.3)           # n axis, all sizes


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

    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH_IN["full"], 1.85 * nrows + 0.5),
                              constrained_layout=True, squeeze=False, sharex=True, sharey=True)
    ax_flat = axes.flatten()
    for ax in ax_flat[n_panels:]:
        ax.set_visible(False)

    for ax, size_nm in zip(ax_flat, sizes):
        ax.axhline(1.0, color="black", linewidth=1.2, linestyle=_DASH_THEORY, zorder=2)
        sub = file_standard[file_standard["particle_size_nm"] == size_nm]
        for condition, style in (("water", STYLE_WATER), ("hydrogel", STYLE_HYDROGEL)):
            g = sub[sub["condition"] == condition]
            if g.empty:
                continue
            d_vals = g["D_um2_per_s"].to_numpy(dtype=float)
            is_clipped = d_vals > d_upper
            d_clipped = np.clip(d_vals, None, d_upper)
            ax.scatter(d_clipped[~is_clipped], g["exponent"].to_numpy(dtype=float)[~is_clipped], s=16, alpha=0.6,
                       facecolor=style["face"], edgecolor=style["edge"], linewidth=0.6, zorder=3)
            if is_clipped.any():
                ax.scatter(np.full(is_clipped.sum(), d_upper),
                           g["exponent"].to_numpy(dtype=float)[is_clipped], s=20, alpha=0.9,
                           facecolor=style["face"], edgecolor="black", linewidth=0.6, marker=">", zorder=4)
        ax.set_xlim(0.0, d_upper)
        ax.set_ylim(*n_range)
        ax.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm")
        ax.label_outer()

    for ax in ax_flat[:n_panels]:
        ax.set_xlabel(D_LABEL)
        ax.set_ylabel(N_LABEL)
    for ax in ax_flat[:n_panels]:
        ax.label_outer()

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.2, linestyle=_DASH_THEORY, label="n = 1"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STYLE_WATER["face"],
               markeredgecolor=STYLE_WATER["edge"], markersize=4, label="Water"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STYLE_HYDROGEL["face"],
               markeredgecolor=STYLE_HYDROGEL["edge"], markersize=4, label="Hydrogel"),
        Line2D([0], [0], marker=">", color="w", markerfacecolor="#888888", markeredgecolor="black",
               markersize=4, label=rf"$D$ > {d_upper:g} µm$^2$ s$^{{-1}}$ (clipped)"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=4, frameon=False)
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
            patch.set_linewidth(0.8)
        for element in ("whiskers", "caps"):
            for line in bp[element]:
                line.set_color(edge)
                line.set_linewidth(0.8)
    if value_col_is_D:
        ax.set_yscale("log")
        ax.set_ylim(*d_ylim)
        _add_log_minor_ticks(ax)
    else:
        ax.set_ylim(*N_AXIS_RANGE)


def plot_length_bins_one_size(df: pd.DataFrame, size_nm: float, condition: str,
                               dls_labels: dict[float, int], d_ylim: tuple[float, float]) -> plt.Figure:
    face, edge = SIZE_COLORS.get(size_nm, ("#999999", "#555555"))
    sub = df[(df["condition"] == condition) & (df["particle_size_nm"] == size_nm)]
    bin_positions = np.arange(len(LENGTH_BIN_LABELS))

    fig, (ax_d, ax_n) = plt.subplots(2, 1, figsize=(WIDTH_IN["half"], 4.2), constrained_layout=True, sharex=True)
    for ax, value_col, ylabel, is_d in ((ax_d, "D_um2_per_s", D_LABEL, True),
                                          (ax_n, "exponent", N_LABEL, False)):
        groups = [(bin_positions[j], sub.loc[sub["length_bin"] == label, value_col].dropna().to_numpy(), face, edge)
                  for j, label in enumerate(LENGTH_BIN_LABELS)]
        _length_bin_panel(ax, groups, is_d, d_ylim)
        ax.set_ylabel(ylabel)
    ax_d.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm")
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

    fig, (ax_d, ax_n) = plt.subplots(2, 1, figsize=(WIDTH_IN["half"], 4.5), constrained_layout=True, sharex=True)
    for ax, value_col, ylabel, is_d in ((ax_d, "D_um2_per_s", D_LABEL, True),
                                          (ax_n, "exponent", N_LABEL, False)):
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
    ax_d.set_title("Hydrogel")
    ax_n.set_xticks(bin_positions)
    ax_n.set_xticklabels(LENGTH_BIN_LABELS)
    ax_n.set_xlabel("Track length bin (frames)")

    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=face, markeredgecolor=edge, markersize=6,
                       label=f"{dls_labels.get(s, int(s))} nm")
               for s, (face, edge) in sorted(SIZE_COLORS.items()) if s in sizes]
    fig.legend(handles=handles, loc="outside lower center", ncol=max(len(handles), 1), frameon=False)
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


def _correction_group(row: int, size_nm: float) -> str:
    if row == 1:
        return "n"
    return "D_small" if size_nm in CORR_SMALL_SIZES_NM else "D_large"


def _correction_ylims(table: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Fixed axis range per group (user, 2026-10-12); panels of one group share it."""
    return {"n": CORR_N_RANGE, "D_small": CORR_D_RANGE_SMALL, "D_large": CORR_D_RANGE_LARGE}


def plot_correction_before_after(table: pd.DataFrame, dls_labels: dict[float, int]) -> tuple[plt.Figure, pd.DataFrame]:
    """Before/After boxplots per size with fixed axis ranges. Values outside a range stay in the
    box statistics; they are not marked in the figure (user, 2026-10-12) but counted per box in the
    returned table, which main() writes to disk and prints."""
    sizes = sorted(table["particle_size_nm"].dropna().unique())
    fig, axes = plt.subplots(2, len(sizes), figsize=(WIDTH_IN["full"], 4.4), constrained_layout=True, squeeze=False)
    hidden_median = dict(color="none", linewidth=0)
    outside = []
    ylims = _correction_ylims(table)

    for col, size_nm in enumerate(sizes):
        sub = table[table["particle_size_nm"] == size_nm]
        for row, (before_col, after_col, ylabel) in enumerate((
                ("D_before", "D_after", D_LABEL),
                ("n_before", "n_after", N_LABEL))):
            ax = axes[row][col]
            lo, hi = ylims[_correction_group(row, size_nm)]
            positions, box_data, colors, edges, alphas = [], [], [], [], []
            xtick_pos, xtick_labels = [], []
            pos = 0.0
            for condition, style in (("water", STYLE_WATER), ("hydrogel", STYLE_HYDROGEL)):
                g = sub[sub["condition"] == condition]
                if g.empty:
                    continue
                before_vals = g[before_col].dropna().to_numpy()
                after_vals = g[after_col].dropna().to_numpy()
                for vals, stage, alpha, has in ((before_vals, "Before", 0.35, before_vals.size > 0),
                                                (after_vals, "After", 0.85, after_vals.size > 0)):
                    if has:
                        box_data.append(vals); positions.append(pos); colors.append(style["face"])
                        edges.append(style["edge"]); alphas.append(alpha)
                        n_above, n_below = int((vals > hi).sum()), int((vals < lo).sum())
                        outside.append({"particle_size_nm": size_nm, "condition": condition, "stage": stage,
                                        "quantity": "D" if row == 0 else "n", "axis_min": lo, "axis_max": hi,
                                        "n_values": int(vals.size), "n_above_axis": n_above, "n_below_axis": n_below})
                    elif stage == "After" and before_vals.size:
                        ax.text(pos, 0.5, "no σ_loc reference", transform=ax.get_xaxis_transform(),
                                ha="center", va="center", fontsize=8, color="#888888", rotation=90)
                    if has or (stage == "After" and before_vals.size):
                        xtick_pos.append(pos); xtick_labels.append(stage)
                    pos += 1.0 if stage == "Before" else 1.3
            if box_data:
                bp = ax.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True, showfliers=False,
                                 medianprops=hidden_median)
                for patch, color, edge, alpha in zip(bp["boxes"], colors, edges, alphas):
                    patch.set_facecolor(color)
                    patch.set_alpha(alpha)
                    patch.set_edgecolor(edge)
                    patch.set_linewidth(0.8)
                for element in ("whiskers", "caps"):
                    for i, line in enumerate(bp[element]):
                        line.set_color(edges[i // 2])
                        line.set_linewidth(0.8)
            ax.set_ylim(lo, hi)
            ax.set_xticks(xtick_pos)
            ax.set_xticklabels(xtick_labels, rotation=90)
            if row == 0:
                ax.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm")
            if col == 0:
                ax.set_ylabel(ylabel)

    handles = [Patch(facecolor=STYLE_WATER["face"], edgecolor=STYLE_WATER["edge"], alpha=0.6, label="Water"),
               Patch(facecolor=STYLE_HYDROGEL["face"], edgecolor=STYLE_HYDROGEL["edge"], alpha=0.6, label="Hydrogel"),
               Patch(facecolor="#888888", alpha=0.35, label="Without log-error correction"),
               Patch(facecolor="#888888", alpha=0.85, label="After correction")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    return fig, pd.DataFrame(outside)


def _save(fig: plt.Figure, stem: str) -> None:
    """Style guide v2: plots as PDF; PNG (600 dpi) kept as before. Never bbox_inches='tight'."""
    for ext in ("pdf", "png"):
        path = SAVE_PATH / f"{stem}.{ext}"
        safe_savefig(fig, path, format=ext, dpi=600)
        print(f"Plot gespeichert: {path}")
    plt.close(fig)


def main() -> None:
    d0_results = _load_pickle(CACHE_D0, "MSD_FromTrackmate_D0.py")
    hydrogel_results = _load_pickle(CACHE_20MG, "MSD_FromTrackmate_20mg.py")
    locerr_df = _load_pickle(CACHE_LOCERR, "Loc_Error_Analyse_immob_particle.py")
    dls_labels = get_dls_labels()
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    with plt.style.context(str(STYLE_FILE)):
        # Item 7: main (0-15 um^2/s) + zoomed (0-4 um^2/s, for the hydrogel cluster)
        file_standard = build_file_standard_table(d0_results, hydrogel_results)
        for d_upper, suffix in ((D_AXIS_UPPER_MAIN, ""), (D_AXIS_UPPER_ZOOM, "_zoom_0to4")):
            fig7, diag7 = plot_d_vs_n_refined(file_standard, dls_labels, d_upper, N_AXIS_RANGE_D_VS_N)
            print(f"D vs n{suffix}: linear D-range = (0, {diag7['d_upper_um2_s']:g}) um^2/s, "
                  f"n-range = {N_AXIS_RANGE_D_VS_N}. {diag7['n_out_of_range']} file(s) above the upper "
                  "bound -- marked with a triangle, not dropped.")
            _save(fig7, f"C_D_vs_n_scatter_refined{suffix}")

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
            _save(fig8, f"B_length_bin_water_{dls_labels.get(size_nm, int(size_nm))}nm")
        fig8_gel = plot_length_bins_hydrogel(length_bin_table, dls_labels, d_ylim)
        _save(fig8_gel, "B_length_bin_hydrogel")

        # Item 9
        correction_table = build_correction_table_all_sizes(d0_results, hydrogel_results, locerr_df)
        n_missing = int(correction_table.groupby("particle_size_nm")["has_reference"].apply(lambda s: not s.any()).sum())
        print(f"Correction deviation: {correction_table['particle_size_nm'].nunique()} particle size(s) considered, "
              f"{n_missing} without a sigma_loc reference (Before shown, After explicitly marked missing).")
        fig9, outside9 = plot_correction_before_after(correction_table, dls_labels)
        print(f"Correction deviation axis ranges: {_correction_ylims(correction_table)}")
        _save(fig9, "A_correction_deviation_refined")
        outside_path = SAVE_PATH / "A_correction_deviation_refined_values_outside_axes.csv"
        outside9.to_csv(outside_path, index=False)
        shown = outside9[(outside9["n_above_axis"] > 0) | (outside9["n_below_axis"] > 0)]
        print(f"Correction deviation: values outside the fixed axis ranges (kept in the box statistics, "
              f"not marked in the figure): {len(shown)} of {len(outside9)} boxes, table: {outside_path}")
        if not shown.empty:
            print(shown.to_string(index=False))


if __name__ == "__main__":
    main()
