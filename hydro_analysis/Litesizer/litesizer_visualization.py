"""
DLS particle-size reference figure (Anton Paar LiteSizer 500) for the
fluorescent polystyrene particles that are subsequently analysed by SPT.

The DLS data are a reference characterisation, not a main result, so the
output is one compact two-panel figure:

  A  Intensity-weighted size distributions of every accepted replicate
     measurement, each max-normalised and drawn as its own curve in the
     colour of its nominal size (no averaging, no SD band). Dotted vertical
     lines mark the nominal manufacturer diameter.
  B  DLS hydrodynamic diameter d_H vs. nominal diameter: every accepted
     replicate as a small semi-transparent point, plus the mean +-SD
     between replicates, with a grey dashed 1:1 reference line. No
     regression is fitted.

d_H is the Anton Paar-reported "Hydrodynamic diameter" of each measurement
(cumulant z-average, MeasurementData.hydrodynamic_diameter_nm as parsed by
litesizer_parser.py). It is never re-derived from the plotted intensity
distribution; the distribution data are used for panel A only.

Measurements with PDI > MAX_PDI (42 %) are excluded before any statistics
are computed; every exclusion and the remaining count per size are printed.

Legend labels follow the dissertation-wide particle labels
(core.io.get_dls_labels(), e.g. nominal 20 nm -> "35 nm"), read from
Litesizer/cache/dls_reference.pkl (written by litesizer_measurements_mean.py);
if that cache is missing, the nominal size is used as label. Size colours
are SIZE_COLORS from MSD_Trackmate/Validation_Claude/Correlations.py
(Styleguide_Figures_Dissertation.md §11, "Partikelgroessen").

Styling follows Styleguide_Figures_Dissertation.md (v2): plotting happens
inside plt.style.context(hydro_analysis/thesis.mplstyle); the figure is
created in its final printed size, width class "full" (6.30 in, embedded
with \\includegraphics[width=\\linewidth], never rescaled), and saved as
vector PDF without bbox_inches="tight", so font sizes in the thesis are
exactly those set here.

Reads the LiteSizer XLSX directly (no cache of its own, nothing needs to be
run first). Shows the figure first (plt.show(), blocking); only after the
window is closed are the outputs written into SAVE_PATH
(Auswertungsbilder\\DLS_particle_size_reference\\):
  dls_particle_size_reference.pdf
  dls_particle_size_reference_statistics.csv
No other script reads these outputs.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.Litesizer.litesizer_parser import LiteSizerData, MeasurementData, load_litesizer_xlsx
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _DASH_THEORY, _add_log_minor_ticks
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS

# ── Configuration ──────────────────────────────────────────────────────────────
XLSX_PATH  = Path(r"H:\Daten Promotion Sicherung\Lite Sizer Particle Measurements\Size_repitition_All Sizes.xlsx")
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "DLS_particle_size_reference"
FIG_STEM   = "dls_particle_size_reference"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"

SIZES   = [20, 50, 100, 200, 500, 1000]   # nominal manufacturer diameters (nm)
MAX_PDI = 42.0                            # measurements with PDI > this (%) are excluded

SUPPORT_EPS  = 1e-3        # normalised intensity below this is treated as zero (masking)

POINT_ALPHA     = 0.5      # single measurements in panel B (style guide §7, §13)
CURVE_ALPHA     = 0.8      # overlapping replicate curves in panel A
REFERENCE_ALPHA = 0.7      # nominal-diameter lines: reference, thinner and transparent (§12)
COLOR_REFERENCE = "#888888"

# Figure geometry in inches, width class "full" (style guide §2). Both axes share the
# same height; panel B is square (identical log limits on x and y), panel A takes the
# remaining width (~1.4 x B). The height deviates from 1.42:1, as allowed for multi-panel
# full-width figures. PANEL_GAP_IN holds panel B's y tick labels and y label plus free
# space, so that label sits visibly closer to its own axes than to panel A. All margins
# must contain every label, because the PDF is saved without bbox_inches="tight".
FIG_WIDTH_IN     = 6.30
AX_HEIGHT_IN     = 2.05
MARGIN_LEFT_IN   = 0.50
MARGIN_RIGHT_IN  = 0.06
MARGIN_BOTTOM_IN = 0.42
MARGIN_TOP_IN    = 0.30    # room for the shared legend above both panels
PANEL_GAP_IN     = 0.80

A_X_LIM      = (10.0, 3000.0)             # diameter range shown in panel A (nm)
B_AXIS_LIM   = (12.0, 2500.0)             # identical x and y limits for panel B
B_TICKS      = [20, 50, 100, 200, 500, 1000]

# Per-size colours: the dissertation-wide (face, edge) mapping from
# Validation_Claude/Correlations.py (1000 nm blue ... 20 nm orange), shared with the
# Figures_Refined scripts. Face colours fill SD bands and markers; edge (dark) colours
# draw lines, error bars and indicators, because the light faces are not legible as
# thin lines on white.
SIZE_COLOR_FALLBACK = ("#999999", "#555555")

CSV_COLUMNS = [
    "nominal_size_nm",
    "dls_diameter_mean_nm",
    "dls_diameter_sd_nm",
    "dls_diameter_sem_nm",
    "diffusion_mean_um2s",
    "diffusion_sd_um2s",
    "pdi_mean_pct",
    "pdi_sd_pct",
    "n_measurements",
]


def calculate_statistics(values: List[float]) -> Tuple[float, float, float]:
    """Calculate mean, standard deviation, and standard error.

    Returns
    -------
    mean, std, sem : tuple of floats
    """
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # Sample std
    sem = std / np.sqrt(len(arr))  # Standard error of mean
    return mean, std, sem


# ── Grouping and exclusion ─────────────────────────────────────────────────────

_NOMINAL_RE = re.compile(r"^\s*(\d+)\s*nm\b")


def _nominal_size_from_name(name: str) -> Optional[int]:
    """Return the nominal size encoded at the start of a measurement name ("500 nm 2" -> 500)."""
    match = _NOMINAL_RE.match(name)
    return int(match.group(1)) if match else None


def group_accepted_measurements(
    data: LiteSizerData,
    sizes: List[int],
    max_pdi: float = MAX_PDI,
) -> Dict[int, List[MeasurementData]]:
    """Group measurements by nominal size and apply the PDI exclusion.

    A measurement is excluded when its Anton Paar polydispersity exceeds
    max_pdi. A PDI this high means the cumulant analysis does not describe
    a single narrow population (aggregates, dust or multimodality), so its
    z-average d_H is not a meaningful reference diameter. Measurements
    without a reported d_H are excluded as well. Every exclusion is printed
    with its reason; sizes without any (accepted) measurement are reported
    and left out, never substituted.

    Returns
    -------
    dict
        {nominal_nm: [accepted MeasurementData, ...]} for sizes with at
        least one accepted measurement, in the order of `sizes`.
    """
    found: Dict[int, List[MeasurementData]] = {s: [] for s in sizes}
    for m in data.measurements:
        nominal = _nominal_size_from_name(m.name)
        if nominal in found:
            found[nominal].append(m)
        else:
            print(f"  [IGNORED] '{m.name}': nominal size not in SIZES {sizes}")

    print(f"\nMeasurement exclusion (criterion: PDI > {max_pdi:.0f} %)")
    print("-" * 70)
    accepted: Dict[int, List[MeasurementData]] = {}
    for size in sizes:
        kept = []
        for m in found[size]:
            if m.hydrodynamic_diameter_nm is None:
                print(f"  [EXCLUDED] {size:>4} nm  '{m.name}': no hydrodynamic diameter reported")
            elif m.polydispersity_pct is not None and m.polydispersity_pct > max_pdi:
                print(f"  [EXCLUDED] {size:>4} nm  '{m.name}': PDI = {m.polydispersity_pct:.1f} % > {max_pdi:.0f} %"
                      f"  (d_H = {m.hydrodynamic_diameter_nm:.1f} nm)")
            else:
                kept.append(m)
        if kept:
            accepted[size] = kept

    print("\nAccepted measurements per nominal size")
    print("-" * 70)
    for size in sizes:
        n_total = len(found[size])
        n_kept = len(accepted.get(size, []))
        if n_total == 0:
            note = "  <-- MISSING: no measurement with this nominal size in the XLSX"
        elif n_kept == 0:
            note = "  <-- NO ACCEPTED MEASUREMENT: size omitted from figure and CSV"
        elif n_kept == 1:
            note = "  <-- only one measurement: SD/SEM undefined"
        else:
            note = ""
        print(f"  {size:>4} nm: {n_kept} of {n_total} accepted ({n_total - n_kept} excluded){note}")

    if not accepted:
        raise ValueError("No accepted DLS measurement for any configured particle size.")
    return accepted


# ── Statistics ─────────────────────────────────────────────────────────────────

def summarize_size_statistics(groups: Dict[int, List[MeasurementData]]) -> pd.DataFrame:
    """Per-size statistics over the accepted replicate measurements.

    d_H is the Anton Paar cumulant z-average of each measurement. The SD is
    the sample standard deviation (ddof = 1) between independent replicate
    measurements, i.e. the measurement-to-measurement variation shown as
    error bar in panel B; SEM = SD / sqrt(n) is exported for completeness.
    SD and SEM are NaN when only one measurement was accepted.
    """
    rows = []
    for size, ms in groups.items():
        d_h = [m.hydrodynamic_diameter_nm for m in ms]
        diff = [m.diffusion_coefficient_um2s for m in ms if m.diffusion_coefficient_um2s is not None]
        pdi = [m.polydispersity_pct for m in ms if m.polydispersity_pct is not None]
        n = len(d_h)

        d_mean, d_sd, d_sem = calculate_statistics(d_h) if n > 1 else (float(d_h[0]), np.nan, np.nan)
        diff_mean, diff_sd, _ = calculate_statistics(diff) if len(diff) > 1 else (
            float(diff[0]) if diff else np.nan, np.nan, np.nan)
        pdi_mean, pdi_sd, _ = calculate_statistics(pdi) if len(pdi) > 1 else (
            float(pdi[0]) if pdi else np.nan, np.nan, np.nan)

        rows.append({
            "nominal_size_nm": size,
            "dls_diameter_mean_nm": d_mean,
            "dls_diameter_sd_nm": d_sd,
            "dls_diameter_sem_nm": d_sem,
            "diffusion_mean_um2s": diff_mean,
            "diffusion_sd_um2s": diff_sd,
            "pdi_mean_pct": pdi_mean,
            "pdi_sd_pct": pdi_sd,
            "n_measurements": n,
        })
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def prepare_normalized_replicates(
    groups: Dict[int, List[MeasurementData]],
) -> Dict[int, List[Tuple[np.ndarray, np.ndarray]]]:
    """Max-normalised intensity distribution of every accepted replicate.

    Normalisation: each intensity-weighted distribution is divided by its own
    maximum so that its peak equals 1. This is purely a graphical
    normalisation for comparing shapes and positions of populations with very
    different scattering intensities; the diameter values are the measured
    LiteSizer grid, unchanged. Replicates are not averaged or interpolated.

    Returns
    -------
    dict
        {nominal_nm: [(diameter_nm, normalised_intensity), ...]}, one entry
        per replicate measurement.
    """
    curves: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for size, ms in groups.items():
        for m in ms:
            df = m.size_distribution_intensity
            if df.empty:
                print(f"  [NO DISTRIBUTION] {size} nm '{m.name}': intensity distribution missing, "
                      "skipped in panel A only")
                continue
            d = df["diameter_nm"].to_numpy(dtype=float)
            y = df["frequency_pct"].to_numpy(dtype=float)
            valid = d > 0
            d, y = d[valid], y[valid]
            if y.max() <= 0:
                print(f"  [NO DISTRIBUTION] {size} nm '{m.name}': intensity distribution is all zero, "
                      "skipped in panel A only")
                continue
            order = np.argsort(d)
            curves.setdefault(size, []).append((d[order], y[order] / y.max()))
    return curves


# ── Plotting ───────────────────────────────────────────────────────────────────

def _size_colors(size: int) -> Tuple[str, str]:
    """(face, edge) colour of a nominal size, as in all other dissertation figures."""
    return SIZE_COLORS.get(float(size), SIZE_COLOR_FALLBACK)


def _create_panel_axes() -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Figure with two axes of identical height at fixed inch positions.

    The constrained layout of thesis.mplstyle is switched off for this figure:
    with panel B's equal aspect it shrinks B vertically, so the two panels
    would end up with different heights.
    """
    width_b = AX_HEIGHT_IN
    width_a = FIG_WIDTH_IN - MARGIN_LEFT_IN - MARGIN_RIGHT_IN - PANEL_GAP_IN - width_b
    fig_height = MARGIN_BOTTOM_IN + AX_HEIGHT_IN + MARGIN_TOP_IN
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height), layout="none")

    def rect(left_in: float, width_in: float) -> List[float]:
        return [left_in / FIG_WIDTH_IN, MARGIN_BOTTOM_IN / fig_height,
                width_in / FIG_WIDTH_IN, AX_HEIGHT_IN / fig_height]

    ax_a = fig.add_axes(rect(MARGIN_LEFT_IN, width_a))
    ax_b = fig.add_axes(rect(MARGIN_LEFT_IN + width_a + PANEL_GAP_IN, width_b))
    return fig, ax_a, ax_b


def _particle_labels(sizes: List[int]) -> Dict[int, str]:
    """Dissertation-wide legend labels ("35 nm" for nominal 20 nm, ...), nominal as fallback."""
    try:
        labels = get_dls_labels()
    except FileNotFoundError:
        print("  [NOTE] DLS label cache missing; legend uses nominal sizes.")
        labels = {}
    return {s: f"{labels.get(float(s), s)} nm" for s in sizes}


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="left", va="top")


def plot_normalized_dls_distributions(
    ax: plt.Axes,
    replicates: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Panel A: one max-normalised intensity distribution per replicate measurement.

    All replicates of a nominal size share its colour and line style, so
    replicate-to-replicate agreement is visible directly. A dotted line marks
    each nominal manufacturer diameter.
    """
    for size, reps in replicates.items():
        _, dark = _size_colors(size)
        for d, y in reps:
            # Plot only where the distribution is non-zero (plus one grid point on each side),
            # so the curves do not merge into a thick coloured baseline at y = 0.
            nonzero = y > SUPPORT_EPS
            shown = nonzero | np.roll(nonzero, 1) | np.roll(nonzero, -1)
            ax.plot(d, np.where(shown, y, np.nan), color=dark, linewidth=1.0, alpha=CURVE_ALPHA, zorder=3)
        ax.axvline(size, color=dark, linestyle=(0, (1, 2)), linewidth=0.8, alpha=REFERENCE_ALPHA, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(*A_X_LIM)
    ax.set_ylim(0.0, 1.08)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(mticker.LogLocator(subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=100))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.set_xlabel(r"Hydrodynamic diameter $d_\mathrm{H}$ (nm)")
    ax.set_ylabel("Normalized intensity (a.u.)")


def plot_nominal_vs_dls(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    groups: Dict[int, List[MeasurementData]],
) -> None:
    """Panel B: DLS d_H of every replicate and their mean +-SD vs. nominal diameter.

    Individual replicates are drawn exactly at the nominal x value, small and
    semi-transparent, so coinciding repeats read as a darker spot; the mean
    +-SD (sample SD between replicates) is drawn on top. The grey dashed line
    is the identity d_H = nominal, a reference only; no regression is fitted.
    Both axes are logarithmic with identical limits, ticks and aspect, so
    vertical distance to the 1:1 line reads as the same relative deviation
    for every size (a linear axis would compress the 20-100 nm particles
    into one corner).
    """
    lo, hi = B_AXIS_LIM
    ax.plot([lo, hi], [lo, hi], color=COLOR_REFERENCE, linestyle=_DASH_THEORY, linewidth=1.2, zorder=1)

    for row in summary_df.itertuples(index=False):
        size = int(row.nominal_size_nm)
        base, dark = _size_colors(size)
        d_h = [m.hydrodynamic_diameter_nm for m in groups[size]]
        ax.scatter(np.full(len(d_h), size), d_h, s=9, marker="o", alpha=POINT_ALPHA,
                   facecolor=base, edgecolor=dark, linewidth=0.6, zorder=2)
        yerr = None if np.isnan(row.dls_diameter_sd_nm) else row.dls_diameter_sd_nm
        ax.errorbar(size, row.dls_diameter_mean_nm, yerr=yerr,
                    fmt="o", markersize=4, markerfacecolor=base, markeredgecolor=dark,
                    markeredgewidth=0.6, ecolor=dark, elinewidth=0.8, capsize=2.0, capthick=0.8,
                    linestyle="None", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    _add_log_minor_ticks(ax)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.FixedLocator(B_TICKS))
        axis.set_major_formatter(mticker.FixedFormatter([str(t) for t in B_TICKS]))
    ax.set_xlabel("Nominal diameter (nm)")
    ax.set_ylabel(r"DLS diameter $d_\mathrm{H}$ (nm)")

    grey_face, grey_edge = "#bbbbbb", "#555555"
    legend_handles = [
        Line2D([], [], marker="o", markersize=3, linestyle="None", markerfacecolor=grey_face,
               markeredgecolor=grey_edge, markeredgewidth=0.6, alpha=POINT_ALPHA),
        ax.errorbar([], [], yerr=[], fmt="o", markersize=4, markerfacecolor=grey_face,
                    markeredgecolor=grey_edge, markeredgewidth=0.6, ecolor=grey_edge,
                    elinewidth=0.8, capsize=2.0, capthick=0.8, linestyle="None"),
        Line2D([], [], color=COLOR_REFERENCE, linestyle=_DASH_THEORY, linewidth=1.2),
    ]
    ax.legend(legend_handles, ["Single measurement", "Mean ± SD", r"$d_\mathrm{H}$ = nominal"],
              loc="lower right", frameon=False, handlelength=2.0, borderaxespad=0.3)


def plot_dls_reference_figure(
    replicates: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    summary_df: pd.DataFrame,
    groups: Dict[int, List[MeasurementData]],
) -> plt.Figure:
    """Build the two-panel DLS reference figure; call inside plt.style.context(STYLE_PATH).

    The legend is placed above both panels because the size colours encode
    the same populations in A and B; the nominal-diameter indicator of panel A
    is explained once in the same legend.
    """
    sizes = list(summary_df["nominal_size_nm"])
    labels = _particle_labels(sizes)

    fig, ax_a, ax_b = _create_panel_axes()
    plot_normalized_dls_distributions(ax_a, replicates)
    plot_nominal_vs_dls(ax_b, summary_df, groups)
    _panel_label(ax_a, "A")
    _panel_label(ax_b, "B")

    handles = [Line2D([], [], color=_size_colors(s)[1], linewidth=1.2) for s in sizes]
    texts = [labels[s] for s in sizes]
    handles.append(Line2D([], [], color="#555555", linestyle=(0, (1, 2)), linewidth=0.8))
    texts.append("Nominal diameter")
    # Anchored just above the common top edge of both axes, centred over their span.
    x0, x1 = ax_a.get_position().x0, ax_b.get_position().x1
    fig.legend(handles, texts, loc="lower center", bbox_to_anchor=((x0 + x1) / 2, ax_a.get_position().y1 + 0.005),
               ncol=len(handles), frameon=False, handlelength=1.5, columnspacing=1.0, handletextpad=0.4,
               borderaxespad=0.0, borderpad=0.2)
    return fig


def print_panel_b_values(summary_df: pd.DataFrame, groups: Dict[int, List[MeasurementData]]) -> None:
    """Print the values plotted in panel B together with the individual d_H replicates."""
    print("\nPanel B values (Anton Paar hydrodynamic diameter, accepted measurements)")
    print("-" * 70)
    print(f"  {'nominal':>7}  {'d_H mean':>9}  {'SD':>7}  {'SEM':>7}  {'n':>2}   replicates d_H (nm)")
    for row in summary_df.itertuples(index=False):
        size = int(row.nominal_size_nm)
        reps = ", ".join(f"{m.hydrodynamic_diameter_nm:.2f}" for m in groups[size])
        print(f"  {size:>4} nm  {row.dls_diameter_mean_nm:>9.2f}  {row.dls_diameter_sd_nm:>7.2f}  "
              f"{row.dls_diameter_sem_nm:>7.2f}  {row.n_measurements:>2}   {reps}")
    print("-" * 70)


def main():
    """Build the DLS reference figure and statistics CSV."""
    print(f"Loading data from: {XLSX_PATH}")
    data = load_litesizer_xlsx(XLSX_PATH)

    groups = group_accepted_measurements(data, SIZES, MAX_PDI)
    summary_df = summarize_size_statistics(groups)
    replicates = prepare_normalized_replicates(groups)
    print_panel_b_values(summary_df, groups)

    with plt.style.context(STYLE_PATH):
        fig = plot_dls_reference_figure(replicates, summary_df, groups)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            pdf_path = SAVE_PATH / f"{FIG_STEM}.pdf"
            fig.savefig(pdf_path, format="pdf")   # final size, no bbox_inches="tight" (style guide §3)
            csv_path = SAVE_PATH / f"{FIG_STEM}_statistics.csv"
            summary_df.to_csv(csv_path, index=False)
            print(f"Plot saved: {pdf_path}")
            print(f"Statistics saved: {csv_path}")

    return data, summary_df


if __name__ == "__main__":
    main()
