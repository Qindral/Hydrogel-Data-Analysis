"""
SEM vs. DLS particle size distributions -- mean Feret diameter from SEM
compared with the number-weighted LiteSizer (DLS) size distributions, one
panel per nominal particle size (2 x 3 panels, width class "full").

SEM: number-weighted by construction (every particle counts once); size
measure d_F = (Feret_max + Feret_min) / 2 of the dry particle. Read from
hydro_analysis/SEM_Particles/cache/sem_feret_particles.pkl (written by
sem_feret_measure.py, which must be run first); only particles with
used == True (border and fragment exclusions applied upstream). The bin
ranges (COMPARISON_BINS) are wider than those of the single-size histograms
in sem_feret_size_histogram.py, so that the full DLS distributions are shown.

DLS: the number-weighted size distribution of every accepted LiteSizer
replicate, read directly from the LiteSizer XLSX with the parser and the PDI
exclusion (PDI > 42 %) of Litesizer/litesizer_visualization.py. The
number-weighted distribution is the DLS weighting comparable to an SEM count;
it is derived by the instrument from the intensity distribution (Mie
conversion) and describes the hydrodynamic diameter in water, so a systematic
offset to the dry SEM diameter is expected. The intensity-based z-average d_H
(the dissertation-wide particle label, e.g. "35 nm") is reported in the
statistics table for reference.

Figure normalisation: the SEM histogram and every DLS replicate are each
divided by their own maximum, so peak positions and widths are compared on a
common 0-1 scale; the diameter values themselves are unchanged. Statistics
are computed from the unnormalised data: SEM mean, SD, median of the pooled
particles; DLS mean and SD of each replicate's number distribution
(frequency-weighted over the LiteSizer grid), then averaged over replicates.

Writes into SAVE_PATH:
  sem_vs_dls_size_distribution.pdf     2 x 3 panel figure
  sem_vs_dls_size_statistics.xlsx      per-size SEM and DLS statistics
No other script reads these outputs.

Styling follows Styleguide_Figures_Dissertation.md (v2): plotting inside
plt.style.context(hydro_analysis/thesis.mplstyle), figure created in its final
printed size and saved as vector PDF without bbox_inches="tight". SEM data in
the size colours (SIZE_COLORS), DLS in the DLS reference colour (§11).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from hydro_analysis.Litesizer.litesizer_parser import MeasurementData, load_litesizer_xlsx
from hydro_analysis.Litesizer.litesizer_visualization import MAX_PDI, XLSX_PATH, group_accepted_measurements
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS
from hydro_analysis.SEM_Particles.sem_feret_size_histogram import load_particles, size_label

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data"
) / "SEM_particle_size"
FIG_STEM = "sem_vs_dls_size_distribution"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
SHOW_FIGURE = True

SIZES = [20, 50, 100, 200, 500, 1000]
FIG_SIZE_IN = (6.30, 4.20)             # width class "full"; multi-panel height (style guide §2)
COLOR_DLS = "#da00bd"                  # DLS reference (style guide §11)
DLS_ALPHA = 0.8                        # overlapping replicate curves
SEM_ALPHA = 0.6                        # histogram under the DLS curves (style guide §8)

# nominal size -> (lower edge, upper edge, bin width) in nm; 40 bins each (style guide §8).
COMPARISON_BINS = {
    20: (10.0, 50.0, 1.0),
    50: (10.0, 90.0, 2.0),
    100: (40.0, 160.0, 3.0),
    200: (60.0, 380.0, 8.0),
    500: (150.0, 750.0, 15.0),
    1000: (400.0, 1600.0, 30.0),
}


# ── Statistics ─────────────────────────────────────────────────────────────────

def dls_number_distribution(m: MeasurementData) -> tuple[np.ndarray, np.ndarray]:
    """Diameter grid (nm) and number-weighted frequency (%) of one LiteSizer measurement."""
    df = m.size_distribution_number
    d = df["diameter_nm"].to_numpy(dtype=float)
    f = df["frequency_pct"].to_numpy(dtype=float)
    valid = d > 0
    return d[valid], f[valid]


def weighted_mean_sd(d: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    mean = float(np.sum(f * d) / np.sum(f))
    return mean, float(np.sqrt(np.sum(f * (d - mean) ** 2) / np.sum(f)))


def compare_statistics(particles: pd.DataFrame, dls: dict[int, list[MeasurementData]]) -> pd.DataFrame:
    """One row per size: SEM distribution statistics next to DLS number-distribution statistics."""
    rows = []
    for nominal in SIZES:
        d = particles.loc[particles["nominal_nm"] == nominal, "feret_mean_nm"].to_numpy()
        reps = [weighted_mean_sd(*dls_number_distribution(m)) for m in dls.get(nominal, [])]
        rep_means = np.array([r[0] for r in reps])
        rep_sds = np.array([r[1] for r in reps])
        d_h = np.array([m.hydrodynamic_diameter_nm for m in dls.get(nominal, [])])
        row = {
            "nominal_nm": nominal,
            "label": size_label(nominal),
            "sem_n_images": particles.loc[particles["nominal_nm"] == nominal, "image"].nunique(),
            "sem_n_particles": d.size,
            "sem_mean_nm": d.mean(),
            "sem_sd_nm": d.std(ddof=1),
            "sem_cv_pct": 100.0 * d.std(ddof=1) / d.mean(),
            "sem_median_nm": np.median(d),
            "dls_n_measurements": len(reps),
            "dls_number_mean_nm": rep_means.mean() if reps else np.nan,
            "dls_number_mean_sd_between_reps_nm": rep_means.std(ddof=1) if len(reps) > 1 else np.nan,
            "dls_number_sd_nm": rep_sds.mean() if reps else np.nan,
            "dls_number_cv_pct": 100.0 * rep_sds.mean() / rep_means.mean() if reps else np.nan,
            "dls_z_average_nm": d_h.mean() if reps else np.nan,
            "dls_z_average_sd_between_reps_nm": d_h.std(ddof=1) if len(reps) > 1 else np.nan,
        }
        row["ratio_sem_to_dls_number"] = row["sem_mean_nm"] / row["dls_number_mean_nm"]
        row["ratio_sem_to_dls_z_average"] = row["sem_mean_nm"] / row["dls_z_average_nm"]
        rows.append(row)
    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_panel(ax: plt.Axes, nominal: int, d_sem: np.ndarray, dls_reps: list[MeasurementData],
               stats: pd.Series) -> None:
    """SEM histogram and DLS number-weighted replicate curves of one size, each max-normalised."""
    base, dark = SIZE_COLORS[float(nominal)]
    lo, hi, width = COMPARISON_BINS[nominal]
    edges = np.arange(lo, hi + 0.5 * width, width)
    counts, _ = np.histogram(d_sem, bins=edges)
    counts = counts / counts.max()
    ax.stairs(counts, edges, fill=True, color=base, alpha=SEM_ALPHA, zorder=1)
    ax.stairs(counts, edges, color=dark, linewidth=0.5, zorder=2)

    for m in dls_reps:
        d, f = dls_number_distribution(m)
        ax.plot(d, f / f.max(), color=COLOR_DLS, linewidth=1.0, alpha=DLS_ALPHA, zorder=3)

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0.0, 1.45)                         # headroom for the statistics text
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.set_title(size_label(nominal))
    ax.text(0.97, 0.96,
            f"SEM {stats.sem_mean_nm:.0f} ± {stats.sem_sd_nm:.0f} nm\n"
            f"DLS {stats.dls_number_mean_nm:.0f} ± {stats.dls_number_sd_nm:.0f} nm",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)


def plot_comparison(particles: pd.DataFrame, dls: dict[int, list[MeasurementData]],
                    stats: pd.DataFrame) -> plt.Figure:
    """2 x 3 panel figure; call inside plt.style.context(STYLE_PATH)."""
    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE_IN, sharey=True)
    for ax, nominal in zip(axes.flat, SIZES):
        d_sem = particles.loc[particles["nominal_nm"] == nominal, "feret_mean_nm"].to_numpy()
        plot_panel(ax, nominal, d_sem, dls.get(nominal, []), stats.set_index("nominal_nm").loc[nominal])
    for ax in axes[1]:
        ax.set_xlabel("Diameter (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized frequency")

    handles = [Patch(facecolor="#bbbbbb", edgecolor="#555555", linewidth=0.5),
               Line2D([], [], color=COLOR_DLS, linewidth=1.0)]
    fig.legend(handles, [r"SEM, mean Feret diameter $d_\mathrm{F}$", "DLS, number-weighted (per replicate)"],
               loc="outside upper center", ncol=2, frameon=False)
    return fig


def main(show: bool = SHOW_FIGURE):
    """Build the SEM vs. DLS comparison figure and statistics workbook."""
    particles = load_particles()
    dls = group_accepted_measurements(load_litesizer_xlsx(XLSX_PATH), SIZES, MAX_PDI)
    stats = compare_statistics(particles, dls)

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.precision", 2):
        print()
        print(stats[["label", "sem_n_particles", "sem_mean_nm", "sem_sd_nm", "sem_median_nm",
                     "dls_n_measurements", "dls_number_mean_nm", "dls_number_sd_nm", "dls_z_average_nm",
                     "ratio_sem_to_dls_number", "ratio_sem_to_dls_z_average"]].to_string(index=False))

    with plt.style.context(STYLE_PATH):
        fig = plot_comparison(particles, dls, stats)
        if show:
            plt.show()
        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            fig.savefig(SAVE_PATH / f"{FIG_STEM}.pdf", format="pdf")
            stats.to_excel(SAVE_PATH / "sem_vs_dls_size_statistics.xlsx", index=False)
            print(f"Saved: {SAVE_PATH / FIG_STEM}.pdf and sem_vs_dls_size_statistics.xlsx")
    return stats


if __name__ == "__main__":
    main()
