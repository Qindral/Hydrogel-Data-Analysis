"""
SEM vs. DLS particle size distributions -- SEM Feret diameter histogram with
the LiteSizer (DLS) intensity-weighted size distribution, one panel per
nominal particle size (2 x 3 panels, width class "full").

SEM: Feret diameter (Feret_max, largest caliper width, ImageJ "Feret") of
every used particle, number-weighted by construction; relative-frequency
histogram with N_BINS bins (from sem_feret_size_histogram.py) over
X_RANGES. X_RANGES are wider than the ranges of the single-size histograms:
they cover the DLS curve down to about 5 % of its peak (the far tail of the
20 nm sample near 17 um, dust, is left out) together with the SEM data. Read
from hydro_analysis/SEM_Particles/cache/sem_feret_particles.pkl (written by
sem_feret_measure.py, which must be run first); border and fragment
exclusions are applied upstream.

DLS: the data of Litesizer/litesizer_visualization.py, selected, normalised
and summarised with that script's own functions, so no DLS result differs
from it: accepted replicates (PDI <= 42 %, group_accepted_measurements),
intensity-weighted distributions divided by their own maximum
(prepare_normalized_replicates), and the d_H mean +- SD between replicates
(summarize_size_statistics, Anton Paar z-average). For the figure, the
normalised replicates of one size are summed on the common LiteSizer grid
into one curve (combined_dls_curves) and scaled so that its peak equals the
highest histogram bar -- a graphical normalisation only. Intensity weighting
emphasises large particles and d_H is the hydrodynamic diameter in water, so
the DLS curve is expected to lie above the dry, number-weighted SEM data.

Writes into SAVE_PATH:
  sem_vs_dls_size_distribution.pdf     2 x 3 panel figure
  sem_vs_dls_size_statistics.xlsx      per-size SEM statistics next to the
                                       unchanged LiteSizer statistics
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

from hydro_analysis.Litesizer.litesizer_parser import load_litesizer_xlsx
from hydro_analysis.Litesizer.litesizer_visualization import (
    MAX_PDI, SUPPORT_EPS, XLSX_PATH, group_accepted_measurements, prepare_normalized_replicates,
    summarize_size_statistics,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS
from hydro_analysis.SEM_Particles.sem_feret_size_histogram import N_BINS, load_particles, size_label

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data"
) / "SEM_particle_size"
FIG_STEM = "sem_vs_dls_size_distribution"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
SHOW_FIGURE = True

SIZES = [20, 50, 100, 200, 500, 1000]
SEM_COLUMN = "feret_max_nm"            # Feret diameter
FIG_SIZE_IN = (6.30, 4.20)             # width class "full"; multi-panel height (style guide §2)
COLOR_DLS = "#da00bd"                  # DLS reference (style guide §11)
Y_HEADROOM = 1.45                      # y limit / highest bar, leaves room for the statistics text

X_RANGES = {                           # nominal size -> (lower, upper) x limit and histogram edge in nm
    20: (10.0, 90.0),
    50: (15.0, 115.0),
    100: (50.0, 150.0),
    200: (100.0, 400.0),
    500: (250.0, 750.0),
    1000: (600.0, 2000.0),
}


# ── DLS reference curve ────────────────────────────────────────────────────────

def combined_dls_curves(sizes: list[int]) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict]:
    """One intensity-weighted DLS curve per size: sum of the max-normalised replicates, peak = 1.

    Replicates are selected and normalised exactly as in litesizer_visualization.py
    (PDI exclusion, division by each replicate's own maximum); the sum is divided by its
    maximum. Returns the curves {nominal: (diameter_nm, curve)} and the accepted
    measurement groups of group_accepted_measurements().
    """
    groups = group_accepted_measurements(load_litesizer_xlsx(XLSX_PATH), sizes, MAX_PDI)
    curves = {}
    for size, reps in prepare_normalized_replicates(groups).items():
        d = reps[0][0]
        if any(r[0].shape != d.shape or not np.allclose(r[0], d) for r in reps):
            raise ValueError(f"DLS replicates of {size} nm do not share one diameter grid; cannot sum them.")
        total = np.sum([r[1] for r in reps], axis=0)
        curves[size] = (d, total / total.max())
    return curves, groups


def plot_dls_curve(ax: plt.Axes, d: np.ndarray, y: np.ndarray, peak: float) -> None:
    """Draw a peak-1 DLS curve scaled to `peak`.

    Only where the distribution is non-zero (plus one grid point on each side), as in
    litesizer_visualization.py, so the curve does not form a baseline at y = 0.
    """
    nonzero = y > SUPPORT_EPS
    shown = nonzero | np.roll(nonzero, 1) | np.roll(nonzero, -1)
    ax.plot(d, np.where(shown, y * peak, np.nan), color=COLOR_DLS, linewidth=1.2, zorder=3)


# ── Statistics ─────────────────────────────────────────────────────────────────

def compare_statistics(particles: pd.DataFrame, dls_summary: pd.DataFrame) -> pd.DataFrame:
    """SEM Feret statistics per size next to the LiteSizer statistics of litesizer_visualization.py."""
    rows = []
    for nominal in SIZES:
        grp = particles[particles["nominal_nm"] == nominal]
        f_max, f_mean = grp["feret_max_nm"].to_numpy(), grp["feret_mean_nm"].to_numpy()
        rows.append({
            "nominal_nm": nominal,
            "label": size_label(nominal),
            "sem_n_images": grp["image"].nunique(),
            "sem_n_particles": len(grp),
            "sem_feret_mean_nm": f_max.mean(),
            "sem_feret_sd_nm": f_max.std(ddof=1),
            "sem_feret_median_nm": np.median(f_max),
            "sem_mean_feret_mean_nm": f_mean.mean(),
            "sem_mean_feret_sd_nm": f_mean.std(ddof=1),
            "sem_mean_feret_median_nm": np.median(f_mean),
        })
    stats = pd.DataFrame(rows).merge(dls_summary.rename(columns={"nominal_size_nm": "nominal_nm"}),
                                     on="nominal_nm", how="left")
    stats["ratio_sem_feret_to_dls"] = stats["sem_feret_mean_nm"] / stats["dls_diameter_mean_nm"]
    stats["ratio_sem_mean_feret_to_dls"] = stats["sem_mean_feret_mean_nm"] / stats["dls_diameter_mean_nm"]
    return stats


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_panel(ax: plt.Axes, nominal: int, d_sem: np.ndarray,
               dls_curve: tuple[np.ndarray, np.ndarray] | None, stats: pd.Series) -> None:
    """SEM Feret-diameter histogram and the peak-matched summed DLS curve of one size."""
    base, dark = SIZE_COLORS[float(nominal)]
    edges = np.linspace(*X_RANGES[nominal], N_BINS + 1)
    freq, _ = np.histogram(d_sem, bins=edges)
    freq = 100.0 * freq / d_sem.size
    ax.stairs(freq, edges, fill=True, color=base, zorder=1)
    ax.stairs(freq, edges, color=dark, linewidth=0.5, zorder=2)
    if dls_curve is not None:
        plot_dls_curve(ax, *dls_curve, peak=freq.max())

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0.0, Y_HEADROOM * freq.max())
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.set_title(size_label(nominal))
    ax.text(0.97, 0.96,
            f"SEM {stats.sem_feret_mean_nm:.1f} ± {stats.sem_feret_sd_nm:.1f} nm\n"
            rf"$d_\mathrm{{H}}$ {stats.dls_diameter_mean_nm:.1f} ± {stats.dls_diameter_sd_nm:.1f} nm",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)


def plot_comparison(particles: pd.DataFrame, dls_curves: dict[int, tuple[np.ndarray, np.ndarray]],
                    stats: pd.DataFrame) -> plt.Figure:
    """2 x 3 panel figure; call inside plt.style.context(STYLE_PATH)."""
    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE_IN)
    for ax, nominal in zip(axes.flat, SIZES):
        d_sem = particles.loc[particles["nominal_nm"] == nominal, SEM_COLUMN].to_numpy()
        plot_panel(ax, nominal, d_sem, dls_curves.get(nominal), stats.set_index("nominal_nm").loc[nominal])
    for ax in axes[1]:
        ax.set_xlabel("Feret diameter (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative frequency (%)")

    handles = [Patch(facecolor="#bbbbbb", edgecolor="#555555", linewidth=0.5),
               Line2D([], [], color=COLOR_DLS, linewidth=1.2)]
    fig.legend(handles, ["SEM, Feret diameter", "DLS, intensity-weighted (sum of replicates)"],
               loc="outside upper center", ncol=2, frameon=False)
    return fig


def main(show: bool = SHOW_FIGURE):
    """Build the SEM vs. DLS comparison figure and statistics workbook."""
    particles = load_particles()
    dls_curves, dls_groups = combined_dls_curves(SIZES)
    stats = compare_statistics(particles, summarize_size_statistics(dls_groups))

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.precision", 2):
        print()
        print(stats[["label", "sem_n_particles", "sem_feret_mean_nm", "sem_feret_sd_nm",
                     "sem_mean_feret_mean_nm", "sem_mean_feret_sd_nm", "n_measurements",
                     "dls_diameter_mean_nm", "dls_diameter_sd_nm", "pdi_mean_pct",
                     "ratio_sem_feret_to_dls", "ratio_sem_mean_feret_to_dls"]].to_string(index=False))

    with plt.style.context(STYLE_PATH):
        fig = plot_comparison(particles, dls_curves, stats)
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
