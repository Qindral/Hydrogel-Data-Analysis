"""
SEM particle size distributions with the LiteSizer (DLS) reference -- one
frequency histogram per nominal particle size, in two variants of the SEM
size measure:
  Feret diameter        Feret_max, the largest caliper width (ImageJ "Feret")
  Mean Feret diameter   (Feret_max + Feret_min) / 2, the diameter of the
                        best-fitting sphere
Each histogram has N_BINS bins over the fixed range HIST_RANGES of its size.

SEM data: consumer stage, reads only
hydro_analysis/SEM_Particles/cache/sem_feret_particles.pkl (written by
sem_feret_measure.py, which must be run first) and never touches the SEM
images or annotations. Only particles with used == True are shown, i.e.
particles touching the scanned-area border (including labels in the FEI
databar) and sub-MIN_AREA_PX fragments are excluded upstream. The images of
one nominal size are pooled; the printed and annotated mean +- SD include the
particles outside the plotted range.

DLS data: the same data as panel A of Litesizer/litesizer_visualization.py,
i.e. the intensity-weighted size distributions of the accepted replicates
(PDI <= 42 %), read and max-normalised with that script's own functions
(group_accepted_measurements, prepare_normalized_replicates). The
max-normalised replicates of one size are summed on the common LiteSizer
diameter grid into one curve (combined_dls_curves), which is scaled so that
its peak equals the highest histogram bar. This is a graphical normalisation
for comparing peak positions and widths only; the DLS diameters and the
LiteSizer results (d_H, PDI) are unchanged. Intensity weighting emphasises
large particles, so the DLS curve is expected to lie above the
number-weighted SEM histogram.

Writes into SAVE_PATH, two figures per size (width class "half", 3.07 in):
  sem_feret_diameter_<nominal>nm.pdf
  sem_mean_feret_diameter_<nominal>nm.pdf
load_particles(), size_label(), combined_dls_curves(), plot_dls_curve(),
HIST_RANGES and N_BINS are imported by sem_vs_dls_size_distribution.py. No other script reads the PDFs.

Styling follows Styleguide_Figures_Dissertation.md (v2): plotting inside
plt.style.context(hydro_analysis/thesis.mplstyle), figures created in their
final printed size and saved as vector PDF without bbox_inches="tight". Size
colours are SIZE_COLORS (Validation_Claude/Correlations.py), DLS lines the
DLS reference colour (§11), labels the DLS labels from core.io.get_dls_labels().
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.Litesizer.litesizer_parser import load_litesizer_xlsx
from hydro_analysis.Litesizer.litesizer_visualization import (
    MAX_PDI, SUPPORT_EPS, XLSX_PATH, group_accepted_measurements, prepare_normalized_replicates,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "sem_feret_particles.pkl"
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data"
) / "SEM_particle_size"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
SHOW_FIGURE = True

FIG_SIZE_IN = (3.07, 3.07 / 1.42)      # width class "half" (style guide §2)
COLOR_DLS = "#da00bd"                  # DLS reference (style guide §11)
Y_HEADROOM = 1.45                      # y limit / highest bar, leaves room for legend and text

N_BINS = 40
HIST_RANGES = {                        # nominal size -> (lower, upper) histogram edge in nm
    20: (10.0, 50.0),
    50: (15.0, 85.0),
    100: (75.0, 125.0),
    200: (100.0, 300.0),
    500: (400.0, 600.0),
    1000: (800.0, 1200.0),
}

# particle-table column -> (file stem, x-axis label)
DIAMETER_VARIANTS = {
    "feret_max_nm": ("sem_feret_diameter", "Feret diameter (nm)"),
    "feret_mean_nm": ("sem_mean_feret_diameter", "Mean Feret diameter (nm)"),
}


def load_particles(cache_path: Path = CACHE_PATH) -> pd.DataFrame:
    """Used particles of the per-particle table written by sem_feret_measure.py."""
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path} not found; run sem_feret_measure.py first.")
    with open(cache_path, "rb") as fh:
        particles = pickle.load(fh)["particles"]
    return particles[particles["used"]]


def size_label(nominal_nm: int) -> str:
    """Dissertation-wide particle label ("35 nm" for nominal 20 nm)."""
    return f"{get_dls_labels().get(float(nominal_nm), nominal_nm)} nm"


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


def plot_size_histogram(d_nm: np.ndarray, nominal_nm: int, x_label: str,
                        dls_curve: tuple[np.ndarray, np.ndarray] | None) -> plt.Figure:
    """Relative-frequency histogram with the peak-matched DLS curve; call inside the style context."""
    base, dark = SIZE_COLORS[float(nominal_nm)]
    edges = np.linspace(*HIST_RANGES[nominal_nm], N_BINS + 1)
    freq, _ = np.histogram(d_nm, bins=edges)
    freq = 100.0 * freq / d_nm.size

    fig, ax = plt.subplots(figsize=FIG_SIZE_IN)
    ax.stairs(freq, edges, fill=True, color=base, zorder=1)
    ax.stairs(freq, edges, color=dark, linewidth=0.5, zorder=2)
    if dls_curve is not None:
        plot_dls_curve(ax, *dls_curve, peak=freq.max())

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, Y_HEADROOM * freq.max())
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Relative frequency (%)")
    ax.text(0.97, 0.95,
            f"{size_label(nominal_nm)}\n$n$ = {d_nm.size}\n{d_nm.mean():.1f} ± {d_nm.std(ddof=1):.1f} nm",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    handles = [Patch(facecolor=base, edgecolor=dark, linewidth=0.5), Line2D([], [], color=COLOR_DLS, linewidth=1.2)]
    ax.legend(handles, ["SEM", "DLS (intensity)"], loc="upper left", handlelength=1.5, borderaxespad=0.4)
    return fig


def main(show: bool = SHOW_FIGURE):
    """Plot and save both diameter variants for every nominal size."""
    particles = load_particles()
    dls, _ = combined_dls_curves(list(HIST_RANGES))

    figures = {}
    with plt.style.context(STYLE_PATH):
        for column, (stem, x_label) in DIAMETER_VARIANTS.items():
            print(f"\n{x_label}")
            for nominal, grp in particles.groupby("nominal_nm"):
                d_nm = grp[column].to_numpy()
                lo, hi = HIST_RANGES[nominal]
                outside = int(((d_nm < lo) | (d_nm > hi)).sum())
                print(f"  {nominal:>5} nm: n = {d_nm.size}, mean = {d_nm.mean():.1f} nm, "
                      f"SD = {d_nm.std(ddof=1):.1f} nm, median = {np.median(d_nm):.1f} nm, "
                      f"{outside} ({100 * outside / d_nm.size:.1f} %) outside {lo:g}-{hi:g} nm")
                figures[f"{stem}_{nominal}nm"] = plot_size_histogram(d_nm, nominal, x_label, dls.get(nominal))

        if show:
            plt.show()
        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(SAVE_PATH / f"{name}.pdf", format="pdf")
            print(f"Saved {len(figures)} histograms to {SAVE_PATH}")
    return figures


if __name__ == "__main__":
    main()
