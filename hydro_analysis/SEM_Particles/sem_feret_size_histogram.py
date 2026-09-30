"""
SEM particle size distributions -- one frequency histogram per nominal
particle size, in two variants of the SEM size measure:
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

Writes into SAVE_PATH, two figures per size (width class "half", 3.07 in):
  sem_feret_diameter_<nominal>nm.pdf
  sem_mean_feret_diameter_<nominal>nm.pdf
load_particles(), size_label() and N_BINS are imported by
sem_vs_dls_size_distribution.py, which shows the comparison with the
LiteSizer (DLS) data. No other script reads the PDFs.

Styling follows Styleguide_Figures_Dissertation.md (v2): plotting inside
plt.style.context(hydro_analysis/thesis.mplstyle), figures created in their
final printed size and saved as vector PDF without bbox_inches="tight". Size
colours are SIZE_COLORS (Validation_Claude/Correlations.py), labels the DLS
labels from core.io.get_dls_labels().
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "sem_feret_particles.pkl"
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data"
) / "SEM_particle_size"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
SHOW_FIGURE = True

FIG_SIZE_IN = (3.07, 3.07 / 1.42)      # width class "half" (style guide §2)
Y_HEADROOM = 1.4                       # y limit / highest bar, keeps the statistics text clear of the bars

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


def plot_size_histogram(d_nm: np.ndarray, nominal_nm: int, x_label: str) -> plt.Figure:
    """Relative-frequency histogram of one size and diameter variant; call inside the style context."""
    base, dark = SIZE_COLORS[float(nominal_nm)]
    edges = np.linspace(*HIST_RANGES[nominal_nm], N_BINS + 1)
    freq, _ = np.histogram(d_nm, bins=edges)
    freq = 100.0 * freq / d_nm.size

    fig, ax = plt.subplots(figsize=FIG_SIZE_IN)
    ax.stairs(freq, edges, fill=True, color=base, zorder=1)
    ax.stairs(freq, edges, color=dark, linewidth=0.5, zorder=2)

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
    return fig


def main(show: bool = SHOW_FIGURE):
    """Plot and save both diameter variants for every nominal size."""
    particles = load_particles()

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
                figures[f"{stem}_{nominal}nm"] = plot_size_histogram(d_nm, nominal, x_label)

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
