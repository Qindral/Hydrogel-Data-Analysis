"""
Effect of the manual segmentation review on the SEM size distributions --
histograms before and after sem_segmentation_review.py, one panel per nominal
particle size (2 x 3 panels, width class "full"), for both size measures
(Feret diameter and mean Feret diameter).

Consumer stage: reads only hydro_analysis/SEM_Particles/cache/sem_feret_particles.pkl
(written by sem_feret_measure.py after the review). "Before" are all
annotated particles without the automatic exclusions (touches_border,
too_small) and without the ellipses added by hand, i.e. the data as they were
used before the review; "after" are the particles with used == True, which
additionally excludes removed_manually and includes the added ellipses. Both are relative frequencies
(each normalised to its own particle count) with the bins of the single-size
histograms (HIST_RANGES, N_BINS of sem_feret_size_histogram.py).

Writes into SAVE_PATH:
  sem_review_before_after_feret_diameter.pdf
  sem_review_before_after_mean_feret_diameter.pdf
No other script reads these outputs.

Styling follows Styleguide_Figures_Dissertation.md (v2): plotting inside
plt.style.context(hydro_analysis/thesis.mplstyle), final printed size, vector
PDF without bbox_inches="tight"; "after" in the size colours (SIZE_COLORS),
"before" as a grey outline.
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

from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS
from hydro_analysis.SEM_Particles.sem_feret_measure import CACHE_PATH
from hydro_analysis.SEM_Particles.sem_feret_size_histogram import HIST_RANGES, N_BINS, size_label

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data"
) / "SEM_particle_size"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
SHOW_FIGURE = True

SIZES = [20, 50, 100, 200, 500, 1000]
FIG_SIZE_IN = (6.30, 4.20)             # width class "full"; multi-panel height (style guide §2)
COLOR_BEFORE = "#555555"
Y_HEADROOM = 1.45                      # y limit / highest bar, leaves room for the statistics text

# particle-table column -> (file stem suffix, x-axis label)
DIAMETER_VARIANTS = {
    "feret_max_nm": ("feret_diameter", "Feret diameter (nm)"),
    "feret_mean_nm": ("mean_feret_diameter", "Mean Feret diameter (nm)"),
}


def load_before_after(cache_path: Path = CACHE_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(before, after) particle tables: without automatic exclusions / with all exclusions incl. review."""
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path} not found; run sem_feret_measure.py first.")
    with open(cache_path, "rb") as fh:
        particles = pickle.load(fh)["particles"]
    if "removed_manually" not in particles:
        raise ValueError("Cache predates the segmentation review; rerun sem_feret_measure.py.")
    added = particles["added_manually"] if "added_manually" in particles else False
    before = particles[~(particles["touches_border"] | particles["too_small"] | added)]
    return before, particles[particles["used"]]


def change_table(before: pd.DataFrame, after: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for nominal in SIZES:
        b = before.loc[before["nominal_nm"] == nominal, column]
        a = after.loc[after["nominal_nm"] == nominal, column]
        n_added = int(after.loc[after["nominal_nm"] == nominal, "added_manually"].sum())             if "added_manually" in after else 0
        n_removed = len(b) - (len(a) - n_added)
        rows.append({"label": size_label(nominal), "n_before": len(b), "n_after": len(a),
                     "removed": n_removed, "removed_pct": 100.0 * n_removed / len(b), "added": n_added,
                     "mean_before": b.mean(), "mean_after": a.mean(),
                     "sd_before": b.std(ddof=1), "sd_after": a.std(ddof=1),
                     "median_before": b.median(), "median_after": a.median()})
    return pd.DataFrame(rows)


def plot_before_after(before: pd.DataFrame, after: pd.DataFrame, column: str, x_label: str,
                      table: pd.DataFrame) -> plt.Figure:
    """2 x 3 panels of before (grey outline) and after (filled) histograms; call inside the style context."""
    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE_IN)
    for ax, nominal, row in zip(axes.flat, SIZES, table.itertuples(index=False)):
        base, dark = SIZE_COLORS[float(nominal)]
        edges = np.linspace(*HIST_RANGES[nominal], N_BINS + 1)
        d_b = before.loc[before["nominal_nm"] == nominal, column].to_numpy()
        d_a = after.loc[after["nominal_nm"] == nominal, column].to_numpy()
        f_b = 100.0 * np.histogram(d_b, bins=edges)[0] / d_b.size
        f_a = 100.0 * np.histogram(d_a, bins=edges)[0] / d_a.size
        ax.stairs(f_a, edges, fill=True, color=base, zorder=1)
        ax.stairs(f_a, edges, color=dark, linewidth=0.5, zorder=2)
        ax.stairs(f_b, edges, color=COLOR_BEFORE, linewidth=1.0, zorder=3)

        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0.0, Y_HEADROOM * max(f_a.max(), f_b.max()))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.set_title(size_label(nominal))
        ax.text(0.97, 0.96,
                f"Before {row.mean_before:.1f} ± {row.sd_before:.1f} nm\n"
                f"After {row.mean_after:.1f} ± {row.sd_after:.1f} nm\n"
                f"Removed {row.removed} ({row.removed_pct:.1f} %)" + (f", added {row.added}" if row.added else ""),
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(x_label)
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative frequency (%)")
    handles = [Line2D([], [], color=COLOR_BEFORE, linewidth=1.0),
               Patch(facecolor="#bbbbbb", edgecolor="#555555", linewidth=0.5)]
    fig.legend(handles, ["Before review", "After review"], loc="outside upper center", ncol=2, frameon=False)
    return fig


def main(show: bool = SHOW_FIGURE):
    """Print the changes per size and plot both diameter variants before vs. after the review."""
    before, after = load_before_after()
    figures = {}
    with plt.style.context(STYLE_PATH):
        for column, (stem, x_label) in DIAMETER_VARIANTS.items():
            table = change_table(before, after, column)
            with pd.option_context("display.width", 200, "display.precision", 1):
                print(f"\n{x_label}")
                print(table.to_string(index=False))
            figures[stem] = plot_before_after(before, after, column, x_label, table)
        if show:
            plt.show()
        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            for stem, fig in figures.items():
                fig.savefig(SAVE_PATH / f"sem_review_before_after_{stem}.pdf", format="pdf")
            print(f"Saved {len(figures)} figures to {SAVE_PATH}")
    return figures


if __name__ == "__main__":
    main()
