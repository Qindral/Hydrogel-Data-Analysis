"""
DLS-measured diffusion coefficient D0 vs. particle size -- individual
repeat measurements (not the aggregate mean), one point per LiteSizer
measurement/file.

The ONLY Dissertation_Figures script that reads raw data directly rather
than a pickle cache: individual (per-repeat) DLS measurements are never
persisted anywhere -- Litesizer/cache/dls_reference.pkl only stores the
already-aggregated per-size mean (core.io.get_dls_diffusion_means(), used
by every other script in this folder). This script instead reuses
litesizer_measurements_mean.py's parse_litesizer()/aggregate() -- the
actual XLSX parsing and PDI-filtering logic -- rather than reimplementing
it. Importing that module runs its own mpl.rcParams.update() as a
module-level side effect, but this script's own plotting happens inside
plt.rc_context(_RC) regardless of matplotlib's ambient defaults, so the
side effect never reaches the saved figure.

Each point is plotted at that one measurement's own z_average_nm
(hydrodynamic diameter) and diffusion_coeff_um2s -- not the aggregate DLS
mean position -- so repeat-to-repeat DLS measurement spread is visible, the
same way SPT per-file spread is visible in D0_Diffusion_vs_Size.py. No
error bars (file-wise plot). Particle size axis ticks are still the
aggregate DLS label (core.io.get_dls_labels()) for readability.

Always reads the LiteSizer XLSX fresh (H:\\Daten Promotion Sicherung\\Lite
Sizer Particle Measurements\\Size_repitition_All Sizes.xlsx via
litesizer_measurements_mean.XLSX_PATH) -- there is no cache here to go
stale, so nothing needs to be rerun first, but the XLSX itself must be
reachable.

Shows the figure first (plt.show(), blocking); only after the window is
closed does it save into its own Auswertungsbilder\\DLS_Diffusion_vs_Size\\
subfolder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

from hydro_analysis.core.io import get_dls_sizes, get_dls_labels, get_dls_reference_maps
from hydro_analysis.core.physics import calculate_theoretical_diffusion
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY, _add_log_minor_ticks
from hydro_analysis.Litesizer.litesizer_measurements_mean import (
    parse_litesizer, aggregate, XLSX_PATH, SIZES, MAX_PDI,
)

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "DLS_Diffusion_vs_Size"

X_MIN, X_MAX = 15.0, 1500.0

COLOR_DLS, COLOR_DLS_DARK = "#da00bd", "#9b5191"   # Style-guide Markierung A
COLOR_THEORY = "black"

POINT_ALPHA = 0.7


def main() -> None:
    records = parse_litesizer(XLSX_PATH)
    print(f"XLSX geladen: {XLSX_PATH} ({len(records)} Messungen)")
    agg = aggregate(records, sizes=SIZES, max_pdi=MAX_PDI)

    dls_sizes     = get_dls_sizes()
    dls_labels    = get_dls_labels()
    size_override = get_dls_reference_maps()["size_override_nm"]

    def _map_size(size: float) -> float:
        return float(dls_sizes.get(size, size_override.get(size, size)))

    # ── One (z_average_nm, diffusion_coeff_um2s) row per individual DLS measurement ──
    rows: list[tuple[float, float]] = []
    for size_nm, entry in agg.items():
        for rec in entry["records"]:
            D = rec.get("diffusion_coeff_um2s")
            z = rec.get("z_average_nm")
            if D is None or z is None:
                continue
            rows.append((float(z), float(D)))
    print(f"{len(rows)} individuelle DLS-Messungen mit gültigem D-Wert.")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

        # ── Stokes-Einstein theory line for D0 ──────────────────────────────
        sizes_range = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 200)
        D_theory_range = [calculate_theoretical_diffusion(s) for s in sizes_range]
        ax.plot(sizes_range, D_theory_range, color=COLOR_THEORY, linewidth=1.2,
                linestyle=_DASH_THEORY, zorder=3)

        # ── Individual DLS measurements -- own z_average_nm, no error bars ──
        if rows:
            xs = [z for z, D in rows]
            ys = [D for z, D in rows]
            ax.scatter(xs, ys, s=60, alpha=POINT_ALPHA, marker="*",
                       facecolor=COLOR_DLS, edgecolor=COLOR_DLS_DARK, linewidth=0.6, zorder=5)

        # ── Axes: DLS-mapped tick positions, DLS labels (not nominal names) ──
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(X_MIN, X_MAX)
        _add_log_minor_ticks(ax)

        tick_nominal = sorted(dls_labels.keys())
        tick_pos     = [_map_size(s) for s in tick_nominal]
        tick_text    = [str(dls_labels[s]) for s in tick_nominal]
        ax.xaxis.set_major_locator(FixedLocator(tick_pos))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_text))

        ax.set_xlabel("Particle size (nm)")
        ax.set_ylabel(r"Diffusion coefficient $D$ (µm²/s)")

        legend_elements = [
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
                   label="Stokes–Einstein theory (D0)"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor=COLOR_DLS,
                   markeredgecolor=COLOR_DLS_DARK, markersize=10, markeredgewidth=0.6,
                   label="DLS D0 (water, per measurement)", linestyle="None"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / "dls_diffusion_vs_size.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
