"""
Diffusion coefficient D vs. particle size — water (D0) reference measurements.

Reads msd_d0_results.pkl (produced by MSD_FromTrackmate_D0.py) and plots one
point per file: D (µm²/s) against particle size (nm). Overlaid for reference:
  - the DLS-measured D0 (water) points at each particle size, with their
    DLS size/diffusion uncertainties (get_dls_reference_maps())
  - the continuous Stokes-Einstein theory line for D0 across the size range

This is the water-side counterpart to MSD_Diffusion_vs_Size_20mg.py: since
these files ARE the D0 (water) measurements, there is no Surface
loading/Injection split (that distinction only applies to the hydrogel
dataset) -- every file is plotted in one series, so this is effectively a
quality-control chart of the per-file MSD measurements against the DLS/
theory reference.

Sizes are handled the same way as MSD_per_file_20mg.py / MSD_perfile_D0.py:
the real DLS-measured diameter (get_dls_sizes(), z-average, falling back to
size_override_nm) is used for the x-position and the theory/D0 calculation,
and the DLS label (get_dls_labels()) is shown on the axis -- never the
nominal folder name ("20 nm" etc.), which is a rounded target, not the
actual particle size.

Run MSD_FromTrackmate_D0.py first (or after any change to the raw data) to
refresh msd_d0_results.pkl.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

from hydro_analysis.core.io import (
    get_dls_reference_maps, get_dls_sizes, get_dls_labels,
)
from hydro_analysis.core.physics import calculate_theoretical_diffusion
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY, _add_log_minor_ticks

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent / "cache" / "msd_d0_results.pkl"
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)

X_MIN, X_MAX = 15.0, 1500.0

# Same _RC/_DASH_THEORY/_add_log_minor_ticks and figsize as
# MSD_Diffusion_vs_Size_20mg.py, for a consistent look across both charts.
COLOR_MEASURED      = "#3B6E8C"   # desaturated blue -- per-file D0 measurement
COLOR_MEASURED_DARK = "#2A4F66"
COLOR_DLS           = "#da00bd"   # Style-guide Markierung A base
COLOR_DLS_DARK      = "#9b5191"
COLOR_THEORY        = "black"

POINT_ALPHA = 0.5   # overlapping files at the same size show up as denser/darker


def load_results() -> dict:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_FILE}\nBitte zuerst MSD_FromTrackmate_D0.py ausführen."
        )
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def main() -> None:
    results = load_results()
    print(f"Cache geladen: {CACHE_FILE} ({len(results)} Dateien)")

    dls_sizes     = get_dls_sizes()               # {nominal_nm: real DLS z-average diameter}
    dls_labels    = get_dls_labels()               # {nominal_nm: DLS label for axis display}
    dls_maps      = get_dls_reference_maps()
    size_override = dls_maps["size_override_nm"]   # fallback if a size is missing from dls_sizes
    size_err      = dls_maps["size_err_nm"]
    dls_D         = dls_maps["dls_D_um2_per_s"]
    dls_D_err     = dls_maps["dls_D_err_um2_per_s"]

    def _map_size(size: float) -> float:
        # Real measured diameter, not the nominal folder name -- same priority
        # order as MSD_per_file_20mg.py's `mapped_size`.
        return float(dls_sizes.get(size, size_override.get(size, size)))

    def _sz_err(size: float) -> float:
        return float(size_err.get(size, 0.0))

    # ── Collect one (size, D, D_err) row per file ───────────────────────────
    rows: list[tuple[float, float, float]] = []
    n_skipped = 0
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        fit = r.get("fit_results_MSD")
        D = fit.get("D_um2_per_s") if fit else None
        if size_nm is None or D is None:
            n_skipped += 1
            continue
        rows.append((float(size_nm), float(D), float(fit.get("D_error", np.nan))))
    print(f"{len(rows)} Dateien mit gültigem D-Wert, {n_skipped} übersprungen.")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

        # ── Stokes-Einstein theory line for D0 ──────────────────────────────
        sizes_range = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 200)
        D_theory_range = [calculate_theoretical_diffusion(s) for s in sizes_range]
        ax.plot(sizes_range, D_theory_range, color=COLOR_THEORY, linewidth=1.2,
                linestyle=_DASH_THEORY, zorder=3)

        # ── DLS-measured D0 (water) reference points ────────────────────────
        dls_ref_sizes = sorted(dls_D.keys())
        dls_x    = [_map_size(s) for s in dls_ref_sizes]
        dls_y    = [dls_D[s] for s in dls_ref_sizes]
        dls_xerr = [_sz_err(s) for s in dls_ref_sizes]
        dls_yerr = [dls_D_err.get(s, 0.0) for s in dls_ref_sizes]
        ax.errorbar(dls_x, dls_y, xerr=dls_xerr, yerr=dls_yerr, fmt="*", markersize=10,
                    markerfacecolor=COLOR_DLS, markeredgecolor=COLOR_DLS_DARK, markeredgewidth=0.8,
                    ecolor=COLOR_DLS_DARK, elinewidth=0.8, capsize=2.0, capthick=0.8,
                    linestyle="None", zorder=8)

        # ── Per-file D0 measurements -- exact x, no offset ──────────────────
        # Alpha lets overlapping files at the same size show up as denser/darker.
        xs    = [_map_size(s) for s, D, Derr in rows]
        ys    = [D for s, D, Derr in rows]
        yerrs = [Derr if np.isfinite(Derr) else 0.0 for s, D, Derr in rows]
        ax.errorbar(xs, ys, yerr=yerrs, fmt="o", markersize=6, alpha=POINT_ALPHA,
                    markerfacecolor=COLOR_MEASURED, markeredgecolor=COLOR_MEASURED_DARK,
                    markeredgewidth=0.6, ecolor=COLOR_MEASURED_DARK, elinewidth=0.8,
                    capsize=2.0, linestyle="None", zorder=5)

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
                   markeredgecolor=COLOR_DLS_DARK, markersize=10, markeredgewidth=0.8,
                   label="DLS D0 (water)", linestyle="None"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MEASURED,
                   markeredgecolor=COLOR_MEASURED_DARK, markersize=7, markeredgewidth=0.6,
                   label="MSD D0 (per file)", linestyle="None"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / "diffusion_vs_size_d0.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
