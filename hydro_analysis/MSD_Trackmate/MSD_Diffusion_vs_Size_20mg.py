"""
Diffusion coefficient D vs. particle size — 20 mg/mL C16 hydrogel.

Reads msd_20mg_files.pkl (produced by MSD_FromTrackmate_20mg.py) and plots
one point per file: D (µm²/s) against particle size (nm), colored by whether
the file was measured at "Surface loading" (an "A<n>" token in the filename)
or "Injection" ("B<n>"), via core.io.condition_label_from_filename(). Files
with neither token are shown in grey.

Overlaid for reference:
  - the DLS-measured D0 (water) points at each particle size, with their
    DLS size/diffusion uncertainties (get_dls_reference_maps())
  - the continuous Stokes-Einstein theory line for D0 across the size range

Sizes are handled the same way as MSD_per_file_20mg.py: the real DLS-measured
diameter (get_dls_sizes(), z-average, falling back to size_override_nm) is
used for the x-position and the theory/D0 calculation, and the DLS label
(get_dls_labels()) is shown on the axis -- never the nominal folder name
("20 nm" etc.), which is a rounded target, not the actual particle size.

Dissertation version (user, 2026-10-14; Styleguide v2, width class narrow, PDF,
SAVE_PATH): x positions at the DLS diameters in full precision (get_dls_sizes()),
hydrogel markers in the dark SIZE_COLORS colour of each size, DLS reference as a
square #da00bd with x and y error bars, "Buffer" instead of "Water".

Run MSD_FromTrackmate_20mg.py first (or after any change to the raw data)
to refresh msd_20mg_files.pkl.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, FuncFormatter, LogLocator

from hydro_analysis.core.io import (
    get_dls_reference_maps, get_dls_sizes, get_dls_labels, condition_label_from_filename,
)
from hydro_analysis.core.physics import calculate_theoretical_diffusion
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _DASH_THEORY, _add_log_minor_ticks
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent / "cache" / "msd_20mg_files.pkl"
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\SPT"
)
STYLE = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
FIG_SIZE = (4.72, 4.72 / 1.42)       # Styleguide width class "narrow"

X_MIN, X_MAX = 15.0, 1500.0

# Same _RC/_DASH_THEORY/_add_log_minor_ticks as MSD_per_file_publication.py,
# but figsize=(7.15, 5.00) -- this is a dataset-level summary chart, so it
# follows the same size/settings as the other size-comparison plots that use
# this _RC (e.g. MSD_AnomalousExponent_SaxtonFit.py), not the smaller
# (5.5, 4.0) used only for the per-file multi-panel figure with insets.
COLOR_SURFACE      = "#3B8C8C"   # desaturated green-blue (teal) -- Surface loading
COLOR_SURFACE_DARK = "#2A6666"
COLOR_INJECTION    = "#D98C3D"   # desaturated orange -- Injection
COLOR_INJECTION_DARK = "#A6672D"
COLOR_OTHER        = "#999999"   # no A/B token in filename
COLOR_OTHER_DARK   = "#555555"
COLOR_DLS          = "#da00bd"   # Style-guide Markierung A base
COLOR_DLS_DARK     = "#9b5191"
COLOR_THEORY       = "black"

# No x-offset: points sit exactly at their real size, with alpha so that
# overlapping files at the same size show up as visibly denser/darker.
# Hydrogel points in the dark SIZE_COLORS colour of their size; the marker shows the loading type
CONDITION_STYLE = {
    "Surface loading": {"marker": "o"},
    "Injection":       {"marker": "s"},
}
POINT_ALPHA = 0.7


def load_results() -> dict:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_FILE}\nBitte zuerst MSD_FromTrackmate_20mg.py ausführen."
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

    # ── Collect one (size, D, D_err, condition) row per file ────────────────
    rows: list[tuple[float, float, float, str | None]] = []
    n_skipped = 0
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        fit = r.get("fit_results_MSD")
        D = fit.get("D_um2_per_s") if fit else None
        if size_nm is None or D is None:
            n_skipped += 1
            continue
        condition = condition_label_from_filename(r.get("base_name", ""))
        rows.append((float(size_nm), float(D), float(fit.get("D_error", np.nan)), condition))
    print(f"{len(rows)} Dateien mit gültigem D-Wert, {n_skipped} übersprungen.")

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIG_SIZE)

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
        ax.errorbar(dls_x, dls_y, xerr=dls_xerr, yerr=dls_yerr, fmt="s", markersize=5,
                    markerfacecolor=COLOR_DLS, markeredgecolor=COLOR_DLS_DARK, markeredgewidth=0.6,
                    ecolor=COLOR_DLS_DARK, elinewidth=0.8, capsize=2.0, capthick=0.8,
                    linestyle="None", zorder=8)

        # ── Per-file D values, colored by condition -- exact x, no offset ───
        # Alpha lets overlapping files at the same size show up as denser/darker.
        for label, style in CONDITION_STYLE.items():
            group = [row for row in rows if row[3] == label]
            if not group:
                continue
            for s, D, Derr, c in group:
                dark = SIZE_COLORS[s][1]
                ax.errorbar([_map_size(s)], [D], yerr=[Derr if np.isfinite(Derr) else 0.0], fmt=style["marker"],
                            markersize=4, alpha=POINT_ALPHA, markerfacecolor=dark, markeredgecolor=dark,
                            markeredgewidth=0.6, ecolor=dark, elinewidth=0.8, capsize=2.0, linestyle="None",
                            zorder=5)

        # Files without a recognizable A<n>/B<n> token
        other = [row for row in rows if row[3] is None]
        if other:
            xs = [_map_size(s) for s, D, Derr, c in other]
            ys = [D for s, D, Derr, c in other]
            ax.scatter(xs, ys, s=20, facecolor=COLOR_OTHER, edgecolor=COLOR_OTHER_DARK,
                       linewidth=0.5, alpha=POINT_ALPHA, zorder=4)

        # ── Axes: DLS-mapped tick positions, DLS labels (not nominal names) ──
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(X_MIN, X_MAX)
        _add_log_minor_ticks(ax)
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))

        tick_nominal = sorted(dls_labels.keys())
        tick_pos     = [_map_size(s) for s in tick_nominal]
        tick_text    = [str(dls_labels[s]) for s in tick_nominal]
        ax.xaxis.set_major_locator(FixedLocator(tick_pos))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_text))

        ax.set_xlabel("Particle size (nm)")
        ax.set_ylabel(r"Diffusion coefficient $D$ (µm$^2$ s$^{-1}$)")

        legend_elements = [
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
                   label=r"Stokes–Einstein $D_0$"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_DLS,
                   markeredgecolor=COLOR_DLS_DARK, markersize=5, markeredgewidth=0.6,
                   label=r"DLS $D_0$ (buffer)", linestyle="None"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="0.3",
                   markeredgecolor="0.3", markersize=4, markeredgewidth=0.6,
                   label="Hydrogel, surface loading", linestyle="None"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="0.3",
                   markeredgecolor="0.3", markersize=4, markeredgewidth=0.6,
                   label="Hydrogel, injection", linestyle="None"),
        ]
        if other:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_OTHER,
                       markeredgecolor=COLOR_OTHER_DARK, markersize=6,
                       label="Unknown (no A/B code)", linestyle="None")
            )
        ax.legend(handles=legend_elements, loc="upper right", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            pdf_path = SAVE_PATH / "diffusion_vs_size_20mg.pdf"
            fig.savefig(pdf_path)
            print(f"Plot gespeichert: {pdf_path}")


if __name__ == "__main__":
    main()
