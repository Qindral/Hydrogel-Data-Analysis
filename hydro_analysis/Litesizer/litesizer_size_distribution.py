"""
Plot normalized size distributions from a LiteSizer XLSX file.

Each measurement is normalized so its area integrates to 1 (probability density).
Individual curves are shown as thin semi-transparent lines; the ensemble mean
per size group is overlaid as a bold line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from hydro_analysis.core.io import get_dls_reference_maps
from hydro_analysis.Litesizer.litesizer_parser import MeasurementData, load_litesizer_xlsx

# ── Configuration ─────────────────────────────────────────────────────────────
XLSX_PATH  = Path(r"H:\Daten Promotion Sicherung\Lite Sizer Particle Measurements\Size_repitition_All Sizes.xlsx")
SAVE_PATH  = None          # Set to a Path to save figures, e.g. Path(r"E:\...\Plots")
DIST_TYPE  = "intensity"   # "volume" | "intensity" | "number"
MAX_PDI    = 42.0          # measurements with PDI > this are excluded
SMOOTH_SIGMA = 2.0         # Gaussian smoothing sigma for mean curve (points); 0 = off
SIZES      = [20, 50, 100, 200, 500, 1000]  # nominal sizes to plot (nm)

# Style guide colours (base + dark variant)
COLORS = {
    "base": ["#0000da", "#004cff", "#00c4ff", "#49ffad", "#adff49", "#ffd700", "#ff6800", "#da0000"],
    "dark": ["#000099", "#0035b2", "#0089b2", "#33b279", "#79b233", "#b29600", "#b24900", "#990000"],
}

mpl.rcParams.update({
    "figure.figsize":      (7.15, 5.00),
    "figure.dpi":          150,
    "savefig.dpi":         600,
    "savefig.bbox":        "tight",
    "font.family":         "Source Sans 3",
    "font.size":           9,
    "axes.titlesize":      12,
    "axes.titleweight":    "semibold",
    "axes.labelsize":      10,
    "axes.linewidth":      1.0,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.major.size":    4.0,
    "ytick.major.size":    4.0,
    "xtick.minor.size":    2.0,
    "ytick.minor.size":    2.0,
    "xtick.major.width":   1.0,
    "ytick.major.width":   1.0,
    "lines.linewidth":     1.8,
    "lines.markersize":    4.5,
    "legend.fontsize":     9,
    "legend.frameon":      False,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_distribution(m: MeasurementData, dist_type: str):
    """Return (diameter_nm, frequency) arrays for the requested distribution type."""
    if dist_type == "volume":
        df = m.size_distribution_volume
    elif dist_type == "intensity":
        df = m.size_distribution_intensity
    else:
        df = m.size_distribution_number
    if df.empty:
        return None, None
    x = df["diameter_nm"].to_numpy(dtype=float)
    y = df["frequency_pct"].to_numpy(dtype=float)
    mask = y > 0
    return x[mask], y[mask]


def _normalize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normalize y so the area under the curve (trapezoid) equals 1."""
    area = np.trapz(y, x)
    if area <= 0:
        return y
    return y / area


def _group_by_size(
    measurements: list[MeasurementData],
    sizes: list[int],
    dist_type: str,
    max_pdi: float,
) -> dict[int, list[tuple[np.ndarray, np.ndarray, str]]]:
    """Return {size_nm: [(x, y_norm, name), ...]} after PDI filtering."""
    groups: dict[int, list] = {s: [] for s in sizes}
    n_excluded = 0

    for m in measurements:
        if m.polydispersity_pct is not None and m.polydispersity_pct > max_pdi:
            print(f"  [EXCLUDED PDI={m.polydispersity_pct:.1f}%] {m.name}")
            n_excluded += 1
            continue

        x, y = _get_distribution(m, dist_type)
        if x is None or len(x) < 2:
            continue

        y_norm = _normalize(x, y)

        for size in sizes:
            if f"{size} nm" in m.name:
                groups[size].append((x, y_norm, m.name))
                break

    if n_excluded:
        print(f"  → {n_excluded} measurement(s) excluded (PDI > {max_pdi:.0f}%)")
    return groups


# ── Plotting ──────────────────────────────────────────────────────────────────

def _fwhm(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (x_peak, x_left_half, x_right_half) from linear interpolation."""
    peak_idx = int(np.argmax(y))
    half_max  = float(y[peak_idx]) / 2.0

    # Left crossing: scan left from peak
    x_left = float(x[0])
    for i in range(peak_idx, 0, -1):
        if y[i - 1] <= half_max:
            x_left = float(np.interp(half_max, [y[i - 1], y[i]], [x[i - 1], x[i]]))
            break

    # Right crossing: scan right from peak
    x_right = float(x[-1])
    for i in range(peak_idx, len(y) - 1):
        if y[i + 1] <= half_max:
            x_right = float(np.interp(half_max, [y[i + 1], y[i]], [x[i + 1], x[i]]))
            break

    return float(x[peak_idx]), x_left, x_right


def _ensemble_mean(curves: list[tuple[np.ndarray, np.ndarray, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_grid, y_mean_normalized) over the shared non-zero support."""
    x_all = np.unique(np.concatenate([c[0] for c in curves]))
    x_all.sort()
    y_interp = np.array([
        np.interp(x_all, c[0], c[1], left=0.0, right=0.0) for c in curves
    ])
    y_mean = y_interp.mean(axis=0)
    if SMOOTH_SIGMA > 0:
        y_mean = gaussian_filter1d(y_mean, sigma=SMOOTH_SIGMA)
    mask = y_mean > 0
    return x_all[mask], _normalize(x_all[mask], y_mean[mask])


def plot_all_sizes(
    groups: dict[int, list],
    dist_type: str,
    save_path: Path | None,
    size_subset: list[int],
    dls_size_map: dict[float, float],
    fig_suffix: str = "all",
) -> None:
    """Plot a subset of particle sizes on a single axes with FWHM errorbars."""
    active = {s: c for s, c in groups.items() if c and s in size_subset}
    if not active:
        print(f"No data to plot for sizes {size_subset}.")
        return

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

    for ci, (size_nm, curves) in enumerate(sorted(active.items())):
        color_base = COLORS["base"][ci % len(COLORS["base"])]
        color_dark  = COLORS["dark"][ci % len(COLORS["dark"])]

        # Individual curves – unsmoothed, thin, transparent
        for x, y, _ in curves:
            ax.plot(x, y, color=color_base, linewidth=0.9, alpha=0.25)

        # Smoothed ensemble mean – bold, labeled
        if len(curves) > 1:
            x_m, y_m = _ensemble_mean(curves)
        else:
            x_m, y_m = curves[0][0], curves[0][1]

        dls_label = dls_size_map.get(float(size_nm), float(size_nm))
        ax.plot(x_m, y_m, color=color_dark, linewidth=2.2,
                label=f"{dls_label:.1f} nm (N={len(curves)})")

        # FWHM-based errorbar: peak position = mean, half-widths = uncertainty
        x_peak, x_left, x_right = _fwhm(x_m, y_m)
        peak_y = float(y_m[np.argmax(y_m)])
        ax.errorbar(x_peak, peak_y,
                    xerr=[[x_peak - x_left], [x_right - x_peak]],
                    fmt="o", color="black", markersize=3.5,
                    capsize=3.0, elinewidth=1.0, capthick=1.0, zorder=5)
        fwhm_val = x_right - x_left
        ax.text(x_peak, peak_y * 1.06,
                f"{x_peak:.1f} nm\nFWHM {fwhm_val:.1f} nm",
                ha="center", va="bottom", fontsize=7.5, color="black")

    ax.set_xscale("log")
    ax.set_xlabel("Particle diameter (nm)")
    ax.set_ylabel("Probability density (1/nm)")
    ax.set_title(f"Normalized size distributions ({dist_type})")
    ax.minorticks_on()
    ax.legend(loc="upper right")

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        base = f"size_dist_{dist_type}_{fig_suffix}"
        fig.savefig(save_path / f"{base}.png", dpi=600)
        fig.savefig(save_path / f"{base}.pdf")
        print(f"  Saved: {base}.png / .pdf")
        plt.close(fig)
    else:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    data = load_litesizer_xlsx(XLSX_PATH)
    print(f"Loaded: {XLSX_PATH.name}  ({len(data.measurements)} measurements)")

    groups = _group_by_size(data.measurements, SIZES, DIST_TYPE, MAX_PDI)

    for size_nm, curves in sorted(groups.items()):
        n = len(curves)
        print(f"  {size_nm} nm: {n} measurement(s)" if n else f"  [SKIP] {size_nm} nm – no data")

    dls      = get_dls_reference_maps()
    size_map = dls["size_override_nm"]

    plot_all_sizes(groups, DIST_TYPE, SAVE_PATH,
                   size_subset=[20, 50, 100],
                   dls_size_map=size_map,
                   fig_suffix="small")

    plot_all_sizes(groups, DIST_TYPE, SAVE_PATH,
                   size_subset=[200, 500, 1000],
                   dls_size_map=size_map,
                   fig_suffix="large")


if __name__ == "__main__":
    main()
