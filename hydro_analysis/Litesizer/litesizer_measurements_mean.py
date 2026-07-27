"""
Litesizer Measurements Mean
===========================
Aggregates three DLS repeat measurements per nominal particle size from an
Anton Paar LiteSizer XLSX export (Kalliope software), computes combined
statistics with proper SEM, fits a log-normal to each combined distribution
for FWHM reporting, and produces a publication-quality figure plus a
summary table.

Usage (CLI):
    python litesizer_measurements_mean.py [path/to/file.xlsx]

Default XLSX_PATH is used when no argument is given.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from hydro_analysis.Litesizer.litesizer_parser import load_litesizer_xlsx

# ── Configuration ──────────────────────────────────────────────────────────────
XLSX_PATH      = Path(r"H:\Daten Promotion Sicherung\Lite Sizer Particle Measurements\Size_repitition_All Sizes.xlsx")
SAVE_PATH      = None           # Set to Path to export figures; None → plt.show()
SIZES          = [20, 50, 100, 200, 500, 1000]
MAX_PDI        = 42.0
DLS_CACHE_FILE = Path(__file__).parent / "cache" / "dls_reference.pkl"

# Renamed labels for figures: nominal (nm) → label (nm)
PARTICLE_LABELS = {20: 35, 50: 50, 100: 100, 200: 240, 500: 560, 1000: 1370}

COLORS = {
    "base": ["#0000da", "#004cff", "#00c4ff", "#49ffad", "#adff49", "#ffd700", "#ff6800", "#da0000"],
    "dark": ["#000099", "#0035b2", "#0089b2", "#33b279", "#79b233", "#b29600", "#b24900", "#990000"],
}

mpl.rcParams.update({
    "figure.figsize":    (7.15, 5.00),
    "figure.dpi":        150,
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
    "font.family":       "Source Sans 3",
    "font.size":         9,
    "axes.titlesize":    12,
    "axes.titleweight":  "semibold",
    "axes.labelsize":    10,
    "axes.linewidth":    1.0,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.major.size":  4.0,
    "ytick.major.size":  4.0,
    "xtick.minor.size":  2.0,
    "ytick.minor.size":  2.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "lines.linewidth":   1.8,
    "lines.markersize":  4.5,
    "legend.fontsize":   9,
    "legend.frameon":    False,
})

_SQRT2LN2 = np.sqrt(2.0 * np.log(2.0))


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_litesizer(xlsx_path: str | Path) -> list[dict]:
    """Load a LiteSizer XLSX export and return one dict per measurement.

    Parameters
    ----------
    xlsx_path : str or Path
        Path to the Kalliope-exported XLSX file.

    Returns
    -------
    list[dict]
        One dict per measurement sheet. Raises FileNotFoundError if the
        file is missing, or ValueError if mandatory fields are absent.
    """
    data = load_litesizer_xlsx(xlsx_path)
    records: list[dict] = []

    for m in data.measurements:
        if m.hydrodynamic_diameter_nm is None:
            raise ValueError(
                f"Hydrodynamic diameter missing in measurement '{m.name}'. "
                "Check the file or the parser."
            )
        if m.polydispersity_pct is None:
            raise ValueError(
                f"Polydispersity missing in measurement '{m.name}'. "
                "Check the file or the parser."
            )

        records.append({
            "name":               m.name,
            "z_average_nm":       m.hydrodynamic_diameter_nm,
            "polydispersity_pct": m.polydispersity_pct,
            "intercept_g1sq":     m.intercept_g2,
            "diffusion_coeff_um2s": m.diffusion_coefficient_um2s,
            "processed_runs":     m.processed_runs,
            # Peak positions (first peak of each weighting)
            "peak_intensity_nm":  (m.peak_analysis_intensity.peak_values[0]
                                   if m.peak_analysis_intensity.peak_values else None),
            "peak_volume_nm":     (m.peak_analysis_volume.peak_values[0]
                                   if m.peak_analysis_volume.peak_values else None),
            "peak_number_nm":     (m.peak_analysis_number.peak_values[0]
                                   if m.peak_analysis_number.peak_values else None),
            "sd_peak_intensity":  (m.peak_analysis_intensity.std_values[0]
                                   if m.peak_analysis_intensity.std_values else None),
            "sd_peak_volume":     (m.peak_analysis_volume.std_values[0]
                                   if m.peak_analysis_volume.std_values else None),
            "sd_peak_number":     (m.peak_analysis_number.std_values[0]
                                   if m.peak_analysis_number.std_values else None),
            # D values
            "d10_intensity": m.d_values.d10_intensity,
            "d50_intensity": m.d_values.d50_intensity,
            "d90_intensity": m.d_values.d90_intensity,
            "d10_volume":    m.d_values.d10_volume,
            "d50_volume":    m.d_values.d50_volume,
            "d90_volume":    m.d_values.d90_volume,
            "d10_number":    m.d_values.d10_number,
            "d50_number":    m.d_values.d50_number,
            "d90_number":    m.d_values.d90_number,
            # Size distributions
            "dist_intensity": m.size_distribution_intensity,
            "dist_volume":    m.size_distribution_volume,
            "dist_number":    m.size_distribution_number,
        })

    return records


# ── Aggregation helpers ────────────────────────────────────────────────────────

def _nominal_size(name: str, sizes: list[int]) -> int | None:
    """Return the first nominal size (nm) found in the measurement name."""
    for s in sorted(sizes, reverse=True):   # longest match first (1000 before 100)
        if f"{s} nm" in name:
            return s
    return None


def _combine_distributions(records: list[dict], key: str) -> pd.DataFrame | None:
    """Interpolate individual distributions onto a shared grid and average them."""
    dfs = [r[key] for r in records
           if r.get(key) is not None and not r[key].empty]
    if not dfs:
        return None

    x_grid = np.unique(np.concatenate([df["diameter_nm"].to_numpy() for df in dfs]))
    x_grid.sort()

    y_stack = np.array([
        np.interp(x_grid,
                  df["diameter_nm"].to_numpy(),
                  df["frequency_pct"].to_numpy(),
                  left=0.0, right=0.0)
        for df in dfs
    ])
    return pd.DataFrame({"diameter_nm": x_grid, "frequency_pct": y_stack.mean(axis=0)})


def _fit_lognormal(dist_df: pd.DataFrame) -> dict:
    """Fit a Gaussian on log(diameter) and return peak/FWHM in nm.

    The fit describes the *shape* of the distribution, not the measurement
    uncertainty. Its output (mode, FWHM) represents sample polydispersity.

    Returns
    -------
    dict with keys: mode_nm, fwhm_nm, x_left_half, x_right_half, mu, sigma
    """
    x = dist_df["diameter_nm"].to_numpy(dtype=float)
    y = dist_df["frequency_pct"].to_numpy(dtype=float)
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    if len(x) < 5:
        raise ValueError("Fewer than 5 non-zero bins — cannot fit log-normal.")

    peak_idx = int(np.argmax(y))
    p0 = [np.log(float(x[peak_idx])), 0.5, float(y[peak_idx])]

    def _model(xv, mu, sigma, A):
        return A * np.exp(-((np.log(xv) - mu) ** 2) / (2.0 * sigma ** 2))

    popt, _ = curve_fit(_model, x, y, p0=p0, maxfev=8000)
    mu, sigma, _ = popt
    sigma = abs(sigma)

    x_left  = np.exp(mu - _SQRT2LN2 * sigma)
    x_right = np.exp(mu + _SQRT2LN2 * sigma)

    return {
        "mu":         mu,
        "sigma":      sigma,
        "mode_nm":    float(np.exp(mu)),
        "fwhm_nm":    float(x_right - x_left),
        "x_left_half":  float(x_left),
        "x_right_half": float(x_right),
    }


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(
    measurements: list[dict],
    sizes: list[int] = SIZES,
    max_pdi: float = MAX_PDI,
) -> dict:
    """Group measurements by nominal size and compute combined statistics.

    Reported uncertainty — sigma_dist, NOT SEM
    ──────────────────────────────────────────────────────────────────────
    The uncertainty on z_avg is sigma_dist = z_avg * PDI% / 100, which is
    the standard deviation of the particle SIZE distribution itself.
    Anton Paar's "Polydispersity [%]" is the coefficient of variation
    sigma/mu in size space, so the conversion is exact.

    This is intentionally NOT the SEM (std / sqrt(n)) of the z_avg across
    repeat measurements. The SEM would shrink with more repeats and would
    describe measurement precision, not sample polydispersity. For single-
    particle tracking, where each tracked particle is one draw from the
    population, sigma_dist is the physically relevant quantity.

    sigma_D [µm²/s] = D [µm²/s] * PDI% / 100   (same relative width)
    ──────────────────────────────────────────────────────────────────────

    Parameters
    ----------
    measurements : list[dict]
        Output of parse_litesizer().
    sizes : list[int]
        Nominal particle sizes to recognise.
    max_pdi : float
        Exclude individual measurements with PDI above this threshold.
        Populations with fewer than 2 remaining measurements are dropped.

    Returns
    -------
    dict
        Keyed by nominal size (int). Each value contains z_mean,
        sigma_dist_mean, D_mean, sigma_D_mean, individual records,
        combined distributions, and the log-normal fit result.
    """
    grouped: dict[int, list[dict]] = {s: [] for s in sizes}
    dropped_files: list[tuple[str, float]] = []

    for r in measurements:
        nom = _nominal_size(r["name"], sizes)
        if nom is None:
            print(f"  [SKIP no size match] {r['name']}")
            continue
        if r["polydispersity_pct"] > max_pdi:
            dropped_files.append((r["name"], r["polydispersity_pct"]))
            grouped[nom]  # ensure key exists but don't append
            continue
        grouped[nom].append(r)

    # ── Diagnostic block ──────────────────────────────────────────────────────
    print("\n── PDI filter diagnostic ─────────────────────────────────────────────")
    if dropped_files:
        for name, pdi in dropped_files:
            print(f"  [DROPPED  PDI={pdi:.1f}%] {name}")
    else:
        print("  No files dropped by PDI filter.")

    dropped_populations: list[int] = []
    for size_nm, records in sorted(grouped.items()):
        if len(records) < 2:
            dropped_populations.append(size_nm)
            tag = "NO valid measurements" if not records else f"only {len(records)} valid measurement"
            print(f"  [DROP population] {size_nm} nm — {tag}, excluded entirely")

    if not dropped_populations:
        print("  No populations dropped.")

    print("\n── Kept measurements per population ──────────────────────────────────")
    for size_nm, records in sorted(grouped.items()):
        if size_nm in dropped_populations or not records:
            continue
        print(f"  {size_nm} nm ({len(records)} measurements):")
        for r in records:
            sigma_d = r["z_average_nm"] * r["polydispersity_pct"] / 100.0
            print(f"    {r['name']}: z={r['z_average_nm']:.1f} nm, "
                  f"PDI={r['polydispersity_pct']:.1f}%, σ_dist={sigma_d:.1f} nm")
    print()

    result: dict[int, dict] = {}
    for size_nm, records in grouped.items():
        if size_nm in dropped_populations or not records:
            continue

        z_vals   = np.array([r["z_average_nm"]       for r in records])
        pdi_vals = np.array([r["polydispersity_pct"] for r in records])
        n = len(records)

        z_mean          = float(np.mean(z_vals))
        # sigma_dist = z * PDI/100 per measurement; report the mean across measurements
        sigma_dist_vals = z_vals * pdi_vals / 100.0
        sigma_dist_mean = float(np.mean(sigma_dist_vals))

        d_records = [(r["diffusion_coeff_um2s"], r["polydispersity_pct"])
                     for r in records if r["diffusion_coeff_um2s"] is not None]
        D_mean      = float(np.mean([d for d, _ in d_records])) if d_records else None
        sigma_D_mean = float(np.mean([d * p / 100.0 for d, p in d_records])) if d_records else None

        dist_intensity = _combine_distributions(records, "dist_intensity")
        dist_volume    = _combine_distributions(records, "dist_volume")
        dist_number    = _combine_distributions(records, "dist_number")

        fit = None
        if dist_number is not None and not dist_number.empty:
            try:
                fit = _fit_lognormal(dist_number)
            except Exception as exc:
                print(f"  [WARNING] Log-normal fit failed for {size_nm} nm: {exc}")

        result[size_nm] = {
            "nominal_nm":      size_nm,
            "n":               n,
            "z_mean":          z_mean,
            "sigma_dist_mean": sigma_dist_mean,
            "D_mean":          D_mean,
            "sigma_D_mean":    sigma_D_mean,
            "records":         records,
            "dist_intensity":  dist_intensity,
            "dist_volume":     dist_volume,
            "dist_number":     dist_number,
            "lognormal_fit":   fit,
        }

    return result


# ── Table ──────────────────────────────────────────────────────────────────────

def make_table(agg: dict) -> tuple[pd.DataFrame, str]:
    """Build a 4-column summary DataFrame and a LaTeX/booktabs table string.

    Columns: nominal size (nm) | measured size mean ± sigma_dist (nm) |
             D mean ± sigma_D (µm²/s) | N measurements

    sigma_dist = z_avg * PDI% / 100 (coefficient of variation in size space).
    sigma_D    = D * PDI% / 100      (same relative width for D).
    """
    rows: list[dict] = []

    for size_nm, g in sorted(agg.items()):
        z   = g["z_mean"]
        sig = g["sigma_dist_mean"]
        D   = g["D_mean"]
        sD  = g["sigma_D_mean"]
        d_str = f"{D:.4f} ± {sD:.4f}" if D is not None else "—"
        rows.append({
            "Nominal size (nm)":    size_nm,
            "Measured size (nm)":   f"{z:.1f} ± {sig:.1f}",
            "D (µm²/s)":            d_str,
            "N":                    g["n"],
        })

    df = pd.DataFrame(rows)

    # LaTeX — exact format requested in the task spec
    data_lines = []
    for _, row in df.iterrows():
        nom  = row["Nominal size (nm)"]
        size = row["Measured size (nm)"].replace("±", r"\pm")
        d    = row["D (µm²/s)"].replace("±", r"\pm")
        n    = row["N"]
        data_lines.append(
            rf"  {nom} & ${size}$ & ${d}$ & {n} \\"
        )

    latex = "\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{LiteSizer DLS summary. "
        r"Uncertainty is $\sigma_\text{dist} = z_\text{avg} \times \text{PDI}/100$, "
        r"the standard deviation of the particle-size distribution.}",
        r"\label{tab:litesizer_summary}",
        r"\begin{tabular}{cccc}",
        r"  \toprule",
        r"  Particle size nominal & Particle size measured & $D$ & Measurements \\",
        r"  (nm) & (nm) & ($\mu\text{m}^2/\text{s}$) & \\",
        r"  \midrule",
        *data_lines,
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return df, latex


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _draw_population_on_ax(
    ax: plt.Axes,
    ci: int,
    size_nm: int,
    g: dict,
    dist_key: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Draw one population's elements onto `ax`.

    Elements (in draw order):
      1. Individual repeat measurement curves — thin, semi-transparent,
         plotted only where frequency ≥ 0.9 % (tails excluded)
      2. Bold mean curve — labeled with nominal size for the axes legend
      3. Dashed vertical z-mean line, truncated at the mean curve's own height
      4. Peak marker (filled circle) + text showing z_mean ± sigma_dist
         to 0.1 nm precision

    Returns (x_m, y_m) of the thresholded mean curve, or None if absent.
    """
    color_base = COLORS["base"][ci % len(COLORS["base"])]
    color_dark  = COLORS["dark"][ci % len(COLORS["dark"])]

    # 2. Mean curve — threshold at 0.9 % to exclude noisy tails
    mean_df = g.get(dist_key)
    if mean_df is None or mean_df.empty:
        return None
    x_m = mean_df["diameter_nm"].to_numpy(dtype=float)
    y_m = mean_df["frequency_pct"].to_numpy(dtype=float)
    mask = y_m >= 0.9
    x_m, y_m = x_m[mask], y_m[mask]
    if len(x_m) < 2:
        return None

    # 1. Individual thin lines (same 0.9 % threshold applied per curve)
    for r in g["records"]:
        df_ind = r.get(dist_key)
        if df_ind is None or df_ind.empty:
            continue
        xi = df_ind["diameter_nm"].to_numpy(dtype=float)
        yi = df_ind["frequency_pct"].to_numpy(dtype=float)
        mi = yi >= 0.9
        if mi.sum() >= 2:
            ax.plot(xi[mi], yi[mi], color=color_base, linewidth=0.9, alpha=0.40)

    ax.plot(x_m, y_m, color=color_dark, linewidth=2.2, label=f"{size_nm} nm")

    peak_idx = int(np.argmax(y_m))
    peak_x   = float(x_m[peak_idx])
    peak_y   = float(y_m[peak_idx])

    # 3. Truncated dashed z-mean line (stops at this population's own curve height)
    z_mean = g["z_mean"]
    y_stop = float(np.interp(z_mean, x_m, y_m, left=0.0, right=0.0))
    if y_stop > 0:
        ax.plot([z_mean, z_mean], [0.0, y_stop],
                color=color_dark, linewidth=0.8, linestyle="--", alpha=0.55, zorder=3)

    # 4. Peak marker + annotation (0.1 nm precision)
    ax.plot(peak_x, peak_y, "o", color=color_dark, markersize=4.0, zorder=6)
    sigma_dist = g["sigma_dist_mean"]
    ax.text(peak_x, peak_y * 1.04,
            f"{z_mean:.1f} ± {sigma_dist:.1f} nm",
            ha="center", va="bottom", fontsize=7.0, color=color_dark)

    return x_m, y_m


# ── Overlay figure ─────────────────────────────────────────────────────────────

def plot_combined(
    agg: dict,
    weighting: str = "intensity",
    save_path: Path | None = None,
) -> plt.Figure:
    """Overlay figure — all nominal sizes on a single log-x axes (Figure A).

    Caption note: On the intensity-weighted plot the z-average dashed line
    sits noticeably to the right of each distribution's mode. This is not a
    plotting error; it reflects the Rayleigh d⁶ bias inherent to DLS, which
    up-weights larger particles in the scattered intensity. The number-weighted
    figure shows the z-average much closer to the mode.

    Parameters
    ----------
    agg : dict
        Output of aggregate().
    weighting : str
        "intensity", "volume", or "number".
    save_path : Path, optional
        If given, PNG + PDF are saved there; otherwise plt.show() is called.
    """
    dist_key = f"dist_{weighting}"
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

    for ci, (size_nm, g) in enumerate(sorted(agg.items())):
        _draw_population_on_ax(ax, ci, size_nm, g, dist_key)

    ax.set_xscale("log")
    ax.set_xlabel("Particle diameter (nm)")
    ax.set_ylabel(f"Frequency, {weighting}-weighted (%)")
    ax.set_title(f"LiteSizer size distributions — {weighting}-weighted")
    ax.minorticks_on()

    # Per-size colored legend (from ax.plot labels set in _draw_population_on_ax)
    ax.legend(loc="upper right", fontsize=8, title="Nominal size", title_fontsize=8)

    _save_or_show(fig, save_path, f"litesizer_overlay_{weighting}")
    return fig


# ── Faceted figure ─────────────────────────────────────────────────────────────

def plot_faceted(
    agg: dict,
    weighting: str = "number",
    save_path: Path | None = None,
) -> plt.Figure:
    """Faceted figure — one panel per nominal size, 2 × 3 grid (Figure B).

    Y-axis: independent per panel. Sharing a single y-range is impractical
    because intensity-weighted distributions span two orders of magnitude in
    peak height (d⁶ bias), and even number-weighted distributions vary
    substantially. Each panel therefore fills its own vertical space so
    distribution shape is readable.

    X-axis: independent log-scale per panel, auto-ranged to the population's
    non-zero support with a 20 % margin on each side in log-space.

    Parameters
    ----------
    agg : dict
        Output of aggregate().
    weighting : str
        "intensity", "volume", or "number".
    save_path : Path, optional
        If given, PNG + PDF are saved; otherwise plt.show() is called.
    """
    dist_key = f"dist_{weighting}"
    sizes_sorted = sorted(agg.keys())
    n = len(sizes_sorted)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7.15 * ncols / 2.0, 5.00 * nrows / 2.0),
        constrained_layout=True,
        squeeze=False,
    )

    for idx, size_nm in enumerate(sizes_sorted):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        g  = agg[size_nm]
        ci = idx  # colour index consistent with overlay figure

        result = _draw_population_on_ax(ax, ci, size_nm, g, dist_key)

        ax.set_xscale("log")
        ax.set_title(f"{size_nm} nm", fontsize=10, fontweight="semibold")
        ax.set_xlabel("Diameter (nm)", fontsize=8)
        ax.set_ylabel(f"{weighting}-weighted (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.minorticks_on()

        # Set x-range to population support + 20 % log-margin on each side
        if result is not None:
            x_m, _ = result
            log_lo = np.log10(x_m.min()) - 0.20
            log_hi = np.log10(x_m.max()) + 0.20
            ax.set_xlim(10 ** log_lo, 10 ** log_hi)

    # Hide any unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"LiteSizer size distributions — {weighting}-weighted (faceted)",
        fontsize=11, fontweight="semibold",
    )

    _save_or_show(fig, save_path, f"litesizer_faceted_{weighting}")
    return fig


# ── Save helper ────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: Path | None, base: str) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"{base}.png", dpi=600)
        print(f"  Saved: {base}.png")
        plt.close(fig)
    else:
        plt.show()


# ── Results text file ─────────────────────────────────────────────────────────

def write_results_txt(agg: dict, save_path: Path) -> None:
    """Write z_mean ± sigma_dist and D ± sigma_D per size to a plain-text file."""
    header = (
        f"{'Nominal':>10}  {'z_mean':>10}  {'sigma_dist':>10}  "
        f"{'D':>12}  {'sigma_D':>12}  {'N':>3}\n"
        f"{'(nm)':>10}  {'(nm)':>10}  {'(nm)':>10}  "
        f"{'(µm²/s)':>12}  {'(µm²/s)':>12}  {'':>3}\n"
        + "-" * 65
    )
    lines = ["LiteSizer DLS Results", "=" * 65, header]
    for size_nm, g in sorted(agg.items()):
        z   = g["z_mean"]
        sig = g["sigma_dist_mean"]
        D   = g["D_mean"]
        sD  = g["sigma_D_mean"]
        n   = g["n"]
        if D is not None:
            d_part = f"{D:>12.4f}  {sD:>12.4f}"
        else:
            d_part = f"{'—':>12}  {'—':>12}"
        lines.append(
            f"{size_nm:>10}  {z:>10.1f}  {sig:>10.1f}  {d_part}  {n:>3}"
        )
    lines.append("=" * 65)
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Results written: {save_path}")


# ── Summary figure ─────────────────────────────────────────────────────────────

def plot_summary(agg: dict, save_path: Path | None = None) -> plt.Figure:
    """Two-panel summary figure.

    Left:  z_mean ± sigma_dist vs nominal size; dashed y = x reference.
    Right: D ± sigma_D vs measured size (z_mean), with x and y error bars, all black.
    """
    sizes    = sorted(agg.keys())
    x_nom    = np.array(sizes, dtype=float)
    z_vals   = np.array([agg[s]["z_mean"]          for s in sizes])
    sig_vals = np.array([agg[s]["sigma_dist_mean"] for s in sizes])

    d_sizes  = [s for s in sizes if agg[s]["D_mean"] is not None]
    z_d      = np.array([agg[s]["z_mean"]          for s in d_sizes])
    sig_d_x  = np.array([agg[s]["sigma_dist_mean"] for s in d_sizes])
    d_vals   = np.array([agg[s]["D_mean"]          for s in d_sizes])
    sd_vals  = np.array([agg[s]["sigma_D_mean"]    for s in d_sizes])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 4.00), constrained_layout=True)

    # ── Left: measured size vs nominal size ───────────────────────────────────
    col_z = COLORS["dark"][0]
    ref_lo = min(float(x_nom.min()), float(z_vals.min())) * 0.7
    ref_hi = max(float(x_nom.max()), float(z_vals.max())) * 1.4
    ax1.plot([ref_lo, ref_hi], [ref_lo, ref_hi],
             "--", color="#aaaaaa", linewidth=1.0)
    ax1.errorbar(x_nom, z_vals, yerr=sig_vals,
                 fmt="o", color=col_z, markersize=5,
                 capsize=3.0, elinewidth=1.2, capthick=1.2, zorder=5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(ref_lo, ref_hi)
    ax1.set_ylim(ref_lo, ref_hi)
    ax1.set_xlabel("Nominal size (nm)")
    ax1.set_ylabel("Measured particle size (nm)")
    ax1.minorticks_on()

    # ── Right: D vs measured size, all black, x and y error bars ─────────────
    if len(d_sizes) > 0:
        ax2.errorbar(z_d, d_vals, xerr=sig_d_x, yerr=sd_vals,
                     fmt="o", color="black", markersize=5,
                     capsize=3.0, elinewidth=1.2, capthick=1.2, zorder=5)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Measured particle size (nm)")
        ax2.set_ylabel("D (µm²/s)")
        ax2.minorticks_on()

    _save_or_show(fig, save_path, "litesizer_summary")
    return fig


# ── Weighting comparison figure (C) ───────────────────────────────────────────

_WEIGHTINGS = [
    ("dist_intensity", "peak_intensity_nm", "sd_peak_intensity", "Intensity"),
    ("dist_volume",    "peak_volume_nm",    "sd_peak_volume",    "Volume"),
    ("dist_number",    "peak_number_nm",    "sd_peak_number",    "Number"),
]


def plot_weighting_comparison(
    agg: dict,
    save_path: Path | None = None,
) -> plt.Figure:
    """Figure C — 2 × 3 grid illustrating the dual-algorithm DLS output.

    Rows: most polydisperse population (top) and near-monodisperse (bottom),
    selected at runtime by PDI = sigma_dist / z_mean.
    Columns: intensity, volume, number weighting.

    Each panel shows:
      - Individual repeat-measurement curves (alpha 0.30, full y>0 support)
      - Bold mean curve
      - Solid vertical line at the mean distribution-peak position
      - Symmetric horizontal bar at half peak height: Peak ± mean(sd_peak)
      - Dashed vertical z-mean line (same x in all three panels of a row)

    Reading: the dashed z-avg line does NOT move across a row; the solid peak
    line DOES. The rightward offset on the intensity panel is the d⁶ bias.
    """
    by_pdi = sorted(
        agg.items(),
        key=lambda kv: kv[1]["sigma_dist_mean"] / kv[1]["z_mean"],
    )
    size_mono, g_mono = by_pdi[0]
    size_poly, g_poly = by_pdi[-1]

    sizes_sorted = sorted(agg.keys())
    populations  = [
        (size_poly, g_poly, sizes_sorted.index(size_poly)),  # row 0: most polydisperse
        (size_mono, g_mono, sizes_sorted.index(size_mono)),  # row 1: near-monodisperse
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.72, 7.00), constrained_layout=True)

    for row_idx, (size_nm, g, ci) in enumerate(populations):
        color_base = COLORS["base"][ci % len(COLORS["base"])]
        color_dark  = COLORS["dark"][ci % len(COLORS["dark"])]
        pdi_pct = g["sigma_dist_mean"] / g["z_mean"] * 100.0
        z_mean  = g["z_mean"]

        for col_idx, (dist_key, peak_key, sd_key, wlabel) in enumerate(_WEIGHTINGS):
            ax = axes[row_idx][col_idx]

            mean_df = g.get(dist_key)
            if mean_df is None or mean_df.empty:
                ax.set_visible(False)
                continue

            x_m = mean_df["diameter_nm"].to_numpy(dtype=float)
            y_m = mean_df["frequency_pct"].to_numpy(dtype=float)
            mask = y_m > 0
            x_m, y_m = x_m[mask], y_m[mask]
            if len(x_m) < 2:
                continue

            # Individual curves — full y > 0 support to show shape across weightings
            for r in g["records"]:
                df_ind = r.get(dist_key)
                if df_ind is None or df_ind.empty:
                    continue
                xi = df_ind["diameter_nm"].to_numpy(dtype=float)
                yi = df_ind["frequency_pct"].to_numpy(dtype=float)
                mi = yi > 0
                if mi.sum() >= 2:
                    ax.plot(xi[mi], yi[mi], color=color_base,
                            linewidth=0.8, alpha=0.30)

            # Mean curve
            ax.plot(x_m, y_m, color=color_dark, linewidth=2.0)
            peak_y = float(np.max(y_m))

            # Distribution peak and width from parsed peak_analysis fields
            peaks = [r[peak_key] for r in g["records"] if r.get(peak_key) is not None]
            sds   = [r[sd_key]   for r in g["records"] if r.get(sd_key)   is not None]

            if peaks:
                mean_peak  = float(np.mean(peaks))
                peak_label = "Distribution peak" if col_idx == 0 else "_nolegend_"
                ax.axvline(mean_peak, color=color_dark, linewidth=1.2,
                           linestyle="-", zorder=4, label=peak_label)
                if sds:
                    mean_sd = float(np.mean(sds))
                    ax.errorbar(mean_peak, peak_y / 2.0,
                                xerr=[[mean_sd], [mean_sd]],
                                fmt="none", color=color_base,
                                capsize=3.0, elinewidth=1.4, capthick=1.4,
                                alpha=0.85, zorder=5)

            # Dashed cumulant z-average — same x across all three panels in this row
            zmean_label = "Cumulant z-avg" if col_idx == 0 else "_nolegend_"
            ax.axvline(z_mean, color=color_dark, linewidth=0.9,
                       linestyle="--", alpha=0.70, zorder=3, label=zmean_label)

            ax.set_xscale("log")
            ax.set_xlabel("Particle diameter (nm)", fontsize=8)
            # Row identifier on the leftmost y-axis label
            if col_idx == 0:
                ax.set_ylabel(
                    f"{size_nm} nm beads (PDI {pdi_pct:.0f} %)\n{wlabel}-weighted (%)",
                    fontsize=8,
                )
            else:
                ax.set_ylabel(f"{wlabel}-weighted (%)", fontsize=8)
            ax.set_title(wlabel, fontsize=9, fontweight="semibold")
            ax.tick_params(labelsize=7)
            ax.minorticks_on()

            if col_idx == 0:
                ax.legend(fontsize=7, loc="upper left")

    _save_or_show(fig, save_path, "litesizer_weighting_comparison")
    return fig


# ── Cumulant overlay figure (D) ────────────────────────────────────────────────

def plot_cumulant_overlay(
    agg: dict,
    save_path: Path | None = None,
) -> plt.Figure | None:
    """Figure D — cumulant model vs. distribution analysis for the most monomodal population.

    The number-weighted distribution (what NNLS recovers) is shown alongside the
    implicit cumulant Gaussian (Gaussian on LINEAR d, mean = z_avg, sigma = z_avg * PDI/100).
    Both are normalized so the Gaussian peak equals the distribution peak height.
    If the population is truly monomodal the two overlap; visible disagreement
    reveals non-Gaussianity that the z-average is averaging over.
    """
    by_pdi = sorted(
        agg.items(),
        key=lambda kv: kv[1]["sigma_dist_mean"] / kv[1]["z_mean"],
    )
    size_nm, g = by_pdi[0]  # lowest PDI = most monomodal

    sizes_sorted = sorted(agg.keys())
    ci = sizes_sorted.index(size_nm)
    color_dark = COLORS["dark"][ci % len(COLORS["dark"])]

    dist_df = g.get("dist_number")
    if dist_df is None or dist_df.empty:
        print(f"[WARNING] No number-weighted distribution for {size_nm} nm — skipping Figure D.")
        return None

    x_m = dist_df["diameter_nm"].to_numpy(dtype=float)
    y_m = dist_df["frequency_pct"].to_numpy(dtype=float)
    mask = y_m > 0
    x_m, y_m = x_m[mask], y_m[mask]

    peak_y    = float(np.max(y_m))
    z_mean    = g["z_mean"]
    sigma     = g["sigma_dist_mean"]
    pdi_pct   = sigma / z_mean * 100.0

    # Plot range: cover the distribution support and ±4 σ around z_mean
    x_lo = min(float(x_m.min()), max(1.0, z_mean - 4.0 * sigma)) * 0.7
    x_hi = max(float(x_m.max()), z_mean + 4.0 * sigma) * 1.4

    # Cumulant Gaussian on LINEAR diameter — evaluated on a fine uniform grid
    x_gauss = np.linspace(max(0.5, x_lo), x_hi, 2000)
    y_gauss = peak_y * np.exp(-((x_gauss - z_mean) ** 2) / (2.0 * sigma ** 2))

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

    ax.plot(x_m, y_m, color=color_dark, linewidth=2.2,
            label="Distribution analysis (NNLS)")
    ax.plot(x_gauss, y_gauss, color="#888888", linewidth=1.6, linestyle="--",
            label="Cumulant approximation (Gaussian)")

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)

    # Annotations with arrows
    dist_peak_idx = int(np.argmax(y_m))
    dist_peak_x   = float(x_m[dist_peak_idx])
    ax.annotate(
        "what distribution\nanalysis recovers",
        xy=(dist_peak_x, peak_y),
        xytext=(0.22, 0.65),
        xycoords="data",
        textcoords="axes fraction",
        fontsize=7.5, color=color_dark, ha="center",
        arrowprops=dict(arrowstyle="->", color=color_dark, lw=0.8),
    )
    ax.annotate(
        "what cumulant\nanalysis assumes",
        xy=(z_mean, peak_y),
        xytext=(0.75, 0.65),
        xycoords="data",
        textcoords="axes fraction",
        fontsize=7.5, color="#555555", ha="center",
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
    )

    ax.set_xlabel("Particle diameter (nm)")
    ax.set_ylabel("Number-weighted frequency (%)")
    ax.set_title(
        f"Cumulant vs. distribution analysis — {size_nm} nm beads "
        f"(PDI {pdi_pct:.0f} %)"
    )
    ax.minorticks_on()
    ax.legend(fontsize=8)

    _save_or_show(fig, save_path, "litesizer_cumulant_overlay")
    return fig


# ── DLS cache ──────────────────────────────────────────────────────────────────

def save_dls_cache(agg: dict, path: Path = DLS_CACHE_FILE) -> None:
    """Persist DLS z_mean and D per nominal size for use by SPT scripts."""
    cache = {
        int(nom): {
            "z_mean_nm":     g["z_mean"],
            "sigma_dist_nm": g["sigma_dist_mean"],
            "D_mean_um2s":   g["D_mean"],
            "sigma_D_um2s":  g["sigma_D_mean"],
            "nominal_nm":    int(nom),
            "label_nm":      PARTICLE_LABELS.get(int(nom), int(nom)),
        }
        for nom, g in agg.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  DLS cache saved: {path}  ({len(cache)} sizes)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load, aggregate, print table, and produce three figures."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else XLSX_PATH

    print(f"Loading: {path}")
    measurements = parse_litesizer(path)
    print(f"  Parsed {len(measurements)} measurements\n")

    agg = aggregate(measurements)
    save_dls_cache(agg)

    print("── Per-size summary ──────────────────────────────────────────────────")
    for size_nm, g in sorted(agg.items()):
        fit = g.get("lognormal_fit")
        fwhm_str = f"  FWHM = {fit['fwhm_nm']:.1f} nm" if fit else ""
        D_mean   = g.get("D_mean")
        sigma_D  = g.get("sigma_D_mean")
        d_str = f"  D = {D_mean:.4f} ± {sigma_D:.4f} µm²/s (σ_D)" if D_mean is not None else ""
        sigma_dist = g["sigma_dist_mean"]
        pdi_pct    = sigma_dist / g["z_mean"] * 100.0
        print(
            f"  {size_nm:>5} nm  n={g['n']}"
            f"  z = {g['z_mean']:.1f} ± {sigma_dist:.1f} nm (σ_dist)"
            f"  PDI ≈ {pdi_pct:.1f}%"
            f"{d_str}"
            f"{fwhm_str}"
        )

    df, latex = make_table(agg)
    print("\n── Summary table ─────────────────────────────────────────────────────")
    print(df.to_string(index=False))
    print("\n── LaTeX ─────────────────────────────────────────────────────────────")
    print(latex)

    # Write results txt next to the XLSX file
    txt_path = path.parent / (path.stem + "_results.txt")
    write_results_txt(agg, txt_path)

    # Figure A — intensity-weighted overlay (Rayleigh d⁶ bias visible)
    plot_combined(agg, weighting="intensity", save_path=SAVE_PATH)

    # Figure B — number-weighted overlay (z-mean closer to peak; SPT-relevant)
    plot_combined(agg, weighting="number", save_path=SAVE_PATH)

    # Figure B2 — number-weighted faceted (cleaner single-population reading)
    plot_faceted(agg, weighting="number", save_path=SAVE_PATH)

    # Figure E — summary: measured size and D vs nominal size
    plot_summary(agg, save_path=SAVE_PATH)

    # Figure C — 2×3 weighting comparison: most polydisperse vs. near-monodisperse
    plot_weighting_comparison(agg, save_path=SAVE_PATH)

    # Figure D — cumulant Gaussian vs. number-weighted distribution (most monomodal pop.)
    plot_cumulant_overlay(agg, save_path=SAVE_PATH)


if __name__ == "__main__":
    main()
