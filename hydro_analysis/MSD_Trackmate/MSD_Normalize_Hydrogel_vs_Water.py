"""
Normalize MSD diffusion coefficients: D_hydrogel / D_water.

Uses:
  - Pickle cache from MSD_FromTrackmate_20mg.py  (hydrogel 20 mg/mL)
  - Pickle cache from MSD_FromTrackmate_13mg.py  (hydrogel 13 mg/mL, optional)
  - Pickle cache from MSD_FromTrackmate_D0.py    (water)
  - DLS size measurements from core.io.get_dls_reference_maps (x-axis)
"""

from __future__ import annotations

from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit, OptimizeWarning

from hydro_analysis.core.io import get_dls_reference_maps

# ── Style (Style_guide.txt) ───────────────────────────────────────
mpl.rcParams.update({
    "figure.figsize": (7.15, 5.00),
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "font.family": "Source Sans 3",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
    "legend.fontsize": 9,
    "legend.frameon": False,
})

# ── Configuration ──────────────────────────────────────────────────
CACHE_20MG = Path(__file__).parent / "cache" / "msd_20mg_results.pkl"
CACHE_13MG = Path(__file__).parent / "cache" / "msd_13mg_results.pkl"
CACHE_D0   = Path(__file__).parent / "cache" / "msd_d0_results.pkl"

# Set to False to exclude 13 mg/mL data from the plot
INCLUDE_13MG =False

SAVE_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")

RF_NM = 4.0
PARTICLE_SIZES_NM = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
IMMOBILE_THRESHOLD_NM = 99
X_LIM = (0.1, 150)
Y_LIM = (0.0, 1.0)

# Colours per series  (style guide: blue family for 20 mg, orange for 13 mg)
STYLE_20MG = dict(
    face="#0000da", edge="#000099",
    ecolor="#000099", fit_color="#000099",
    label="20 mg/mL",
)
STYLE_13MG = dict(
    face="#da6000", edge="#994500",
    ecolor="#994500", fit_color="#994500",
    label="13 mg/mL",
)



# ── Helpers ────────────────────────────────────────────────────────

def _load_cache(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_D_by_size(results: dict,
                       size_override: dict[float, float] | None = None) -> pd.DataFrame:
    """Extract (particle_size_nm, D_MSD) pairs from cache results.

    size_override maps nominal sizes → DLS-measured sizes (e.g. 20.0 → 34.8).
    When provided, particle_size_nm is remapped so all downstream operations
    (aggregation, ratio building, plotting) use the real physical sizes.
    """
    rows = []
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        d_msd   = r.get("D_MSD")
        if size_nm is None or d_msd is None:
            continue
        size_nm = float(size_nm)
        if size_override:
            size_nm = float(size_override.get(size_nm, size_nm))
        rows.append({"particle_size_nm": size_nm, "D_MSD": float(d_msd)})
    return pd.DataFrame(rows, columns=["particle_size_nm", "D_MSD"])


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group by particle size -> mean, std, count of D_MSD."""
    g = df.groupby("particle_size_nm")["D_MSD"]
    return g.agg(["mean", "std", "count"]).reset_index()


def _ratio_with_error(d_num, d_den, err_num, err_den):
    """Compute ratio and propagated error (Gaussian)."""
    ratio = d_num / d_den
    rel_err = np.sqrt((err_num / d_num) ** 2 + (err_den / d_den) ** 2)
    return ratio, ratio * rel_err


def _fit_amsden(sizes_nm, ratios, errors):
    """Fit Amsden obstruction model; returns (x_line, y_line, label) or None."""
    mask = np.isfinite(ratios) & (ratios > 0) & (sizes_nm > 0)
    rs_fit = sizes_nm[mask] / 2.0
    y_fit = ratios[mask]
    sigma = errors[mask] if np.all(np.isfinite(errors[mask])) else None

    if rs_fit.size < 2:
        return None

    def model(rs, psi):
        return np.exp(-np.pi * ((rs + RF_NM) / (psi +RF_NM)) ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(
            model, rs_fit, y_fit,
            p0=[50.0],
            sigma=sigma,
            absolute_sigma=False,
            bounds=([1e-6], [1e6]),
            maxfev=10000,
        )
    psi_fit = popt[0]
    psi_err = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else 0.0
    print(f"  Amsden fit:  ξ = {psi_fit:.3g} ± {psi_err:.3g} nm")
    x_line  = np.logspace(np.log10(X_LIM[0]), np.log10(X_LIM[1]), 200)
    y_line  = model(x_line / 2.0, psi_fit)
    y_upper = model(x_line / 2.0, psi_fit + psi_err)
    y_lower = model(x_line / 2.0, max(psi_fit - psi_err, 1e-6))
    return x_line, y_line, "Amsden fit", y_lower, y_upper


# ── Helpers (continued) ────────────────────────────────────────────

def _build_ratios(agg_h: pd.DataFrame, agg_w: pd.DataFrame,
                  dls_sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (ratios, errors) arrays aligned to dls_sizes.

    dls_sizes: DLS-mapped sizes in nm (same order as PARTICLE_SIZES_NM).
    The immobile threshold is applied against the nominal sizes (PARTICLE_SIZES_NM)
    so that particles ≥ 100 nm nominal are always forced to 0.
    """
    nominal = np.array(PARTICLE_SIZES_NM)
    h_map   = agg_h.set_index("particle_size_nm")
    w_map   = agg_w.set_index("particle_size_nm")

    ratios = np.full(len(dls_sizes), np.nan)
    errors = np.full(len(dls_sizes), np.nan)

    for i, (s_dls, s_nom) in enumerate(zip(dls_sizes, nominal)):
        if s_nom > IMMOBILE_THRESHOLD_NM:
            ratios[i], errors[i] = 0.0, 0.0
        elif s_dls in h_map.index and s_dls in w_map.index:
            d_h, d_h_err = h_map.loc[s_dls, "mean"], h_map.loc[s_dls, "std"]
            d_w, d_w_err = w_map.loc[s_dls, "mean"], w_map.loc[s_dls, "std"]
            ratios[i], errors[i] = _ratio_with_error(d_h, d_w, d_h_err, d_w_err)

    return ratios, errors


def _plot_series(ax, xs, xerr, ratios, errors, style: dict,
                 fit=None, show_fit_label: bool = True) -> None:
    """Plot one hydrogel series (data + optional Amsden fit)."""
    ax.errorbar(
        xs, ratios, yerr=errors, xerr=xerr,
        fmt="o",
        markersize=np.sqrt(26),
        markerfacecolor=style["face"],
        markeredgecolor=style["edge"],
        markeredgewidth=0.8,
        ecolor=style["ecolor"],
        elinewidth=1.2,
        capsize=3.0,
        capthick=1.2,
        zorder=3,
        label=style["label"],
    )
    if fit is not None:
        fit_label = f"{style['label']} {fit[2]}" if show_fit_label else None
        ax.plot(fit[0], fit[1], color=style["fit_color"], linewidth=1.2,
                label=fit_label)
        if len(fit) > 3:
            ax.fill_between(fit[0], fit[3], fit[4],
                            color=style["fit_color"], alpha=0.15, linewidth=0)
            ax.plot(fit[0], fit[3], color=style["fit_color"], linewidth=0.7,
                    linestyle="--", alpha=0.5)
            ax.plot(fit[0], fit[4], color=style["fit_color"], linewidth=0.7,
                    linestyle="--", alpha=0.5)


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    # DLS measured sizes: map nominal → real physical size (e.g. 20 nm → 34.8 nm)
    dls = get_dls_reference_maps()
    size_map  = dls["size_override_nm"]   # {nominal_nm: dls_nm}
    size_errs = dls["size_err_nm"]

    nominal_sizes = np.array(PARTICLE_SIZES_NM)
    xs   = np.array([float(size_map.get(float(s), float(s)))   for s in nominal_sizes])  # DLS positions
    xerr = np.array([float(size_errs.get(float(s), 0.0)) for s in nominal_sizes])

    out_dir = Path(SAVE_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 20 mg/mL series ────────────────────────────────────────────
    agg_w  = _aggregate(_extract_D_by_size(_load_cache(CACHE_D0),   size_map))
    agg_20 = _aggregate(_extract_D_by_size(_load_cache(CACHE_20MG), size_map))
    ratios_20, errors_20 = _build_ratios(agg_20, agg_w, xs)

    fit_20 = _fit_amsden(xs, ratios_20, errors_20)

    summary_20 = pd.DataFrame({
        "size_nm": nominal_sizes,
        "D_20mg/D_water": ratios_20,
        "err_20mg": errors_20,
    })
    print("── 20 mg/mL ──")
    print(summary_20.to_string(index=False))
    summary_20.to_csv(out_dir / "msd_ratio_20mg_vs_water.csv", index=False)

    # ── 13 mg/mL series (optional) ─────────────────────────────────
    ratios_13 = errors_13 = fit_13 = None
    if INCLUDE_13MG:
        agg_13 = _aggregate(_extract_D_by_size(_load_cache(CACHE_13MG), size_map))
        ratios_13, errors_13 = _build_ratios(agg_13, agg_w, xs)
        fit_13 = _fit_amsden(xs, ratios_13, errors_13)

        summary_13 = pd.DataFrame({
            "size_nm": nominal_sizes,
            "D_13mg/D_water": ratios_13,
            "err_13mg": errors_13,
        })
        print("\n── 13 mg/mL ──")
        print(summary_13.to_string(index=False))
        summary_13.to_csv(out_dir / "msd_ratio_13mg_vs_water.csv", index=False)

    # ── Plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(constrained_layout=True)

    _plot_series(ax, xs, xerr, ratios_20, errors_20, STYLE_20MG, fit_20)

    if INCLUDE_13MG and ratios_13 is not None:
        _plot_series(ax, xs, xerr, ratios_13, errors_13, STYLE_13MG, fit_13)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)

    ax.xaxis.set_major_locator(MultipleLocator(10))

    ax.set_xlabel("Particle size (nm)")
    ax.set_ylabel(r"$D_\mathrm{hydrogel}\,/\,D_\mathrm{water}$")
    ax.set_title("Hydrogel / Water (MSD)")
    ax.minorticks_on()
    ax.legend(loc="best")

    fig.savefig(out_dir / "msd_ratio_hydrogel_vs_water.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / "msd_ratio_hydrogel_vs_water.pdf")

    # ── Weitere Grafiken ──────────────────────────────────────────────

    # Filter auf nominale Größen ≤ 100 nm (DLS: ~35, 50.5, 102.5 nm)
    mask_100  = np.array(PARTICLE_SIZES_NM) <= 100.0
    xs_100    = xs[mask_100];    xerr_100  = xerr[mask_100]
    r20_100   = ratios_20[mask_100]; e20_100 = errors_20[mask_100]
    fit_20_100 = _fit_amsden(xs_100, r20_100, e20_100)

    r13_100 = e13_100 = fit_13_100 = None
    if INCLUDE_13MG and ratios_13 is not None:
        r13_100    = ratios_13[mask_100]; e13_100  = errors_13[mask_100]
        fit_13_100 = _fit_amsden(xs_100, r13_100, e13_100)

    X_ALL = (25.0, 1300.0)   # alle Partikel (35–1266 nm DLS)
    X_100 = (25.0, 115.0)    # nur 35–100 nm

    def _decorate(ax, xlim, legend=True):
        ax.set_xlim(*xlim)
        ax.set_ylim(*Y_LIM)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_xlabel("Particle size (nm)")
        ax.set_ylabel(r"$D_\mathrm{hydrogel}\,/\,D_\mathrm{water}$")
        ax.minorticks_on()
        if legend:
            ax.legend(loc="best")

    # ── A: 20 mg – alle Partikel, kein Fit ───────────────────────────
    fig_a, ax = plt.subplots(constrained_layout=True)
    _plot_series(ax, xs, xerr, ratios_20, errors_20, STYLE_20MG)
    _decorate(ax, X_ALL)
    fig_a.savefig(out_dir / "msd_ratio_20mg_all.png",  dpi=600, bbox_inches="tight")
    fig_a.savefig(out_dir / "msd_ratio_20mg_all.pdf")

    # ── B: 13 mg – alle Partikel, kein Fit ───────────────────────────
    fig_b = None
    if INCLUDE_13MG and ratios_13 is not None:
        fig_b, ax = plt.subplots(constrained_layout=True)
        _plot_series(ax, xs, xerr, ratios_13, errors_13, STYLE_13MG)
        _decorate(ax, X_ALL)
        fig_b.savefig(out_dir / "msd_ratio_13mg_all.png",  dpi=600, bbox_inches="tight")
        fig_b.savefig(out_dir / "msd_ratio_13mg_all.pdf")

    # ── C: 20 mg – 35–100 nm, mit Fit ────────────────────────────────
    fig_c, ax = plt.subplots(constrained_layout=True)
    _plot_series(ax, xs_100, xerr_100, r20_100, e20_100, STYLE_20MG, fit=fit_20_100)
    _decorate(ax, X_100)
    fig_c.savefig(out_dir / "msd_ratio_20mg_fit.png",  dpi=600, bbox_inches="tight")
    fig_c.savefig(out_dir / "msd_ratio_20mg_fit.pdf")

    # ── D: 13 mg – 35–100 nm, mit Fit ────────────────────────────────
    fig_d = None
    if INCLUDE_13MG and r13_100 is not None:
        fig_d, ax = plt.subplots(constrained_layout=True)
        _plot_series(ax, xs_100, xerr_100, r13_100, e13_100, STYLE_13MG, fit=fit_13_100)
        _decorate(ax, X_100)
        fig_d.savefig(out_dir / "msd_ratio_13mg_fit.png",  dpi=600, bbox_inches="tight")
        fig_d.savefig(out_dir / "msd_ratio_13mg_fit.pdf")

    # ── E: 13 + 20 mg – 35–100 nm, Datenpunkte ───────────────────────
    fig_e, ax = plt.subplots(constrained_layout=True)
    _plot_series(ax, xs_100, xerr_100, r20_100, e20_100, STYLE_20MG)
    if INCLUDE_13MG and r13_100 is not None:
        _plot_series(ax, xs_100, xerr_100, r13_100, e13_100, STYLE_13MG)
    _decorate(ax, X_100)
    fig_e.savefig(out_dir / "msd_ratio_comparison.png",  dpi=600, bbox_inches="tight")
    fig_e.savefig(out_dir / "msd_ratio_comparison.pdf")

    # ── F: 13 + 20 mg – 35–100 nm, mit Fits (keine Parameter) ────────
    fig_f, ax = plt.subplots(constrained_layout=True)
    _plot_series(ax, xs_100, xerr_100, r20_100, e20_100, STYLE_20MG,
                 fit=fit_20_100, show_fit_label=False)
    if INCLUDE_13MG and r13_100 is not None:
        _plot_series(ax, xs_100, xerr_100, r13_100, e13_100, STYLE_13MG,
                     fit=fit_13_100, show_fit_label=False)
    _decorate(ax, X_100)
    fig_f.savefig(out_dir / "msd_ratio_comparison_fit.png",  dpi=600, bbox_inches="tight")
    fig_f.savefig(out_dir / "msd_ratio_comparison_fit.pdf")

    plt.show()


if __name__ == "__main__":
    main()
