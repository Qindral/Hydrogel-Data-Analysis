"""
Normalize MSD diffusion coefficients: D_hydrogel / D_water.

Uses:
  - Pickle cache from MSD_FromTrackmate_20mg.py (hydrogel)
  - Pickle cache from MSD_FromTrackmate_D0.py (water)
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
CACHE_D0 = Path(__file__).parent / "cache" / "msd_d0_results.pkl"

SAVE_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")

RF_NM = 8.0
PARTICLE_SIZES_NM = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
IMMOBILE_THRESHOLD_NM = 99
X_LIM = (0.1, 1000.0)
Y_LIM = (0.0, 1.0)



# ── Helpers ────────────────────────────────────────────────────────

def _load_cache(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_D_by_size(results: dict) -> pd.DataFrame:
    """Extract (particle_size_nm, D_MSD) pairs from cache results."""
    rows = [
        {"particle_size_nm": float(r["particle_size_nm"]), "D_MSD": float(r["D_MSD"])}
        for r in results.values()
        if r.get("particle_size_nm") is not None and r.get("D_MSD") is not None
    ]
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
        return np.exp(-np.pi * ((rs + RF_NM) / (psi + 2.0 * RF_NM)) ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, _ = curve_fit(
            model, rs_fit, y_fit,
            p0=[50.0],
            sigma=sigma,
            absolute_sigma=False,
            bounds=([1e-6], [1e6]),
            maxfev=10000,
        )
    psi_fit = popt[0]
    x_line = np.logspace(np.log10(X_LIM[0]), np.log10(X_LIM[1]), 200)
    y_line = model(x_line / 2.0, psi_fit)
    return x_line, y_line, rf"Amsden fit ($\xi$={psi_fit:.3g} nm)"


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    # Load data
    df_h = _extract_D_by_size(_load_cache(CACHE_20MG))
    df_w = _extract_D_by_size(_load_cache(CACHE_D0))

    agg_h = _aggregate(df_h)
    agg_w = _aggregate(df_w)

    # DLS measured sizes for x-axis mapping
    dls = get_dls_reference_maps()
    size_map = dls["size_override_nm"]
    size_errs = dls["size_err_nm"]

    # Build ratio table for all nominal sizes
    all_sizes = np.array(PARTICLE_SIZES_NM)
    h_map = agg_h.set_index("particle_size_nm")
    w_map = agg_w.set_index("particle_size_nm")

    ratios = np.full_like(all_sizes, np.nan)
    errors = np.full_like(all_sizes, np.nan)

    for i, s in enumerate(all_sizes):
        if s > IMMOBILE_THRESHOLD_NM:
            ratios[i], errors[i] = 0.0, 0.0
        elif s in h_map.index and s in w_map.index:
            d_h, d_h_err = h_map.loc[s, "mean"], h_map.loc[s, "std"]
            d_w, d_w_err = w_map.loc[s, "mean"], w_map.loc[s, "std"]
            ratios[i], errors[i] = _ratio_with_error(d_h, d_w, d_h_err, d_w_err)

    # Print summary
    summary = pd.DataFrame({
        "size_nm": all_sizes,
        "D_h/D_water": ratios,
        "err": errors,
    })
    print(summary.to_string(index=False))

    # Save CSV
    out_dir = Path(SAVE_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "msd_ratio_hydrogel_vs_water.csv", index=False)

    # Amsden fit (only mobile particles)
    fit = _fit_amsden(
        np.array([size_map.get(s, s) for s in all_sizes]),
        ratios, errors,
    )

    # ── Plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(constrained_layout=True)

    xs = np.array([size_map.get(s, s) for s in all_sizes])
    xerr = np.array([size_errs.get(s, 0.0) for s in all_sizes])

    ax.errorbar(
        xs, ratios, yerr=errors, xerr=xerr,
        fmt="o",
        markersize=np.sqrt(26),
        markerfacecolor="#0000da",
        markeredgecolor="#000099",
        markeredgewidth=0.8,
        ecolor="#000099",
        elinewidth=1.2,
        capsize=3.0,
        capthick=1.2,
        zorder=3,
        label="Mean \u00b1 SD",
    )

    if fit is not None:
        ax.plot(fit[0], fit[1], color="#000000", linewidth=1.2, label=fit[2])

    #ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)

    tick_x = [min(size_map.get(s, s), X_LIM[1]) for s in PARTICLE_SIZES_NM]
    ax.set_xticks(tick_x)
    ax.set_xticklabels([f"{int(s)}" for s in PARTICLE_SIZES_NM])

    ax.set_xlabel("Particle size (nm)")
    ax.set_ylabel(r"$D_\mathrm{hydrogel}\,/\,D_\mathrm{water}$")
    ax.set_title("Hydrogel / Water (MSD)")
    ax.minorticks_on()
    ax.legend(loc="best")

    plt.show()

    fig.savefig(out_dir / "msd_ratio_hydrogel_vs_water.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
