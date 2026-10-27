"""
SPT D_eff / DLS D0 vs. particle size — 20 mg/mL C16 hydrogel, with an Amsden
obstruction-model fit and its uncertainty band.

Version of MSD_Deff_over_D0_vs_Size_20mg_WeightedAvg.py (user, 2026-10-26):
same data, axes, markers and reference line (data loading, ratio and error
propagation are imported from that script, not copied), plus

  - the obstruction model in the form of Amsden 2022 (Macromolecules 55, 8399,
    eq 3; after Amsden 1999 and Hadjiev & Amsden 2015) (user, 2026-10-26, E038)
        D_eff / D0 = exp[-pi * ((r_s + r_f) / (xi + 2 r_f))**2]
    with r_s = particle radius (DLS z-average diameter / 2), r_f = hydrated
    radius of the polymer chain / fibril and xi = mesh size. This is the same
    function as the earlier form exp[-(pi/4) ((r_s + r_f) / (r_bar + r_f))**2]
    with xi = 2 r_bar; the fit now uses xi directly;
  - weighted least squares (weights 1 / sigma**2, sigma = propagated
    D_eff/D0 error bar, absolute), in one of two modes (R_F_MODE):
        "fixed": r_f = FIBRIL_DIAMETER_NM / 2, free parameter xi
                 (user, 2026-10-26: first 6 nm, then 1.5 nm, E037 C);
        "free":  xi and r_f free, r_f bounded to R_F_BOUNDS_NM = 3-15 nm
                 (user, 2026-10-26). With two data points this fit has no
                 degree of freedom left. If r_f ends on a bound, it is reported
                 as such, and xi and its sigma are taken from the one-parameter
                 fit with r_f fixed at that bound;
  - one shaded band, +-1 sigma with absolute errors (user, 2026-10-26, E036 A):
    with r_f fixed (or on its bound) the band lies between the model curves at
    xi - sigma and xi + sigma (the model is monotonic in xi); with r_f free
    inside its bounds it is the +-1 sigma band of the model value, propagated
    from the full 2x2 covariance (linear error propagation).

The DLS size error (0.1-0.3 nm for 35/50 nm) is not propagated into the fit.
In "free" mode a chi2 profile over r_f (xi refitted at each r_f) is written
to the fit summary. For each point alone, the mesh size that puts the model
through that point (same r_f) and its 1 sigma from the point's error bar
(linear propagation) are written to the table and the summary (user, 2026-10-27).

Before the real fit, test_amsden_fit_synthetic() fits ratios generated from
the model with known parameters (noise-free, and with Gaussian noise of the
real relative error bars, fixed seed) and prints recovered vs. true values.
Synthetic values are never plotted or saved.

Reads:  cache/msd_20mg_result.pkl (via the WeightedAvg script's load_averages()).
Writes: <FILE_STEM>.pdf, <FILE_STEM>_fit.csv, <FILE_STEM>_fit_results.txt
        (fit summary, also printed) and <FILE_STEM>_exclusions.csv in SAVE_PATH;
        FILE_STEM = deff_over_d0_vs_size_20mg_amsden_rf<r_f>nm ("fixed",
        e.g. _rf1p5nm) or deff_over_d0_vs_size_20mg_amsden_rf_free ("free").
Run MSD_FromTrackmate_20mg.py first (or after any change to the raw data)
to refresh msd_20mg_result.pkl. Upstream data handling (edge-artifact removal,
fps rule, one analysis per movie, 10-frame filter, 6-lag fit with fixed
localisation error and motion blur, weighting by tracks >= 10 frames) is
documented in that script and in AGENTS.md section 0.2.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit, OptimizeWarning

from hydro_analysis.core.io import get_dls_reference_maps, get_dls_sizes
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _DASH_THEORY
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS
from hydro_analysis.MSD_Trackmate.MSD_Deff_over_D0_vs_Size_20mg_WeightedAvg import (
    CACHE_FILE, SAVE_PATH, STYLE, FIG_SIZE, X_MIN, X_MAX, Y_MIN, Y_MAX, COLOR_THEORY,
    load_averages, _ratio_with_error,
)

# ── Configuration ──────────────────────────────────────────────────────────────
R_F_MODE = "fixed"                 # "free" or "fixed"; decided by user, 2026-10-26 (E037 C)
FIBRIL_DIAMETER_NM = 3.0           # "fixed" mode, r_f = 1.5 nm; decided by user, 2026-10-26 (E037 C; first run used 12 nm)
R_F_BOUNDS_NM = (3.0, 15.0)        # "free" mode, bounds on r_f (radius); decided by user, 2026-10-26
BOUND_TOL_NM = 1e-3                # r_f closer than this to a bound counts as "on the bound" (numerical tolerance)
R_F_PROFILE_STEP_NM = 1.0          # step of the chi2 profile over r_f in the summary (diagnostic only)

# Hydrogel colour (Styleguide section 11, category B): fit line dark, band base
COLOR_FIT  = "#A6672D"
COLOR_BAND = "#D98C3D"
BAND_ALPHA = 0.3

N_CURVE_POINTS = 400

# Synthetic test (never plotted or saved)
SYNTH_XI_NM = 20.0
SYNTH_R_F_NM = 8.0                 # inside R_F_BOUNDS_NM
SYNTH_XI_OUTSIDE_NM = 4.0          # with SYNTH_R_F_OUTSIDE_NM: true r_f above the upper bound
SYNTH_R_F_OUTSIDE_NM = 25.0
SYNTH_N_REPEATS = 2000
SYNTH_SEED = 20261026

FILE_STEM = "deff_over_d0_vs_size_20mg_amsden" + (
    "_rf_free" if R_F_MODE == "free" else f"_rf{FIBRIL_DIAMETER_NM / 2.0:g}nm".replace(".", "p"))


def amsden_ratio(r_s_nm: np.ndarray, xi_nm: float, r_f_nm: float) -> np.ndarray:
    """Obstruction model, Amsden 2022 eq 3: D_eff / D0."""
    return np.exp(-np.pi * ((r_s_nm + r_f_nm) / (xi_nm + 2.0 * r_f_nm)) ** 2)


def _xi_single_point(r_s_nm: np.ndarray, ratios: np.ndarray, r_f_nm: float) -> np.ndarray:
    """xi that makes the model pass exactly through each point, at given r_f."""
    return (r_s_nm + r_f_nm) / np.sqrt(-np.log(ratios) / np.pi) - 2.0 * r_f_nm


def _xi_single_point_err(r_s_nm: np.ndarray, ratios: np.ndarray, ratio_errs: np.ndarray, r_f_nm: float) -> np.ndarray:
    """1 sigma of _xi_single_point() from the D_eff/D0 error bar (linear propagation, r_f fixed):
    d xi / d y = (r_s + r_f) / (2 pi y g^3), g = sqrt(-ln y / pi)."""
    g = np.sqrt(-np.log(ratios) / np.pi)
    return (r_s_nm + r_f_nm) * ratio_errs / (2.0 * np.pi * ratios * g ** 3)


def _fit_xi(r_s_nm: np.ndarray, ratios: np.ndarray, sigma: np.ndarray, r_f_nm: float) -> tuple[float, float]:
    """One-parameter fit of xi at fixed r_f; returns (xi, sigma_xi)."""
    p0 = float(np.mean(_xi_single_point(r_s_nm, ratios, r_f_nm)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        popt, pcov = curve_fit(
            lambda r, xi: amsden_ratio(r, xi, r_f_nm), r_s_nm, ratios, p0=[p0],
            sigma=sigma, absolute_sigma=True, bounds=([-2.0 * r_f_nm + 1e-9], [np.inf]),
        )
    return float(popt[0]), float(np.sqrt(pcov[0, 0]))


def _chi2(r_s_nm, ratios, sigma, xi_nm, r_f_nm) -> float:
    return float(np.sum(((ratios - amsden_ratio(r_s_nm, xi_nm, r_f_nm)) / sigma) ** 2))


def fit_amsden(r_s_nm: np.ndarray, ratios: np.ndarray, sigma: np.ndarray, mode: str | None = None) -> dict:
    """Weighted least-squares Amsden fit with absolute sigma (see module docstring).
    mode defaults to R_F_MODE."""
    mode = R_F_MODE if mode is None else mode
    cov = None
    r_f_err = float("nan")
    on_bound = ""
    if mode == "fixed":
        r_f = FIBRIL_DIAMETER_NM / 2.0
        xi, xi_err = _fit_xi(r_s_nm, ratios, sigma, r_f)
        n_free = 1
    elif mode == "free":
        lo, hi = R_F_BOUNDS_NM
        r_f0 = 0.5 * (lo + hi)
        p0 = [float(np.mean(_xi_single_point(r_s_nm, ratios, r_f0))), r_f0]
        with warnings.catch_warnings():
            # Covariance is checked below; it is only used when r_f is inside its bounds
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                amsden_ratio, r_s_nm, ratios, p0=p0, sigma=sigma, absolute_sigma=True,
                bounds=([-2.0 * lo + 1e-9, lo], [np.inf, hi]),
            )
        xi, r_f = float(popt[0]), float(popt[1])
        if abs(r_f - lo) < BOUND_TOL_NM or abs(r_f - hi) < BOUND_TOL_NM:
            on_bound = "lower" if abs(r_f - lo) < BOUND_TOL_NM else "upper"
            r_f = lo if on_bound == "lower" else hi
            xi, xi_err = _fit_xi(r_s_nm, ratios, sigma, r_f)
            n_free = 1
        else:
            if not np.all(np.isfinite(pcov)):
                raise ValueError("Amsden fit with free r_f: covariance could not be estimated.")
            cov = pcov
            xi_err, r_f_err = float(np.sqrt(pcov[0, 0])), float(np.sqrt(pcov[1, 1]))
            n_free = 2
    else:
        raise ValueError(f"R_F_MODE must be 'fixed' or 'free', got {mode!r}.")

    chi2 = _chi2(r_s_nm, ratios, sigma, xi, r_f)
    dof = len(ratios) - n_free
    return {
        "mode": mode,
        "xi_nm": xi,
        "xi_err_nm": xi_err,
        "r_f_nm": r_f,
        "r_f_err_nm": r_f_err,
        "on_bound": on_bound,
        "cov": cov,
        "n_free": n_free,
        "chi2": chi2,
        "dof": dof,
        "chi2_red": chi2 / dof if dof > 0 else float("nan"),
        "xi_single_point_nm": _xi_single_point(r_s_nm, ratios, r_f),
        "xi_single_point_err_nm": _xi_single_point_err(r_s_nm, ratios, sigma, r_f),
    }


def band_curves(res: dict, r_s_line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper edge of the +-1 sigma band (E036 A)."""
    xi, r_f = res["xi_nm"], res["r_f_nm"]
    if res["cov"] is None:
        hw = res["xi_err_nm"]
        if xi - hw <= -2.0 * r_f:
            raise ValueError(f"xi - sigma = {xi - hw:.3f} nm gives xi + 2 r_f <= 0; model undefined.")
        return amsden_ratio(r_s_line, xi - hw, r_f), amsden_ratio(r_s_line, xi + hw, r_f)
    # Linear propagation of the 2x2 covariance, numerical Jacobian
    y = amsden_ratio(r_s_line, xi, r_f)
    h = 1e-6
    jac = np.column_stack([
        (amsden_ratio(r_s_line, xi + h, r_f) - amsden_ratio(r_s_line, xi - h, r_f)) / (2 * h),
        (amsden_ratio(r_s_line, xi, r_f + h) - amsden_ratio(r_s_line, xi, r_f - h)) / (2 * h),
    ])
    sigma_y = np.sqrt(np.einsum("ij,jk,ik->i", jac, res["cov"], jac))
    return y - sigma_y, y + sigma_y


def chi2_profile(r_s_nm: np.ndarray, ratios: np.ndarray, sigma: np.ndarray) -> list[tuple[float, float, float, float]]:
    """(r_f, xi, sigma_xi, chi2) with xi refitted at each r_f over R_F_BOUNDS_NM."""
    lo, hi = R_F_BOUNDS_NM
    rows = []
    for r_f in np.arange(lo, hi + 0.5 * R_F_PROFILE_STEP_NM, R_F_PROFILE_STEP_NM):
        xi, xi_err = _fit_xi(r_s_nm, ratios, sigma, float(r_f))
        rows.append((float(r_f), xi, xi_err, _chi2(r_s_nm, ratios, sigma, xi, float(r_f))))
    return rows


def fit_summary_text(res: dict, sizes: list, xs: np.ndarray, ratios: np.ndarray,
                     ratio_errs: np.ndarray) -> str:
    """Plain-text summary of the Amsden fit (printed and written beside the figure)."""
    xi, r_f = res["xi_nm"], res["r_f_nm"]
    r_s = xs / 2.0
    model = amsden_ratio(r_s, xi, r_f)
    lines = [
        "Obstruction model fit (Amsden 2022, Macromolecules 55, 8399, eq 3), D_eff/D0 vs. particle size, "
        "20 mg/mL C16 hydrogel",
        "Model: D_eff/D0 = exp[-pi * ((r_s + r_f) / (xi + 2 r_f))^2], xi = mesh size, r_s = d_DLS / 2",
        "Fit: weighted least squares (weights 1/sigma^2, absolute sigma)",
    ]
    if res["mode"] == "fixed":
        lines.append(f"r_f fixed: {r_f:g} nm (diameter {FIBRIL_DIAMETER_NM:g} nm); free parameter xi")
    else:
        lines.append(f"r_f free within {R_F_BOUNDS_NM[0]:g} ... {R_F_BOUNDS_NM[1]:g} nm; free parameters xi, r_f")
    lines += [f"Data points: {len(ratios)}", ""]

    if res["on_bound"]:
        lines.append(f"r_f          = {r_f:.3f} nm  ON THE {res['on_bound'].upper()} BOUND "
                     f"(best fit within the bounds; no sigma for r_f)")
        lines.append(f"xi           = {xi:.3f} ± {res['xi_err_nm']:.3f} nm (absolute sigma, r_f held at the bound)")
    else:
        r_f_txt = f"{r_f:.3f} nm" + (f" ± {res['r_f_err_nm']:.3f} nm" if np.isfinite(res["r_f_err_nm"]) else " (fixed)")
        lines.append(f"r_f          = {r_f_txt}")
        lines.append(f"xi           = {xi:.3f} ± {res['xi_err_nm']:.3f} nm (absolute sigma)")
        if res["cov"] is not None:
            corr = res["cov"][0, 1] / np.sqrt(res["cov"][0, 0] * res["cov"][1, 1])
            lines.append(f"corr(xi, r_f) = {corr:.3f}")
    lines += [
        f"chi2         = {res['chi2']:.3f}",
        f"dof          = {res['dof']} (n - {res['n_free']} free parameter(s))",
        f"chi2_red     = {res['chi2_red']:.3f}" if res["dof"] > 0 else "chi2_red     = undefined (dof = 0)",
        "",
        "Band: +-1 sigma, absolute errors (E036 A)" + (
            f": xi {xi - res['xi_err_nm']:.3f} ... {xi + res['xi_err_nm']:.3f} nm at r_f = {r_f:.3f} nm"
            if res["cov"] is None else ": linear propagation of the (xi, r_f) covariance"),
        "",
        "Per point:",
        f"  {'nominal':>8} {'d_DLS':>8} {'r_s':>7} {'D_eff/D0':>9} {'error':>7} {'model':>7} "
        f"{'(y-m)/err':>9} {'xi single':>10} {'± 1 sigma':>10}",
        f"  {'(nm)':>8} {'(nm)':>8} {'(nm)':>7} {'':>9} {'':>7} {'':>7} {'':>9} {'(nm)':>10} {'(nm)':>10}",
    ]
    for s, d, rs, y, e, m, x1, x1e in zip(sizes, xs, r_s, ratios, ratio_errs, model, res["xi_single_point_nm"],
                                          res["xi_single_point_err_nm"]):
        lines.append(f"  {s:>8g} {d:>8.2f} {rs:>7.2f} {y:>9.4f} {e:>7.4f} {m:>7.4f} {(y - m) / e:>9.3f} "
                     f"{x1:>10.3f} {x1e:>10.3f}")
    lines.append("  xi single: mesh size from each point alone (model through the point, same r_f); "
                 "± 1 sigma from its D_eff/D0 error bar.")

    if res["mode"] == "free":
        lines += ["", f"chi2 profile over r_f (xi refitted at each r_f, step {R_F_PROFILE_STEP_NM:g} nm):",
                  f"  {'r_f (nm)':>9} {'xi (nm)':>9} {'sigma (nm)':>11} {'chi2':>8}"]
        for r_f_p, x_p, x_err, c2 in chi2_profile(r_s, ratios, ratio_errs):
            lines.append(f"  {r_f_p:>9.1f} {x_p:>9.3f} {x_err:>11.3f} {c2:>8.3f}")
        if len(ratios) == 2:
            g = np.sqrt(-np.log(ratios) / np.pi)
            a = (r_s[1] - r_s[0]) / (g[1] - g[0])          # a = xi + 2 r_f
            r_f_exact = g[0] * a - r_s[0]
            lines.append(f"  Without bounds the model passes exactly through both points (chi2 = 0) at "
                         f"r_f = {r_f_exact:.3f} nm, xi = {a - 2.0 * r_f_exact:.3f} nm.")
    return "\n".join(lines) + "\n"


def test_amsden_fit_synthetic(r_s_nm: np.ndarray, rel_err: np.ndarray) -> None:
    """Recover known parameters from model-generated ratios at the real sizes and
    relative errors. Prints recovered vs. true values; nothing is plotted or saved."""
    r_f_true = SYNTH_R_F_NM if R_F_MODE == "free" else FIBRIL_DIAMETER_NM / 2.0
    y_true = amsden_ratio(r_s_nm, SYNTH_XI_NM, r_f_true)
    sigma = rel_err * y_true

    exact = fit_amsden(r_s_nm, y_true, sigma)
    print(f"  Synthetic, noise-free: xi true {SYNTH_XI_NM:.4f} nm, recovered {exact['xi_nm']:.4f} "
          f"± {exact['xi_err_nm']:.4f} nm; r_f true {r_f_true:.4f} nm, recovered {exact['r_f_nm']:.4f} nm"
          f"{' (on ' + exact['on_bound'] + ' bound)' if exact['on_bound'] else ''}")

    if R_F_MODE == "free":
        y_out = amsden_ratio(r_s_nm, SYNTH_XI_OUTSIDE_NM, SYNTH_R_F_OUTSIDE_NM)
        out = fit_amsden(r_s_nm, y_out, rel_err * y_out)
        print(f"  Synthetic, true r_f {SYNTH_R_F_OUTSIDE_NM:g} nm above the bound: fitted r_f {out['r_f_nm']:.4f} nm, "
              f"on bound: '{out['on_bound'] or 'no'}' (expected 'upper')")

    rng = np.random.default_rng(SYNTH_SEED)
    xi_fit, xi_err, n_bound = [], [], 0
    for _ in range(SYNTH_N_REPEATS):
        y = y_true + rng.normal(0.0, sigma)
        if np.any(y <= 0) or np.any(y >= 1):
            raise ValueError("Synthetic draw produced a ratio outside (0, 1); reduce noise or change seed.")
        res = fit_amsden(r_s_nm, y, sigma)
        xi_fit.append(res["xi_nm"])
        xi_err.append(res["xi_err_nm"])
        n_bound += bool(res["on_bound"])
    xi_fit, xi_err = np.array(xi_fit), np.array(xi_err)
    pulls = (xi_fit - SYNTH_XI_NM) / xi_err
    print(f"  Synthetic, noisy (n = {SYNTH_N_REPEATS}, seed {SYNTH_SEED}): "
          f"median recovered xi {np.median(xi_fit):.3f} nm, SD of estimates {xi_fit.std(ddof=1):.3f} nm, "
          f"median reported sigma {np.median(xi_err):.3f} nm, "
          f"coverage of ±1 sigma {np.mean(np.abs(pulls) <= 1.0) * 100:.1f} % (expected 68.3 %), "
          f"fits with r_f on a bound: {n_bound}")


def main() -> None:
    averages = load_averages()
    print(f"Cache geladen: {CACHE_FILE} ({len(averages)} Größen)")

    dls_sizes     = get_dls_sizes()
    dls_maps      = get_dls_reference_maps()
    size_override = dls_maps["size_override_nm"]
    dls_D         = dls_maps["dls_D_um2_per_s"]
    dls_D_err     = dls_maps["dls_D_err_um2_per_s"]

    def _map_size(size: float) -> float:
        return float(dls_sizes.get(size, size_override.get(size, size)))

    # ── D_eff / D0 per size; sizes without both values are logged, not dropped silently
    sizes, ratios, ratio_errs, exclusions = [], [], [], []
    for size_nm in sorted(averages.keys()):
        fit = averages[size_nm].get("fit_results_MSD") or {}
        d_eff, d_eff_err = fit.get("D_um2_per_s"), fit.get("D_error")
        d0, d0_err = dls_D.get(size_nm), dls_D_err.get(size_nm, 0.0)
        if d_eff is None or d0 is None or not np.isfinite(d_eff) or not np.isfinite(d0) or d0 == 0:
            exclusions.append({"file": f"size {size_nm:g} nm", "step": "ratio",
                               "parameter": "D_eff and D0 required", "n_before": 1, "n_after": 0,
                               "reason": "D_eff or DLS D0 missing"})
            continue
        ratio, ratio_err = _ratio_with_error(d_eff, d0, d_eff_err or 0.0, d0_err)
        if not np.isfinite(ratio_err) or ratio_err <= 0:
            raise ValueError(f"Size {size_nm:g} nm: D_eff/D0 error is {ratio_err}; a weighted fit needs a positive error.")
        sizes.append(size_nm)
        ratios.append(ratio)
        ratio_errs.append(ratio_err)
        print(f"  {int(size_nm):>5} nm  D_eff/D0 = {ratio:.4g} ± {ratio_err:.2g}  "
              f"(n_files = {averages[size_nm].get('n_files')}, "
              f"n_tracks = {averages[size_nm].get('n_particles_pooled')})")

    xs = np.array([_map_size(s) for s in sizes])
    ratios, ratio_errs = np.array(ratios), np.array(ratio_errs)
    r_s = xs / 2.0

    # ── Verification on synthetic data, then the real fit ───────────────────
    print(f"Test des Fits an synthetischen Daten (Modus {R_F_MODE}):")
    test_amsden_fit_synthetic(r_s, ratio_errs / ratios)

    res = fit_amsden(r_s, ratios, ratio_errs)
    xi, xi_err, r_f = res["xi_nm"], res["xi_err_nm"], res["r_f_nm"]
    summary = fit_summary_text(res, sizes, xs, ratios, ratio_errs)
    print("\n" + summary)

    x_line = np.linspace(X_MIN, X_MAX, N_CURVE_POINTS)
    y_best = amsden_ratio(x_line / 2.0, xi, r_f)
    y_lo, y_hi = band_curves(res, x_line / 2.0)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        ax.axhline(1.0, color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY, zorder=3)

        ax.fill_between(x_line, y_lo, y_hi, color=COLOR_BAND, alpha=BAND_ALPHA, linewidth=0, zorder=1)
        ax.plot(x_line, y_best, color=COLOR_FIT, linewidth=1.5, zorder=4)

        for x, y, yerr, s in zip(xs, ratios, ratio_errs, sizes):
            dark = SIZE_COLORS[s][1]
            ax.errorbar([x], [y], yerr=[yerr], fmt="o", markersize=5,
                        markerfacecolor=dark, markeredgecolor=dark, markeredgewidth=0.6,
                        ecolor=dark, elinewidth=0.8, capsize=2.0, capthick=0.8, linestyle="None", zorder=6)

        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        ax.set_xlabel("Particle size (nm)")
        ax.set_ylabel(r"$D_\mathrm{eff}$ (SPT) / $D_0$ (DLS)")

        legend_elements = [
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
                   label="Unobstructed diffusion ($D_\\mathrm{eff}/D_0$ = 1)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="0.3",
                   markeredgecolor="0.3", markersize=5, markeredgewidth=0.6,
                   label=r"Hydrogel 20 mg mL$^{-1}$ (weighted average)", linestyle="None"),
            Line2D([0], [0], color=COLOR_FIT, linewidth=1.5,
                   label=rf"Amsden fit ($r_f$ = {r_f:g} nm, $\xi$ = {xi:.1f} nm)"),
            Patch(facecolor=COLOR_BAND, alpha=BAND_ALPHA, linewidth=0, label=r"$\pm$1$\sigma$"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 0.88), frameon=False)

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            pdf_path = SAVE_PATH / f"{FILE_STEM}.pdf"
            fig.savefig(pdf_path)
            print(f"Plot gespeichert: {pdf_path}")

            table = pd.DataFrame({
                "nominal_size_nm": sizes,
                "dls_diameter_nm": xs,
                "r_s_nm": r_s,
                "deff_over_d0": ratios,
                "deff_over_d0_err": ratio_errs,
                "model_at_best_fit": amsden_ratio(r_s, xi, r_f),
                "xi_single_point_nm": res["xi_single_point_nm"],
                "xi_single_point_err_nm": res["xi_single_point_err_nm"],
                "n_files": [averages[s].get("n_files") for s in sizes],
                "n_tracks_weight": [averages[s].get("n_particles_pooled") for s in sizes],
            })
            table["model"] = "Amsden 2022 eq 3"
            table["r_f_mode"] = R_F_MODE
            table["r_f_bounds_nm"] = f"{R_F_BOUNDS_NM[0]:g}-{R_F_BOUNDS_NM[1]:g}" if R_F_MODE == "free" else ""
            table["r_f_nm"] = r_f
            table["r_f_err_nm"] = res["r_f_err_nm"]
            table["r_f_on_bound"] = res["on_bound"]
            table["xi_nm"] = xi
            table["xi_err_nm"] = xi_err
            table["chi2"] = res["chi2"]
            table["dof"] = res["dof"]
            table.to_csv(SAVE_PATH / f"{FILE_STEM}_fit.csv", index=False)
            pd.DataFrame(exclusions, columns=["file", "step", "parameter", "n_before", "n_after", "reason"]) \
                .to_csv(SAVE_PATH / f"{FILE_STEM}_exclusions.csv", index=False)
            (SAVE_PATH / f"{FILE_STEM}_fit_results.txt").write_text(summary, encoding="utf-8")
            print(f"Tabellen gespeichert: {SAVE_PATH / (FILE_STEM + '_fit.csv')}, "
                  f"{SAVE_PATH / (FILE_STEM + '_fit_results.txt')}, "
                  f"{SAVE_PATH / (FILE_STEM + '_exclusions.csv')}")

        plt.show()


if __name__ == "__main__":
    main()
