"""
Anomalous diffusion exponent n vs. particle size — quadratic obstruction fit.

Loads the per-size, trajectory-count-weighted average cached by
MSD_FromTrackmate_20mg.py's compute_weighted_average_per_size() (cache/msd_20mg_result.pkl).
Each file's own D/n fit (from msd_20mg_files.pkl) stays completely
independent -- nothing here pools raw trajectories or refits anything -- but
a plain per-file average would give a noisy low-trajectory file the same
vote as a solid high-trajectory one, so files are weighted by their
trajectory count instead. This script only reads that already-averaged
result; it does not repeat the averaging or refit anything itself.

A quadratic small-obstruction model is fit to n(r) (r = particle radius,
nm):

    n(r) = 1 - c * (r / xi) ** 2

where xi is the effective hydrogel mesh size (nm). n -> 1 (free diffusion)
as r -> 0 and falls to n = 0 at r = xi/sqrt(c). The fit curve is drawn out
to that zero crossing, not cut off at an arbitrary x-range.

The "best fit" is the lowest-SSE result of a multi-start least-squares
search over (c, xi) — with only a handful of particle sizes, a
single-start fit can land in the wrong local minimum. The uncertainty in
both x (DLS size) and y (eMSD exponent error) is propagated by refitting
on ~1500 Monte-Carlo-perturbed draws of the data. A perturbed refit only
enters the band if its chi^2 against the data (evaluated back at the true
particle sizes) is no worse than the best fit's own chi^2 + 2.30 -- the
standard Delta-chi^2 definition of a 68% joint confidence region for 2
free parameters (Numerical Recipes, Table 15.6.1). The surviving family
of curves is shown as a shaded 16th-84th percentile band: other solutions
consistent with the data uncertainty, never wider than what the
measurement itself allows.

A residuals inset ((data - fit) / sigma at each particle size) shows how
well the single best fit matches the data it was built from.

Tracks feeding into the underlying MSD fits are pre-filtered via
core.io.single_file_data() -> remove_edge_artifacts(): detections within
3% of the frame border are dropped and trajectories split at the gap, to
correct spurious TrackMate linking of near-edge detections (see
MSD_FromTrackmate_20mg.py).
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.optimize import curve_fit, OptimizeWarning

from hydro_analysis.core.io import get_dls_reference_maps, get_dls_sizes, get_dls_labels

# ── Style (publication design, matches MSD_per_file_publication.py) ────────
_RC = {
    "text.usetex":       False,
    "mathtext.fontset":  "dejavusans",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Open Sans", "Arial", "DejaVu Sans"],
    "axes.linewidth":    0.8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.major.size":  5.0,
    "ytick.major.size":  5.0,
    "xtick.minor.size":  3.0,
    "ytick.minor.size":  3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "legend.fontsize":   8,
}


def _inset_style(ax: plt.Axes) -> None:
    ax.tick_params(which="both", direction="in",
                   top=True, right=True, labelsize=7, width=0.6)
    ax.tick_params(which="major", length=3.5)
    ax.tick_params(which="minor", length=2.0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


# ── Configuration ─────────────────────────────────────────────────────────
RESULT_CACHE_FILE = Path(__file__).parent / "cache" / "msd_20mg_result.pkl"
SAVE_PATH = Path(
    rf"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16"
    rf"\AnomalousExponent_{pd.Timestamp.now().strftime('%Y%m%d')}"
)

X_LIM = (0.0, 120.0)
Y_LIM = (0.0, 1.0)
N_MC = 1500
MC_SEED = 42
R_ZERO_CAP_NM = 1.0e5  # search cap for the zero-crossing (100 um), not a physical bound
# Delta-chi^2 for a 68% (1-sigma-equivalent) joint confidence region with 2 free
# parameters (Press et al., Numerical Recipes, Table 15.6.1).
DELTA_CHI2_1SIGMA_2PARAM = 2.30

COL_DATA      = "#0000da"
COL_DATA_DARK = "#000099"
COL_THEORY    = "black"

# Bounds [c, xi_nm] for the quadratic fit
_QUAD_BOUNDS = ([0.0, 1.0], [10.0, 5000.0])


# ── Model ─────────────────────────────────────────────────────────────────

def quadratic_model(r_nm: np.ndarray, c: float, xi_nm: float) -> np.ndarray:
    """Quadratic small-obstruction expansion: n = 1 - c*(r/xi)**2."""
    return 1.0 - c * (r_nm / xi_nm) ** 2


# ── Data loading ────────────────────────────────────────────────────────

def _load_cache(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_pooled_exponents(pooled: dict) -> pd.DataFrame:
    """Read n/n_err straight out of the trajectory-weighted-average cache (one
    entry per particle size, each a weighted average of that size's
    independent per-file fits from MSD_FromTrackmate_20mg.py's
    compute_weighted_average_per_size()).
    """
    rows = []
    for size_nm, entry in pooled.items():
        fit = entry.get("fit_results_MSD")
        if not fit or fit.get("exponent") is None:
            continue
        rows.append({
            "particle_size_nm": float(size_nm),
            "n_emsd": float(fit["exponent"]),
            "n_emsd_err": float(fit["exponent_error"]),
            "n_files": entry.get("n_files"),
            "n_particles": entry.get("n_particles_pooled"),
        })
    return pd.DataFrame(rows).sort_values("particle_size_nm").reset_index(drop=True)


# ── Fitting ───────────────────────────────────────────────────────────────

def _fit_best(model_func, bounds, p0_grid, r_nm: np.ndarray, n_vals: np.ndarray, n_err: np.ndarray):
    """Multi-start least-squares search; returns (popt, pcov, sse, sorted_candidates)."""
    sigma = n_err if np.all(np.isfinite(n_err)) and np.all(n_err > 0) else None

    candidates = []
    for p0 in p0_grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(
                    model_func, r_nm, n_vals,
                    p0=p0,
                    sigma=sigma, absolute_sigma=sigma is not None,
                    bounds=bounds,
                    maxfev=20000,
                )
        except Exception:
            continue
        resid = n_vals - model_func(r_nm, *popt)
        sse = float(np.sum(resid ** 2))
        if np.isfinite(sse):
            candidates.append({"popt": popt, "pcov": pcov, "sse": sse})

    if not candidates:
        raise RuntimeError("Fit did not converge for any initial guess.")

    candidates.sort(key=lambda c: c["sse"])
    best = candidates[0]
    return best["popt"], best["pcov"], best["sse"], candidates


def _mc_band(model_func, bounds, r_nm, r_err_nm, n_vals, n_err, popt0, r_line,
            n_mc=N_MC, seed=MC_SEED, delta_chi2=DELTA_CHI2_1SIGMA_2PARAM):
    """Refit on Monte-Carlo-perturbed (x, y) draws; return 16/84th percentile band.

    A perturbed refit is only accepted into the band if, evaluated back at
    the *original* data positions r_nm, its chi^2 against the data is no
    worse than the best fit's own chi^2 + delta_chi2 (the standard
    Delta-chi^2 definition of a joint confidence region, Numerical Recipes
    Table 15.6.1). Without this, a curve fit through a jointly (x, y)
    -jittered draw can wander arbitrarily far from the data once mapped
    back to the true x -- the band is meant to show other solutions
    consistent with the measured uncertainty, never looser than the data
    itself allows. Requiring an exact per-point match instead (a stricter,
    non-statistical criterion) can reject every candidate whenever the
    best fit itself already has nonzero residual, which happens here
    because the quadratic model's (c, xi) aren't both independently
    identifiable from only 2 particle sizes.
    """
    rng = np.random.default_rng(seed)

    finite_err = n_err[np.isfinite(n_err) & (n_err > 0)]
    fallback_err = float(np.median(finite_err)) if finite_err.size else 0.03
    n_err_safe = np.where(np.isfinite(n_err) & (n_err > 0), n_err, fallback_err)
    r_err_safe = np.where(np.isfinite(r_err_nm), r_err_nm, 0.0)

    def _chi2(popt):
        resid = (model_func(r_nm, *popt) - n_vals) / n_err_safe
        return float(np.sum(resid ** 2))

    chi2_best = _chi2(popt0)

    curves = []
    n_rejected = 0
    for _ in range(n_mc):
        r_pert = np.clip(rng.normal(r_nm, r_err_safe), 0.5, None)
        n_pert = rng.normal(n_vals, n_err_safe)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(
                    model_func, r_pert, n_pert,
                    p0=popt0, bounds=bounds, maxfev=5000,
                )
        except Exception:
            continue

        if _chi2(popt) > chi2_best + delta_chi2:
            n_rejected += 1
            continue
        curves.append(model_func(r_line, *popt))

    if not curves:
        return None, None, 0, n_rejected

    curves = np.array(curves)
    lo, hi = np.nanpercentile(curves, [16, 84], axis=0)
    return lo, hi, curves.shape[0], n_rejected


def _zero_crossing_r(model_func, popt, r_min: float, cap: float = R_ZERO_CAP_NM,
                     plateau_tol: float = 1e-3) -> tuple[float, bool]:
    """Right-edge r for plotting a monotonically decreasing model curve out
    to where it stops being interesting: the zero crossing (n = 0) if the
    curve reaches it, found by geometric expansion + bisection (log-spaced);
    otherwise the point where the curve has flattened onto its asymptotic
    plateau (change per step < plateau_tol), so a fit that never reaches
    zero doesn't blow up the x-range out to `cap`.

    Returns (r_end, reached_zero).
    """
    r_prev = max(r_min, 1e-3)
    y_prev = float(model_func(np.array([r_prev]), *popt)[0])
    r = r_prev
    for _ in range(300):
        r *= 1.15
        if r > cap:
            return cap, False
        y = float(model_func(np.array([r]), *popt)[0])
        if y <= 0:
            lo, hi = r_prev, r
            for _ in range(60):
                mid = np.sqrt(lo * hi)
                if model_func(np.array([mid]), *popt)[0] > 0:
                    lo = mid
                else:
                    hi = mid
            return hi, True
        if abs(y - y_prev) < plateau_tol:
            return r, False
        r_prev, y_prev = r, y
    return cap, False


# ── Plot ──────────────────────────────────────────────────────────────────

def plot_exponent_vs_size(
    agg: pd.DataFrame,
    xs: np.ndarray,
    xerr: np.ndarray,
    data_tick_labels: list[str],
    all_tick_pos: list[float],
    all_tick_labels: list[str],
    quad_fit: tuple,
    save_path: Path | None,
) -> None:
    r_line_quad, lo_quad, hi_quad, popt_quad, n_ok_quad = quad_fit

    x_line_quad = r_line_quad * 2.0
    y_quad = quadratic_model(r_line_quad, *popt_quad)

    r_data = xs / 2.0
    n_emsd = agg["n_emsd"].to_numpy()
    n_emsd_err = agg["n_emsd_err"].to_numpy()
    resid_sigma = (n_emsd - quadratic_model(r_data, *popt_quad)) / n_emsd_err

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

        ax.axhline(1.0, color=COL_THEORY, linestyle="--", linewidth=1.2, zorder=1,
                   label="Free diffusion (n = 1)")

        if lo_quad is not None:
            ax.fill_between(x_line_quad, lo_quad, hi_quad, color=COL_DATA_DARK, alpha=0.18,
                            linewidth=0, zorder=2,
                            label=f"Quadratic fit — 16–84th pct. (N={n_ok_quad} MC)")
        ax.plot(x_line_quad, y_quad, color=COL_DATA_DARK, linewidth=2.2, zorder=4,
               label=r"Quadratic fit: $n=1-c\,(r/\xi)^2$")

        ax.errorbar(
            xs, n_emsd, yerr=n_emsd_err, xerr=xerr,
            fmt="o", markersize=np.sqrt(26),
            markerfacecolor=COL_DATA, markeredgecolor=COL_DATA_DARK, markeredgewidth=0.8,
            ecolor=COL_DATA_DARK, elinewidth=1.2, capsize=3.0, capthick=1.2,
            zorder=6, label="eMSD n (per size)",
        )

        # Faint reference ticks for the DLS particle sizes that have no eMSD data yet
        for pos in all_tick_pos:
            if X_LIM[0] < pos < X_LIM[1] and not np.any(np.isclose(pos, xs, rtol=1e-6)):
                ax.axvline(pos, color="#999999", linewidth=0.6, linestyle=":", zorder=0)

        ax.set_xscale("linear")
        ax.set_xlim(*X_LIM)
        ax.set_ylim(*Y_LIM)
        ax.xaxis.set_major_locator(FixedLocator(all_tick_pos))
        ax.xaxis.set_major_formatter(FixedFormatter(all_tick_labels))

        ax.set_xlabel("Particle size (nm, DLS)")
        ax.set_ylabel(r"Anomalous exponent $n$  (MSD $\propto \tau^{n}$)")
        ax.set_title("Anomalous Diffusion Exponent vs. Particle Size — 20 mg/mL C16", fontsize=11)
        ax.legend(loc="lower left")

        # ── Inset: fit residuals (kept as before — log x, own range) ────
        x_min_inset = min(all_tick_pos) * 0.6
        x_max_inset = max(all_tick_pos) * 1.15
        axin = ax.inset_axes([0.60, 0.60, 0.36, 0.32])
        axin.axhline(0.0, color=COL_THEORY, linewidth=0.8, linestyle="--", zorder=1)
        axin.axhspan(-1, 1, color=COL_DATA_DARK, alpha=0.10, zorder=0)
        axin.scatter(xs, resid_sigma, s=20, facecolors=COL_DATA, edgecolors=COL_DATA_DARK,
                    linewidths=0.7, zorder=2)
        axin.set_xscale("log")
        axin.set_xlim(x_min_inset, x_max_inset)
        y_max_resid = max(1.5, float(np.nanmax(np.abs(resid_sigma))) * 1.3) if resid_sigma.size else 1.5
        axin.set_ylim(-y_max_resid, y_max_resid)
        axin.xaxis.set_major_locator(FixedLocator(list(xs)))
        axin.xaxis.set_major_formatter(FixedFormatter(data_tick_labels))
        axin.xaxis.set_minor_locator(FixedLocator([]))
        _inset_style(axin)
        axin.set_ylabel(r"Residual ($\sigma$)", fontsize=7)
        axin.set_title("Fit residuals", fontsize=8)

        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / "anomalous_exponent_vs_size.png", dpi=600, bbox_inches="tight")
            print(f"Saved: {save_path / 'anomalous_exponent_vs_size.png'}")

        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    pooled = _load_cache(RESULT_CACHE_FILE)
    agg = _extract_pooled_exponents(pooled)
    if agg.empty:
        print("No eMSD exponent data found in cache — run MSD_FromTrackmate_20mg.py first.")
        return

    # DLS reference: prefer the measured z-average from the DLS cache (pkl),
    # falling back to the hardcoded size_override_nm map only if a size is
    # missing there — same priority order used in MSD_FromTrackmate_20mg.py.
    dls_sizes  = get_dls_sizes()
    dls_labels = get_dls_labels()
    dls_maps   = get_dls_reference_maps()
    size_override, size_err = dls_maps["size_override_nm"], dls_maps["size_err_nm"]

    xs = np.array([
        float(dls_sizes.get(s, size_override.get(s, s))) for s in agg["particle_size_nm"]
    ])
    xerr = np.array([float(size_err.get(s, 0.0)) for s in agg["particle_size_nm"]])
    data_tick_labels = [str(dls_labels.get(s, int(s))) for s in agg["particle_size_nm"]]
    r_nm  = xs / 2.0
    r_err = xerr / 2.0

    # All DLS reference particle sizes (not just the ones with eMSD data yet)
    # are shown as x-axis ticks, so the plot already carries the full set of
    # sizes planned for this study.
    all_nominal = sorted(set(dls_sizes) | set(size_override))
    all_tick_pos    = [float(dls_sizes.get(s, size_override.get(s, s))) for s in all_nominal]
    all_tick_labels = [str(dls_labels.get(s, int(s))) for s in all_nominal]

    print("── eMSD anomalous exponent n — per particle size (20 mg/mL C16) ──")
    table = agg.copy()
    table["dls_size_nm"] = xs
    print(table.to_string(index=False))

    n_vals = agg["n_emsd"].to_numpy()
    n_err  = agg["n_emsd_err"].to_numpy()

    # ── Quadratic fit ─────────────────────────────────────────────────
    quad_p0 = [(c0, xi0) for c0 in (0.3, 0.7, 1.5, 3.0) for xi0 in np.logspace(0.3, 3.3, 8)]
    popt_quad, pcov_quad, sse_quad, cand_quad = _fit_best(
        quadratic_model, _QUAD_BOUNDS, quad_p0, r_nm, n_vals, n_err)
    perr_quad = np.sqrt(np.diag(pcov_quad)) if np.all(np.isfinite(pcov_quad)) else np.full(2, np.nan)
    print(f"\nBest quadratic fit:  c = {popt_quad[0]:.3f} +/- {perr_quad[0]:.3f}   "
          f"xi = {popt_quad[1]:.1f} +/- {perr_quad[1]:.1f} nm   (SSE={sse_quad:.4g})")
    print("Top 5 candidate quadratic fits (all local minima explored):")
    for c in cand_quad[:5]:
        print(f"  c={c['popt'][0]:.3f}  xi={c['popt'][1]:.1f} nm  SSE={c['sse']:.4g}")

    r_zero_quad, quad_reaches_zero = _zero_crossing_r(quadratic_model, popt_quad, r_nm.min() * 0.5)
    # Draw the fit from r=0 out to its zero crossing (or to the visible plot
    # edge, whichever comes first) -- linear spacing for the linear x-axis.
    r_line_end = min(r_zero_quad, X_LIM[1] / 2.0 * 1.05)
    r_line_quad = np.linspace(0.0, r_line_end, 200)
    lo_quad, hi_quad, n_ok_quad, n_rej_quad = _mc_band(
        quadratic_model, _QUAD_BOUNDS, r_nm, r_err, n_vals, n_err, popt_quad, r_line_quad)
    print(f"Monte-Carlo uncertainty band: {n_ok_quad}/{N_MC} perturbed refits converged "
          f"and stayed within the data error bars ({n_rej_quad} rejected for leaving them).")
    if quad_reaches_zero:
        print(f"Quadratic fit reaches n=0 at particle size x = {2 * r_zero_quad:.1f} nm "
              f"(r = {r_zero_quad:.1f} nm)")
    else:
        plateau = quadratic_model(np.array([r_zero_quad]), *popt_quad)[0]
        print(f"Quadratic fit does NOT reach n=0: plateaus at n={plateau:.3f} "
              f"for x >~ {2 * r_zero_quad:.1f} nm")

    resid_sigma = (n_vals - quadratic_model(r_nm, *popt_quad)) / n_err
    print("\nFit residuals (data - fit) / sigma:")
    for size, r in zip(xs, resid_sigma):
        print(f"  x={size:.1f} nm:  {r:+.2f} sigma")

    plot_exponent_vs_size(
        agg, xs, xerr, data_tick_labels, all_tick_pos, all_tick_labels,
        quad_fit=(r_line_quad, lo_quad, hi_quad, popt_quad, n_ok_quad),
        save_path=SAVE_PATH,
    )


if __name__ == "__main__":
    main()
