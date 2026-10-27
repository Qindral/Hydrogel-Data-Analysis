"""
SPT D_eff / DLS D0 vs. particle size — 20 mg/mL C16 hydrogel, every file as its
own point (no average), with an Amsden obstruction-model fit weighted by the
number of trajectories per file (user, 2026-10-27).

Per-file version of MSD_Deff_over_D0_vs_Size_20mg_Amsden.py (that script, with
one weighted-average point per size, stays unchanged). Same model, r_f, axes,
colours and reference line; model and single-point formula are imported from it:

  - each used file i (one analysis per movie, E031) gives y_i = D_i / D0_DLS(size),
    D_i = corrected per-file D of msd_20mg_files.pkl; no error bars are drawn
    (user, 2026-10-27);
  - Amsden 2022 eq 3, D_eff / D0 = exp[-pi * ((r_s + r_f) / (xi + 2 r_f))**2],
    r_f = FIBRIL_DIAMETER_NM / 2 of the Amsden script (1.5 nm, E037 C), free
    parameter xi;
  - weighted least squares over all files, weight w_i = N_i = num_tracks_fit
    (trajectories >= 10 frames, as in E002 A) (user, 2026-10-27). Implemented as
    curve_fit with sigma_i = 1 / sqrt(N_i) and absolute_sigma=False: the
    covariance is scaled by the weighted residual variance
    s^2 = sum w_i r_i^2 / (n - 1), so sigma_xi follows the scatter between files
    (user, 2026-10-27);
  - one band, model curves at xi - sigma_xi and xi + sigma_xi (as E036, here
    with the scaled sigma);
  - marker area proportional to sqrt(N_i) (user, 2026-10-27); a size legend shows
    reference track counts;
  - for comparison the summary and table also give xi from the files of each
    size alone (same weighting and sigma) and xi through each single file.

Before the real fit, test_perfile_fit_synthetic() fits ratios generated from the
model at the real sizes and track counts (noise SD SYNTH_NOISE_C / sqrt(N_i),
fixed seed) and prints recovered vs. true xi and the coverage of +-1 sigma.
Synthetic values are never plotted or saved.

Reads:  cache/msd_20mg_files.pkl (per-file results of MSD_FromTrackmate_20mg.py).
Writes: <FILE_STEM>.pdf, <FILE_STEM>_per_file.csv, <FILE_STEM>_fit_results.txt
        (fit summary, also printed) and <FILE_STEM>_exclusions.csv in SAVE_PATH.
Run MSD_FromTrackmate_20mg.py first to refresh msd_20mg_files.pkl. Upstream data
handling (edge-artifact removal, fps rule, one analysis per movie, 10-frame
filter, 6-lag fit with fixed localisation error and motion blur) is documented
there and in AGENTS.md section 0.2.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

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
    SAVE_PATH, STYLE, FIG_SIZE, X_MIN, X_MAX, Y_MIN, Y_MAX, COLOR_THEORY,
)
from hydro_analysis.MSD_Trackmate.MSD_Deff_over_D0_vs_Size_20mg_Amsden import (
    FIBRIL_DIAMETER_NM, COLOR_FIT, COLOR_BAND, BAND_ALPHA, N_CURVE_POINTS,
    amsden_ratio, _xi_single_point,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FILES_CACHE = Path(__file__).parent / "cache" / "msd_20mg_files.pkl"
WEIGHT_KEY = "num_tracks_fit"          # weight = trajectories >= 10 frames; decided by user, 2026-10-27
R_F_NM = FIBRIL_DIAMETER_NM / 2.0      # 1.5 nm, decided by user, 2026-10-26 (E037 C)

MARKER_AREA_PER_SQRT_TRACK = 4.0       # scatter area s = 4 sqrt(N) (pt^2), s = 16 at N = 16; area ~ sqrt(N) decided by user, 2026-10-27
MARKER_ALPHA = 0.6                     # overlapping points, Styleguide 0.5-0.7
LEGEND_TRACK_COUNTS = (10, 50, 200)    # reference markers in the size legend (display only)

# Synthetic test (never plotted or saved)
SYNTH_XI_NM = 22.0
SYNTH_NOISE_C = 0.3                    # noise SD of a file = C / sqrt(N)
SYNTH_N_REPEATS = 2000
SYNTH_SEED = 20261027

FILE_STEM = f"deff_over_d0_vs_size_20mg_amsden_per_file_rf{R_F_NM:g}nm".replace(".", "p")


def load_files() -> dict:
    if not FILES_CACHE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {FILES_CACHE}\nBitte zuerst MSD_FromTrackmate_20mg.py ausführen."
        )
    with open(FILES_CACHE, "rb") as f:
        return pickle.load(f)


def fit_xi_weighted(r_s_nm: np.ndarray, ratios: np.ndarray, n_tracks: np.ndarray) -> dict:
    """Weighted least squares, w_i = N_i, covariance scaled by the weighted residual variance."""
    sigma = 1.0 / np.sqrt(n_tracks)
    ok = (ratios > 0) & (ratios < 1)                 # start value only; all points enter the fit
    p0 = float(np.average(_xi_single_point(r_s_nm[ok], ratios[ok], R_F_NM), weights=n_tracks[ok]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        popt, pcov = curve_fit(
            lambda r, xi: amsden_ratio(r, xi, R_F_NM), r_s_nm, ratios, p0=[p0],
            sigma=sigma, absolute_sigma=False, bounds=([-2.0 * R_F_NM + 1e-9], [np.inf]),
        )
    xi = float(popt[0])
    resid = ratios - amsden_ratio(r_s_nm, xi, R_F_NM)
    ssr_w = float(np.sum(n_tracks * resid ** 2))
    dof = len(ratios) - 1
    return {"xi_nm": xi, "xi_err_nm": float(np.sqrt(pcov[0, 0])), "ssr_w": ssr_w, "dof": dof,
            "s2": ssr_w / dof if dof > 0 else float("nan"), "n_points": len(ratios),
            "n_tracks": int(np.sum(n_tracks))}


def test_perfile_fit_synthetic(r_s_nm: np.ndarray, n_tracks: np.ndarray) -> None:
    """Recover a known xi from model ratios at the real sizes and track counts.
    Prints recovered vs. true values; nothing is plotted or saved."""
    y_true = amsden_ratio(r_s_nm, SYNTH_XI_NM, R_F_NM)
    exact = fit_xi_weighted(r_s_nm, y_true, n_tracks)
    print(f"  Synthetic, noise-free: xi true {SYNTH_XI_NM:.4f} nm, recovered {exact['xi_nm']:.4f} nm")

    rng = np.random.default_rng(SYNTH_SEED)
    sd = SYNTH_NOISE_C / np.sqrt(n_tracks)
    xi_fit, xi_err = [], []
    for _ in range(SYNTH_N_REPEATS):
        y = y_true + rng.normal(0.0, sd)
        res = fit_xi_weighted(r_s_nm, y, n_tracks)
        xi_fit.append(res["xi_nm"])
        xi_err.append(res["xi_err_nm"])
    xi_fit, xi_err = np.array(xi_fit), np.array(xi_err)
    pulls = (xi_fit - SYNTH_XI_NM) / xi_err
    print(f"  Synthetic, noisy (n = {SYNTH_N_REPEATS}, seed {SYNTH_SEED}, SD = {SYNTH_NOISE_C:g}/sqrt(N)): "
          f"median recovered xi {np.median(xi_fit):.3f} nm, SD of estimates {xi_fit.std(ddof=1):.3f} nm, "
          f"median reported sigma {np.median(xi_err):.3f} nm, "
          f"coverage of ±1 sigma {np.mean(np.abs(pulls) <= 1.0) * 100:.1f} % (expected 68.3 %)")


def fit_summary_text(res: dict, per_size: dict, table: pd.DataFrame) -> str:
    xi, xi_err = res["xi_nm"], res["xi_err_nm"]
    lines = [
        "Obstruction model fit (Amsden 2022, Macromolecules 55, 8399, eq 3), D_eff/D0 per file vs. particle size, "
        "20 mg/mL C16 hydrogel",
        "Model: D_eff/D0 = exp[-pi * ((r_s + r_f) / (xi + 2 r_f))^2], xi = mesh size, r_s = d_DLS / 2",
        f"r_f fixed: {R_F_NM:g} nm (diameter {FIBRIL_DIAMETER_NM:g} nm); free parameter xi",
        f"Fit: weighted least squares over all files, weight w_i = {WEIGHT_KEY} (trajectories >= 10 frames);",
        "     covariance scaled by s^2 = sum w_i r_i^2 / (n - 1) (relative weights)",
        f"Data points: {res['n_points']} files, {res['n_tracks']} trajectories",
        "",
        f"xi           = {xi:.3f} ± {xi_err:.3f} nm",
        f"sum w r^2    = {res['ssr_w']:.4f}",
        f"dof          = {res['dof']}",
        f"s^2          = {res['s2']:.5f}",
        "",
        f"Band: +-1 sigma: xi {xi - xi_err:.3f} ... {xi + xi_err:.3f} nm at r_f = {R_F_NM:g} nm",
        "",
        "Per size (files of that size only, same weighting and sigma):",
        f"  {'nominal':>8} {'d_DLS':>8} {'files':>6} {'tracks':>7} {'xi (nm)':>9} {'± 1 sigma':>10} "
        f"{'weighted mean y':>16}",
    ]
    for s, ps in per_size.items():
        lines.append(f"  {s:>8g} {ps['d_dls_nm']:>8.2f} {ps['n_points']:>6d} {ps['n_tracks']:>7d} "
                     f"{ps['xi_nm']:>9.3f} {ps['xi_err_nm']:>10.3f} {ps['y_weighted_mean']:>16.4f}")
    lines += ["", "Per file:",
              f"  {'nominal':>8} {'file':<42} {'N':>5} {'D_eff/D0':>9} {'model':>7} {'xi single (nm)':>15}"]
    for _, r in table.iterrows():
        lines.append(f"  {r['nominal_size_nm']:>8g} {r['base_name']:<42} {r['n_tracks_weight']:>5d} "
                     f"{r['deff_over_d0']:>9.4f} {r['model_at_best_fit']:>7.4f} {r['xi_single_file_nm']:>15.3f}")
    lines.append("  xi single: mesh size that puts the model through that file alone (same r_f).")
    return "\n".join(lines) + "\n"


def main() -> None:
    files = load_files()
    print(f"Cache geladen: {FILES_CACHE} ({len(files)} Dateien)")

    dls_sizes     = get_dls_sizes()
    dls_maps      = get_dls_reference_maps()
    size_override = dls_maps["size_override_nm"]
    dls_D         = dls_maps["dls_D_um2_per_s"]

    def _map_size(size: float) -> float:
        return float(dls_sizes.get(size, size_override.get(size, size)))

    # ── D_eff / D0 per file; files without D, D0 or tracks are logged ───────
    rows, exclusions = [], []
    for xml_path, rd in files.items():
        size_nm = float(rd["particle_size_nm"])
        d_eff = (rd.get("fit_results_MSD") or {}).get("D_um2_per_s")
        d0 = dls_D.get(size_nm)
        n = rd.get(WEIGHT_KEY)
        if d_eff is None or d0 is None or not np.isfinite(d_eff) or not np.isfinite(d0) or d0 == 0 or not n:
            exclusions.append({"file": str(xml_path), "step": "ratio",
                               "parameter": f"D_eff, DLS D0 and {WEIGHT_KEY} > 0 required",
                               "n_before": 1, "n_after": 0, "reason": "D_eff, D0 or track count missing"})
            continue
        y = d_eff / d0
        if not 0.0 < y < 1.0:
            raise ValueError(f"{rd['base_name']}: D_eff/D0 = {y:.4f} lies outside (0, 1); the Amsden model "
                             "cannot reach it.")
        rows.append({"nominal_size_nm": size_nm, "dls_diameter_nm": _map_size(size_nm),
                     "base_name": rd["base_name"], "xml_path": str(xml_path), "n_tracks_weight": int(n),
                     "D_eff_um2_per_s": d_eff, "D_eff_error": rd["fit_results_MSD"]["D_error"],
                     "D0_dls_um2_per_s": d0, "deff_over_d0": y})
    table = pd.DataFrame(rows).sort_values(["nominal_size_nm", "base_name"]).reset_index(drop=True)
    sizes = sorted(table["nominal_size_nm"].unique())
    for s in sizes:
        sub = table[table["nominal_size_nm"] == s]
        print(f"  {int(s):>5} nm  {len(sub)} Dateien, {sub['n_tracks_weight'].sum()} Trajektorien")

    r_s = table["dls_diameter_nm"].to_numpy() / 2.0
    ratios = table["deff_over_d0"].to_numpy()
    n_tracks = table["n_tracks_weight"].to_numpy(dtype=float)

    # ── Verification on synthetic data, then the real fit ───────────────────
    print("Test des Fits an synthetischen Daten (Gewicht = Trajektorien, skalierte Kovarianz):")
    test_perfile_fit_synthetic(r_s, n_tracks)

    res = fit_xi_weighted(r_s, ratios, n_tracks)
    xi, xi_err = res["xi_nm"], res["xi_err_nm"]

    per_size = {}
    for s in sizes:
        m = (table["nominal_size_nm"] == s).to_numpy()
        ps = fit_xi_weighted(r_s[m], ratios[m], n_tracks[m])
        ps["d_dls_nm"] = float(table.loc[m, "dls_diameter_nm"].iloc[0])
        ps["y_weighted_mean"] = float(np.average(ratios[m], weights=n_tracks[m]))
        per_size[s] = ps

    table["model_at_best_fit"] = amsden_ratio(r_s, xi, R_F_NM)
    table["xi_single_file_nm"] = _xi_single_point(r_s, ratios, R_F_NM)
    table["xi_size_alone_nm"] = table["nominal_size_nm"].map({s: p["xi_nm"] for s, p in per_size.items()})
    table["xi_size_alone_err_nm"] = table["nominal_size_nm"].map({s: p["xi_err_nm"] for s, p in per_size.items()})
    table["model"] = "Amsden 2022 eq 3"
    table["weight"] = WEIGHT_KEY
    table["r_f_nm"] = R_F_NM
    table["xi_nm"] = xi
    table["xi_err_nm"] = xi_err

    summary = fit_summary_text(res, per_size, table)
    print("\n" + summary)

    x_line = np.linspace(X_MIN, X_MAX, N_CURVE_POINTS)
    y_best = amsden_ratio(x_line / 2.0, xi, R_F_NM)
    if xi - xi_err <= -2.0 * R_F_NM:
        raise ValueError(f"xi - sigma = {xi - xi_err:.3f} nm gives xi + 2 r_f <= 0; model undefined.")
    y_lo, y_hi = amsden_ratio(x_line / 2.0, xi - xi_err, R_F_NM), amsden_ratio(x_line / 2.0, xi + xi_err, R_F_NM)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        ax.axhline(1.0, color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY, zorder=3)
        ax.fill_between(x_line, y_lo, y_hi, color=COLOR_BAND, alpha=BAND_ALPHA, linewidth=0, zorder=1)
        ax.plot(x_line, y_best, color=COLOR_FIT, linewidth=1.5, zorder=4)

        for s in sizes:
            sub = table[table["nominal_size_nm"] == s]
            base, dark = SIZE_COLORS[s]
            ax.scatter(sub["dls_diameter_nm"], sub["deff_over_d0"],
                       s=MARKER_AREA_PER_SQRT_TRACK * np.sqrt(sub["n_tracks_weight"]),
                       facecolor=base, edgecolor=dark, linewidth=0.6, alpha=MARKER_ALPHA, zorder=6)

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
            Line2D([0], [0], color=COLOR_FIT, linewidth=1.5,
                   label=rf"Amsden fit ($r_f$ = {R_F_NM:g} nm, $\xi$ = {xi:.1f} nm)"),
            Patch(facecolor=COLOR_BAND, alpha=BAND_ALPHA, linewidth=0, label=r"$\pm$1$\sigma$"),
        ]
        legend_elements += [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", markeredgecolor="0.3",
                   markeredgewidth=0.6, linestyle="None", alpha=MARKER_ALPHA,
                   markersize=np.sqrt(MARKER_AREA_PER_SQRT_TRACK * np.sqrt(n)), label=f"{n} trajectories")
            for n in LEGEND_TRACK_COUNTS
        ]
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 0.92), frameon=False)

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            pdf_path = SAVE_PATH / f"{FILE_STEM}.pdf"
            fig.savefig(pdf_path)
            print(f"Plot gespeichert: {pdf_path}")
            table.to_csv(SAVE_PATH / f"{FILE_STEM}_per_file.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(exclusions, columns=["file", "step", "parameter", "n_before", "n_after", "reason"]) \
                .to_csv(SAVE_PATH / f"{FILE_STEM}_exclusions.csv", index=False)
            (SAVE_PATH / f"{FILE_STEM}_fit_results.txt").write_text(summary, encoding="utf-8")
            print(f"Tabellen gespeichert: {SAVE_PATH / (FILE_STEM + '_per_file.csv')}, "
                  f"{SAVE_PATH / (FILE_STEM + '_fit_results.txt')}, "
                  f"{SAVE_PATH / (FILE_STEM + '_exclusions.csv')}")

        plt.show()


if __name__ == "__main__":
    main()
