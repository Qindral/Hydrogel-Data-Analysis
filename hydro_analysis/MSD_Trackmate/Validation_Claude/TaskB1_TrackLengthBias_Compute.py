"""
Per-track (not per-file) D and n, for the track-length bias analysis.

Pure consumer of msd_d0_results.pkl / msd_20mg_files.pkl -- never touches
raw XML/TIFF. Nonetheless recomputes and unconditionally overwrites its own
derived pickle every run, mirroring SPT_Aggregation_Uncertainty_D0.py's
documented precedent for this exact hybrid pattern (a "consumer" of raw
caches that still behaves like a compute stage towards ITS OWN derived
cache).

For every trajectory already present in a file's cached
fit_results_MSD["imsd"] (one column per track, produced by the original
ensemble analysis -- no re-linking, no re-filtering of raw trajectories),
fits the model of the ensemble fit with motion blur and 4 sigma^2 (core.analysis.
fit_powerlaw_blur_locerror_general, E046) over the SAME
fit_points already used for that file's ensemble fit, with sigma and t_E of
that file (fit_results_MSD "sigma_loc_nm", "exposure_ms"), to get a genuinely
per-track D and n (user, 2026-10-27, E045 A; before: power law 4 D tau^n). This is a different, new fit from the existing
per-track *linear*-only helper in
../Dissertation_Figures/Deff_Histogram.py::_fit_track_D() (D only, no n) --
Task B explicitly needs both D and n per track, so it needs its own
per-track power-law fit.

Tracks whose fit window contains NaN or non-positive MSD values, and tracks
whose fit does not converge, are left out and counted per file in
EXCLUSIONS_CSV.
Track length comes from tracks_df.groupby('particle').size().

Writes cache/track_length_bias_d0.pkl and cache/track_length_bias_20mg.pkl,
unconditionally overwritten every run. Consumed by iMSD_histograms.py (per-
track D/n panels + track-length panel). Correlations.py's track-length
comparison (length-bin ensemble refits) and the former
TaskB2_LengthBias_Scatter.py / TaskB3_MinTrackLength_Sensitivity.py (moved
to obsolete/) no longer use this cache.

Run MSD_FromTrackmate_D0.py and MSD_FromTrackmate_20mg.py first (or after
any raw-data change) to refresh both upstream inputs.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_analysis.core.analysis import fit_msd_like_ensemble

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_D0 = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
CACHE_20MG = Path(__file__).parent.parent / "cache" / "msd_20mg_files.pkl"
OUT_D0 = Path(__file__).parent.parent / "cache" / "track_length_bias_d0.pkl"
OUT_20MG = Path(__file__).parent.parent / "cache" / "track_length_bias_20mg.pkl"
EXCLUSIONS_CSV = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data"
) / "SPT" / "track_length_bias" / "track_length_bias_exclusions.csv"


def _load_pickle(path: Path, compute_script: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def per_track_fits(results: dict, condition: str) -> tuple[pd.DataFrame, dict, list[dict]]:
    rows, excl = [], []
    n_total = 0
    n_excluded = 0
    n_failed = 0
    for r in results.values():
        fit = r["fit_results_MSD"]
        tracks_df = r["tracks_df"]
        imsd = fit["imsd"]
        fit_points = fit["fit_points"]
        size_nm = r.get("particle_size_nm")
        lengths = tracks_df.groupby("particle").size()
        n_win = n_fail = 0

        for col in imsd.columns:
            n_total += 1
            window = imsd[col].iloc[0:fit_points]
            if window.isna().any() or (window <= 0).any():
                n_excluded += 1
                n_win += 1
                continue
            try:
                tf = fit_msd_like_ensemble(window, fit_points, fit)
            except ValueError:
                n_failed += 1
                n_fail += 1
                continue
            length = int(lengths.get(col, window.count()))

            rows.append({
                "xml_path": r.get("xml_path"),
                "base_name": r.get("base_name"),
                "particle_size_nm": float(size_nm) if size_nm is not None else np.nan,
                "condition": condition,
                "particle_id": col,
                "track_length_frames": length,
                "D_um2_per_s": tf["D"],
                "D_err": tf["D_err"],
                "exponent": tf["n"],
                "exponent_err": tf["n_err"],
                "r_squared": tf["r_squared"],
                "fit_points": fit_points,
                "sigma_loc_nm": float(fit["sigma_loc_nm"]),
                "exposure_ms": float(fit["exposure_ms"]),
            })
        n_tracks = len(imsd.columns)
        if n_win:
            excl.append({"file": r.get("xml_path"), "step": "per_track_window", "parameter": f"{fit_points} points",
                         "n_before": n_tracks, "n_after": n_tracks - n_win, "reason": "NaN/<=0 in the fit window"})
        if n_fail:
            excl.append({"file": r.get("xml_path"), "step": "per_track_fit_failed", "parameter": f"{fit_points} points",
                         "n_before": n_tracks - n_win, "n_after": n_tracks - n_win - n_fail,
                         "reason": "fit with localisation error and motion blur did not converge"})

    df = pd.DataFrame(rows)
    stats = {"n_total": n_total, "n_excluded": n_excluded, "n_failed": n_failed, "n_valid": len(df)}
    return df, stats, excl


def main() -> None:
    d0_results = _load_pickle(CACHE_D0, "MSD_FromTrackmate_D0.py")
    print(f"Cache geladen: {CACHE_D0} ({len(d0_results)} Dateien)")
    hydrogel_results = _load_pickle(CACHE_20MG, "MSD_FromTrackmate_20mg.py")
    print(f"Cache geladen: {CACHE_20MG} ({len(hydrogel_results)} Dateien)")

    df_d0, stats_d0, excl_d0 = per_track_fits(d0_results, "water")
    print(f"Water: {stats_d0['n_valid']} von {stats_d0['n_total']} Tracks gültig "
          f"({stats_d0['n_excluded']} ausgeschlossen wegen NaN/<=0 im Fit-Fenster, "
          f"{stats_d0['n_failed']} Fits nicht konvergiert)")

    df_20mg, stats_20mg, excl_20mg = per_track_fits(hydrogel_results, "hydrogel")
    print(f"Hydrogel: {stats_20mg['n_valid']} von {stats_20mg['n_total']} Tracks gültig "
          f"({stats_20mg['n_excluded']} ausgeschlossen wegen NaN/<=0 im Fit-Fenster, "
          f"{stats_20mg['n_failed']} Fits nicht konvergiert)")
    EXCLUSIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(excl_d0 + excl_20mg, columns=["file", "step", "parameter", "n_before", "n_after", "reason"]) \
        .to_csv(EXCLUSIONS_CSV, index=False)
    print(f"Ausschlüsse gespeichert: {EXCLUSIONS_CSV}")

    for label, df in (("Water", df_d0), ("Hydrogel", df_20mg)):
        if df.empty:
            continue
        summary = df.groupby("particle_size_nm").agg(
            n_tracks=("particle_id", "count"),
            D_median=("D_um2_per_s", "median"),
            n_median=("exponent", "median"),
            length_median=("track_length_frames", "median"),
        ).reset_index()
        print(f"\n{label} per-size summary:")
        print(summary.to_string(index=False))

    for out_path, df in ((OUT_D0, df_d0), (OUT_20MG, df_20mg)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(df, f)
        print(f"\nCache gespeichert: {out_path}")


if __name__ == "__main__":
    main()
