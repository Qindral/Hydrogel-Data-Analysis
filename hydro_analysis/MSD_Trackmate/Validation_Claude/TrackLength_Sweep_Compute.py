"""
Minimum-track-length sweep: ensemble D and n per file as a function of the
minimum trajectory length L_min.

Continuous replacement for the fixed length bins of Correlations.py
(B_length_bin_comparison.png). For every file and every L_min in
L_MIN_GRID, only trajectories with at least L_min frames are kept and the
file's eMSD is refitted with core.analysis.perform_msd_analysis (reused
unmodified, same fit_points as the file's standard fit). The subsets are
cumulative (tracks >= L_min), not disjoint bins, so consecutive sweep points
share most of their trajectories -- the curve shows a trend, the points are
not statistically independent.

The reference value of each file is its own sweep point at L_min = 10
(= core.analysis.MIN_TRACK_LENGTH, i.e. the standard analysis), computed by
the same call so that drift subtraction and stub filtering are identical to
all other sweep points. D_rel = D(L_min) / D(10) is stored per file, the
normalisation happens before any averaging over files.

A file is swept only while at least MIN_TRACKS_PER_POINT trajectories
remain; the sweep for that file stops at the first L_min below this
threshold (and the file is skipped entirely if already L_min = 10 falls
below it). Edge-artifact filtering (core.io.remove_edge_artifacts) was
already applied upstream when the tracks were loaded.

Pure consumer of cache/msd_d0_results.pkl (water) and
cache/msd_20mg_files.pkl (hydrogel) -- never touches raw XML/TIFF -- but
recomputes and unconditionally overwrites its own derived cache
cache/track_length_sweep.pkl on every run (same hybrid pattern as
TaskB1_TrackLengthBias_Compute.py). Consumed by
Figures_Refined/TrackLength_Sweep_Refined.py.

Run MSD_FromTrackmate_D0.py and MSD_FromTrackmate_20mg.py first.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_analysis.core.analysis import perform_msd_analysis, DEFAULT_MSD_FIT_POINTS, MIN_TRACK_LENGTH
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import CACHE_D0, CACHE_20MG, _load_pickle, _fit_key

# ── Configuration ──────────────────────────────────────────────────────────────
OUT_SWEEP = Path(__file__).parent.parent / "cache" / "track_length_sweep.pkl"

L_MIN_GRID = np.arange(MIN_TRACK_LENGTH, 121, 2)   # frames
MIN_TRACKS_PER_POINT = 20                            # sweep of a file stops below this


def sweep_file(r: dict, condition: str) -> list[dict]:
    tracks_df = r.get("tracks_df")
    mpp, fps = r.get("mpp"), r.get("fps")
    fit = r.get("fit_results_MSD")
    size_nm = r.get("particle_size_nm")
    if tracks_df is None or tracks_df.empty or mpp is None or fps is None or fit is None or size_nm is None:
        return []
    fit_points = fit.get("fit_points", DEFAULT_MSD_FIT_POINTS)
    lengths = tracks_df.groupby("particle").size()
    n_tracks_total = int((lengths >= MIN_TRACK_LENGTH).sum())

    rows = []
    D_ref = None
    for L_min in L_MIN_GRID:
        particle_ids = lengths[lengths >= L_min].index
        if len(particle_ids) < MIN_TRACKS_PER_POINT:
            break
        temp_result = {"tracks_df": tracks_df[tracks_df["particle"].isin(particle_ids)], "mpp": mpp, "fps": fps}
        perform_msd_analysis(temp_result, fit_points=fit_points)
        sweep_fit = temp_result.get(_fit_key(fit_points))
        if sweep_fit is None or sweep_fit.get("D_um2_per_s") is None:
            break
        D = float(sweep_fit["D_um2_per_s"])
        if D_ref is None:
            D_ref = D
        rows.append({
            "xml_path": r.get("xml_path"), "base_name": r.get("base_name"),
            "particle_size_nm": float(size_nm), "condition": condition,
            "L_min": int(L_min), "n_tracks": int(len(particle_ids)), "n_tracks_total": n_tracks_total,
            "D_um2_per_s": D, "D_rel": D / D_ref if D_ref > 0 else np.nan,
            "exponent": float(sweep_fit["exponent"]),
        })
    return rows


def main() -> None:
    frames = []
    for cache, compute_script, condition in ((CACHE_D0, "MSD_FromTrackmate_D0.py", "water"),
                                             (CACHE_20MG, "MSD_FromTrackmate_20mg.py", "hydrogel")):
        results = _load_pickle(cache, compute_script)
        t0 = time.time()
        rows = []
        for r in results.values():
            rows.extend(sweep_file(r, condition))
        df = pd.DataFrame(rows)
        n_files = df["xml_path"].nunique() if not df.empty else 0
        print(f"{condition}: {n_files}/{len(results)} Dateien gesweept, {len(df)} Fits, {time.time() - t0:.0f} s.")
        frames.append(df)

    sweep = pd.concat(frames, ignore_index=True)
    OUT_SWEEP.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SWEEP, "wb") as f:
        pickle.dump({"sweep": sweep, "L_min_grid": L_MIN_GRID,
                     "min_tracks_per_point": MIN_TRACKS_PER_POINT}, f)
    print(f"Cache gespeichert: {OUT_SWEEP}")


if __name__ == "__main__":
    main()
