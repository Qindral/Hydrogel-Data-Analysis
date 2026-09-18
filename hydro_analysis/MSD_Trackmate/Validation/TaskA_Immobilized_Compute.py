"""
Localization error from immobilized particles -- per-track positional SD.

Compute stage: scans both immobilized-particle TrackMate XML folders via
core.io.build_datasets() (raw XML + .rec calibration, unmodified) and
computes, for every sufficiently long track, the positional standard
deviation about that track's own mean position (the "static" localization-
precision estimator) plus a frame-to-frame-difference cross-check. Does NOT
read the Analysis\\ subfolders (raw un-suffixed TrackMate project XML) --
that schema is not what core.io.read_trackmate_xml() parses; only the
filtered Tracks\\*_Tracks.xml exports are used, exactly as elsewhere in
this project.

Unconditionally overwrites cache/localization_error_immobilized.pkl every
run. Consumed by TaskA_Figure1_SigmaLoc_Distribution.py,
TaskA2_A4_Intercept_Comparison.py, and TaskA3_MSD_Correction.py.

Definitions used (see also core.io.remove_edge_artifacts, applied
automatically inside build_datasets/single_file_data):
  sigma_x_static_nm, sigma_y_static_nm:
      std(x_um - mean(x_um)) * 1000, i.e. the standard deviation of the
      track's own position about its own mean -- the standard "static"
      localization-precision estimator for an immobilized particle.
  sigma_x_frame2frame_nm, sigma_y_frame2frame_nm:
      std(diff(x_um)) / sqrt(2) * 1000 -- an independent cross-check: for
      uncorrelated per-frame positional noise, Var(step) = 2*Var(position),
      so dividing by sqrt(2) puts this on the same scale as the static
      estimator. Reported alongside, never used to override the static one.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_analysis.core.io import build_datasets

# ── Configuration ──────────────────────────────────────────────────────────────
IMMOB_FOLDERS: dict[str, Path] = {
    "2026.01.31_immobilized": Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.31_immobilized\Tracks"),
    "2026.01.28_immobilized_locerror": Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.28 _ Immobilized\Tracks"),
}
CACHE_FILE = Path(__file__).parent.parent / "cache" / "localization_error_immobilized.pkl"

MIN_POINTS_FOR_SIGMA = 3   # need >=3 positions for a meaningful std

# core.io.extract_particle_size_from_path()'s generic fallback regex reads
# "1k_Tracks.xml" as size=1 (matches the "1" in "1k", never reaching its
# "1000" branch). Not fixed in core/io.py (existing, widely-used function,
# not to be changed without an explicit request) -- corrected locally here
# instead, since "1k" unambiguously means 1000 nm in this dataset's naming.
_SIZE_NAME_OVERRIDES = {"1k": 1000.0}


def _corrected_particle_size(base_name: str, particle_size_nm: float | None) -> float | None:
    stem = base_name.lower().replace("_tracks.xml", "").replace(".xml", "")
    for token, corrected in _SIZE_NAME_OVERRIDES.items():
        if stem == token or stem.startswith(token + "_") or stem.startswith(token + "."):
            if particle_size_nm != corrected:
                print(f"  [SIZE FIX] {base_name}: particle_size_nm {particle_size_nm} -> {corrected}")
            return corrected
    return particle_size_nm


def _track_sigma_rows(result: dict, dataset_label: str) -> list[dict]:
    tracks_df = result.get("tracks_df")
    mpp = result.get("mpp")
    fps = result.get("fps")
    base_name = result.get("base_name")
    particle_size_nm = _corrected_particle_size(base_name, result.get("particle_size_nm"))
    if tracks_df is None or tracks_df.empty or mpp is None:
        return []

    rows = []
    for pid, g in tracks_df.groupby("particle"):
        g = g.sort_values("frame")
        n = len(g)
        if n < MIN_POINTS_FOR_SIGMA:
            continue
        x_um = g["x"].to_numpy() * mpp
        y_um = g["y"].to_numpy() * mpp

        sigma_x_static_nm = float(np.std(x_um - x_um.mean(), ddof=1) * 1000.0)
        sigma_y_static_nm = float(np.std(y_um - y_um.mean(), ddof=1) * 1000.0)

        dx = np.diff(x_um)
        dy = np.diff(y_um)
        sigma_x_f2f_nm = float(np.std(dx, ddof=1) / np.sqrt(2) * 1000.0) if len(dx) >= 2 else np.nan
        sigma_y_f2f_nm = float(np.std(dy, ddof=1) / np.sqrt(2) * 1000.0) if len(dy) >= 2 else np.nan

        rows.append({
            "dataset_label": dataset_label,
            "movie": base_name,
            "particle_id": int(pid),
            "particle_size_nm": particle_size_nm,
            "track_length_frames": int(n),
            "sigma_x_static_nm": sigma_x_static_nm,
            "sigma_y_static_nm": sigma_y_static_nm,
            "sigma_x_frame2frame_nm": sigma_x_f2f_nm,
            "sigma_y_frame2frame_nm": sigma_y_f2f_nm,
            "mpp": mpp,
            "fps": fps,
        })
    return rows


def main() -> None:
    all_rows: list[dict] = []
    for dataset_label, folder in IMMOB_FOLDERS.items():
        if not folder.exists():
            print(f"  [SKIP] Ordner nicht gefunden: {folder}")
            continue
        datasets = build_datasets(folder)
        print(f"{dataset_label}: {len(datasets)} Dateien geladen aus {folder}")
        for _, result in datasets.items():
            if result is None:
                continue
            all_rows.extend(_track_sigma_rows(result, dataset_label))

    df = pd.DataFrame(all_rows)
    print(f"\nGesamt: {len(df)} Tracks mit gültigem sigma_loc (>= {MIN_POINTS_FOR_SIGMA} Punkte).")

    if not df.empty:
        summary = df.groupby(["dataset_label", "particle_size_nm"]).agg(
            n_tracks=("particle_id", "count"),
            sigma_x_mean_nm=("sigma_x_static_nm", "mean"),
            sigma_x_std_nm=("sigma_x_static_nm", "std"),
            sigma_y_mean_nm=("sigma_y_static_nm", "mean"),
            sigma_y_std_nm=("sigma_y_static_nm", "std"),
        ).reset_index()
        print(summary.to_string(index=False))

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(df, f)
    print(f"\nCache gespeichert: {CACHE_FILE}")


if __name__ == "__main__":
    main()
