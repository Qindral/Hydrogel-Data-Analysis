"""
Localization error from immobilized particles -- redone with REAL headless
TrackMate (trackmate_headless_runner.py) instead of the original
TaskA_Immobilized_Compute.py's approach of trusting the already-exported
Tracks\\*_Tracks.xml files (whose own detector/tracker parameters were
never confirmed -- same "no candidate settings file/folder found" gap
TrackMate_Settings_Inventory.py flagged for the water movies). Both
immobilized folders' Analysis\\ subfolders contain full raw TrackMate
session XMLs with real, confirmed parameters, so this version drives an
actual TrackMate run per movie instead.

Detector: both immobilized datasets use LOG_DETECTOR (not DOG_DETECTOR, the
water movies' detector) -- confirmed by inspection of Analysis\\20nm_immob.xml
et al. Track filters here also include TRACK_DISPLACEMENT and
TRACK_MEAN_SPEED (not just NUMBER_SPOTS, as in the water movies) -- sensible
for immobilized data, since it explicitly selects genuinely-stationary
tracks. Every filter feature is passed through to real TrackMate as-is (see
_trackmate_settings_parser.py's docstring): no Python-side feature
computation needed, since Settings.addAllAnalyzers() computes everything
TrackMate itself needs before filtering.

Compute stage: reuses Compare_Tracks_vs_RawAnalysis.py::match_tracks_to_analysis()
unmodified (exact .tif-filename matching, not fuzzy) to pair each Tracks\\
export with its real Analysis\\ session XML, runs (or reuses a cached run
of) real headless TrackMate per movie, then computes the same sigma
definitions as the original script from the resulting tracks_df:

  sigma_x_static_nm, sigma_y_static_nm:
      std(x_um - mean(x_um)) * 1000 -- the static localization-precision
      estimator for an immobilized particle.
  sigma_x_frame2frame_nm, sigma_y_frame2frame_nm:
      std(diff(x_um)) / sqrt(2) * 1000 -- cross-check, reported alongside,
      never used to override the static one.

remove_edge_artifacts() (core.io, unmodified) is applied to each movie's
tracks_df before computing sigma, consistent with the rest of this project.

Each movie's headless run is cached under
cache/trackmate_headless_immobilized/ (output XML + result JSON); rerunning
this script without changing IMMOB_DATASETS reuses the existing runs.

Unconditionally overwrites cache/localization_error_immobilized_headless.pkl
every run (the ORIGINAL cache/localization_error_immobilized.pkl from
TaskA_Immobilized_Compute.py is left untouched, so the two approaches can be
compared directly -- this script prints that comparison at the end if the
original cache exists).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hydro_analysis.core.io import find_rec_tif_files, remove_edge_artifacts, extract_particle_size_from_path
from hydro_analysis.MSD_Trackmate.Validation_Claude._trackmate_settings_parser import parse_trackmate_full_settings
from hydro_analysis.MSD_Trackmate.Validation_Claude.trackmate_headless_runner import (
    run_trackmate_headless, params_from_settings,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Compare_Tracks_vs_RawAnalysis import match_tracks_to_analysis
from hydro_analysis.MSD_Trackmate.Validation_GPT.trackmate_session import read_trackmate_session

# ── Configuration ──────────────────────────────────────────────────────────────
IMMOB_DATASETS: dict[str, dict[str, Path]] = {
    "2026.01.31_immobilized": {
        "tracks_folder": Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.31_immobilized\Tracks"),
        "analysis_folder": Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.31_immobilized\Analysis"),
    },
    "2026.01.28_immobilized_locerror": {
        "tracks_folder": Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.28 _ Immobilized\Tracks"),
        "analysis_folder": Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.28 _ Immobilized\Analysis"),
    },
}

CACHE_DIR = Path(__file__).parent.parent / "cache" / "trackmate_headless_immobilized"
CACHE_FILE = Path(__file__).parent.parent / "cache" / "localization_error_immobilized_headless.pkl"
OLD_CACHE_FILE = Path(__file__).parent.parent / "cache" / "localization_error_immobilized.pkl"

MIN_POINTS_FOR_SIGMA = 3

_SIZE_NAME_OVERRIDES = {"1k": 1000.0}


def _corrected_particle_size(base_name: str, particle_size_nm: float | None) -> float | None:
    stem = base_name.lower().replace("_tracks.xml", "").replace(".xml", "")
    for token, corrected in _SIZE_NAME_OVERRIDES.items():
        if stem == token or stem.startswith(token + "_") or stem.startswith(token + "."):
            return corrected
    return particle_size_nm


def run_movie(dataset_label: str, tracks_xml: Path, analysis_xml: Path) -> dict[str, Any] | None:
    movie_tag = tracks_xml.stem.replace("_Tracks", "").replace(" ", "_")
    cache_key = f"{dataset_label}_{movie_tag}"
    out_xml = CACHE_DIR / f"{cache_key}.xml"

    calib = find_rec_tif_files(tracks_xml)
    if calib["tiff_file"] is None or calib["mpp"] is None:
        print(f"  [SKIP] {tracks_xml.name}: calibration could not be resolved")
        return None

    try:
        settings = parse_trackmate_full_settings(analysis_xml)
    except NotImplementedError as exc:
        print(f"  [SKIP] {analysis_xml.name}: {exc}")
        return None

    params = params_from_settings(settings, str(calib["tiff_file"]), str(out_xml))
    try:
        result = run_trackmate_headless(params, CACHE_DIR / f"{cache_key}_params.json")
    except RuntimeError as exc:
        print(f"  [SKIP] {cache_key}: headless run failed ({exc})")
        return None

    try:
        session = read_trackmate_session(out_xml)
    except ValueError as exc:
        print(f"  [SKIP] {cache_key}: raw session unusable ({exc})")
        return None

    traj = session.trajectories.copy()
    traj["particle"] = pd.factorize(traj["trajectory_id"])[0] + 1
    tracks_df = traj[["frame", "particle", "x", "y"]].sort_values(["particle", "frame"]).reset_index(drop=True)

    if not tracks_df.empty:
        tracks_df, edge_stats = remove_edge_artifacts(tracks_df, settings["width_px"], settings["height_px"])
        if edge_stats["n_removed"]:
            print(f"     [EDGE FILTER] {cache_key}: removed {edge_stats['n_removed']} detections near border, "
                  f"+{edge_stats['n_splits']} tracks from splits")

    particle_size_nm = _corrected_particle_size(tracks_xml.name, extract_particle_size_from_path(tracks_xml))
    print(f"  [OK] {cache_key}: n_spots={result['n_spots']} n_tracks_filtered={result['n_tracks_filtered']} "
          f"-> {tracks_df['particle'].nunique() if not tracks_df.empty else 0} tracks post-edge-filter, "
          f"size={particle_size_nm} nm")

    return {
        "dataset_label": dataset_label, "movie": tracks_xml.name, "tracks_df": tracks_df,
        "mpp": calib["mpp"], "fps": calib["fps"], "particle_size_nm": particle_size_nm,
    }


def _track_sigma_rows(result: dict[str, Any]) -> list[dict]:
    tracks_df, mpp = result["tracks_df"], result["mpp"]
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

        dx, dy = np.diff(x_um), np.diff(y_um)
        sigma_x_f2f_nm = float(np.std(dx, ddof=1) / np.sqrt(2) * 1000.0) if len(dx) >= 2 else np.nan
        sigma_y_f2f_nm = float(np.std(dy, ddof=1) / np.sqrt(2) * 1000.0) if len(dy) >= 2 else np.nan

        rows.append({
            "dataset_label": result["dataset_label"], "movie": result["movie"], "particle_id": int(pid),
            "particle_size_nm": result["particle_size_nm"], "track_length_frames": int(n),
            "sigma_x_static_nm": sigma_x_static_nm, "sigma_y_static_nm": sigma_y_static_nm,
            "sigma_x_frame2frame_nm": sigma_x_f2f_nm, "sigma_y_frame2frame_nm": sigma_y_f2f_nm,
            "mpp": mpp, "fps": result["fps"],
        })
    return rows


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for dataset_label, folders in IMMOB_DATASETS.items():
        print(f"\n=== {dataset_label} ===")
        matched, unmatched_tracks, unmatched_analysis = match_tracks_to_analysis(
            folders["tracks_folder"], folders["analysis_folder"])
        print(f"{len(matched)} matched movie pairs, {len(unmatched_tracks)} Tracks file(s) unmatched, "
              f"{len(unmatched_analysis)} Analysis file(s) without a Tracks counterpart")
        for t, tif_name, n_cand in unmatched_tracks:
            note = "no raw analysis for this movie" if n_cand == 0 else f"{n_cand} ambiguous raw analysis candidates"
            print(f"  [UNMATCHED] {t.name} (tif={tif_name}): {note}")

        for tracks_xml, analysis_xml, _ in matched:
            result = run_movie(dataset_label, tracks_xml, analysis_xml)
            if result is not None:
                all_rows.extend(_track_sigma_rows(result))

    df = pd.DataFrame(all_rows)
    print(f"\nGesamt: {len(df)} Tracks mit gueltigem sigma_loc (>= {MIN_POINTS_FOR_SIGMA} Punkte).")

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

    if OLD_CACHE_FILE.exists() and not df.empty:
        with open(OLD_CACHE_FILE, "rb") as f:
            old_df = pickle.load(f)
        print(f"\n--- Vergleich mit der urspruenglichen (Tracks.xml-basierten) Berechnung ---")
        old_summary = old_df.groupby("particle_size_nm")["sigma_x_static_nm"].mean()
        new_summary = df.groupby("particle_size_nm")["sigma_x_static_nm"].mean()
        compare = pd.DataFrame({"sigma_x_mean_nm_old": old_summary, "sigma_x_mean_nm_headless": new_summary})
        print(compare.to_string())


if __name__ == "__main__":
    main()
