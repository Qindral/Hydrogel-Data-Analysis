"""
Localization-error analysis for immobilized particles: per-track sigma_loc,
recentered trajectory overlays, and a loc-error-vs-step-size comparison
against real diffusion measurements -- consolidated into one script.
Replaces TaskA_Immobilized_Compute.py + TaskA_Figure1_SigmaLoc_Distribution.py
+ TaskA_MSD_Trajectories_by_Size.py (removed). TaskA2_A4_Intercept_
Comparison.py and TaskA3_MSD_Correction.py are a separate concern (intercept
comparison, MSD correction) and are untouched -- they only read the pickle
cache below, unmodified schema. Frame_Sharpness_Immobilized.py stays its own
module (imported, not copied in) since it is also useful standalone.

Compute stage: scans both immobilized-particle TrackMate XML folders
(IMMOB_FOLDERS) via _build_immob_datasets() (calibration/metadata only --
IMMOB_FOLDERS points at raw/full TrackMate session XML, e.g.
Analysis_immob_NEW\\, not the dedicated Tracks\\*.xml export
core.io.build_datasets()/read_trackmate_xml() expect; the actual track
positions are parsed per movie by clean_movie_tracks()'s own
_load_tracks_from_new_session() call below). Each movie's sigma_loc rows
(cache/locerror_vs_stepsize_ratio.png) go through clean_movie_tracks()'s
full cleaning pipeline exactly once. The trajectory overlay
(trajectories_overlay_by_size.png) is sourced SEPARATELY and deliberately
NOT from that cleaned result -- see "Track-position source" below -- so
it is independent of clean_movie_tracks() and of what that pipeline drops.

Track-position source: Analysis_immob_NEW\\ (current/authoritative, DOG
detector, confirmed by the user) when available for a movie, else the
older Analysis\\/Tracks\\ pair (LOG detector); falls back further to the
Tracks\\*.xml positions alone if neither raw session parses. Every
fallback is logged explicitly, never silent. clean_movie_tracks() (sigma_loc
rows, cache, locerror_vs_stepsize_ratio.png) applies core.io.
remove_edge_artifacts() to Analysis_immob_NEW\\ positions as usual --
unchanged. trajectories_overlay_by_size.png instead uses
_load_raw_trajectory_data(), which skips that entirely (no edge-artifact
removal/splitting) and applies only MIN_FRAME_COVERAGE_FRACTION (a track
must cover >= 4/5 of the movie's total frames, from ALL detected spots, to
be kept) -- a completeness-only filter, it never splits a track or moves a
detection. This keeps the two outputs on intentionally different cleaning
regimes: sigma_loc/step-size numbers stay on the established pipeline,
while the trajectory picture shows positions as close to raw as possible.

Cleaning pipeline per movie, in this order:
  1. Blurry-frame removal (Frame_Sharpness_Immobilized.py's variance-of-
     Laplacian focus metric) -- a defocused frame inflates apparent
     positional spread without that being real localization noise.
  2. Step-outlier splitting, 500 nm dataset only (STEP_OUTLIER_THRESHOLD_NM
     = 200 nm): a track is split wherever its own frame-to-frame step
     exceeds this -- a one-frame glitch, not real motion, and not
     necessarily a whole-track mis-link (step 3 handles that). Same
     "split the link, don't guess which side is wrong" convention as
     core.io.remove_edge_artifacts().
  3. Outlier-trajectory removal (OUTLIER_RANGE_FACTOR = 10x the movie's
     median track range): a track spanning tens of micrometers is almost
     certainly a mis-linked trajectory, not a stationary particle. Runs
     before the close-pair check and drift subtraction so it cannot
     contaminate either.
  4. Close-pair contamination removal: two particles closer than
     CLOSE_PAIR_DIST_FACTOR * r_px (2r) in some shared frame have
     overlapping PSFs that bias both centroids; every particle within
     EXCLUSION_RADIUS_FACTOR * r_px (3r, i.e. diameter 6r) of a flagged
     particle's mean position is dropped. r_px is read from the movie's
     own real detector settings (LOG_DETECTOR or DOG_DETECTOR). Skipped
     entirely for the "1k" (1000 nm) dataset -- deliberately dense by
     design, all particles kept regardless of separation.
  5. Ensemble drift subtraction (trackpy.subtract_drift()) -- removes any
     common-mode (bulk stage/focus) drift shared by every particle, so
     remaining per-track spread reflects independent localization noise.

Two outputs (Auswertungsbilder\\LocError_Immobilized\\) plus the trajectory
overlay -- no separate sigma_x/sigma_y figure (the combined metric below is
the one that matters here; per-axis anisotropy is still in the pickle/CSV
for anyone who needs it):

  sigma_loc_combined_by_size.png
      sigma_xy_static_nm (RMS of sigma_x/sigma_y), black boxes, one per
      (particle_size_nm, mpp) group. FILTERED to only the mpp values
      actually used for that same particle size in the real water/hydrogel
      diffusion measurements (WATER_HYDROGEL_FOLDERS below, the same
      folder set TaskC_Detectability_Compute.py/SNR_DoG_Compute.py use) --
      matched at 2-decimal mpp precision, not exact (see
      _qualifying_groups()'s docstring for why). The underlying
      cache/localization_error_immobilized.pkl and its printed summary
      table are NOT filtered -- every cleaned track is kept there.

  trajectories_overlay_by_size.png
      Trajectory overlay recentered to each track's OWN STARTING POINT
      (x - x[0], y - y[0], nm -- not its mean position), axes labelled
      Delta x / Delta y accordingly. Tracks come from _load_raw_trajectory_
      data() (see "Track-position source" above), NOT from clean_movie_
      tracks()'s cleaned result -- independent of the sigma_loc pipeline.
      Same (particle_size_nm, mpp) filtering as sigma_loc_combined_by_
      size.png. Every panel uses the same fixed display window
      (TRAJECTORY_WINDOW_NM), not a size-dependent one.
      _drop_tracks_exceeding_window() additionally drops (for this overlay
      only, per movie, logged) any track whose own excursion from its
      recentred start exceeds half the display window in x or y -- a real
      immobilized particle cannot move hundreds of nm in one frame, so this
      is a mis-link (most likely a close-pair swap; the close-pair filter
      itself is deliberately not applied to this raw overlay source, see
      "Track-position source" above), not localization noise, and left in
      would otherwise draw one dominant streak across the panel that
      obscures the origin-clustered tracks it exists to be compared
      against.

  locerror_vs_stepsize_ratio.png
      Boxplot (one box per condition per particle size), not a single
      point: every qualifying track's own sigma_xy_static_nm divided by
      that particle size's real diffusive step size -- the median
      Schrittweite per particle size (pooled across water+hydrogel, µm ->
      nm here), computed once by iMSD_histograms.py from its own per-track
      table and handed over via cache/imsd_median_stepsize_by_size.pkl
      (_load_median_stepsize_nm() below). Both conditions' boxes are
      divided by the SAME per-size reference (no separate water/hydrogel
      step-size scan here anymore). A ratio near or above 1 means
      localization noise is comparable to or larger than the real
      per-frame motion being measured at that size: the diffusion
      measurement there is noise-limited, not motion-limited.

Unconditionally overwrites cache/localization_error_immobilized.pkl every
run -- same schema as the predecessor scripts (plus sigma_xy_static_nm/
sigma_xy_frame2frame_nm, already present since that combined-metric
request), so TaskA2_A4_Intercept_Comparison.py / TaskA3_MSD_Correction.py
keep working unmodified.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import trackpy as tp
import matplotlib.pyplot as plt

from hydro_analysis.core.io import (
    remove_edge_artifacts, get_dls_labels, find_rec_tif_files,
    extract_particle_size_from_path, parse_rec_file,
)
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude.Frame_Sharpness_Immobilized import (
    compute_sharpness_series, flag_blurry_frames,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude._trackmate_settings_parser import (
    parse_trackmate_full_settings,
)
from hydro_analysis.MSD_Trackmate.Validation_GPT.trackmate_session import read_trackmate_session

# ── Configuration ──────────────────────────────────────────────────────────────
IMMOB_FOLDERS: dict[str, Path] = {
    "2026.01.31_immobilized": Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.31_immobilized\Analysis_immob_NEW"),#Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.31_immobilized\Tracks"),
    #"2026.01.28_immobilized_locerror": Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.28 _ Immobilized\Tracks"),
}

# Tracks\+Analysis\ folder pairs for match_tracks_to_analysis() -- needed by
# PSF_Analysis.py (moved here from the now-obsolete
# TaskA_Immobilized_Compute_Headless.py; this is the single source for both
# IMMOB_DATASETS and _corrected_particle_size() now, no longer duplicated).
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

CACHE_FILE = Path(__file__).parent.parent / "cache" / "localization_error_immobilized.pkl"

# Qualifying-mpp bookkeeping only (_scan_qualifying_mpp()): which mpp was
# actually used per particle size in water/hydrogel, read from the
# already-computed step-size caches (Schrittweiten_methode_D0.py /
# Schrittweiten_methode_20mg.py) instead of re-scanning
# WATER_HYDROGEL_FOLDERS' raw XML here. Run those two scripts first (or
# after any raw-data change) to refresh these pickles. The step-size
# reference VALUES themselves come from CACHE_MEDIAN_STEPSIZE below.
CACHE_STEPSIZE_D0 = Path(__file__).parent.parent / "cache" / "stepsize_d0_results.pkl"
CACHE_STEPSIZE_DEFF = Path(__file__).parent.parent / "cache" / "stepsize_20mg_results.pkl"

# Median Schrittweite je Partikelgroesse (Wasser+Hydrogel gepoolt), aus
# iMSD_histograms.py -- ersetzt die fruehere eigene KDE-Median-Berechnung
# innerhalb dieses Skripts (siehe _load_median_stepsize_nm()).
CACHE_MEDIAN_STEPSIZE = Path(__file__).parent.parent / "cache" / "imsd_median_stepsize_by_size.pkl"

SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "LocError_Immobilized"

NEW_ANALYSIS_SUBFOLDER = "Analysis_immob_NEW"   # current/authoritative reanalysis, 2026.01.31_immobilized only

MIN_POINTS_FOR_SIGMA = 3        # need >=3 positions for a meaningful std
DEFAULT_LOG_RADIUS_PX = 3.0     # used only if a movie's own settings XML cannot be matched
CLOSE_PAIR_DIST_FACTOR = 2.0
EXCLUSION_RADIUS_FACTOR = 3.0
OUTLIER_RANGE_FACTOR = 10.0
STEP_OUTLIER_SIZE_NM = 500.0
STEP_OUTLIER_THRESHOLD_NM = 200.0

# _load_raw_trajectory_data()'s minimal-manipulation track source for
# trajectories_overlay_by_size.png ONLY (main sigma_loc/cache pipeline via
# clean_movie_tracks() is unaffected, still applies edge-artifact removal):
# a track must cover >= 4/5 of the movie's total frames to be kept.
MIN_FRAME_COVERAGE_FRACTION = 0.02

# Reuses the exact per-size raw-XML folder paths already established in
# TaskC_Detectability_Compute.py::MOVIE_FOLDERS (re-declared here rather than
# imported, to avoid a circular import -- TaskC_Detectability_Compute.py
# imports IMMOB_FOLDERS etc. from THIS script).
_HYDROGEL_ROOT = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")


COLOR_XY, COLOR_XY_DARK = "#000000", "#000000"
COLOR_TRK = "#000000"
COLOR_RATIO = "#3030d6"
TRAJECTORY_WINDOW_NM = 700.0   # fixed +-300 nm display window, every panel


# ── Naming / size correction ────────────────────────────────────────────────────

def _matches_1k_stem(base_name: str) -> bool:
    stem = base_name.lower().replace("_tracks.xml", "").replace(".xml", "")
    return stem == "1k" or stem.startswith("1k_") or stem.startswith("1k.")


def _corrected_particle_size(base_name: str, particle_size_nm: float | None) -> float | None:
    # core.io.extract_particle_size_from_path()'s generic fallback regex reads
    # "1k_Tracks.xml" as size=1 -- corrected locally, "1k" unambiguously means
    # 1000 nm in this dataset's naming (not changing core/io.py for this).
    if _matches_1k_stem(base_name):
        if particle_size_nm != 1000.0:
            print(f"  [SIZE FIX] {base_name}: particle_size_nm {particle_size_nm} -> 1000.0")
        return 1000.0
    return particle_size_nm


def _is_1k_dataset(base_name: str) -> bool:
    return _matches_1k_stem(base_name)


def _norm_stem(s: str) -> str:
    return re.sub(r"[\s_]", "", s).lower()


def _find_xml_in_subfolder(tracks_xml: Path, subfolder: str) -> Path | None:
    """Locate a raw TrackMate project XML for a Tracks\\*.xml file in a
    sibling subfolder ("Analysis" or "Analysis_immob_NEW"). Exact-stem
    match first; falls back to whitespace/underscore-stripped, lowercased
    comparison (raw exports frequently drop spaces, e.g. "100 nm_2" ->
    "100nm_2.xml"). An ambiguous normalized match is reported, never
    guessed."""
    stem = tracks_xml.stem.replace("_Tracks", "")
    folder = tracks_xml.parent.parent / subfolder
    if not folder.exists():
        return None
    exact = folder / f"{stem}.xml"
    if exact.exists():
        return exact
    target = _norm_stem(stem)
    candidates = [p for p in folder.glob("*.xml") if _norm_stem(p.stem) == target]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"  [WARN] {tracks_xml.name}: ambiguous match in {folder}: {[c.name for c in candidates]}")
    return None


# ── Close-pair contamination removal ────────────────────────────────────────────

def _detection_radius_px(tracks_xml: Path) -> float:
    settings_xml = (
        _find_xml_in_subfolder(tracks_xml, NEW_ANALYSIS_SUBFOLDER)
        or _find_xml_in_subfolder(tracks_xml, "Analysis")
    )
    if settings_xml is None:
        print(f"  [NOTE] {tracks_xml.name}: no settings XML found, using default radius "
              f"{DEFAULT_LOG_RADIUS_PX} px")
        return DEFAULT_LOG_RADIUS_PX
    try:
        settings = parse_trackmate_full_settings(settings_xml)
    except NotImplementedError as exc:
        print(f"  [NOTE] {tracks_xml.name}: {exc} Using default radius {DEFAULT_LOG_RADIUS_PX} px")
        return DEFAULT_LOG_RADIUS_PX
    return float(settings["radius"])


def _find_close_pair_particles(tracks_df: pd.DataFrame, r_px: float) -> dict[int, tuple[float, float]]:
    """Frame-by-frame (not mean-position-only: split track fragments of the
    SAME stationary particle sit at near-identical means without ever
    sharing a frame, which a mean-position-only check would misflag)."""
    flagged_pids: set[int] = set()
    for _, g in tracks_df.groupby("frame"):
        if len(g) < 2:
            continue
        xy = g[["x", "y"]].to_numpy()
        pids = g["particle"].to_numpy()
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        np.fill_diagonal(d, np.inf)
        close = np.any(d < CLOSE_PAIR_DIST_FACTOR * r_px, axis=1)
        flagged_pids.update(int(p) for p in pids[close])
    if not flagged_pids:
        return {}
    means = tracks_df[tracks_df["particle"].isin(flagged_pids)].groupby("particle")[["x", "y"]].mean()
    return {int(pid): (float(row["x"]), float(row["y"])) for pid, row in means.iterrows()}


def _exclude_near_hotspots(
    tracks_df: pd.DataFrame, hotspots: dict[int, tuple[float, float]], r_px: float,
) -> tuple[pd.DataFrame, set[int]]:
    if not hotspots:
        return tracks_df, set()
    hs_xy = np.array(list(hotspots.values()))
    means = tracks_df.groupby("particle")[["x", "y"]].mean()
    pids = means.index.to_numpy()
    xy = means.to_numpy()
    d_to_hotspot = np.min(
        np.hypot(xy[:, None, 0] - hs_xy[None, :, 0], xy[:, None, 1] - hs_xy[None, :, 1]), axis=1,
    )
    excluded_pids = set(pids[d_to_hotspot < EXCLUSION_RADIUS_FACTOR * r_px].tolist())
    kept = tracks_df[~tracks_df["particle"].isin(excluded_pids)].reset_index(drop=True)
    return kept, excluded_pids


# ── Blurry-frame removal ────────────────────────────────────────────────────────

def _remove_blurry_frames(tracks_df: pd.DataFrame, tif_path: str | None, base_name: str) -> tuple[pd.DataFrame, int]:
    if tif_path is None or not Path(tif_path).exists():
        print(f"  [NOTE] {base_name}: no TIFF found, skipping blurry-frame filter")
        return tracks_df, 0
    try:
        scores = compute_sharpness_series(Path(tif_path))
    except Exception as exc:
        print(f"  [WARN] {base_name}: sharpness computation failed ({exc}), skipping blurry-frame filter")
        return tracks_df, 0
    is_blurry, _ = flag_blurry_frames(scores)
    blurry_frames = set(np.flatnonzero(is_blurry).tolist())
    if not blurry_frames:
        return tracks_df, 0
    mask = tracks_df["frame"].isin(blurry_frames)
    n_removed = int(mask.sum())
    return tracks_df.loc[~mask].reset_index(drop=True), n_removed


# ── NEW-reanalysis track source (2026.01.31_immobilized only) ──────────────────

def _frame_shape_px(tif_path: str | None) -> tuple[float, float] | None:
    if tif_path is None or not Path(tif_path).exists():
        return None
    with tifffile.TiffFile(tif_path) as tif:
        page0 = tif.series[0]
        h, w = page0.shape[-2], page0.shape[-1]
    return float(w), float(h)


def _filter_short_tracks(traj: pd.DataFrame, n_frames: int, base_name: str) -> pd.DataFrame:
    """Drop trajectories covering less than MIN_FRAME_COVERAGE_FRACTION of
    the movie's total frame count -- a completeness filter only (track
    length vs. n_frames), never looks at position/step size/shape, so it
    cannot split a track or reposition a detection. n_frames comes from
    ALL detected spots in the session (not just tracked ones), so it
    reflects the movie's real length even for a particle never tracked
    through the final frames."""
    if traj.empty:
        return traj
    min_len = MIN_FRAME_COVERAGE_FRACTION * n_frames
    lengths = traj.groupby("particle").size()
    short_pids = lengths[lengths < min_len].index
    if len(short_pids):
        print(f"  [FRAME COVERAGE] {base_name}: {len(short_pids)} track(s) below "
              f"{MIN_FRAME_COVERAGE_FRACTION:.0%} frame coverage ({min_len:.0f}/{n_frames} frames) -- removed")
    return traj[~traj["particle"].isin(short_pids)].reset_index(drop=True)


def _load_tracks_from_new_session(
    tracks_xml: Path, tif_path: str | None, base_name: str,
    *, apply_edge_filter: bool = True, apply_frame_coverage_filter: bool = False,
) -> pd.DataFrame | None:
    """Positions from the current/authoritative Analysis_immob_NEW\\
    reanalysis, read via its raw session XML (spatialunits=pixel, matches
    Tracks\\'s x/y convention). Two independent, optional manipulations on
    top of read_trackmate_session()'s own FilteredTracks/AllTracks edge-walk
    reconstruction:
      apply_edge_filter=True (default): core.io.remove_edge_artifacts() --
        drops near-border detections and splits each affected trajectory at
        the gap. This is clean_movie_tracks()'s own call (main sigma_loc/
        cache/locerror_vs_stepsize_ratio.png pipeline), unchanged.
      apply_frame_coverage_filter=True: MIN_FRAME_COVERAGE_FRACTION -- a
        track must cover >= 4/5 of the movie's total frame count (from ALL
        detected spots, not just tracked ones) to be kept. Completeness
        only, never splits a track or repositions a detection. Used ONLY by
        _load_raw_trajectory_data() below, for trajectories_overlay_by_
        size.png specifically -- deliberately NOT part of the main
        clean_movie_tracks() pipeline, which keeps its original cleaning.
    Returns None (caller falls back to Tracks\\ data, with a console note)
    if no matching file exists or it fails to parse."""
    session_xml = _find_xml_in_subfolder(tracks_xml, NEW_ANALYSIS_SUBFOLDER)
    if session_xml is None:
        return None
    try:
        session = read_trackmate_session(session_xml)
    except ValueError as exc:
        print(f"  [WARN] {base_name}: {NEW_ANALYSIS_SUBFOLDER}\\{session_xml.name} unusable ({exc}); "
              f"falling back to Tracks\\ data")
        return None

    traj = session.trajectories[["trajectory_id", "frame", "x", "y"]].copy()
    id_map = {tid: i + 1 for i, tid in enumerate(sorted(traj["trajectory_id"].unique()))}
    traj["particle"] = traj["trajectory_id"].map(id_map)
    traj["frame"] = traj["frame"].astype(int)
    traj = traj[["frame", "particle", "x", "y"]].sort_values(["particle", "frame"]).reset_index(drop=True)

    if apply_frame_coverage_filter:
        n_frames = int(session.spots["frame"].max()) + 1
        traj = _filter_short_tracks(traj, n_frames, base_name)
        if traj.empty:
            print(f"  {base_name}: no tracks left after frame-coverage filter")
            return traj

    if apply_edge_filter:
        shape = _frame_shape_px(tif_path)
        if shape is not None:
            w, h = shape
            traj, edge_stats = remove_edge_artifacts(traj, w, h)
            if edge_stats["n_removed"]:
                print(f"  [EDGE FILTER] {base_name} ({NEW_ANALYSIS_SUBFOLDER}): removed {edge_stats['n_removed']} "
                      f"detections near border ({w:.0f}x{h:.0f} px), +{edge_stats['n_splits']} tracks from splits")
        else:
            print(f"  [NOTE] {base_name}: no TIFF found for edge-artifact filtering of the {NEW_ANALYSIS_SUBFOLDER} data")

    print(f"  [NEW ANALYSIS] {base_name}: using {NEW_ANALYSIS_SUBFOLDER}\\{session_xml.name} "
          f"({len(traj)} detections, {traj['particle'].nunique()} tracks)")
    return traj


def _load_raw_trajectory_data(tracks_xml: Path, tif_path: str | None, base_name: str) -> pd.DataFrame | None:
    """Trajectory-overlay-only track source (trajectories_overlay_by_size.png):
    Analysis_immob_NEW\\ positions taken as close to raw as possible -- no
    edge-artifact removal/splitting, only MIN_FRAME_COVERAGE_FRACTION
    (track must cover >= 4/5 of the movie's total frames). Independent of
    clean_movie_tracks()'s main pipeline (sigma_loc rows, the pkl cache,
    locerror_vs_stepsize_ratio.png), which is unchanged and keeps its
    original edge-artifact/blurry-frame/outlier-range/close-pair cleaning."""
    return _load_tracks_from_new_session(
        tracks_xml, tif_path, base_name,
        apply_edge_filter=False, apply_frame_coverage_filter=True,
    )


# ── Dataset scan for raw TrackMate session folders (IMMOB_FOLDERS) ─────────────

# core.io.find_rec_tif_files() matches the .rec/.tif calibration filename to
# the XML stem exactly. In 2026.01.31_immobilized\, the calibration files for
# these XML stems were recorded with a space before "nm" (e.g. "100nm.xml" ->
# "100 nm.tif.rec", "1000nm.xml" -> "1000 nm .tif.rec", note the extra
# trailing space in that last one) while the XML itself has none -- a fixed
# naming mismatch in this already-recorded dataset, not something that will
# change, so it is spelled out explicitly rather than pattern-matched.
# 20/50 nm and "1k" are named identically in both places and need no entry.
_REC_FILENAME_OVERRIDES: dict[str, str] = {
    "100nm": "100 nm.tif.rec",
    "100nm_2": "100 nm_2.tif.rec",
    "100nm_3": "100 nm_3.tif.rec",
    "200nm_immob": "200 nm_immob.tif.rec",
    "200nm_immob_2": "200 nm_immob_2.tif.rec",
    "200nm_immob_3": "200 nm_immob_3.tif.rec",
    "200nm_immob_4": "200 nm_immob_4.tif.rec",
    "500nm_2": "500 nm_2.tif.rec",
    "500nm_3": "500 nm_3.tif.rec",
    "500nm_4": "500 nm_4.tif.rec",
    "500nm_5": "500 nm_5.tif.rec",
    "500nm_6": "500 nm_6.tif.rec",
    "500nm_7": "500 nm_7.tif.rec",
    "1000nm": "1000 nm .tif.rec",
}


def _calibration_for(xml_file: Path) -> dict:
    """Calibration (fps/mpp/tif/rec) for one Analysis_immob_NEW\\ XML file:
    the explicit _REC_FILENAME_OVERRIDES entry for this file's known
    calibration-filename mismatch, else core.io.find_rec_tif_files()'s
    exact-name match (checked first so the override cases never trigger
    find_rec_tif_files()'s own "Missing calibration" print for a filename
    it was never going to find)."""
    override_name = _REC_FILENAME_OVERRIDES.get(xml_file.stem)
    if override_name is None:
        return find_rec_tif_files(xml_file)
    rec_path = xml_file.parent.parent / override_name
    rec_info = parse_rec_file(rec_path)
    tif_path = rec_path.with_name(override_name[: -len(".tif.rec")] + ".tif")
    return {
        "fps": rec_info.get("fps"),
        "mpp": rec_info.get("mpp"),
        "size_x": rec_info.get("size_x"),
        "size_y": rec_info.get("size_y"),
        "tiff_file": tif_path if tif_path.exists() else None,
        "rec_file": rec_path if rec_path.exists() else None,
    }


def _build_immob_datasets(folder: Path) -> dict[str, dict]:
    """Same result_dict shape core.io.build_datasets() produces, but for a
    folder of raw/full TrackMate session XML (Analysis_immob_NEW\\ style --
    Model/AllSpots/AllTracks schema) rather than the dedicated Tracks\\*.xml
    export core.io.read_trackmate_xml() parses. Calling build_datasets()
    directly on such a folder crashes in core.io._build_result_dict() (the
    schema mismatch makes read_trackmate_xml() return an empty, columnless
    DataFrame). tracks_df is left None here -- clean_movie_tracks() parses
    the session XML itself via its existing _load_tracks_from_new_session()
    call, which already handles this schema."""
    datasets: dict[str, dict] = {}
    for xml_file in sorted(folder.glob("*.xml")):
        calib = _calibration_for(xml_file)
        if calib["fps"] is None or calib["mpp"] is None:
            print(f"     ERROR: Missing calibration for {xml_file.name} "
                  f"(fps={calib['fps']}, mpp={calib['mpp']})")
            continue
        datasets[xml_file.stem] = {
            "xml_path": str(xml_file),
            "tif_path": str(calib["tiff_file"]) if calib["tiff_file"] else None,
            "rec_path": str(calib["rec_file"]) if calib["rec_file"] else None,
            "base_name": xml_file.name,
            "tracks_df": None,
            "mpp": calib["mpp"],
            "fps": calib["fps"],
            "particle_size_nm": extract_particle_size_from_path(xml_file),
        }
    return datasets


# ── Outlier-trajectory removal ──────────────────────────────────────────────────

def _remove_outlier_range_tracks(tracks_df: pd.DataFrame, mpp: float, base_name: str) -> tuple[pd.DataFrame, int]:
    ranges: dict[int, float] = {}
    for pid, g in tracks_df.groupby("particle"):
        x_nm = g["x"].to_numpy(dtype=float) * mpp * 1000.0
        y_nm = g["y"].to_numpy(dtype=float) * mpp * 1000.0
        ranges[int(pid)] = float(max(x_nm.max() - x_nm.min(), y_nm.max() - y_nm.min()))
    if not ranges:
        return tracks_df, 0
    median_range = float(np.median(list(ranges.values())))
    if median_range <= 0:
        return tracks_df, 0
    threshold = OUTLIER_RANGE_FACTOR * median_range
    outlier_pids = {pid for pid, r in ranges.items() if r > threshold}
    if not outlier_pids:
        return tracks_df, 0
    for pid in sorted(outlier_pids):
        print(f"  [OUTLIER TRACK] {base_name}: particle {pid} spans {ranges[pid]:.0f} nm "
              f"({ranges[pid] / median_range:.1f}x the movie's median {median_range:.0f} nm) -- removed")
    kept = tracks_df[~tracks_df["particle"].isin(outlier_pids)].reset_index(drop=True)
    return kept, len(outlier_pids)


# ── Trajectory-overlay-only excursion filter ────────────────────────────────────

def _drop_tracks_exceeding_window(
    traj_df: pd.DataFrame, mpp: float, window_nm: float, base_name: str,
) -> tuple[pd.DataFrame, int]:
    """trajectories_overlay_by_size.png-only filter, applied after
    _load_raw_trajectory_data(): drops any track whose excursion from its
    OWN recentred starting point (the same x - x[0], y - y[0] plot_
    trajectories_by_size() itself computes) exceeds half the fixed display
    window in x or y. _remove_outlier_range_tracks()'s 10x-median heuristic
    (used by clean_movie_tracks(), not here) fails on a movie where most
    tracks are already contaminated, since the median itself is then
    inflated -- this uses the window's own absolute scale instead. A real
    immobilized particle cannot move hundreds of nm in one frame, so a track
    that does is a mis-link (most likely a close-pair swap; the close-pair
    filter itself is deliberately not applied to this raw overlay source,
    see module docstring), not localization noise, and left in would
    otherwise draw one dominant streak across the panel that obscures the
    origin-clustered tracks it exists to be compared against."""
    if traj_df.empty:
        return traj_df, 0
    half_window = window_nm / 2.0
    drop_pids = []
    for pid, g in traj_df.groupby("particle"):
        g = g.sort_values("frame")
        x_nm = g["x"].to_numpy(dtype=float) * mpp * 1000.0
        y_nm = g["y"].to_numpy(dtype=float) * mpp * 1000.0
        if np.max(np.abs(x_nm - x_nm[0])) > half_window or np.max(np.abs(y_nm - y_nm[0])) > half_window:
            drop_pids.append(int(pid))
    if not drop_pids:
        return traj_df, 0
    print(f"  [EXCURSION FILTER] {base_name}: {len(drop_pids)} track(s) jump > {half_window:.0f} nm "
          f"from their own start (trajectory overlay only, likely mis-link) -- removed")
    kept = traj_df[~traj_df["particle"].isin(drop_pids)].reset_index(drop=True)
    return kept, len(drop_pids)


# ── Step-outlier splitting (500 nm dataset only) ────────────────────────────────

def _split_step_outliers(tracks_df: pd.DataFrame, mpp: float, threshold_nm: float) -> tuple[pd.DataFrame, int]:
    if tracks_df.empty:
        return tracks_df, 0
    df = tracks_df.sort_values(["particle", "frame"]).reset_index(drop=True)
    x_nm = df["x"].to_numpy(dtype=float) * mpp * 1000.0
    y_nm = df["y"].to_numpy(dtype=float) * mpp * 1000.0
    particle = df["particle"].to_numpy()
    same_particle = np.concatenate([[False], particle[1:] == particle[:-1]])
    step = np.concatenate([[0.0], np.hypot(np.diff(x_nm), np.diff(y_nm))])
    outlier_step = same_particle & (step > threshold_nm)
    new_segment = (~same_particle) | outlier_step
    df["particle"] = new_segment.cumsum()
    return df, int(outlier_step.sum())


# ── Per-file cleaning pipeline ───────────────────────────────────────────────────

def clean_movie_tracks(result: dict) -> tuple[pd.DataFrame | None, float | None, dict]:
    """Full per-movie cleaning pipeline (see module docstring for the
    rationale behind each step). Returns (cleaned_tracks_df, particle_size_nm,
    stats); cleaned_tracks_df is None if nothing survives."""
    tracks_df = result.get("tracks_df")
    mpp = result.get("mpp")
    base_name = result.get("base_name")
    xml_path = result.get("xml_path")
    tif_path = result.get("tif_path")
    particle_size_nm = _corrected_particle_size(base_name, result.get("particle_size_nm"))

    stats = {"n_before": 0, "n_blurry_removed": 0, "n_outlier_removed": 0, "n_close_pair_removed": 0}
    if mpp is None:
        return None, particle_size_nm, stats

    # tracks_df may already be None here (IMMOB_FOLDERS pointing straight at a
    # raw-session folder, see _build_immob_datasets()) -- that is fine as long
    # as _load_tracks_from_new_session() below finds something to parse.
    if tracks_df is not None and tracks_df.empty:
        tracks_df = None

    xml_path = Path(xml_path)
    new_analysis_dir_exists = (xml_path.parent.parent / NEW_ANALYSIS_SUBFOLDER).exists()
    new_tracks_df = _load_tracks_from_new_session(xml_path, tif_path, base_name)
    if new_tracks_df is not None:
        if new_tracks_df.empty:
            print(f"  {base_name}: {NEW_ANALYSIS_SUBFOLDER} data empty after edge-artifact filtering")
            return None, particle_size_nm, stats
        tracks_df = new_tracks_df
    elif new_analysis_dir_exists:
        print(f"  [NOTE] {base_name}: no usable {NEW_ANALYSIS_SUBFOLDER} entry for this file, "
              f"using Tracks\\ data (superseded LOG-detector run)")

    if tracks_df is None or tracks_df.empty:
        print(f"  {base_name}: no track data available (neither {NEW_ANALYSIS_SUBFOLDER} nor Tracks\\ data)")
        return None, particle_size_nm, stats

    stats["n_before"] = len(tracks_df)

    tracks_df, n_blurry_removed = _remove_blurry_frames(tracks_df, tif_path, base_name)
    stats["n_blurry_removed"] = n_blurry_removed
    if tracks_df.empty:
        print(f"  {base_name}: all detections removed by blurry-frame filter")
        return None, particle_size_nm, stats

    if particle_size_nm == STEP_OUTLIER_SIZE_NM:
        tracks_df, n_step_splits = _split_step_outliers(tracks_df, mpp, STEP_OUTLIER_THRESHOLD_NM)
        if n_step_splits:
            print(f"  [STEP OUTLIER] {base_name}: {n_step_splits} step(s) > {STEP_OUTLIER_THRESHOLD_NM:.0f} nm "
                  f"-- track(s) split there")

    tracks_df, n_outlier_removed = _remove_outlier_range_tracks(tracks_df, mpp, base_name)
    stats["n_outlier_removed"] = n_outlier_removed
    if tracks_df.empty:
        print(f"  {base_name}: all particles removed by outlier-range filter")
        return None, particle_size_nm, stats

    if _is_1k_dataset(base_name):
        print(f"  [NOTE] {base_name}: 1k dataset, close-pair filter skipped (particles intentionally dense)")
    else:
        r_px = _detection_radius_px(xml_path)
        hotspots = _find_close_pair_particles(tracks_df, r_px)
        tracks_df, excluded_pids = _exclude_near_hotspots(tracks_df, hotspots, r_px)
        stats["n_close_pair_removed"] = len(excluded_pids)
        if hotspots:
            print(f"  [CLOSE-PAIR FILTER] {base_name}: {len(hotspots)} particle(s) in a close pair "
                  f"(r={r_px:.2f} px) -> {stats['n_close_pair_removed']} particle(s) removed")
    if tracks_df.empty:
        print(f"  {base_name}: all particles removed by close-pair filter")
        return None, particle_size_nm, stats

    # subtract_drift() sets a (frame, particle) MultiIndex with drop=False,
    # leaving 'particle' as both an index level and a column -- reset before
    # any further groupby("particle").
    tracks_df = tracks_df.copy()# tp.subtract_drift(tracks_df.copy()).reset_index(drop=True)

    print(f"  {base_name}: {stats['n_before']} -> {len(tracks_df)} detections "
          f"({stats['n_blurry_removed']} blurry-frame, {stats['n_outlier_removed']} outlier-track, "
          f"{stats['n_close_pair_removed']} close-pair particle detections removed)")
    return tracks_df, particle_size_nm, stats


# ── Per-track sigma_loc ──────────────────────────────────────────────────────────

def _sigma_rows(tracks_df: pd.DataFrame, mpp: float, fps: float, base_name: str,
                 particle_size_nm: float, dataset_label: str) -> list[dict]:
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
        sigma_xy_static_nm = float(np.sqrt((sigma_x_static_nm ** 2 + sigma_y_static_nm ** 2) / 2.0))

        dx = np.diff(x_um)
        dy = np.diff(y_um)
        sigma_x_f2f_nm = float(np.std(dx, ddof=1) / np.sqrt(2) * 1000.0) if len(dx) >= 2 else np.nan
        sigma_y_f2f_nm = float(np.std(dy, ddof=1) / np.sqrt(2) * 1000.0) if len(dy) >= 2 else np.nan
        sigma_xy_f2f_nm = (
            float(np.sqrt((sigma_x_f2f_nm ** 2 + sigma_y_f2f_nm ** 2) / 2.0))
            if np.isfinite(sigma_x_f2f_nm) and np.isfinite(sigma_y_f2f_nm) else np.nan
        )

        rows.append({
            "dataset_label": dataset_label, "movie": base_name, "particle_id": int(pid),
            "particle_size_nm": particle_size_nm, "track_length_frames": int(n),
            "sigma_x_static_nm": sigma_x_static_nm, "sigma_y_static_nm": sigma_y_static_nm,
            "sigma_xy_static_nm": sigma_xy_static_nm,
            "sigma_x_frame2frame_nm": sigma_x_f2f_nm, "sigma_y_frame2frame_nm": sigma_y_f2f_nm,
            "sigma_xy_frame2frame_nm": sigma_xy_f2f_nm,
            "mpp": mpp, "fps": fps,
        })
    return rows



# ── Water/hydrogel reference: which mpp is "real"; median step size ────────────

def _load_stepsize_cache(path: Path, compute_script: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _scan_qualifying_mpp() -> dict[float, set[float]]:
    """{particle_size_nm: {mpp values actually used for that size in
    water/hydrogel}}, pooled across both conditions. Reads
    CACHE_STEPSIZE_D0 (water)/CACHE_STEPSIZE_DEFF (hydrogel) --
    already-computed per-file result_dicts (Schrittweiten_methode_D0.py /
    Schrittweiten_methode_20mg.py) -- only for this (particle_size_nm, mpp)
    bookkeeping, which drives _qualifying_groups() below. The step-size
    reference values themselves (median Schrittweite per particle size) no
    longer come from here -- see _load_median_stepsize_nm(), which reads
    cache/imsd_median_stepsize_by_size.pkl (iMSD_histograms.py)."""
    caches = {
        "water": (CACHE_STEPSIZE_D0, "Schrittweiten_methode_D0.py"),
        "hydrogel": (CACHE_STEPSIZE_DEFF, "Schrittweiten_methode_20mg.py"),
    }
    ref_mpp: dict[float, set[float]] = {}
    for cache_path, compute_script in caches.values():
        results = _load_stepsize_cache(cache_path, compute_script)
        for result in results.values():
            if result is None:
                continue
            size_nm = result.get("particle_size_nm")
            mpp = result.get("mpp")
            if size_nm is None or not mpp:
                continue
            ref_mpp.setdefault(float(size_nm), set()).add(round(mpp, 2))
    return ref_mpp


def _load_median_stepsize_nm() -> dict[float, float]:
    """{particle_size_nm: median Schrittweite (nm)}, aus
    cache/imsd_median_stepsize_by_size.pkl (iMSD_histograms.py -- Median
    ueber Wasser+Hydrogel gepoolt, µm -> nm hier umgerechnet). Ersetzt die
    fruehere eigene KDE-Median-Schrittweiten-Berechnung dieses Skripts."""
    if not CACHE_MEDIAN_STEPSIZE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_MEDIAN_STEPSIZE}\nBitte zuerst iMSD_histograms.py ausführen.")
    with open(CACHE_MEDIAN_STEPSIZE, "rb") as f:
        median_um: dict[float, float] = pickle.load(f)
    return {float(size): float(value) * 1000.0 for size, value in median_um.items()}


# water/20.0 (WATER_HYDROGEL_FOLDERS) includes 2026.01.16\Tracks_20, which
# mixes in a small stray batch of files calibrated at mpp=0.15 alongside the
# otherwise uniform mpp=0.3 used for the actual 20 nm ("35 nm" DLS-labeled)
# water/hydrogel measurements. Without this, that stray 0.15 lets the
# immob_20nm_loc_error mpp=0.149 (immobilized) group also pass
# _qualifying_groups()'s 2-decimal match, even though it is not the
# calibration real 20 nm diffusion measurements were actually taken at --
# only the mpp=0.3 20 nm group is meant to be shown.
_EXCLUDED_QUALIFYING_MPP: dict[float, set[float]] = {20.0: {0.15}}


def _qualifying_groups(df: pd.DataFrame, ref_mpp: dict[float, set[float]]) -> list[tuple[float, float]]:
    """(particle_size_nm, mpp) pairs present in df whose mpp was actually
    used for that same size in water/hydrogel -- see module docstring.
    Matched at 2-decimal precision (not exact): the same nominal
    calibration setting (e.g. "0.15 um/px") is recorded with up to ~1%
    session-to-session rounding noise in the .rec metadata (0.299 vs
    0.300, 0.149 vs 0.150 have both been observed for what is clearly the
    same setup), so an exact-value match would wrongly drop real matches.
    _EXCLUDED_QUALIFYING_MPP overrides this for known stray reference
    calibrations that would otherwise wrongly let a group qualify."""
    pairs = df[["particle_size_nm", "mpp"]].dropna().drop_duplicates()
    qualifying = [
        (float(s), float(m)) for s, m in pairs.to_numpy().tolist()
        if round(m, 2) in ref_mpp.get(s, set())
        and round(m, 2) not in _EXCLUDED_QUALIFYING_MPP.get(s, set())
    ]
    return sorted(qualifying)


def _size_mpp_labels(groups: list[tuple[float, float]], dls_labels: dict[float, int]) -> list[str]:
    return [f"{dls_labels.get(s, int(s))} nm\n(mpp={m:.3g})" for s, m in groups]


# ── Plot: combined sigma_loc vs particle size ───────────────────────────────────

def _boxplot_by_group(ax, groups, positions, color, edge, width) -> None:
    bp = ax.boxplot(groups, positions=positions, widths=width, patch_artist=True,
                     showfliers=False, manage_ticks=False)
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_edgecolor(edge)
        box.set_alpha(0.6)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(edge)


def plot_combined_sigma_by_size(df: pd.DataFrame, qualifying: list[tuple[float, float]],
                                 dls_labels: dict[float, int]) -> plt.Figure:
    labels = _size_mpp_labels(qualifying, dls_labels)
    xy_groups = [df.loc[(df["particle_size_nm"] == s) & (df["mpp"] == m), "sigma_xy_static_nm"].dropna().to_numpy()
                 for s, m in qualifying]
    positions = np.arange(len(qualifying), dtype=float)

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    _boxplot_by_group(ax, xy_groups, positions, COLOR_XY, COLOR_XY_DARK, 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(r"$\sigma_{loc}$ (nm, combined x/y, static estimator)")
    ax.set_xlabel("Particle size (nm), by pixel calibration (mpp used in water/hydrogel)")
    return fig


# ── Plot: trajectory overlay, recentered to each track's own starting point ────

def _grid(n: int) -> tuple[plt.Figure, np.ndarray]:
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.15 / 3, nrows * 5.00 / 2),
                              constrained_layout=True, squeeze=False)
    ax_flat = axes.flatten()
    for ax in ax_flat[n:]:
        ax.set_visible(False)
    return fig, ax_flat


def plot_trajectories_by_size(
    by_group: dict[tuple[float, float], list[pd.DataFrame]], qualifying: list[tuple[float, float]],
    dls_labels: dict[float, int],
) -> plt.Figure:
    fig, ax_flat = _grid(len(qualifying))
    for ax, (size_nm, mpp) in zip(ax_flat, qualifying):
        for traj_df in by_group.get((size_nm, mpp), []):
            for _, g in traj_df.groupby("particle"):
                g = g.sort_values("frame")
                x_nm = g["x"].to_numpy(dtype=float) * mpp * 1000.0
                y_nm = g["y"].to_numpy(dtype=float) * mpp * 1000.0
                # Recentered to this track's own starting point (frame 0), not
                # its mean position -- delta from where it started, not spread
                # around its average.
                ax.plot(x_nm - x_nm[0], y_nm - y_nm[0], color=COLOR_TRK,
                        linewidth=0.6, alpha=0.6, rasterized=True)

        ax.set_xlim(-TRAJECTORY_WINDOW_NM, TRAJECTORY_WINDOW_NM)
        ax.set_ylim(-TRAJECTORY_WINDOW_NM, TRAJECTORY_WINDOW_NM)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\Delta x$ (nm)")
        ax.set_ylabel(r"$\Delta y$ (nm)")
        ax.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm (mpp={mpp:.3g})", fontsize=9)

    fig.suptitle("Trajectory overlay, immobilized particles (from each track's own start)",
                 fontsize=11, fontweight="semibold")
    return fig


# ── Plot: loc_error / step_size vs particle size ────────────────────────────────

def plot_locerror_vs_stepsize_ratio(
    df: pd.DataFrame, qualifying: list[tuple[float, float]], step_size_nm: dict[float, float],
    dls_labels: dict[float, int],
) -> plt.Figure:
    """Boxplot, not a single point per size: every qualifying immobilized
    track's own sigma_xy_static_nm divided by that particle size's median
    step-size reference (water+hydrogel pooled, see
    _load_median_stepsize_nm()) -- one box per size (the immobilized
    sigma_loc data itself has no water/hydrogel split, and since the
    reference step size is now pooled rather than per-condition, a
    second per-condition box would just duplicate the first)."""
    sizes = sorted({s for s, _ in qualifying})
    mpp_by_size: dict[float, list[float]] = {}
    for s, m in qualifying:
        mpp_by_size.setdefault(s, []).append(m)

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    positions = np.arange(len(sizes), dtype=float)

    boxes = []
    for i, s in enumerate(sizes):
        step = step_size_nm.get(s)
        if step is None or not np.isfinite(step) or step <= 0:
            continue
        sub = df.loc[(df["particle_size_nm"] == s) & df["mpp"].isin(mpp_by_size[s]), "sigma_xy_static_nm"]
        ratios = (sub / step).dropna().to_numpy()
        if ratios.size:
            boxes.append((positions[i], ratios))
    if boxes:
        _boxplot_by_group(ax, [r for _, r in boxes], np.array([p for p, _ in boxes]), COLOR_RATIO, COLOR_RATIO, 0.5)

    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{dls_labels.get(s, int(s))} nm" for s in sizes])
    ax.set_xlabel("Particle size (nm)")
    ax.set_ylabel(r"$\sigma_{loc}$ / step size (dimensionless)")
    ax.legend(handles=[
        plt.Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", label=r"$\sigma_{loc}$ = step size"),
    ], loc="upper right", frameon=False)
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    dls_labels = get_dls_labels()

    all_sigma_rows: list[dict] = []
    trajectory_by_group: dict[tuple[float, float], list[pd.DataFrame]] = {}

    for dataset_label, folder in IMMOB_FOLDERS.items():
        if not folder.exists():
            print(f"  [SKIP] Ordner nicht gefunden: {folder}")
            continue
        datasets = _build_immob_datasets(folder)
        print(f"{dataset_label}: {len(datasets)} Dateien geladen aus {folder}")
        for result in datasets.values():
            if result is None:
                continue
            mpp, fps, base_name = result.get("mpp"), result.get("fps"), result.get("base_name")
            xml_path, tif_path = result.get("xml_path"), result.get("tif_path")

            # Main sigma_loc pipeline: clean_movie_tracks()'s full cleaning,
            # unchanged (edge-artifact removal/splitting still applies here).
            tracks_df, particle_size_nm, _ = clean_movie_tracks(result)
            if tracks_df is not None:
                all_sigma_rows.extend(_sigma_rows(tracks_df, mpp, fps, base_name, particle_size_nm, dataset_label))

            # Trajectory overlay: separate, minimal-manipulation source (see
            # module docstring) -- independent of whether the sigma_loc
            # pipeline above found any usable tracks for this movie.
            if mpp is not None and xml_path is not None:
                raw_traj = _load_raw_trajectory_data(Path(xml_path), tif_path, base_name)
                if raw_traj is not None and not raw_traj.empty:
                    raw_traj, _ = _drop_tracks_exceeding_window(raw_traj, mpp, TRAJECTORY_WINDOW_NM, base_name)
                if raw_traj is not None and not raw_traj.empty:
                    traj_df = raw_traj[["particle", "frame", "x", "y"]].copy()
                    key = (float(particle_size_nm), round(float(mpp), 3))
                    trajectory_by_group.setdefault(key, []).append(traj_df)

    df = pd.DataFrame(all_sigma_rows)
    if not df.empty:
        df["mpp"] = df["mpp"].round(3)
        df = df.sort_values(["particle_size_nm", "mpp"]).reset_index(drop=True)
    print(f"\nGesamt: {len(df)} Tracks mit gültigem sigma_loc (>= {MIN_POINTS_FOR_SIGMA} Punkte).")
    if df.empty:
        print("Keine Daten -- Abbruch.")
        return

    summary = df.groupby(["particle_size_nm", "mpp", "dataset_label"]).agg(
        n_tracks=("particle_id", "count"),
        sigma_x_mean_nm=("sigma_x_static_nm", "mean"), sigma_x_std_nm=("sigma_x_static_nm", "std"),
        sigma_y_mean_nm=("sigma_y_static_nm", "mean"), sigma_y_std_nm=("sigma_y_static_nm", "std"),
        sigma_xy_mean_nm=("sigma_xy_static_nm", "mean"), sigma_xy_std_nm=("sigma_xy_static_nm", "std"),
    ).reset_index()
    print(summary.to_string(index=False))

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(df, f)
    print(f"\nCache gespeichert: {CACHE_FILE}")

    print("\nScanning water/hydrogel reference data (used mpp)...")
    ref_mpp = _scan_qualifying_mpp()
    step_size_nm = _load_median_stepsize_nm()
    qualifying = _qualifying_groups(df, ref_mpp)
    dropped = len(df[["particle_size_nm", "mpp"]].drop_duplicates()) - len(qualifying)
    print(f"  {len(qualifying)} (size, mpp) groups qualify for the figures "
          f"({dropped} dropped: mpp not used in water/hydrogel for that size)")

    with plt.rc_context(_RC):
        figs = {
            "sigma_loc_combined_by_size.png": plot_combined_sigma_by_size(df, qualifying, dls_labels),
            "trajectories_overlay_by_size.png": plot_trajectories_by_size(
                trajectory_by_group, qualifying, dls_labels),
            "locerror_vs_stepsize_ratio.png": plot_locerror_vs_stepsize_ratio(
                df, qualifying, step_size_nm, dls_labels),
        }
        for filename, fig in figs.items():
            plt.figure(fig.number)
            plt.show()
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / filename
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")

    csv_path = SAVE_PATH / "sigma_loc_summary_by_size_mpp.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
