"""
Apparent particle size (2D Gaussian FWHM) and SNR for every tracked
detection in water, hydrogel and immobilized samples -- raw image and
DoG-filtered image.

Compute stage. Tracks are exactly the detections that survived
core.io.single_file_data() -> remove_edge_artifacts(); nothing further is
eliminated:
  water        cache/msd_d0_results.pkl   (written by MSD_FromTrackmate_D0.py)
  hydrogel     cache/msd_20mg_files.pkl   (written by MSD_FromTrackmate_20mg.py)
  immobilized  every *_Tracks.xml of Loc_Error_Analyse_immob_particle
               .IMMOB_DATASETS, loaded here with single_file_data() (no
               MSD cache exists for them); "1k" names corrected to 1000 nm
Each movie is used once (deduplicate): variants of the same movie in one
folder ("_processed" TIFFs, "Resultof"/"filtered_" exports) keep the
shortest TIFF name, then the shortest Tracks name; copies of a movie in
another folder (same name and TIFF size) are dropped. Dropped files are
listed in the "duplicates" table.

Per movie, the TIFF (result_dict["tif_path"]) and the raw TrackMate
session XML are read. Several sessions can exist per movie (e.g. Analysis and
Analysis_new for 50 nm hydrogel, where the tracks come from Tracks_new);
every candidate is read and the one whose spots coincide with the most track
detections is used (choose_session). Files without any session XML are
skipped and listed in "failures" -- their detection radius would otherwise
have to be guessed.

Per detection:
  1. Matched to its session spot (same frame, nearest within MATCH_TOL_PX;
     match_distance_px records the offset, 0 for the exact run),
     which provides TrackMate's own spot features: QUALITY, RADIUS,
     SNR_CH1, CONTRAST_CH1 and the intensity statistics. TrackMate's
     SNR_CH1 is (mean inside the spot radius - mean of the ring out to twice
     the radius) / std of that ring.
  2. 2D Gaussian fit on the raw frame with PSF_Analysis.fit_frame_psf()
     (unchanged): crop half width max(CROP_HALF_PX, 3 R), start width and
     upper bound (2 R) from the TrackMate detection radius R ("fitting pixel
     size"), centre bounded to +-min(1 px, R/2) around the XML position,
     neighbouring AllSpots masked with 2 R. Stored: amplitude (height),
     background, sigma_x/y and FWHM_x/y = 2 sqrt(2 ln 2) sigma in nm, R^2,
     fit status. Overlapping detections (centre distance < R_i + R_j) are
     not fitted and marked "overlapping_particles".
  3. Annulus SNR on the raw frame as in snr_signal_profile.py:
     (peak within R - mean of 1.5 R..3 R annulus) / std of that annulus,
     other spots masked. Also amplitude / annulus std from the fit.
  4. Steps 2-3 again on the DoG-filtered frame (snr_signal_profile.dog_filter,
     TrackMate/ImgLib2 sigma convention, session detector radius), columns
     prefixed "dog_". The DoG is a band-pass: the fitted DoG width is set
     largely by the filter scale and the negative ring, so dog_fwhm is NOT a
     particle size -- it is kept for comparison only. dog_snr describes the
     detectability as seen by a DoG detector.
  Blurry frames (Frame_Sharpness_Immobilized, MAD threshold) are flagged in
  "is_blurry_frame", not removed.

Modes (--mode):
  frames10  per file NUM_TEST_FRAMES evenly spaced frames, all detections in
            them -- used for the runtime estimate
  first     each track once, at its first detection ("new particles")
  all       every detection of every track; immobilized movies only every
            IMMOBILIZED_FRAME_STEP-th frame (stationary particle = repeated
            measurement of the same object)

Fit acceptance, both stored per detection:
  status == "accepted"   strict, fit_frame_psf(): R^2 >= 0.5, relative sigma
                         error <= 0.5, not at a bound
  accepted_relaxed       strict OR a "poor_fit" whose relative sigma error is
                         <= RELAXED_MAX_REL_SIGMA_ERROR and whose amplitude /
                         annulus std >= RELAXED_MIN_SNR_FIT -- for weak spots
                         (hydrogel) R^2 stays low from pixel noise alone
All sizes are stored in nm (FWHM, FWHM error, detection diameter 2 R);
pixel columns are kept only as the raw fit parameters.

Output (pickle, dict of DataFrames), cache/particle_size_snr.pkl for
--mode all, cache/particle_size_snr_<mode>.pkl otherwise:
  meta        parameters, mode, runtime, source caches
  files       one row per movie: paths, mpp, fps, detector settings, counts,
              runtime per stage
  detections  one row per detection: identifiers, TrackMate features, raw
              and DoG fit results, SNR values
  tracks      one row per track: medians over the strictly accepted
              (fwhm_nm, ...) and the relaxed (fwhm_relaxed_nm, ...) fits,
              detection_diameter_nm
  failures    skipped files with reason
  duplicates  movies left out as duplicates, with the file kept instead
Consumers: SNR_Analysis.py, Figures_Refined/SNR_Refined.py,
Figures_Refined/PSF_Refined.py.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from hydro_analysis.core.io import single_file_data
from hydro_analysis.MSD_Trackmate.Validation_Claude.Frame_Sharpness_Immobilized import (
    compute_sharpness_series, flag_blurry_frames,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.PSF_Analysis import fit_frame_psf
from hydro_analysis.MSD_Trackmate.Validation_Claude.Loc_Error_Analyse_immob_particle import (
    IMMOB_DATASETS, _corrected_particle_size,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.snr_signal_profile import dog_filter, _norm

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
SOURCE_CACHES = {
    "water": CACHE_DIR / "msd_d0_results.pkl",
    "hydrogel": CACHE_DIR / "msd_20mg_files.pkl",
}
OUTPUT_STEM = "particle_size_snr"
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))   # one movie per process

NUM_TEST_FRAMES = 10
MATCH_EXACT_PX = 0.05        # same TrackMate run: identical float positions
MATCH_TOL_PX = 1.0          # same particle from a slightly different TrackMate run of the movie
CROP_HALF_PX = 8             # as PSF_Analysis.CROP_HALF_PX
NEIGHBOUR_MASK_FACTOR = 2.0  # neighbouring spots masked out to 2 R, as in PSF_Analysis
ANNULUS_INNER = 1.5          # annulus in units of R, as in snr_signal_profile
ANNULUS_OUTER = 3.0
MIN_ANNULUS_PIXELS = 10
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
IMMOBILIZED_FRAME_STEP = 10  # immobilized: only every 10th frame -- the same particle in every
                             # frame is a repeated measurement; keeps the full run short
# Mild fit criterion ("accepted_relaxed"), in addition to fit_frame_psf's strict one (R^2 >= 0.5):
# for weak, noise-dominated spots R^2 stays low even when the width is well determined, so a
# "poor_fit" also counts if the width is precise and the peak clearly above the background noise.
RELAXED_MAX_REL_SIGMA_ERROR = 0.25
RELAXED_MIN_SNR_FIT = 3.0
TRACKMATE_FEATURES = ("QUALITY", "RADIUS", "SNR_CH1", "CONTRAST_CH1", "MEAN_INTENSITY_CH1",
                      "MEDIAN_INTENSITY_CH1", "MAX_INTENSITY_CH1", "MIN_INTENSITY_CH1",
                      "STD_INTENSITY_CH1", "TOTAL_INTENSITY_CH1")


# ============================================================================
# Session XML
# ============================================================================

def read_session(path: Path) -> tuple[pd.DataFrame, dict]:
    """All spots of a TrackMate session (positions in px) and its detector
    settings. Positions are converted with ImageData pixelwidth only when the
    model is stored in physical units."""
    root = ET.parse(path).getroot()
    image = root.find("Settings/ImageData")
    detector = root.find("Settings/DetectorSettings")
    model = root.find("Model")
    units = (model.get("spatialunits", "pixel") if model is not None else "pixel").lower()
    pw = float(image.get("pixelwidth", 1.0)) if image is not None else 1.0
    scale = 1.0 if "pixel" in units else 1.0 / pw
    settings = {f"detector_{k.lower()}": v for k, v in (detector.attrib.items() if detector is not None else [])}
    default_radius = float(settings.get("detector_radius", "nan")) * scale

    rows = []
    for spot in root.iterfind("Model/AllSpots/SpotsInFrame/Spot"):
        a = spot.attrib
        row = {"spot_id": int(a["ID"]), "frame": int(float(a["FRAME"])),
               "x_px": float(a["POSITION_X"]) * scale, "y_px": float(a["POSITION_Y"]) * scale,
               "visibility": int(float(a.get("VISIBILITY", 1)))}
        for key in TRACKMATE_FEATURES:
            row[f"tm_{key.lower()}"] = float(a[key]) if key in a else np.nan
        rows.append(row)
    spots = pd.DataFrame(rows)
    if not spots.empty:
        spots["tm_radius"] = spots["tm_radius"] * scale
        spots["radius_px"] = spots["tm_radius"].where(spots["tm_radius"] > 0, default_radius)
    settings.update(session_units=units, detector_radius_px=default_radius, session_xml=str(path))
    return spots, settings


def _nearest_spots(det: pd.DataFrame, spots: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(spot_id, distance_px) of the nearest session spot in the same frame."""
    by_frame = {f: (g["x_px"].to_numpy(), g["y_px"].to_numpy(), g["spot_id"].to_numpy())
                for f, g in spots.groupby("frame")}
    ids = np.full(len(det), -1, dtype=np.int64)
    dist = np.full(len(det), np.inf)
    for i, (f, x, y) in enumerate(det[["frame", "x", "y"]].itertuples(index=False)):
        g = by_frame.get(int(f))
        if g is None:
            continue
        d2 = (g[0] - x) ** 2 + (g[1] - y) ** 2
        j = int(np.argmin(d2))
        ids[i], dist[i] = int(g[2][j]), float(np.sqrt(d2[j]))
    return ids, dist


def match_detections(det: pd.DataFrame, spots: pd.DataFrame) -> pd.DataFrame:
    """Attach the nearest session spot (same frame, <= MATCH_TOL_PX). The
    measurement itself always uses the track position; the session spot only
    supplies TrackMate's features and radius. match_distance_px > 0 means the
    session comes from a slightly different TrackMate run of the same movie."""
    ids, dist = _nearest_spots(det, spots)
    ids[dist > MATCH_TOL_PX] = -1
    det = det.assign(spot_id=ids, match_distance_px=np.where(ids >= 0, dist, np.nan))
    return det.merge(spots.drop(columns=["frame"]), on="spot_id", how="left")


def _session_candidates(tracks_xml: Path, tif_path: Path) -> list[Path]:
    """All session XMLs of this movie: *.xml in every sub folder whose name
    contains "analys" and in the movie folder itself, whose normalised stem
    equals the Tracks stem (with and without a filtered_/Resultof prefix)."""
    stem = re.sub(r"_tracks$", "", tracks_xml.stem, flags=re.IGNORECASE)
    targets = {_norm(stem), _norm(re.sub(r"^(filtered_|resultof)", "", stem, flags=re.IGNORECASE)),
               _norm(tif_path.stem)}
    movie_dir = tracks_xml.parent.parent
    folders = [movie_dir] + [p for p in movie_dir.iterdir() if p.is_dir() and "analys" in p.name.lower()]
    return [p for folder in folders for p in folder.glob("*.xml")
            if _norm(p.stem) in targets and "tracks" not in p.stem.lower()]


def choose_session(tracks_xml: Path, tif_path: Path, det: pd.DataFrame) -> tuple:
    """Session with the most exactly matching detections (then the most
    within MATCH_TOL_PX). Returns (spots, settings, stats) or None."""
    best = None
    for cand in _session_candidates(tracks_xml, tif_path):
        try:
            spots, settings = read_session(cand)
        except (ET.ParseError, KeyError, ValueError):
            continue
        if spots.empty:
            continue
        _, dist = _nearest_spots(det, spots)
        stats = {"session_candidates": 0, "session_exact_fraction": float(np.mean(dist <= MATCH_EXACT_PX)),
                 "session_match_fraction": float(np.mean(dist <= MATCH_TOL_PX))}
        key = (stats["session_exact_fraction"], stats["session_match_fraction"])
        if best is None or key > best[0]:
            best = (key, spots, settings, stats)
    if best is None:
        return None
    best[3]["session_candidates"] = len(_session_candidates(tracks_xml, tif_path))
    return best[1], best[2], best[3]


# ============================================================================
# Per-detection measurements
# ============================================================================

def _annulus_snr(img: np.ndarray, x: float, y: float, r: float, others: np.ndarray) -> dict:
    h, w = img.shape
    half = int(np.ceil(ANNULUS_OUTER * r)) + 1
    xi, yi = int(round(x)), int(round(y))
    x_lo, x_hi, y_lo, y_hi = max(xi - half, 0), min(xi + half + 1, w), max(yi - half, 0), min(yi + half + 1, h)
    patch = img[y_lo:y_hi, x_lo:x_hi]
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    dist = np.hypot(xx - x, yy - y)
    ring = (dist >= ANNULUS_INNER * r) & (dist <= ANNULUS_OUTER * r)
    for ox, oy, orad in others:
        ring &= np.hypot(xx - ox, yy - oy) > ANNULUS_INNER * orad
    inner = dist <= r
    if ring.sum() < MIN_ANNULUS_PIXELS or not inner.any():
        return {"peak": np.nan, "bg_mean": np.nan, "bg_std": np.nan, "snr": np.nan}
    bg = patch[ring]
    peak = float(patch[inner].max())
    std = float(bg.std(ddof=1))
    return {"peak": peak, "bg_mean": float(bg.mean()), "bg_std": std,
            "snr": (peak - float(bg.mean())) / std if std > 0 else np.nan}


def _fit(img: np.ndarray, x: float, y: float, r: float, others: np.ndarray, mpp: float) -> dict:
    half = max(CROP_HALF_PX, int(np.ceil(3 * r)))
    xi, yi = int(round(x)), int(round(y))
    if xi - half < 0 or yi - half < 0 or xi + half >= img.shape[1] or yi + half >= img.shape[0]:
        return {"status": "image_border"}
    left, top = xi - half, yi - half
    crop = img[top:yi + half + 1, left:xi + half + 1]
    yy, xx = np.indices(crop.shape)
    mask = np.ones(crop.shape, dtype=bool)
    for ox, oy, orad in others:
        mask &= np.hypot(xx + left - ox, yy + top - oy) > NEIGHBOUR_MASK_FACTOR * orad
    fit = fit_frame_psf(crop, x - left, y - top, r, r, mask)
    out = {k: fit.get(k, np.nan) for k in ("amplitude", "background", "sigma_x_px", "sigma_y_px",
                                              "sigma_x_error_px", "sigma_y_error_px", "r2")}
    out["status"] = fit["status"]
    out["fwhm_x_nm"] = FWHM_FACTOR * out["sigma_x_px"] * mpp * 1000.0
    out["fwhm_y_nm"] = FWHM_FACTOR * out["sigma_y_px"] * mpp * 1000.0
    out["fwhm_nm"] = 0.5 * (out["fwhm_x_nm"] + out["fwhm_y_nm"])
    out["fwhm_error_nm"] = 0.5 * FWHM_FACTOR * mpp * 1000.0 * np.hypot(out["sigma_x_error_px"], out["sigma_y_error_px"])
    out["rel_sigma_error"] = np.nanmax([out["sigma_x_error_px"] / out["sigma_x_px"],
                                        out["sigma_y_error_px"] / out["sigma_y_px"]])         if np.isfinite(out["sigma_x_px"]) else np.nan
    return out


def measure_frame(raw: np.ndarray, dog: np.ndarray, det: pd.DataFrame, frame_spots: pd.DataFrame,
                  mpp: float) -> list[dict]:
    all_xyr = frame_spots[["x_px", "y_px", "radius_px", "spot_id"]].to_numpy(dtype=float)
    rows = []
    for d in det.itertuples(index=False):
        x, y, r = float(d.x), float(d.y), float(d.radius_px)
        others = all_xyr[all_xyr[:, 3] != d.spot_id][:, :3]
        near = others[np.hypot(others[:, 0] - x, others[:, 1] - y) < 2 * (ANNULUS_OUTER * r + others[:, 2])]
        overlapping = bool(np.any(np.hypot(near[:, 0] - x, near[:, 1] - y) < r + near[:, 2]))
        row = {"spot_id": d.spot_id, "frame": d.frame, "particle": d.particle, "overlapping": overlapping}
        for prefix, img in (("", raw), ("dog_", dog)):
            snr = _annulus_snr(img, x, y, r, near)
            fit = {"status": "overlapping_particles"} if overlapping else _fit(img, x, y, r, near, mpp)
            row.update({f"{prefix}{k}": v for k, v in fit.items()})
            row.update({f"{prefix}{k}": v for k, v in snr.items()})
            amp = fit.get("amplitude", np.nan)
            row[f"{prefix}snr_fit"] = amp / snr["bg_std"] if np.isfinite(amp) and snr["bg_std"] > 0 else np.nan
            row[f"{prefix}accepted_relaxed"] = fit["status"] == "accepted" or (
                fit["status"] == "poor_fit"
                and fit.get("rel_sigma_error", np.inf) <= RELAXED_MAX_REL_SIGMA_ERROR
                and row[f"{prefix}snr_fit"] >= RELAXED_MIN_SNR_FIT)
        rows.append(row)
    return rows


# ============================================================================
# Per-file driver
# ============================================================================

def select_detections(tracks_df: pd.DataFrame, mode: str, condition: str) -> pd.DataFrame:
    if mode == "all":
        if condition == "immobilized":
            return tracks_df[tracks_df["frame"] % IMMOBILIZED_FRAME_STEP == 0]
        return tracks_df
    if mode == "first":
        return tracks_df.sort_values("frame").groupby("particle", as_index=False).head(1)
    frames = np.unique(tracks_df["frame"].to_numpy())
    pick = frames[np.linspace(0, len(frames) - 1, min(NUM_TEST_FRAMES, len(frames))).astype(int)]
    return tracks_df[tracks_df["frame"].isin(pick)]


def process_file(job: dict) -> dict:
    try:
        return _process_file(job)
    except Exception as exc:  # one broken movie must not abort a multi-hour run
        return {"failure": {**job["meta"], "reason": f"{type(exc).__name__}: {exc}"}}


def _process_file(job: dict) -> dict:
    t0 = time.perf_counter()
    rd_meta, tracks_df, mode = job["meta"], job["tracks_df"], job["mode"]
    tracks_xml, tif_path = Path(rd_meta["xml_path"]), Path(rd_meta["tif_path"])
    if not tif_path.exists():
        return {"failure": {**rd_meta, "reason": "TIFF not found"}}
    lengths = tracks_df.groupby("particle").size().rename("track_length")
    det = select_detections(tracks_df, mode, rd_meta["condition"]).merge(lengths, left_on="particle", right_index=True)
    chosen = choose_session(tracks_xml, tif_path, det)
    if chosen is None:
        return {"failure": {**rd_meta, "reason": "no session XML"}}
    spots, settings, session_stats = chosen
    session_xml = Path(settings["session_xml"])
    det = match_detections(det, spots)
    n_unmatched = int((det["spot_id"] < 0).sum())
    det = det[det["spot_id"] >= 0]
    if det.empty:
        return {"failure": {**rd_meta, "reason": f"no detection matched to session spots ({n_unmatched} unmatched, "
                                                 f"session {session_xml.name})"}}
    t_session = time.perf_counter()

    try:
        is_blurry, _ = flag_blurry_frames(compute_sharpness_series(tif_path))
    except (IndexError, ValueError):
        is_blurry = np.zeros(0, dtype=bool)   # corrupt TIFF: no sharpness flag instead of losing the file
    t_sharp = time.perf_counter()

    radius_px = float(settings["detector_radius_px"])
    spots_by_frame = {f: g for f, g in spots.groupby("frame")}
    rows, t_load, t_dog, t_fit = [], 0.0, 0.0, 0.0
    n_unreadable = 0
    with tifffile.TiffFile(tif_path) as tif:
        n_pages = len(tif.pages)
        for frame, fdet in det.groupby("frame"):
            a = time.perf_counter()
            try:
                raw = tif.pages[int(frame)].asarray().astype(np.float64)
            except (IndexError, ValueError, AttributeError):
                # truncated/corrupt TIFFs: pages past the damage are not readable
                raw = None
            if raw is None or raw.ndim != 2:
                n_unreadable += len(fdet)
                rows.extend({"spot_id": s, "frame": frame, "particle": p, "status": "frame_unreadable",
                             "dog_status": "frame_unreadable"}
                            for s, p in fdet[["spot_id", "particle"]].itertuples(index=False))
                continue
            b = time.perf_counter()
            dog = dog_filter(raw, (radius_px, radius_px))[0]
            c = time.perf_counter()
            rows.extend(measure_frame(raw, dog, fdet, spots_by_frame[int(frame)], rd_meta["mpp"]))
            t_load, t_dog, t_fit = t_load + b - a, t_dog + c - b, t_fit + time.perf_counter() - c

    meas = pd.DataFrame(rows)
    out = det.merge(meas, on=["spot_id", "frame", "particle"], how="left")
    for key in ("condition", "particle_size_nm", "file", "mpp", "fps"):
        out[key] = rd_meta[key]
    out["is_blurry_frame"] = [bool(is_blurry[f]) if f < len(is_blurry) else False for f in out["frame"]]

    file_row = {**rd_meta, **settings, **session_stats, "mode": mode, "n_tracks": int(tracks_df["particle"].nunique()),
                "n_detections_total": len(tracks_df), "n_detections_measured": len(out),
                "n_unmatched": n_unmatched, "n_unreadable": n_unreadable, "n_tiff_pages": n_pages,
                "n_frames_measured": int(out["frame"].nunique()),
                "t_session_s": t_session - t0, "t_sharpness_s": t_sharp - t_session, "t_load_s": t_load,
                "t_dog_s": t_dog, "t_fit_s": t_fit, "t_total_s": time.perf_counter() - t0}
    return {"file": file_row, "detections": out}


def build_track_table(det: pd.DataFrame) -> pd.DataFrame:
    ok = det[det["status"] == "accepted"]
    keys = ["condition", "particle_size_nm", "file", "particle"]
    base = det.groupby(keys).agg(n_measured=("frame", "size"), track_length=("track_length", "first"),
                                 radius_px=("radius_px", "median"), mpp=("mpp", "first"),
                                 fps=("fps", "first"),
                                 tm_snr=("tm_snr_ch1", "median"), tm_quality=("tm_quality", "median"),
                                 snr=("snr", "median"), dog_snr=("dog_snr", "median"))
    fits = ok.groupby(keys).agg(n_fit_accepted=("fwhm_nm", "size"), fwhm_nm=("fwhm_nm", "median"),
                                fwhm_x_nm=("fwhm_x_nm", "median"), fwhm_y_nm=("fwhm_y_nm", "median"),
                                amplitude=("amplitude", "median"), snr_fit=("snr_fit", "median"))
    relaxed = det[det["accepted_relaxed"].fillna(False).astype(bool)].groupby(keys).agg(
        n_fit_relaxed=("fwhm_nm", "size"), fwhm_relaxed_nm=("fwhm_nm", "median"),
        amplitude_relaxed=("amplitude", "median"))
    dog_ok = det[det["dog_status"] == "accepted"]
    dog = dog_ok.groupby(keys).agg(dog_fwhm_nm=("dog_fwhm_nm", "median"), dog_amplitude=("dog_amplitude", "median"))
    out = base.join(fits).join(relaxed).join(dog).reset_index()
    out["detection_diameter_nm"] = 2.0 * out["radius_px"] * out["mpp"] * 1000.0
    return out


def _immobilized_result_dicts() -> list[dict]:
    """Immobilized movies are not in any MSD cache: load every *_Tracks.xml
    of IMMOB_DATASETS with core.io.single_file_data() (same edge-artifact
    filtering as the water/hydrogel caches, nothing else removed)."""
    rds = []
    for folders in IMMOB_DATASETS.values():
        for xml in sorted(Path(folders["tracks_folder"]).glob("*_Tracks.xml")):
            rd = single_file_data(xml)
            if rd is not None:
                rd["particle_size_nm"] = _corrected_particle_size(rd.get("base_name", xml.name), rd.get("particle_size_nm"))
                rds.append(rd)
    return rds


def deduplicate(jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    """One job per movie. (1) Same folder, same movie: variants such as
    "_processed" TIFFs or "Resultof"/"filtered_" track exports -- keep the
    shortest TIFF name, then the shortest Tracks name. (2) The same movie
    copied to another folder (same normalised name and TIFF file size) --
    keep the first by path."""
    dropped = []

    def keep_best(jobs, key, order):
        groups: dict = {}
        for j in jobs:
            groups.setdefault(key(j), []).append(j)
        kept = []
        for group in groups.values():
            group.sort(key=order)
            kept.append(group[0])
            dropped.extend({"condition": j["meta"]["condition"], "file": j["meta"]["file"], "xml_path": j["meta"]["xml_path"],
                            "tif_path": j["meta"]["tif_path"], "kept_xml_path": group[0]["meta"]["xml_path"]}
                           for j in group[1:])
        return kept

    def movie_stem(j):
        return _norm(re.sub(r"_processed$", "", Path(j["meta"]["tif_path"]).stem, flags=re.IGNORECASE))

    jobs = keep_best(jobs, lambda j: (j["meta"]["condition"], str(Path(j["meta"]["tif_path"]).parent).lower(), movie_stem(j)),
                     lambda j: (len(Path(j["meta"]["tif_path"]).name), len(Path(j["meta"]["xml_path"]).name)))
    size = lambda j: Path(j["meta"]["tif_path"]).stat().st_size if Path(j["meta"]["tif_path"]).exists() else -1
    jobs = keep_best(jobs, lambda j: (j["meta"]["condition"], movie_stem(j), size(j)), lambda j: j["meta"]["xml_path"])
    return jobs, dropped


def load_jobs(mode: str, only: str | None, max_files: int | None) -> tuple[list[dict], list[dict]]:
    sources: list[tuple[str, dict]] = []
    for condition, path in SOURCE_CACHES.items():
        if not path.exists():
            raise FileNotFoundError(f"Nicht gefunden: {path}\nBitte zuerst das zugehoerige MSD_FromTrackmate-Skript ausfuehren.")
        with open(path, "rb") as f:
            sources += [(condition, rd) for rd in pickle.load(f).values()]
    sources += [("immobilized", rd) for rd in _immobilized_result_dicts()]

    jobs = []
    for condition, rd in sources:
        td = rd.get("tracks_df")
        if td is None or td.empty or not rd.get("tif_path"):
            continue
        meta = dict(condition=condition, particle_size_nm=rd.get("particle_size_nm"), file=rd.get("base_name"),
                    xml_path=str(rd["xml_path"]), tif_path=str(rd["tif_path"]),
                    mpp=float(rd["mpp"]), fps=float(rd["fps"]))
        if only and only.lower() not in f"{meta['file']} {meta['xml_path']}".lower():
            continue
        jobs.append({"meta": meta, "tracks_df": td[["frame", "particle", "x", "y"]].copy(), "mode": mode})
    jobs, dropped = deduplicate(jobs)
    return (jobs[:max_files] if max_files else jobs), dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("frames10", "first", "all"), default="all")
    parser.add_argument("--only", help="nur Dateien, deren Name/Pfad diesen Text enthaelt")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallele Prozesse (je Datei einer)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    jobs, duplicates = load_jobs(args.mode, args.only, args.max_files)
    print(f"{len(jobs)} Dateien ({len(duplicates)} doppelte Filme ausgelassen), Modus {args.mode}, "
          f"{args.workers} Prozess(e)")
    for d in duplicates:
        print(f"  [DOPPELT] {Path(d['xml_path']).name} -> behalten: {Path(d['kept_xml_path']).name}")
    t0 = time.perf_counter()
    files, detections, failures = [], [], []
    runner = ProcessPoolExecutor(args.workers).map if args.workers > 1 else map
    for res in runner(process_file, jobs):
        if "failure" in res:
            failures.append(res["failure"])
            print(f"  [SKIP] {res['failure']['file']}: {res['failure']['reason']}")
            continue
        fr = res["file"]
        files.append(fr)
        detections.append(res["detections"])
        print(f"  {fr['condition']:<8} {fr['particle_size_nm']:>6.0f} nm  {fr['file']:<45} "
              f"{fr['n_detections_measured']:>6} Det.  {fr['t_total_s']:6.1f} s "
              f"(Schaerfe {fr['t_sharpness_s']:.1f}, Fits {fr['t_fit_s']:.1f})")

    det = pd.concat(detections, ignore_index=True) if detections else pd.DataFrame()
    result = {
        "meta": dict(mode=args.mode, created=pd.Timestamp.now().isoformat(timespec="seconds"),
                     runtime_s=time.perf_counter() - t0, workers=args.workers,
                     source_caches={k: str(v) for k, v in SOURCE_CACHES.items()},
                     num_test_frames=NUM_TEST_FRAMES, crop_half_px=CROP_HALF_PX,
                     annulus=(ANNULUS_INNER, ANNULUS_OUTER), neighbour_mask_factor=NEIGHBOUR_MASK_FACTOR,
                     fwhm_factor=FWHM_FACTOR, match_tol_px=MATCH_TOL_PX,
                     immobilized_frame_step=IMMOBILIZED_FRAME_STEP,
                     relaxed_max_rel_sigma_error=RELAXED_MAX_REL_SIGMA_ERROR,
                     relaxed_min_snr_fit=RELAXED_MIN_SNR_FIT),
        "files": pd.DataFrame(files),
        "detections": det,
        "tracks": build_track_table(det) if not det.empty else pd.DataFrame(),
        "failures": pd.DataFrame(failures),
        "duplicates": pd.DataFrame(duplicates),
    }
    out = args.output or CACHE_DIR / (f"{OUTPUT_STEM}.pkl" if args.mode == "all" else f"{OUTPUT_STEM}_{args.mode}.pkl")
    with open(out, "wb") as f:
        pickle.dump(result, f)
    print(f"Gespeichert: {out}  ({len(det)} Detektionen, {result['meta']['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
