"""
Consistency check between the official SPT pipeline's track export and a
from-scratch raw TrackMate re-analysis, run once per (Tracks_X, Analysis_X)
folder pair listed in DATASETS below. Currently covers the 2026.01.19 water
movies at 50 nm (Tracks_50 vs. Analysis_50) and 1000 nm (Tracks_1000 vs.
Analysis_1000); add further entries to DATASETS to cover more sizes/folders
without touching the analysis logic.

Compute stage (reads raw XML + the shared .rec calibration file per movie;
no cache pickle -- a handful of movies per dataset, cheap to recompute
every run):

  extract_tracks_and_parameters(session_xml_path, mpp, fps, ...) is the
  general-purpose function this script was built around: given ANY complete
  raw TrackMate session XML (Model/AllSpots/AllTracks/FilteredTracks +
  Settings/ImageData -- the full Fiji project export, not the simplified
  "<Tracks><particle><detection>" schema read by core.io.read_trackmate_xml),
  it returns a result_dict-shaped dict (tracks_df, mpp, fps, num_tracks,
  D_MSD, fit_results_MSD, ...) directly comparable to what
  core.io.single_file_data() + core.analysis.perform_msd_analysis() produce
  for the official "*_Tracks.xml" exports. It is not specific to any one
  particle size, condition, or folder pair.

  It reuses, unmodified: core.io.remove_edge_artifacts() (border filtering,
  same 3% margin, applied automatically for the official pipeline too),
  core.analysis.perform_msd_analysis()/DEFAULT_MSD_FIT_POINTS (same
  ensemble iMSD/eMSD power-law fit used everywhere else in this repo), and
  Validation_CPT/trackmate_session.py::read_trackmate_session() (already
  written raw-session XML reader: walks AllTracks' Edge list per
  FilteredTracks track ID to reconstruct ordered per-particle trajectories;
  a track's surviving split/merge branches, if any, each become their own
  "particle").

  IMPORTANT: the raw session XML's own Settings/ImageData pixelwidth/
  timeinterval are Fiji PLACEHOLDER values (1.0/1.0) in every dataset
  checked so far -- not real calibration. mpp/fps are therefore taken from
  the same .rec file already resolved for that movie by the official
  pipeline (core.io.find_rec_tif_files() via single_file_data()), so both
  sides of every comparison share one calibration and any difference in
  D/n reflects the detection/tracking itself, not a unit mismatch. Frame
  size (for edge-artifact filtering) is read per-file from each session's
  own ImageData instead, since it can differ between movies in the same
  folder (confirmed: 1000_nm_10.tif is 400x300 px, every other 1000 nm
  movie is 200x150 px).

File matching: match_tracks_to_analysis() connects each Tracks_X file to
its Analysis_X counterpart by the exact source .tif FILENAME -- the
official pipeline's own resolved tif_path vs. the raw session's own
Settings/ImageData filename="..." attribute -- rather than fuzzy stem
matching. The 50 nm folder pair's filenames otherwise differ only by
spacing (e.g. "50 nm water 02_Tracks.xml" vs. "50nmwater02.xml"), the same
issue TrackMate_Settings_Inventory.py's normalized fallback was built for,
but here the raw session states its source TIFF explicitly so an exact
match is possible without normalization. A Tracks_X file is left unmatched
(reported, never guessed) when it has zero or more than one candidate raw
analysis, or when its own calibration can't be resolved at all -- e.g.
"1000_nm_10_Tracks_v2.xml" has no ".rec" under that exact cleaned stem
(core.io.find_rec_tif_files() does not strip "_v2") and is reported as
such rather than silently matched to the same raw analysis as
"1000_nm_10_Tracks.xml". A matched raw analysis with zero surviving tracks
(read_trackmate_session() raises ValueError -- confirmed genuine for
1000_nm_03.xml: both AllTracks and FilteredTracks are empty, not a parsing
issue) is reported as [SKIP], not silently dropped from the printed log.

Also worth stating plainly: every Analysis_X XML found so far (all
modified 2026-09-18) records DETECTOR_NAME="DOG_DETECTOR" -- a different
detector from what produced the original Tracks_X exports, whose own
settings XML was not found for either folder pair (see
TrackMate_Settings_Inventory.py's missing-settings report: "no candidate
settings file/folder found" for 2026.01.19\\Tracks_50 and \\Tracks_1000).
This script is therefore a genuine detector/pipeline consistency check,
not an identity check of the same analysis run twice.

Outputs (Auswertungsbilder\\Tracks_vs_RawAnalysis_<label>\\, one subfolder
per DATASETS entry):
  tracks_vs_raw_analysis_comparison.csv   one row per matched movie: track
                                           count, median/mean track length,
                                           D, n for both pipelines
  num_tracks_official_vs_raw.png          identity scatter, N tracks
  track_length_median_official_vs_raw.png identity scatter, median length
  D_official_vs_raw.png                   identity scatter, D0 (µm²/s)
  n_official_vs_raw.png                   identity scatter, anomalous n
  track_length_distribution.png           pooled track-length histograms
                                           (official vs raw), density

Shows each figure first (plt.show(), blocking); only after the window is
closed does it save.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydro_analysis.core.io import single_file_data, find_rec_tif_files, remove_edge_artifacts
from hydro_analysis.core.analysis import perform_msd_analysis, DEFAULT_MSD_FIT_POINTS
from hydro_analysis.MSD_Trackmate.Validation_GPT.trackmate_session import read_trackmate_session
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC

# ── Configuration ──────────────────────────────────────────────────────────────
MOVIE_ROOT = Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19")

DATASETS: list[dict[str, Any]] = [
    {"label": "50nmWater", "tracks_folder": MOVIE_ROOT / "Tracks_50", "analysis_folder": MOVIE_ROOT / "Analysis_50"},
    {"label": "1000nmWater", "tracks_folder": MOVIE_ROOT / "Tracks_1000", "analysis_folder": MOVIE_ROOT / "Analysis_1000"},
]

SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)

IMAGE_DATA_TAIL_BYTES = 200_000

COLOR_OFFICIAL, COLOR_OFFICIAL_DARK = "#3B6E8C", "#2A4F66"
COLOR_RAW, COLOR_RAW_DARK = "#da0000", "#990000"

TRACK_LENGTH_X_MIN, TRACK_LENGTH_X_MAX = 0.0, 200.0
TRACK_LENGTH_N_BINS = 30

_IMAGE_DATA_TAG_RE = re.compile(r"<ImageData\s[^>]*/>")


def _attr(tag: str, name: str) -> str | None:
    match = re.search(rf'{name}="([^"]*)"', tag)
    return match.group(1) if match else None


def _read_image_data(xml_path: Path, tail_bytes: int = IMAGE_DATA_TAIL_BYTES) -> dict[str, Any]:
    """Tail-read a raw TrackMate session XML for its own Settings/ImageData
    tag (source filename, frame width/height). ImageData sits within the
    last few KB of the file, right before </TrackMate> -- same tail-read
    principle TrackMate_Settings_Inventory.py uses for the <Settings> block
    in general, just with a much smaller window since ImageData is the very
    last element."""
    size = xml_path.stat().st_size
    with open(xml_path, "rb") as f:
        if size > tail_bytes:
            f.seek(-tail_bytes, 2)
        text = f.read().decode("utf-8", errors="ignore")
    match = _IMAGE_DATA_TAG_RE.search(text)
    if match is None:
        raise ValueError(f"{xml_path.name}: no <ImageData> tag found in the last {tail_bytes} bytes.")
    tag = match.group(0)
    width = _attr(tag, "width")
    height = _attr(tag, "height")
    return {
        "filename": _attr(tag, "filename"),
        "width_px": int(width) if width else None,
        "height_px": int(height) if height else None,
    }


def extract_tracks_and_parameters(
    session_xml_path: Path,
    mpp: float,
    fps: float,
    particle_size_nm: float | None = None,
    fit_points: int = DEFAULT_MSD_FIT_POINTS,
) -> dict[str, Any]:
    """General-purpose: read one complete raw TrackMate session XML and
    return a result_dict-shaped dict (calibration must be supplied by the
    caller -- see module docstring). See module docstring for the full
    contract."""
    session = read_trackmate_session(session_xml_path)
    image_data = _read_image_data(session_xml_path)

    traj = session.trajectories.copy()
    traj["particle"] = pd.factorize(traj["trajectory_id"])[0] + 1
    tracks_df = traj[["frame", "particle", "x", "y"]].sort_values(["particle", "frame"]).reset_index(drop=True)

    edge_stats = {"n_removed": 0, "n_splits": 0}
    if not tracks_df.empty and image_data["width_px"] and image_data["height_px"]:
        tracks_df, edge_stats = remove_edge_artifacts(tracks_df, image_data["width_px"], image_data["height_px"])
        if edge_stats["n_removed"]:
            print(f"     [EDGE FILTER] {session_xml_path.name}: removed {edge_stats['n_removed']} "
                  f"detections near border ({image_data['width_px']}x{image_data['height_px']} px), "
                  f"+{edge_stats['n_splits']} tracks from splits")

    result: dict[str, Any] = {
        "xml_path": str(session_xml_path),
        "base_name": session_xml_path.name,
        "raw_tif_filename": image_data["filename"],
        "tracks_df": tracks_df,
        "mpp": mpp,
        "fps": fps,
        "particle_size_nm": particle_size_nm,
        "num_tracks": int(tracks_df["particle"].nunique()) if not tracks_df.empty else 0,
        "num_frames": int(tracks_df["frame"].max() + 1) if not tracks_df.empty else 0,
        "edge_filter_removed": edge_stats["n_removed"],
        "edge_filter_splits": edge_stats["n_splits"],
        "D_MSD": None,
        "fit_results_MSD": None,
    }
    perform_msd_analysis(result, fit_points=fit_points)
    return result


def match_tracks_to_analysis(
    tracks_folder: Path, analysis_folder: Path,
) -> tuple[list[tuple[Path, Path, str]], list[tuple[Path, str | None, int]], list[Path]]:
    """Connect each *_Tracks.xml to its raw Analysis session XML by the
    exact source .tif filename. See module docstring for why this is exact
    rather than normalized matching. Returns (matched, unmatched_tracks,
    unmatched_analysis); see module docstring for their shapes. Never
    guesses between multiple candidates -- an ambiguous, absent, or
    uncalibrated match is reported in unmatched_tracks instead."""
    tracks_files = sorted(tracks_folder.glob("*.xml"))
    analysis_files = sorted(analysis_folder.glob("*.xml"))

    analysis_by_tif: dict[str, list[Path]] = {}
    for a in analysis_files:
        info = _read_image_data(a)
        if info["filename"]:
            analysis_by_tif.setdefault(info["filename"], []).append(a)

    matched: list[tuple[Path, Path, str]] = []
    unmatched_tracks: list[tuple[Path, str | None, int]] = []
    used: set[Path] = set()
    for t in tracks_files:
        calib = find_rec_tif_files(t)
        tif_name = calib["tiff_file"].name if calib["tiff_file"] else None
        candidates = analysis_by_tif.get(tif_name, []) if tif_name else []
        if len(candidates) == 1:
            matched.append((t, candidates[0], tif_name))
            used.add(candidates[0])
        else:
            unmatched_tracks.append((t, tif_name, len(candidates)))

    unmatched_analysis = [a for a in analysis_files if a not in used]
    return matched, unmatched_tracks, unmatched_analysis


def _unmatched_note(tif_name: str | None, n_candidates: int) -> str:
    if tif_name is None:
        return "own .tif/.rec calibration could not be resolved"
    if n_candidates == 0:
        return "no raw analysis for this movie"
    return f"{n_candidates} ambiguous raw analysis candidates"


def _track_lengths(result: dict[str, Any]) -> np.ndarray:
    tracks_df = result.get("tracks_df")
    if tracks_df is None or tracks_df.empty:
        return np.array([])
    return tracks_df.groupby("particle").size().to_numpy()


def _comparison_row(movie_label: str, official: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    off_len = _track_lengths(official)
    raw_len = _track_lengths(raw)
    off_fit = official.get("fit_results_MSD") or {}
    raw_fit = raw.get("fit_results_MSD") or {}
    return {
        "movie": movie_label,
        "num_tracks_official": official["num_tracks"],
        "num_tracks_raw": raw["num_tracks"],
        "track_length_median_official": float(np.median(off_len)) if off_len.size else np.nan,
        "track_length_median_raw": float(np.median(raw_len)) if raw_len.size else np.nan,
        "track_length_mean_official": float(np.mean(off_len)) if off_len.size else np.nan,
        "track_length_mean_raw": float(np.mean(raw_len)) if raw_len.size else np.nan,
        "D_official_um2_s": off_fit.get("D_um2_per_s", np.nan),
        "D_raw_um2_s": raw_fit.get("D_um2_per_s", np.nan),
        "n_official": off_fit.get("exponent", np.nan),
        "n_raw": raw_fit.get("exponent", np.nan),
    }


def _identity_scatter_figure(df: pd.DataFrame, xcol: str, ycol: str, labelcol: str,
                              xlabel: str, ylabel: str) -> plt.Figure:
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    labels = df[labelcol].to_numpy()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, labels = x[finite], y[finite], labels[finite]

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    if x.size:
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        pad = 0.06 * (hi - lo) if hi > lo else max(abs(lo), 1.0) * 0.1
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=1.2, zorder=1)
        ax.scatter(x, y, s=26, facecolor=COLOR_RAW, edgecolor=COLOR_RAW_DARK, linewidth=0.8, zorder=3)
        for xi, yi, label in zip(x, y, labels):
            ax.annotate(str(label), (xi, yi), fontsize=6, xytext=(3, 3),
                        textcoords="offset points", color="#555555")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        if x.size >= 2:
            r = float(np.corrcoef(x, y)[0, 1])
            ax.text(0.03, 0.97, f"Pearson r = {r:.3f}, N = {x.size}", transform=ax.transAxes,
                    fontsize=8, va="top", ha="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_track_length_histograms(lengths_official: np.ndarray, lengths_raw: np.ndarray,
                                  tracks_label: str, analysis_label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    bins = np.linspace(TRACK_LENGTH_X_MIN, TRACK_LENGTH_X_MAX, TRACK_LENGTH_N_BINS + 1)
    ax.hist(lengths_official, bins=bins, density=True, histtype="step", linewidth=1.8,
            color=COLOR_OFFICIAL_DARK, label=f"{tracks_label}, official (N={lengths_official.size})")
    ax.hist(lengths_raw, bins=bins, density=True, histtype="step", linewidth=1.8,
            color=COLOR_RAW_DARK, label=f"{analysis_label}, raw (N={lengths_raw.size})")
    ax.set_xlim(TRACK_LENGTH_X_MIN, TRACK_LENGTH_X_MAX)
    ax.set_xlabel("Track length (frames)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    return fig


def _show_and_save(fig: plt.Figure, save_path: Path, filename: str) -> None:
    plt.show()
    save_path.mkdir(parents=True, exist_ok=True)
    path = save_path / filename
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Plot gespeichert: {path}")


def run_comparison(label: str, tracks_folder: Path, analysis_folder: Path) -> pd.DataFrame | None:
    print(f"\n=== {label}: {tracks_folder.name} vs. {analysis_folder.name} ===")
    save_path = SAVE_PATH_BASE / f"Tracks_vs_RawAnalysis_{label}"

    matched, unmatched_tracks, unmatched_analysis = match_tracks_to_analysis(tracks_folder, analysis_folder)
    print(f"{len(matched)} matched movie pairs, {len(unmatched_tracks)} {tracks_folder.name} file(s) unmatched, "
          f"{len(unmatched_analysis)} {analysis_folder.name} file(s) without a {tracks_folder.name} counterpart")
    for t, tif_name, n_cand in unmatched_tracks:
        print(f"  [UNMATCHED] {t.name} (tif={tif_name}): {_unmatched_note(tif_name, n_cand)}")
    for a in unmatched_analysis:
        print(f"  [UNMATCHED] {a.name}: no {tracks_folder.name} file references this raw analysis")

    rows: list[dict[str, Any]] = []
    lengths_official_all: list[np.ndarray] = []
    lengths_raw_all: list[np.ndarray] = []
    for tracks_xml, analysis_xml, tif_name in matched:
        official = single_file_data(tracks_xml)
        if official is None:
            print(f"  [SKIP] {tracks_xml.name}: official pipeline could not load calibration")
            continue
        perform_msd_analysis(official)
        try:
            raw = extract_tracks_and_parameters(
                analysis_xml, mpp=official["mpp"], fps=official["fps"],
                particle_size_nm=official["particle_size_nm"],
            )
        except ValueError as exc:
            print(f"  [SKIP] {analysis_xml.name}: raw session unusable ({exc})")
            continue
        movie_label = tracks_xml.stem.replace("_Tracks", "")
        rows.append(_comparison_row(movie_label, official, raw))
        lengths_official_all.append(_track_lengths(official))
        lengths_raw_all.append(_track_lengths(raw))

        off_fit = official.get("fit_results_MSD")
        raw_fit = raw.get("fit_results_MSD")
        if off_fit and raw_fit:
            d_str = f"D {off_fit['D_um2_per_s']:.4g} -> {raw_fit['D_um2_per_s']:.4g} um2/s"
        else:
            d_str = "D fit unavailable"
        print(f"  [OK] {movie_label} (tif={tif_name}): tracks {official['num_tracks']} -> {raw['num_tracks']}, {d_str}")

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  [SKIP DATASET] {label}: no matched movie pairs produced usable results.")
        return None

    print(df.to_string(index=False))

    lengths_official = np.concatenate(lengths_official_all) if lengths_official_all else np.array([])
    lengths_raw = np.concatenate(lengths_raw_all) if lengths_raw_all else np.array([])

    with plt.rc_context(_RC):
        fig1 = _identity_scatter_figure(
            df, "num_tracks_official", "num_tracks_raw", "movie",
            f"Number of tracks ({tracks_folder.name}, official)", f"Number of tracks ({analysis_folder.name}, raw)")
        _show_and_save(fig1, save_path, "num_tracks_official_vs_raw.png")

        fig2 = _identity_scatter_figure(
            df, "track_length_median_official", "track_length_median_raw", "movie",
            "Median track length, official (frames)", "Median track length, raw (frames)")
        _show_and_save(fig2, save_path, "track_length_median_official_vs_raw.png")

        fig3 = _identity_scatter_figure(
            df, "D_official_um2_s", "D_raw_um2_s", "movie",
            "D, official (µm²/s)", "D, raw (µm²/s)")
        _show_and_save(fig3, save_path, "D_official_vs_raw.png")

        fig4 = _identity_scatter_figure(
            df, "n_official", "n_raw", "movie",
            "Anomalous exponent n, official", "Anomalous exponent n, raw")
        _show_and_save(fig4, save_path, "n_official_vs_raw.png")

        fig5 = plot_track_length_histograms(lengths_official, lengths_raw, tracks_folder.name, analysis_folder.name)
        _show_and_save(fig5, save_path, "track_length_distribution.png")

    save_path.mkdir(parents=True, exist_ok=True)
    csv_path = save_path / "tracks_vs_raw_analysis_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Tabelle gespeichert: {csv_path}")
    return df


def main() -> None:
    for dataset in DATASETS:
        run_comparison(dataset["label"], dataset["tracks_folder"], dataset["analysis_folder"])


if __name__ == "__main__":
    main()
