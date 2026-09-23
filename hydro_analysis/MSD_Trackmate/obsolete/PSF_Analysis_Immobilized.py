"""
Per-frame 2D Gaussian fits of immobilized particles; never average raw frames.
Movies, particles and frames are randomly ordered without replacement using
RANDOM_SEED. Stop after MAX_FITS_PER_GROUP optimizer calls per (size, mpp),
shared across all movies/datasets. Failed or quality-rejected optimizer calls
count; pre-fit exclusions do not. Exhausted groups may have fewer calls.
Prefer complete Analysis_immob_NEW sessions, falling back explicitly to the
matched Analysis session if missing or unusable. No headless run is required.
Positions, per-spot RADIUS (detector RADIUS fallback), and calibration of XML
coordinates all come from the SAME session; physical nm calibration uses .rec.

Each sharp frame is cropped around its own XML position. Initial Gaussian
sigma = XML radius / 2 is a configurable starting heuristic, not an equality
between detection radius and optical sigma. Bounded centers, neighbor masking,
and rejection of overlapping XML spots reduce neighbor capture. Unresolved
aggregates cannot be identified with certainty from a single Gaussian alone.

Reject nonconverged, boundary-limited, uncertain and poor fits. Before averaging,
remove per-track frame-width outliers using MAD; then remove unusually broad
track means within each movie as possible aggregates. All exclusions are logged.
The arithmetic mean of accepted per-frame sigmas is the per-track PSF estimate;
frame-to-frame standard deviations are exported alongside it. No drift correction
is applied to image coordinates: each frame uses its own tracked position.

Outputs: psf_fits.csv (accepted track means), psf_frame_fits.csv (frame fits and
rejection reasons), psf_detection_settings.csv (actual session provenance),
psf_vs_size.png (600 dpi). No shared analysis caches are changed.

The 575 nm reference is confirmed only for the smallest bead; large particles
include physical particle extent as well as the optical PSF.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import curve_fit, OptimizeWarning
from tqdm.auto import tqdm
from hydro_analysis.MSD_Trackmate.Validation_Claude.Frame_Sharpness_Immobilized import (
    compute_sharpness_series, flag_blurry_frames,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Loc_Error_Analyse_immob_particle import (
    _find_xml_in_subfolder, NEW_ANALYSIS_SUBFOLDER,
)

from hydro_analysis.core.physics import calculate_theoretical_psf_sigma
from hydro_analysis.MSD_Trackmate.Validation_GPT.trackmate_session import read_trackmate_session
from hydro_analysis.MSD_Trackmate.Validation_Claude.TaskA_Immobilized_Compute_Headless import (
    IMMOB_DATASETS, _corrected_particle_size,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Compare_Tracks_vs_RawAnalysis import (
    match_tracks_to_analysis, _read_image_data,
)
from hydro_analysis.core.io import find_rec_tif_files, extract_particle_size_from_path
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "PSF_Analysis_Immobilized"

NUMERICAL_APERTURE = 1.2
WAVELENGTH_NM = 575.0   # confirmed only for the smallest (Nile Red) bead -- see module docstring
DIFFRACTION_LIMITED_SIZES_NM = (20.0,)   # sizes where the point-source comparison is physically meaningful

MAX_FITS_PER_GROUP = 4_000
RANDOM_SEED = 42   # reproducible sampling; use None for a different draw each run
MAX_TRACKS_PER_MOVIE = None   # optional preview limit, applied after shuffling
MAX_FRAMES_PER_TRACK = None   # optional preview limit, applied after shuffling
CROP_HALF_PX = 8   # minimum half-window; enlarged for large XML radii
INITIAL_SIGMA_RADIUS_FACTOR = 0.5   # starting heuristic only
MAX_CENTER_SHIFT_PX = 1.0
MIN_SIGMA_PX = 0.3
MIN_FIT_R2 = 0.5
MAX_RELATIVE_SIGMA_ERROR = 0.5
MIN_ACCEPTED_FRAMES = 3
OUTLIER_MAD_K = 3.5
OUTLIER_RELATIVE_FLOOR = 0.10   # prevents zero/tiny MAD rejecting numerical noise

COLOR_EMPIRICAL = "#3B6E8C"
COLOR_THEORY = "#da0000"


def _gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = coords
    return offset + amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))


def fit_frame_psf(crop, x0, y0, radius_x, radius_y, mask=None):
    """Bounded single-frame fit initialized at the subpixel XML position."""
    yy, xx = np.indices(crop.shape)
    valid = np.isfinite(crop) if mask is None else mask & np.isfinite(crop)
    if valid.sum() < 20:
        return {"status": "insufficient_pixels"}
    z = crop[valid].astype(float)
    background = float(np.percentile(z, 20))
    center = (xx - x0)**2 + (yy - y0)**2 <= max(radius_x, radius_y)**2
    signal = crop[valid & center]
    amplitude = float(np.max(signal) - background) if signal.size else 0.0
    if amplitude <= 0:
        return {"status": "no_signal"}
    shift_x = min(MAX_CENTER_SHIFT_PX, radius_x / 2)
    shift_y = min(MAX_CENTER_SHIFT_PX, radius_y / 2)
    max_sx, max_sy = 2 * radius_x, 2 * radius_y
    if min(max_sx, max_sy) <= MIN_SIGMA_PX:
        return {"status": "invalid_radius"}
    lower = [0, x0-shift_x, y0-shift_y, MIN_SIGMA_PX, MIN_SIGMA_PX, -np.inf]
    upper = [np.inf, x0+shift_x, y0+shift_y, max_sx, max_sy, np.inf]
    sx0 = np.clip(radius_x * INITIAL_SIGMA_RADIUS_FACTOR, MIN_SIGMA_PX*1.01, max_sx*.99)
    sy0 = np.clip(radius_y * INITIAL_SIGMA_RADIUS_FACTOR, MIN_SIGMA_PX*1.01, max_sy*.99)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, pcov = curve_fit(_gaussian_2d, (xx[valid], yy[valid]), z,
                                  p0=[amplitude, x0, y0, sx0, sy0, background],
                                  bounds=(lower, upper), maxfev=3000)
    except (RuntimeError, ValueError, OptimizeWarning, FloatingPointError):
        return {"status": "fit_failed", "fit_attempted": True}
    prediction = _gaussian_2d((xx[valid], yy[valid]), *popt)
    residual = float(np.sum((z-prediction)**2))
    total = float(np.sum((z-z.mean())**2))
    r2 = 1-residual/total if total > 0 else -np.inf
    errors = np.sqrt(np.maximum(np.diag(pcov), 0))
    result = dict(status="accepted", fit_attempted=True, amplitude=popt[0], x0_px=popt[1], y0_px=popt[2],
                  sigma_x_px=popt[3], sigma_y_px=popt[4], background=popt[5],
                  sigma_x_error_px=errors[3], sigma_y_error_px=errors[4], r2=r2,
                  rmse=np.sqrt(residual/len(z)), initial_sigma_x_px=sx0, initial_sigma_y_px=sy0)
    if not np.all(np.isfinite(popt)) or not np.all(np.isfinite(errors)):
        result["status"] = "uncertain_fit"
    elif any(min(popt[i]-lower[i], upper[i]-popt[i]) < .01*(upper[i]-lower[i]) for i in (1,2,3,4)):
        result["status"] = "fit_at_bound"
    elif r2 < MIN_FIT_R2:
        result["status"] = "poor_fit"
    elif max(errors[3]/popt[3], errors[4]/popt[4]) > MAX_RELATIVE_SIGMA_ERROR:
        result["status"] = "uncertain_fit"
    return result


def width_outliers(values, upper_only=False):
    """MAD rejection per axis with a relative floor; need >=5 estimates."""
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return np.zeros(len(values), dtype=bool)
    med = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values-med), axis=0)
    limit = np.maximum(OUTLIER_MAD_K*scale, OUTLIER_RELATIVE_FLOOR*np.abs(med))
    delta = values-med if upper_only else np.abs(values-med)
    return np.any(delta > limit, axis=1)


def select_session(tracks_xml, analysis_xml):
    preferred = _find_xml_in_subfolder(tracks_xml, NEW_ANALYSIS_SUBFOLDER)
    expected_image = _read_image_data(analysis_xml)["filename"]
    for candidate in dict.fromkeys(p for p in (preferred, analysis_xml) if p is not None):
        try:
            if _read_image_data(candidate)["filename"] != expected_image:
                raise ValueError("session references a different source TIFF")
            session = read_trackmate_session(candidate)
            settings = read_detector_values(candidate)
            print(f"  [SESSION] {tracks_xml.name}: {candidate}")
            return session, settings
        except (ValueError, ET.ParseError, KeyError) as exc:
            print(f"  [SESSION SKIP] {candidate.name}: {exc}")
    return None, None


def read_detector_values(xml_path: Path) -> dict[str, Any]:
    """Read configured detection values, not fitted per-spot PSF estimates."""
    root = ET.parse(xml_path).getroot()
    image = root.find("Settings/ImageData")
    detector = root.find("Settings/DetectorSettings")
    if image is None or detector is None:
        raise ValueError(f"{xml_path.name}: ImageData/DetectorSettings missing")
    radius = float(detector.attrib["RADIUS"])
    pixelwidth = float(image.attrib["pixelwidth"])
    pixelheight = float(image.attrib["pixelheight"])
    if not all(np.isfinite(v) and v > 0 for v in (radius, pixelwidth, pixelheight)):
        raise ValueError(f"{xml_path.name}: invalid radius or pixel calibration")
    model = root.find("Model")
    return {
        "analysis_xml": str(xml_path),
        "xml_spatialunits": model.get("spatialunits", "") if model is not None else "",
        **{f"detector_{key.lower()}": value for key, value in detector.attrib.items()},
        "detector_radius_x_px": radius / pixelwidth,
        "detector_radius_y_px": radius / pixelheight,
        "detector_diameter_x_px": 2 * radius / pixelwidth,
        "detector_diameter_y_px": 2 * radius / pixelheight,
    }


def process_movie(dataset_label, tracks_xml, analysis_xml, frame_rows=None, detector_rows=None,
                  fit_counts=None, rng=None):
    frame_rows = [] if frame_rows is None else frame_rows
    fit_counts = {} if fit_counts is None else fit_counts
    rng = np.random.default_rng(RANDOM_SEED) if rng is None else rng
    particle_size_nm = _corrected_particle_size(tracks_xml.name, extract_particle_size_from_path(tracks_xml))
    calib = find_rec_tif_files(tracks_xml)
    tif_path, mpp = calib["tiff_file"], calib["mpp"]
    if tif_path is None or mpp is None or not np.isfinite(mpp) or mpp <= 0:
        print(f"  [SKIP] {tracks_xml.name}: calibration missing")
        return []
    if particle_size_nm is None or not np.isfinite(particle_size_nm):
        print(f"  [SKIP] {tracks_xml.name}: particle size missing")
        return []
    group = (float(particle_size_nm), float(mpp))
    fit_counts.setdefault(group, 0)
    if fit_counts[group] >= MAX_FITS_PER_GROUP:
        print(f"  [BUDGET] {tracks_xml.name}: {group} already has {fit_counts[group]} fits")
        return []
    session, detector_values = select_session(tracks_xml, analysis_xml)
    if session is None:
        return []
    common = dict(dataset_label=dataset_label, movie=tracks_xml.name,
                  particle_size_nm=particle_size_nm, mpp=mpp)
    if detector_rows is not None:
        detector_rows.append({**common, **detector_values})
    blurry, _ = flag_blurry_frames(compute_sharpness_series(tif_path))
    spots = session.spots.copy()
    spots["x_px"] = spots["x"] / session.pixelwidth_um
    spots["y_px"] = spots["y"] / session.pixelheight_um
    radii = pd.to_numeric(spots.get("RADIUS", pd.Series(np.nan, index=spots.index)), errors="coerce")
    radii = radii.where(np.isfinite(radii) & (radii > 0), float(detector_values["detector_radius"]))
    spots["rx"] = radii/session.pixelwidth_um
    spots["ry"] = radii/session.pixelheight_um
    indexed = spots.set_index("spot_id")
    by_frame = {int(f): g for f, g in spots.groupby("frame")}
    tracks = list(session.trajectories.groupby("trajectory_id"))
    tracks = [tracks[i] for i in rng.permutation(len(tracks))]
    if MAX_TRACKS_PER_MOVIE is not None and len(tracks) > MAX_TRACKS_PER_MOVIE:
        tracks = tracks[:MAX_TRACKS_PER_MOVIE]
    rows = []
    seen_spots = set()
    with tqdm(total=MAX_FITS_PER_GROUP, initial=fit_counts[group],
              desc=f"PSF {particle_size_nm:g} nm / {mpp:g} um/px",
              unit="fit", dynamic_ncols=True) as progress:
        for traj_id, g in tracks:
            if fit_counts[group] >= MAX_FITS_PER_GROUP:
                break
            g = g.sort_values("frame").drop_duplicates("frame")
            g = g.iloc[rng.permutation(len(g))]
            if MAX_FRAMES_PER_TRACK is not None and len(g) > MAX_FRAMES_PER_TRACK:
                g = g.iloc[:MAX_FRAMES_PER_TRACK]
            progress.set_postfix(movie=tracks_xml.stem, track=traj_id, refresh=False)
            records = []
            for detection in tqdm(g.itertuples(), total=len(g), desc=f"Track {traj_id}",
                                  unit="frame", leave=False, dynamic_ncols=True):
                if fit_counts[group] >= MAX_FITS_PER_GROUP:
                    break
                if detection.spot_id in seen_spots:
                    continue
                seen_spots.add(detection.spot_id)
                f = int(detection.frame)
                spot = indexed.loc[detection.spot_id]
                x, y, rx, ry = (float(spot[k]) for k in ("x_px", "y_px", "rx", "ry"))
                rec = {**common, "session_xml": str(session.path), "trajectory_id": traj_id,
                       "spot_id": detection.spot_id, "frame": f, "xml_x_px": x, "xml_y_px": y,
                       "xml_radius_x_px": rx, "xml_radius_y_px": ry, "status": "accepted",
                       "fit_attempted": False}
                records.append(rec)
                if f < 0 or f >= len(blurry):
                    rec["status"] = "invalid_frame"
                    continue
                if blurry[f]:
                    rec["status"] = "blurry_frame"
                    continue
                neighbors = by_frame[f].loc[by_frame[f]["spot_id"] != detection.spot_id]
                # Use all XML spots, including those excluded from the final tracks.
                separation = ((neighbors["x_px"]-x)/(neighbors["rx"]+rx))**2 + ((neighbors["y_px"]-y)/(neighbors["ry"]+ry))**2
                if (separation < 1).any():
                    rec["status"] = "overlapping_particles"
                    continue
                half = max(CROP_HALF_PX, int(np.ceil(3*max(rx, ry))))
                xi, yi = int(round(x)), int(round(y))
                img = tifffile.imread(tif_path, key=f)
                if img.ndim != 2:
                    raise ValueError(f"{tif_path}: expected a 2D grayscale frame, got {img.shape}")
                if xi-half < 0 or yi-half < 0 or xi+half >= img.shape[1] or yi+half >= img.shape[0]:
                    rec["status"] = "image_border"
                    continue
                left, top = xi-half, yi-half
                crop = img[top:yi+half+1, left:xi+half+1].astype(float)
                yy, xx = np.indices(crop.shape)
                mask = np.ones(crop.shape, dtype=bool)
                nearby = neighbors.loc[(abs(neighbors["x_px"]-x) < 2*half+2*neighbors["rx"]) &
                                       (abs(neighbors["y_px"]-y) < 2*half+2*neighbors["ry"])]
                for n in nearby.itertuples():
                    mask &= ((xx+left-n.x_px)/(2*n.rx))**2 + ((yy+top-n.y_px)/(2*n.ry))**2 > 1
                fit = fit_frame_psf(crop, x-left, y-top, rx, ry, mask)
                rec.update(fit)
                if fit.get("fit_attempted", False):
                    fit_counts[group] += 1
                    progress.update(1)
                if "x0_px" in fit:
                    rec["x0_px"] += left
                    rec["y0_px"] += top
            valid = [r for r in records if r["status"] == "accepted"]
            outliers = width_outliers([[r["sigma_x_px"], r["sigma_y_px"]] for r in valid])
            for r, rejected in zip(valid, outliers):
                if rejected:
                    r["status"] = "frame_width_outlier"
            valid = [r for r in valid if r["status"] == "accepted"]
            if len(valid) < MIN_ACCEPTED_FRAMES:
                for r in valid:
                    r["status"] = "too_few_valid_frames"
            else:
                values = np.array([[r["sigma_x_px"], r["sigma_y_px"]] for r in valid])
                sx, sy = values.mean(axis=0)
                dx, dy = values.std(axis=0, ddof=1)
                rows.append({**common, **detector_values, "trajectory_id": traj_id, "mpp": mpp,
                             "n_frames_sampled": len(records), "n_frames_fitted": len(valid),
                             "sigma_x_psf_px": sx, "sigma_y_psf_px": sy,
                             "sigma_x_std_px": dx, "sigma_y_std_px": dy,
                             "amplitude_mean": np.mean([r["amplitude"] for r in valid]),
                             "background_mean": np.mean([r["background"] for r in valid]),
                             "r2_mean": np.mean([r["r2"] for r in valid]),
                             "sigma_x_psf_nm": sx*mpp*1000, "sigma_y_psf_nm": sy*mpp*1000})
            frame_rows.extend(records)
    aggregate_mask = width_outliers([[r["sigma_x_psf_px"], r["sigma_y_psf_px"]] for r in rows], upper_only=True)
    rejected_ids = {r["trajectory_id"] for r, bad in zip(rows, aggregate_mask) if bad}
    for rec in frame_rows:
        if rec["dataset_label"] == dataset_label and rec["movie"] == tracks_xml.name and rec["trajectory_id"] in rejected_ids and rec["status"] == "accepted":
            rec["status"] = "broad_track_outlier"
    rows = [r for r in rows if r["trajectory_id"] not in rejected_ids]
    counts = pd.Series([r["status"] for r in frame_rows if r["dataset_label"] == dataset_label and r["movie"] == tracks_xml.name]).value_counts().to_dict()
    print(f"  [OK] {tracks_xml.name}: {len(rows)}/{len(tracks)} tracks; "
          f"group fits {fit_counts[group]}/{MAX_FITS_PER_GROUP}; frame status: {counts}")
    return rows


def plot_psf_vs_size(df: pd.DataFrame, detector_df: pd.DataFrame | None = None) -> plt.Figure:
    if detector_df is None:
        detector_df = df.drop_duplicates(["dataset_label", "movie"])
    sizes = sorted(set(df["particle_size_nm"].dropna()) | set(detector_df["particle_size_nm"].dropna()))
    data = [df.loc[df["particle_size_nm"] == s, ["sigma_x_psf_px", "sigma_y_psf_px"]].to_numpy().ravel()
            for s in sizes]
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    positions = np.arange(len(sizes))
    measured = [(position, values) for position, values in zip(positions, data) if len(values)]
    if measured:
        bp = ax.boxplot([values for _, values in measured],
                        positions=[position for position, _ in measured],
                        widths=0.5, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLOR_EMPIRICAL)
            patch.set_alpha(0.6)

    movies = df.drop_duplicates(["dataset_label", "movie"])
    theory_sigma = calculate_theoretical_psf_sigma(WAVELENGTH_NM, NUMERICAL_APERTURE)
    for i, size in enumerate(sizes):
        selected = detector_df.loc[detector_df["particle_size_nm"] == size]
        radii = selected[["detector_radius_x_px", "detector_radius_y_px"]].to_numpy().ravel()
        ax.scatter(np.full(len(radii), i), radii, marker="s", s=26, color="#ff6800", alpha=0.7,
                   label="XML detection radius (x/y)" if i == 0 else None)
        theory = theory_sigma / (movies.loc[movies["particle_size_nm"] == size, "mpp"].to_numpy() * 1000)
        ax.scatter(np.full(len(theory), i), theory, marker="_", s=100, color=COLOR_THEORY,
                   label="Point-source reference sigma (575 nm, NA=1.2)" if i == 0 else None)

    labels = [f"{s:.0f} nm" + ("*" if s in DIFFRACTION_LIMITED_SIZES_NM else "") for s in sizes]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Nominal particle size (* = point-source comparison valid)")
    ax.set_ylabel("PSF sigma / configured detection radius (pixels)")
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend([Patch(facecolor=COLOR_EMPIRICAL, alpha=0.6)] + handles,
              ["Mean per-frame PSF sigma (x/y)"] + legend_labels,
              loc="upper left", fontsize=8, frameon=False)
    fig.supxlabel("Detection radius is not PSF sigma; wavelength confirmed only for smallest beads.", fontsize=8)
    return fig


def main() -> None:
    all_rows: list[dict] = []
    detector_rows: list[dict] = []
    frame_rows: list[dict] = []
    fit_counts: dict[tuple[float, float], int] = {}
    rng = np.random.default_rng(RANDOM_SEED)
    jobs = []
    for dataset_label, folders in IMMOB_DATASETS.items():
        print(f"\n=== {dataset_label} ===")
        matched, unmatched_tracks, unmatched_analysis = match_tracks_to_analysis(
            folders["tracks_folder"], folders["analysis_folder"])
        print(f"{len(matched)} matched movies, {len(unmatched_tracks)} unmatched track files, "
              f"{len(unmatched_analysis)} unmatched analysis files")
        for tracks_xml, tif_name, n_candidates in unmatched_tracks:
            print(f"  [UNMATCHED] {tracks_xml.name}: TIFF={tif_name}, analysis candidates={n_candidates}")
        for analysis_xml in unmatched_analysis:
            print(f"  [UNMATCHED] {analysis_xml.name}: no uniquely matched track export")
        for tracks_xml, analysis_xml, _ in matched:
            jobs.append((dataset_label, tracks_xml, analysis_xml))

    for i in rng.permutation(len(jobs)):
        dataset_label, tracks_xml, analysis_xml = jobs[i]
        all_rows.extend(process_movie(dataset_label, tracks_xml, analysis_xml, frame_rows,
                                      detector_rows, fit_counts, rng))

    detector_df = pd.DataFrame(detector_rows)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    detector_df.to_csv(SAVE_PATH / "psf_detection_settings.csv", index=False)
    pd.DataFrame(frame_rows).to_csv(SAVE_PATH / "psf_frame_fits.csv", index=False)
    budget_rows = []
    for (size, mpp), count in sorted(fit_counts.items()):
        accepted = sum(r["status"] == "accepted" and r["particle_size_nm"] == size
                       and r["mpp"] == mpp for r in frame_rows)
        budget_rows.append(dict(particle_size_nm=size, mpp=mpp, fits_carried_out=count,
                                fits_accepted=accepted, fit_limit=MAX_FITS_PER_GROUP,
                                limit_reached=count >= MAX_FITS_PER_GROUP, random_seed=RANDOM_SEED))
    budget_df = pd.DataFrame(budget_rows)
    budget_df.to_csv(SAVE_PATH / "psf_sampling_summary.csv", index=False)
    print(budget_df.to_string(index=False))
    df = pd.DataFrame(all_rows)
    df.to_csv(SAVE_PATH / "psf_fits.csv", index=False)
    if df.empty:
        raise RuntimeError("No accepted PSF fits; see psf_frame_fits.csv and session/calibration messages.")

    theory_sigma = calculate_theoretical_psf_sigma(WAVELENGTH_NM, NUMERICAL_APERTURE)
    print(f"\nTheoretical point-source PSF sigma (lambda={WAVELENGTH_NM} nm, NA={NUMERICAL_APERTURE}): "
          f"{theory_sigma:.2f} nm")
    print("NOTE: this wavelength is confirmed only for the smallest (Nile Red) bead; other sizes use a "
          "different, unconfirmed dye, and sizes >= diffraction limit are not point sources (see module docstring).")

    summary = df.groupby(["particle_size_nm", "mpp"]).agg(
        n_tracks=("trajectory_id", "count"),
        sigma_x_mean_px=("sigma_x_psf_px", "mean"), sigma_x_std_px=("sigma_x_psf_px", "std"),
        sigma_y_mean_px=("sigma_y_psf_px", "mean"), sigma_y_std_px=("sigma_y_psf_px", "std"),
        sigma_x_mean_nm=("sigma_x_psf_nm", "mean"), sigma_x_std_nm=("sigma_x_psf_nm", "std"),
        sigma_y_mean_nm=("sigma_y_psf_nm", "mean"), sigma_y_std_nm=("sigma_y_psf_nm", "std"),
    ).reset_index()
    print(summary.to_string(index=False))

    with plt.rc_context(_RC):
        fig = plot_psf_vs_size(df, detector_df)
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        fig_path = SAVE_PATH / "psf_vs_size.png"
        fig.savefig(fig_path, dpi=600, bbox_inches="tight")
        print(f"Plot gespeichert: {fig_path}")
        plt.show()

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    csv_path = SAVE_PATH / "psf_fits.csv"
    df.to_csv(csv_path, index=False)
    print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
