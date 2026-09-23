"""
PSF-Analyse fuer immobilisierte Partikel -- konsolidiertes Skript, ersetzt
PSF_Analysis_Immobilized.py.

Fittet pro Frame einen 2D-Gauss auf die rohe TIFF-Crop um jede XML-Position
eines immobilisierten Partikels (nie auf gemittelte Frames). sigma_fit (nm)
ist der Mittelwert der akzeptierten Pro-Frame-Gauss-Fits eines Tracks, in nm
umgerechnet ueber mpp. sigma_xml (Pixel) ist der TrackMate-Radius direkt aus
der Session-XML (per-Spot RADIUS, sonst DetectorSettings-RADIUS als Fallback),
nur ueber die Pixelkalibrierung in px umgerechnet -- kein erneuter Fit. Eine
zweite Plot-Variante (psf_scatter_vs_theory.png) markiert zusaetzlich die
theoretische beugungsbegrenzte PSF (core.physics.calculate_theoretical_psf_
sigma) bei der kleinsten Groesse, siehe DIFFRACTION_LIMITED_SIZES_NM.

Performance: pro (Bedingung x Partikelgroesse) werden maximal
MAX_SPOTS_PER_GROUP=200 zufaellig ausgewaehlte Tracks gefittet (Bedingung =
IMMOB_DATASETS-Label, da hier ausschliesslich immobilisierte Partikel
vorliegen), reproduzierbar via RANDOM_SEED=42. Je Track werden hoechstens
MAX_FRAMES_PER_TRACK=20 zufaellig ausgewaehlte Frames gefittet -- genug fuer
eine stabile Track-Mittelung (>> MIN_ACCEPTED_FRAMES), begrenzt aber die
Rechenzeit je Gruppe auf maximal 200*20=4000 Fits und verhindert, dass ein
einzelner sehr langer Track (manche immobilisierten Tracks laufen ueber
hunderte Frames) das ganze Gruppenbudget allein aufbraucht und die anderen
199 Tracks der Gruppe verdraengt. Zusaetzlich werden bereits gefittete
Movies in cache/psf_analysis_fits.pkl gepickelt; ein erneuter Lauf fitted
nur Sessions (Tracks-XML-Dateien), die noch nicht im Cache stehen.

Bevorzugt vollstaendige Analysis_immob_NEW-Sessions, faellt sonst auf die
gematchte Analysis-Session zurueck (siehe select_session()). Blurry Frames
werden ausgeschlossen (Frame_Sharpness_Immobilized), ueberlappende
XML-Spots werden maskiert bzw. verworfen, Pro-Frame-Breiten-Ausreisser und
unueblich breite Track-Mittel werden per MAD entfernt.

Outputs (Auswertungsbilder/PSF_Analysis/): psf_fits.csv (akzeptierte
Track-Mittel inkl. sigma_fit_nm/sigma_xml_px), psf_frame_fits.csv (alle
Pro-Frame-Fitversuche inkl. Ablehnungsgrund), psf_detection_settings.csv
(tatsaechliche Session-Herkunft/Detektor-Settings je Movie),
psf_sampling_summary.csv (Budget-Buchhaltung je Gruppe),
psf_scatter_vs_size.png / psf_scatter_vs_theory.png -- Streudiagramm (kein
Violinplot, auf Wunsch entfernt): links jeder Track als Punkt fuer
"PSF width" = 2*sigma_fit (nm, linke Y-Achse), rechts ebenso fuer "fitting
pixel diameter" = 2*sigma_xml (Pixel, rechte Y-Achse) -- beide als
Durchmesser, nicht als Radius/Sigma. Mittelwertstrich je Seite, keine
nominelle Referenzlinie, X-Achse mit realer DLS-Groesse beschriftet. Die
_vs_theory-Variante markiert zusaetzlich die theoretische PSF bei der
kleinsten Groesse.

Ersetzt: PSF_Analysis_Immobilized.py (nach obsolete/ verschoben).
Frame_Sharpness_Immobilized.py, Compare_Tracks_vs_RawAnalysis.py und
Loc_Error_Analyse_immob_particle.py bleiben unveraendert und werden nur
importiert -- IMMOB_DATASETS und _corrected_particle_size() kommen jetzt aus
Loc_Error_Analyse_immob_particle.py (vorher aus dem inzwischen nach
obsolete/ verschobenen TaskA_Immobilized_Compute_Headless.py, dessen eigene
Analyse durch Loc_Error_Analyse_immob_particle.py's gruendlichere Pipeline
ueberholt war -- die beiden gemeinsam gebrauchten Konstanten wurden vorher
umgezogen). test_psf_frame_fits.py importiert die Kernfunktionen
(fit_frame_psf, width_outliers, _gaussian_2d) jetzt von hier.
"""
from __future__ import annotations

import pickle
import xml.etree.ElementTree as ET
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, OptimizeWarning
from tqdm.auto import tqdm

from hydro_analysis.MSD_Trackmate.Validation_Claude.Frame_Sharpness_Immobilized import (
    compute_sharpness_series, flag_blurry_frames,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Loc_Error_Analyse_immob_particle import (
    _find_xml_in_subfolder, NEW_ANALYSIS_SUBFOLDER, IMMOB_DATASETS, _corrected_particle_size,
)
from hydro_analysis.MSD_Trackmate.Validation_GPT.trackmate_session import read_trackmate_session
from hydro_analysis.MSD_Trackmate.Validation_Claude.Compare_Tracks_vs_RawAnalysis import (
    match_tracks_to_analysis, _read_image_data,
)
from hydro_analysis.core.io import find_rec_tif_files, extract_particle_size_from_path, get_dls_labels
from hydro_analysis.core.physics import calculate_theoretical_psf_sigma
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "PSF_Analysis"

CACHE_FILE = Path(__file__).parent.parent / "cache" / "psf_analysis_fits.pkl"

MAX_SPOTS_PER_GROUP = 200   # zufaellig ausgewaehlte Tracks ("Spots") pro (Bedingung x Partikelgroesse)
RANDOM_SEED = 42
# Begrenzt die Rechenzeit je Track: manche immobilisierten Tracks laufen ueber
# hunderte Frames, was bei ungebremster Pro-Frame-Fitting-Schleife x 200 Tracks
# zu sehr langen Laufzeiten fuehrt UND einzelne lange Tracks das Budget einer
# Gruppe allein aufbrauchen wuerden (beobachtet: manche Groessen bekamen dadurch
# nur 1-9 statt bis zu 200 Tracks). 20 Frames reichen fuer eine stabile
# Track-Mittelung (>> MIN_ACCEPTED_FRAMES) und begrenzen die Rechenzeit je
# Gruppe auf maximal 200*20 = 4000 Fits -- vergleichbar mit dem urspruenglichen
# MAX_FITS_PER_GROUP=4000 des Vorgaengerskripts.
MAX_FRAMES_PER_TRACK = 20

CROP_HALF_PX = 8   # minimales halbes Fenster; vergroessert fuer grosse XML-Radien
INITIAL_SIGMA_RADIUS_FACTOR = 0.5   # nur Startheuristik
MAX_CENTER_SHIFT_PX = 1.0
MIN_SIGMA_PX = 0.3
MIN_FIT_R2 = 0.5
MAX_RELATIVE_SIGMA_ERROR = 0.5
MIN_ACCEPTED_FRAMES = 3
OUTLIER_MAD_K = 3.5
OUTLIER_RELATIVE_FLOOR = 0.10   # verhindert, dass Null-/Mini-MAD numerisches Rauschen verwirft

COLOR_FIT = "#0000da"
COLOR_XML = "#da6a00"
COLOR_THEORY = "#da0000"

# Theoretische beugungsbegrenzte PSF (core.physics.calculate_theoretical_psf_sigma),
# nur fuer die psf_scatter_vs_theory.png-Variante (show_theory=True). Die
# Wellenlaenge ist nur fuer die kleinste (Nile-Red-)Perle bestaetigt -- andere
# Groessen verwenden einen anderen, unbestaetigten Farbstoff, daher wird die
# Referenzlinie nur bei DIFFRACTION_LIMITED_SIZES_NM eingezeichnet.
NUMERICAL_APERTURE = 1.2
WAVELENGTH_NM = 575.0
DIFFRACTION_LIMITED_SIZES_NM = (20.0,)


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


def process_movie(dataset_label: str, tracks_xml: Path, analysis_xml: Path,
                   fit_counts: dict[tuple[str, float], int], rng: np.random.Generator,
                   ) -> tuple[list[dict], dict | None, list[dict]]:
    """Fits randomly ordered tracks/frames in one movie, up to the remaining
    per-group frame-fit budget (MAX_SPOTS_PER_GROUP)."""
    rows: list[dict] = []
    frame_rows: list[dict] = []
    particle_size_nm = _corrected_particle_size(tracks_xml.name, extract_particle_size_from_path(tracks_xml))
    calib = find_rec_tif_files(tracks_xml)
    tif_path, mpp = calib["tiff_file"], calib["mpp"]
    if tif_path is None or mpp is None or not np.isfinite(mpp) or mpp <= 0:
        print(f"  [SKIP] {tracks_xml.name}: calibration missing")
        return rows, None, frame_rows
    if particle_size_nm is None or not np.isfinite(particle_size_nm):
        print(f"  [SKIP] {tracks_xml.name}: particle size missing")
        return rows, None, frame_rows
    group = (dataset_label, float(particle_size_nm))
    fit_counts.setdefault(group, 0)
    if fit_counts[group] >= MAX_SPOTS_PER_GROUP:
        print(f"  [BUDGET] {tracks_xml.name}: {group} already has {fit_counts[group]} tracks")
        return rows, None, frame_rows
    session, detector_values = select_session(tracks_xml, analysis_xml)
    if session is None:
        return rows, None, frame_rows
    common = dict(dataset_label=dataset_label, movie=tracks_xml.name,
                  particle_size_nm=particle_size_nm, mpp=mpp)
    detector_row = {**common, **detector_values}
    blurry, _ = flag_blurry_frames(compute_sharpness_series(tif_path))
    spots = session.spots.copy()
    spots["x_px"] = spots["x"] / session.pixelwidth_um
    spots["y_px"] = spots["y"] / session.pixelheight_um
    radii = pd.to_numeric(spots.get("RADIUS", pd.Series(np.nan, index=spots.index)), errors="coerce")
    radii = radii.where(np.isfinite(radii) & (radii > 0), float(detector_values["detector_radius"]))
    spots["rx"] = radii / session.pixelwidth_um
    spots["ry"] = radii / session.pixelheight_um
    indexed = spots.set_index("spot_id")
    by_frame = {int(f): g for f, g in spots.groupby("frame")}
    tracks = list(session.trajectories.groupby("trajectory_id"))
    tracks = [tracks[i] for i in rng.permutation(len(tracks))]
    seen_spots: set = set()

    for traj_id, g in tracks:
        if fit_counts[group] >= MAX_SPOTS_PER_GROUP:
            break
        fit_counts[group] += 1   # this track is one of the randomly selected "Spots" being fit
        g = g.sort_values("frame").drop_duplicates("frame")
        if MAX_FRAMES_PER_TRACK is not None and len(g) > MAX_FRAMES_PER_TRACK:
            g = g.iloc[rng.permutation(len(g))[:MAX_FRAMES_PER_TRACK]]
        records = []
        for detection in g.itertuples():
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
            radii_vals = np.array([[r["xml_radius_x_px"], r["xml_radius_y_px"]] for r in valid])
            rx_mean, ry_mean = radii_vals.mean(axis=0)
            sigma_x_nm, sigma_y_nm = sx*mpp*1000, sy*mpp*1000
            rows.append({**common, **detector_values, "trajectory_id": traj_id, "mpp": mpp,
                         "n_frames_sampled": len(records), "n_frames_fitted": len(valid),
                         "sigma_x_psf_px": sx, "sigma_y_psf_px": sy,
                         "sigma_x_std_px": dx, "sigma_y_std_px": dy,
                         "sigma_x_psf_nm": sigma_x_nm, "sigma_y_psf_nm": sigma_y_nm,
                         "sigma_fit_nm": float(np.mean([sigma_x_nm, sigma_y_nm])),
                         "xml_radius_x_px_mean": rx_mean, "xml_radius_y_px_mean": ry_mean,
                         "sigma_xml_px": float(np.mean([rx_mean, ry_mean]))})
        frame_rows.extend(records)

    aggregate_mask = width_outliers([[r["sigma_x_psf_px"], r["sigma_y_psf_px"]] for r in rows], upper_only=True)
    rejected_ids = {r["trajectory_id"] for r, bad in zip(rows, aggregate_mask) if bad}
    for rec in frame_rows:
        if rec["trajectory_id"] in rejected_ids and rec["status"] == "accepted":
            rec["status"] = "broad_track_outlier"
    rows = [r for r in rows if r["trajectory_id"] not in rejected_ids]
    print(f"  [OK] {tracks_xml.name}: {len(rows)} tracks accepted; group {group} now "
          f"{fit_counts[group]}/{MAX_SPOTS_PER_GROUP} tracks sampled")
    return rows, detector_row, frame_rows


def plot_psf_scatter(df: pd.DataFrame, dls_labels: dict[float, int], show_theory: bool = False) -> plt.Figure:
    """Streudiagramm statt Violinplot (auf Wunsch): links jeder Track als
    eigener Punkt fuer PSF width = 2*sigma_fit (nm), rechts ebenso fuer
    fitting pixel diameter = 2*sigma_xml (Pixel) -- beide als Durchmesser.
    Zeigt die XML-Pixelwerte immer als echte Rohdaten (nie als KDE/Balken,
    die bei einem pro Movie oft konstanten Detektorradius wie eine leere
    Flaeche aussehen koennen), plus Mittelwertstrich je Seite. X-Achse
    nutzt die reale DLS-Partikelgroesse als Beschriftung.

    show_theory=True markiert zusaetzlich die theoretische beugungsbegrenzte
    PSF (core.physics.calculate_theoretical_psf_sigma, WAVELENGTH_NM/
    NUMERICAL_APERTURE) bei der kleinsten Groesse -- diese Wellenlaengen-
    Annahme ist nur fuer die kleinste (Nile-Red-)Perle bestaetigt, siehe
    DIFFRACTION_LIMITED_SIZES_NM."""
    sizes = sorted(df["particle_size_nm"].dropna().unique())
    width = 0.35
    rng = np.random.default_rng(0)
    fig, ax_left = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    ax_right = ax_left.twinx()
    positions = np.arange(len(sizes))

    for x0, size in zip(positions, sizes):
        sub = df.loc[df["particle_size_nm"] == size]
        psf_width_nm = 2.0 * sub["sigma_fit_nm"].to_numpy()
        fitting_diameter_px = 2.0 * sub["sigma_xml_px"].to_numpy()

        jitter_l = (rng.random(len(psf_width_nm)) - 1.0) * width * 0.9   # links von x0
        ax_left.scatter(x0 + jitter_l, psf_width_nm, s=12, alpha=0.55,
                        facecolor=COLOR_FIT, edgecolor="none", zorder=3)
        if len(psf_width_nm):
            mean_val = float(np.mean(psf_width_nm))
            ax_left.plot([x0 - width, x0], [mean_val, mean_val], color="black", linewidth=1.6, zorder=4)

        jitter_r = rng.random(len(fitting_diameter_px)) * width * 0.9   # rechts von x0
        ax_right.scatter(x0 + jitter_r, fitting_diameter_px, s=12, alpha=0.55,
                         facecolor=COLOR_XML, edgecolor="none", zorder=3)
        if len(fitting_diameter_px):
            mean_val = float(np.mean(fitting_diameter_px))
            ax_right.plot([x0, x0 + width], [mean_val, mean_val], color="black", linewidth=1.6, zorder=4)

    if show_theory:
        theory_width_nm = 2.0 * calculate_theoretical_psf_sigma(WAVELENGTH_NM, NUMERICAL_APERTURE)
        for x0, size in zip(positions, sizes):
            if size not in DIFFRACTION_LIMITED_SIZES_NM:
                continue
            ax_left.plot([x0 - width, x0], [theory_width_nm, theory_width_nm], color=COLOR_THEORY,
                        linewidth=1.6, linestyle=(0, (4, 3)), zorder=5)

    ax_left.set_xticks(positions)
    ax_left.set_xticklabels([f"{dls_labels.get(s, int(s))} nm" for s in sizes])
    ax_left.set_xlabel("Partikelgroesse (DLS, Durchmesser)")
    ax_left.set_ylabel("PSF width (nm)", color=COLOR_FIT)
    ax_right.set_ylabel("fitting pixel diameter (px)", color=COLOR_XML)
    ax_left.tick_params(axis="y", colors=COLOR_FIT)
    ax_right.tick_params(axis="y", colors=COLOR_XML)
    ax_left.set_xlim(-0.7, len(sizes) - 0.3)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_FIT, markersize=6,
               label="PSF width (nm, links, pro Track)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_XML, markersize=6,
               label="fitting pixel diameter (px, rechts, pro Track)"),
        Line2D([0], [0], color="black", linewidth=1.6, label="Mittelwert"),
    ]
    if show_theory:
        legend_handles.append(Line2D([0], [0], color=COLOR_THEORY, linewidth=1.6, linestyle=(0, (4, 3)),
                                     label=f"theoretische PSF ({WAVELENGTH_NM:.0f} nm, NA={NUMERICAL_APERTURE}, "
                                           "nur kleinste Perle bestaetigt)"))
    ax_left.legend(handles=legend_handles, loc="upper left", fontsize=7, frameon=False)
    return fig


def _load_cache() -> dict[str, dict]:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def main() -> None:
    jobs: list[tuple[str, Path, Path]] = []
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

    cache = _load_cache()
    fit_counts: dict[tuple[str, float], int] = {}
    all_rows: list[dict] = []
    detector_rows: list[dict] = []
    frame_rows: list[dict] = []

    cached_keys = [str(tracks_xml) for _, tracks_xml, _ in jobs if str(tracks_xml) in cache]
    new_jobs = [(label, tracks_xml, analysis_xml) for label, tracks_xml, analysis_xml in jobs
                if str(tracks_xml) not in cache]

    for key in cached_keys:
        entry = cache[key]
        # Budget counts attempted tracks, not just accepted rows -- a track that
        # was sampled but rejected (too few valid frames, outlier, ...) still
        # consumed one unit of the group budget when it was originally fit.
        attempted_per_group: dict[tuple[str, float], set] = {}
        for rec in entry.get("frame_rows", []):
            group = (rec["dataset_label"], rec["particle_size_nm"])
            attempted_per_group.setdefault(group, set()).add(rec["trajectory_id"])
        for group, traj_ids in attempted_per_group.items():
            fit_counts[group] = fit_counts.get(group, 0) + len(traj_ids)
        all_rows.extend(entry["rows"])
        if entry.get("detector_row") is not None:
            detector_rows.append(entry["detector_row"])
        frame_rows.extend(entry.get("frame_rows", []))

    print(f"\n{len(cached_keys)} Sessions aus Cache geladen, {len(new_jobs)} neue Sessions werden gefittet.")

    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(new_jobs))
    for i in tqdm(order, desc="Neue Sessions", unit="movie"):
        dataset_label, tracks_xml, analysis_xml = new_jobs[i]
        rows, detector_row, movie_frame_rows = process_movie(dataset_label, tracks_xml, analysis_xml,
                                                              fit_counts, rng)
        cache[str(tracks_xml)] = {"rows": rows, "detector_row": detector_row, "frame_rows": movie_frame_rows}
        all_rows.extend(rows)
        if detector_row is not None:
            detector_rows.append(detector_row)
        frame_rows.extend(movie_frame_rows)

    _save_cache(cache)

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detector_rows).to_csv(SAVE_PATH / "psf_detection_settings.csv", index=False)
    pd.DataFrame(frame_rows).to_csv(SAVE_PATH / "psf_frame_fits.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(SAVE_PATH / "psf_fits.csv", index=False)
    if df.empty:
        raise RuntimeError("No accepted PSF fits; see psf_frame_fits.csv and session/calibration messages.")

    budget_rows = [dict(dataset_label=label, particle_size_nm=size, tracks_sampled=count,
                         sample_limit=MAX_SPOTS_PER_GROUP, limit_reached=count >= MAX_SPOTS_PER_GROUP,
                         random_seed=RANDOM_SEED)
                   for (label, size), count in sorted(fit_counts.items())]
    budget_df = pd.DataFrame(budget_rows)
    budget_df.to_csv(SAVE_PATH / "psf_sampling_summary.csv", index=False)
    print(budget_df.to_string(index=False))

    summary = df.groupby(["dataset_label", "particle_size_nm"]).agg(
        n_tracks=("trajectory_id", "count"),
        sigma_fit_mean_nm=("sigma_fit_nm", "mean"), sigma_fit_std_nm=("sigma_fit_nm", "std"),
        sigma_xml_mean_px=("sigma_xml_px", "mean"), sigma_xml_std_px=("sigma_xml_px", "std"),
    ).reset_index()
    print(summary.to_string(index=False))

    dls_labels = get_dls_labels()
    with plt.rc_context(_RC):
        for show_theory, filename in ((False, "psf_scatter_vs_size.png"),
                                       (True, "psf_scatter_vs_theory.png")):
            fig = plot_psf_scatter(df, dls_labels, show_theory=show_theory)
            fig_path = SAVE_PATH / filename
            safe_savefig(fig, fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot gespeichert: {fig_path}")

    print(f"Tabelle gespeichert: {SAVE_PATH / 'psf_fits.csv'}")


if __name__ == "__main__":
    main()
