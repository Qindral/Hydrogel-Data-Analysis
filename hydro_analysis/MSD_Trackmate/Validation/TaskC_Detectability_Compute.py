"""
Particle detectability -- per-detection raw-pixel crops and metrics for
35 nm (nominal "20 nm") particles, water and hydrogel.

Compute stage: reads TrackMate XML via core.io.build_datasets() (unmodified)
for the x,y,frame positions, and raw TIFF pixels directly via
tifffile.imread(path, key=frame_idx) -- single-frame reads, never a whole
multi-hundred-MB stack (unlike ../Visualisation/Detection_Demo_Figures.py's
whole-stack load, which does not scale to sampling many frames across many
movies).

For each movie, also locates and parses its sibling raw TrackMate project
XML (the un-suffixed export next to Tracks/, distinct from the filtered
*_Tracks.xml files) for its actual <DetectorSettings DETECTOR_NAME=
"LOG_DETECTOR" RADIUS=".."> -- confirmed by direct inspection that this
project's real detector is TrackMate's LoG (Laplacian-of-Gaussian), not a
"DOG" (Difference-of-Gaussians) filter, and that RADIUS is in PIXELS
(Model spatialunits="pixel" in the same XML), not calibrated microns. Falls
back to DEFAULT_LOG_RADIUS_PX if no settings XML is found for a movie
(logged explicitly, not silent).

Per detection: a padded crop is extracted, giving
  - local_max, background (median), background_sd (SD) from an inner disk /
    outer annulus of the RAW crop,
  - contrast = (local_max-background)/background, snr = (local_max-background)/background_sd,
  - the actual per-movie LoG response (scipy.ndimage.gaussian_laplace,
    sigma = radius_px/sqrt(2), NEGATED since a bright blob gives a negative
    Laplacian -- see plot_particle_size_note below), with the same
    inner/outer peak/background/SNR metrics computed on that response too.
Background ("no particle") candidates are sampled per frame at least
EXCLUSION_RADIUS_PX from every real detection in that frame (using the
FULL tracks_df, not just the subsample).

Every sampled crop (not just the 3 later selected) is stored -- arrays are
tiny -- in cache/detectability_35nm.pkl, so TaskC_Detectability_Figure.py
and TaskC_Detectability_Distribution.py never re-touch the TIFFs.

IMPORTANT: the ~15-21 px crop window is chosen to comfortably bracket the
apparent fluorescence/PSF extent in image pixels -- it is NOT a claim about
the physical particle diameter (35 nm is far below the diffraction limit;
the visible spot is the point-spread function, not the particle itself).

Run this script whenever the raw TIFF/TrackMate data changes -- it
unconditionally overwrites cache/detectability_35nm.pkl every run.
Consumed by TaskC_Detectability_Figure.py and TaskC_Detectability_Distribution.py.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_laplace

from hydro_analysis.core.io import build_datasets

# ── Configuration ──────────────────────────────────────────────────────────────
MOVIE_FOLDERS: dict[str, list[Path]] = {
    "water": [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\20 nm\Tracks"),
        Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.16\Tracks_20"),
    ],
    "hydrogel": [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\Tracks"),
    ],
}
CACHE_FILE = Path(__file__).parent.parent / "cache" / "detectability_35nm.pkl"

CROP_SIZE_PX = 21          # core metrics window
PAD_PX = 10                # extra margin fed to the LoG filter to avoid edge distortion
INNER_RADIUS_PX = 3        # signal disk for peak search
OUTER_RADIUS_PX = CROP_SIZE_PX // 2   # background annulus, (INNER, OUTER] within the core crop

MAX_DETECTIONS_PER_MOVIE = 300
N_BACKGROUND_PER_FRAME = 3
EXCLUSION_RADIUS_PX = CROP_SIZE_PX     # min distance from ANY detection in that frame

DEFAULT_LOG_RADIUS_PX = 3.0   # used only if a movie's own settings XML cannot be found
SETTINGS_TAIL_BYTES = 2_000_000

_DETECTOR_RE = re.compile(
    r'<DetectorSettings[^>]*DETECTOR_NAME="([A-Z_]+)"[^>]*RADIUS="([\d.eE+-]+)"'
)
_PIPELINE_SUFFIXES = ("_Tracks", "filtered_", "Resultof", "_processed", "_var")


def _clean_stem(xml_path: Path) -> str:
    stem = xml_path.stem
    for suffix in _PIPELINE_SUFFIXES:
        stem = stem.replace(suffix, "")
    return stem


def find_settings_xml(xml_path: Path) -> Path | None:
    """Locate the raw (un-suffixed) TrackMate project XML for this movie -- tries
    the same folder as the .tif, then an 'analysis' subfolder next to Tracks/."""
    stem = _clean_stem(xml_path)
    movie_dir = xml_path.parent.parent   # same convention as core.io.find_rec_tif_files
    candidates = [
        movie_dir / f"{stem}.xml",
        movie_dir / "analysis" / f"{stem}.xml",
        movie_dir / "Analysis" / f"{stem}.xml",
    ]
    for cand in candidates:
        if cand.exists() and cand != xml_path:
            return cand
    return None


def parse_log_radius_px(settings_xml: Path) -> float | None:
    try:
        size = settings_xml.stat().st_size
        with open(settings_xml, "rb") as f:
            if size > SETTINGS_TAIL_BYTES:
                f.seek(-SETTINGS_TAIL_BYTES, 2)
            tail = f.read().decode("utf-8", errors="ignore")
        m = _DETECTOR_RE.search(tail)
        if m is None:
            return None
        detector_name, radius_str = m.group(1), m.group(2)
        if detector_name != "LOG_DETECTOR":
            print(f"  [NOTE] {settings_xml.name}: detector is {detector_name}, not LOG_DETECTOR")
        return float(radius_str)
    except Exception as exc:
        print(f"  [WARN] Could not parse settings XML {settings_xml}: {exc}")
        return None


def _crop_metrics(core: np.ndarray, inner_mask: np.ndarray, outer_mask: np.ndarray) -> dict:
    local_max = float(core[inner_mask].max())
    background = float(np.median(core[outer_mask]))
    background_sd = float(np.std(core[outer_mask], ddof=1))
    contrast = (local_max - background) / background if background != 0 else np.nan
    snr = (local_max - background) / background_sd if background_sd > 0 else np.nan
    return {"local_max": local_max, "background": background, "background_sd": background_sd,
            "contrast": contrast, "snr": snr}


def _extract_row(padded: np.ndarray, radius_px: float, frame: int, movie: str, condition: str,
                  particle_size_nm: float, mpp: float, particle_id, is_background: bool) -> dict | None:
    half = CROP_SIZE_PX // 2
    core = padded[PAD_PX:PAD_PX + CROP_SIZE_PX, PAD_PX:PAD_PX + CROP_SIZE_PX]
    if core.shape != (CROP_SIZE_PX, CROP_SIZE_PX):
        return None

    yy, xx = np.mgrid[0:CROP_SIZE_PX, 0:CROP_SIZE_PX]
    dist = np.hypot(xx - half, yy - half)
    inner_mask = dist <= INNER_RADIUS_PX
    outer_mask = (dist > INNER_RADIUS_PX) & (dist <= OUTER_RADIUS_PX)
    if not inner_mask.any() or not outer_mask.any():
        return None

    raw_metrics = _crop_metrics(core.astype(np.float64), inner_mask, outer_mask)

    sigma = radius_px / np.sqrt(2.0)
    log_response_padded = -gaussian_laplace(padded.astype(np.float64), sigma=sigma)
    log_core = log_response_padded[PAD_PX:PAD_PX + CROP_SIZE_PX, PAD_PX:PAD_PX + CROP_SIZE_PX]
    log_metrics = _crop_metrics(log_core, inner_mask, outer_mask)

    return {
        "movie": movie, "condition": condition, "frame": int(frame), "particle_id": particle_id,
        "particle_size_nm": particle_size_nm, "mpp": mpp, "radius_px": radius_px,
        "is_background": is_background,
        "core_crop": core.copy(), "log_crop": log_core.astype(np.float32),
        "local_max": raw_metrics["local_max"], "background": raw_metrics["background"],
        "background_sd": raw_metrics["background_sd"], "contrast": raw_metrics["contrast"],
        "snr": raw_metrics["snr"],
        "log_peak": log_metrics["local_max"], "log_background": log_metrics["background"],
        "log_background_sd": log_metrics["background_sd"], "log_snr": log_metrics["snr"],
    }


def process_movie(result: dict, condition: str, radius_px: float) -> list[dict]:
    tracks_df = result.get("tracks_df")
    tif_path = result.get("tif_path")
    mpp = result.get("mpp")
    base_name = result.get("base_name")
    particle_size_nm = result.get("particle_size_nm")
    if tracks_df is None or tracks_df.empty or tif_path is None:
        return []
    tif_path = Path(tif_path)
    if not tif_path.exists():
        return []

    h_img, w_img = None, None
    with tifffile.TiffFile(tif_path) as tif:
        page0 = tif.series[0]
        n_frames_total = page0.shape[0] if page0.ndim == 3 else 1
        h_img, w_img = page0.shape[-2], page0.shape[-1]

    n_rows = len(tracks_df)
    step = max(1, int(np.ceil(n_rows / MAX_DETECTIONS_PER_MOVIE)))
    sampled = tracks_df.iloc[::step].copy()

    frame_groups: dict[int, list[tuple]] = {}
    for _, row in sampled.iterrows():
        frame_groups.setdefault(int(row["frame"]), []).append((row["particle"], row["x"], row["y"]))

    rows: list[dict] = []
    rng = np.random.default_rng(abs(hash(base_name)) % (2**32))

    for frame_idx, detections in frame_groups.items():
        if frame_idx < 0 or frame_idx >= n_frames_total:
            continue
        try:
            frame_img = tifffile.imread(tif_path, key=frame_idx)
        except Exception as exc:
            print(f"  [WARN] Could not read frame {frame_idx} of {tif_path.name}: {exc}")
            continue

        pad_h, pad_w = frame_img.shape
        margin = CROP_SIZE_PX // 2 + PAD_PX

        for pid, x, y in detections:
            xi, yi = int(round(x)), int(round(y))
            if xi - margin < 0 or xi + margin >= pad_w or yi - margin < 0 or yi + margin >= pad_h:
                continue
            padded = frame_img[yi - margin:yi + margin + 1, xi - margin:xi + margin + 1]
            padded = padded[:2 * margin + 1, :2 * margin + 1]
            row = _extract_row(padded, radius_px, frame_idx, base_name, condition,
                               particle_size_nm, mpp, pid, is_background=False)
            if row is not None:
                rows.append(row)

        # Background candidates: exclude all real detections in this frame (full tracks_df).
        all_in_frame = tracks_df[tracks_df["frame"] == frame_idx][["x", "y"]].to_numpy()
        n_placed = 0
        attempts = 0
        while n_placed < N_BACKGROUND_PER_FRAME and attempts < 50:
            attempts += 1
            xi = int(rng.integers(margin, pad_w - margin))
            yi = int(rng.integers(margin, pad_h - margin))
            if all_in_frame.size:
                d = np.hypot(all_in_frame[:, 0] - xi, all_in_frame[:, 1] - yi)
                if d.min() < EXCLUSION_RADIUS_PX:
                    continue
            padded = frame_img[yi - margin:yi + margin + 1, xi - margin:xi + margin + 1]
            row = _extract_row(padded, radius_px, frame_idx, base_name, condition,
                               particle_size_nm, mpp, None, is_background=True)
            if row is not None:
                rows.append(row)
                n_placed += 1

    return rows


def main() -> None:
    all_rows: list[dict] = []
    for condition, folders in MOVIE_FOLDERS.items():
        for folder in folders:
            if not folder.exists():
                print(f"  [SKIP] Ordner nicht gefunden: {folder}")
                continue
            datasets = build_datasets(folder)
            print(f"{condition}: {len(datasets)} Dateien aus {folder}")
            for _, result in datasets.items():
                if result is None:
                    continue
                # build_datasets() keys its dict by xml_file.stem, not the full path --
                # the actual path lives in the result_dict itself.
                xml_path = Path(result["xml_path"])
                settings_xml = find_settings_xml(xml_path)
                radius_px = None
                if settings_xml is not None:
                    radius_px = parse_log_radius_px(settings_xml)
                if radius_px is None:
                    radius_px = DEFAULT_LOG_RADIUS_PX
                    print(f"  [FALLBACK] {result.get('base_name')}: no LOG_DETECTOR settings found, "
                          f"using default radius {DEFAULT_LOG_RADIUS_PX} px")
                else:
                    print(f"  {result.get('base_name')}: LOG_DETECTOR radius = {radius_px} px "
                          f"(from {settings_xml.name})")

                rows = process_movie(result, condition, radius_px)
                print(f"    -> {len(rows)} Crops extrahiert "
                      f"({sum(1 for r in rows if not r['is_background'])} Partikel, "
                      f"{sum(1 for r in rows if r['is_background'])} Hintergrund)")
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"\nGesamt: {len(df)} Crops ({(~df['is_background']).sum()} Partikel, "
          f"{df['is_background'].sum()} Hintergrund)")

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(df, f)
    print(f"Cache gespeichert: {CACHE_FILE}")


if __name__ == "__main__":
    main()
