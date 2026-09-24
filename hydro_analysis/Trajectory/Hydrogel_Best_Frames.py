"""
Representative hydrogel microscopy frames (20 mg/mL C16), one per particle
size, with an automatically chosen frame and cutout -- display, cutout and
scalebar only, no trajectories or particle markers.

Reads the raw TIFF stacks and their .rec files directly (listed in IMAGES
below); it neither reads nor writes any pickle cache and does not depend on
any compute script. Rendering reuses Trajectory/trajectory_plotter.py's
primitives unchanged (load_tiff_stack, average_frames, add_scalebar,
_make_fig): gray PowerNorm with gamma 0.6, +-1 frame average, same scalebar
box as Random_Condition_Frames.py. One deliberate difference: the display
range is taken from the cutout's own percentiles (DISPLAY_PERCENTILES)
instead of render_frame_on_ax's [min, 2 * max] of the full frame, which
leaves these hydrogel movies (high, uneven background) nearly uniformly gray.

Frame and cutout selection, per movie:
  1. Blurry frames are excluded with Frame_Sharpness_Immobilized's
     compute_sharpness_series()/flag_blurry_frames() (variance of Laplacian,
     per-movie MAD threshold), unchanged.
  2. For up to MAX_FRAMES_EVAL evenly spaced sharp frames, the +-1 frame
     average is split into square tiles of about TILE_UM. Per tile:
       brightness = smoothed tile maximum, normalised to the movie's
                    1st-99.9th intensity percentile range;
       structure  = maximum of the fine structure (image smoothed by
                    SMOOTH_SIGMA_PX minus its BACKGROUND_SIGMA_UM background),
                    normalised to the movie's 99.9th percentile -- bright
                    particles and edges, not the smooth illumination profile;
       dark       = tile mean below DARK_LEVEL of the intensity range.
  3. Every 1:CUTOUT_RATIO cutout (width >= MIN_WIDTH_UM, in WIDTH_STEP_UM
     steps, positions on the tile grid) is scored as
         score = mean(interest) * dark_factor(dark fraction)
     interest    = (1 - STRUCTURE_WEIGHT) * brightness + STRUCTURE_WEIGHT * structure;
     dark_factor = 1 for DARK_MIN-DARK_MAX dark tiles, 1 - DARK_BONUS without
                   any dark tile, falling to 0 as the dark share approaches
                   100 % -- large dark areas are cut away where possible.
     The bonus is kept small on purpose: in these movies the darkest region
     is usually the vignetting at the field border, and a strong bonus
     pulls the cutout there instead of onto the particles. Dark regions
     inside the chosen cutout still appear dark through the cutout-based
     display range. The highest-scoring (frame, cutout) is used; on ties
     the smaller cutout wins.

Outputs (PNG, 600 dpi) in SAVE_PATH:
  <key>_frame.png          raw frame (+-1 average), cutout, scalebar
  <key>_DoG.png            same frame and cutout after a DoG filter
                           (snr_signal_profile.dog_filter, TrackMate/ImgLib2
                           scale convention, radius DOG_RADIUS_UM)
  50nm_vorher_nachher.png / 50nm_vorher_nachher_DoG.png
                           side-by-side initial vs. 1 d after injection
  Bildinformationen.txt    per image: file paths, acquisition settings,
                           chosen frame and cutout, scores, full .rec comment

Note on the 50 nm movies: the TIFFs are named "B3_inside_..." but the only
matching .rec files are named "B2_inside_..." (same recording dates,
01.10./02.10.); they are assigned explicitly in IMAGES and the text file
marks this assignment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter

from hydro_analysis.core.io import check_text_encoding, parse_rec_file, parse_rec_comment_metadata
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Frame_Sharpness_Immobilized import (
    compute_sharpness_series, flag_blurry_frames,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.snr_signal_profile import dog_filter, MPP_BY_IMAGE_WIDTH_PX
from hydro_analysis.Trajectory.trajectory_plotter import (
    load_tiff_stack, average_frames, add_scalebar, _make_fig,
)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")

# rec=None: <tif>.tif.rec or <tif>.rec next to the TIFF is used.
IMAGES: list[dict] = [
    dict(key="20nm", size_nm=20, label="20 nm",
         tif=DATA_ROOT / r"20 nm\Inj_Spot\20 nm_atinj.tif", rec=None),
    dict(key="50nm_initial", size_nm=50, label="Direkt nach Injektion",
         tif=DATA_ROOT / r"50 nm\50 nm 20 mg\B3_inside_atinjection_50nm_20mg.tif",
         rec=DATA_ROOT / r"50 nm\50 nm 20 mg\B2_inside_atinjection_50nm_20mg.rec"),
    dict(key="50nm_1d", size_nm=50, label="1 Tag nach Injektion",
         tif=DATA_ROOT / r"50 nm\50 nm 20 mg\B3_inside_1d_nearinjection_50_20mg.tif",
         rec=DATA_ROOT / r"50 nm\50 nm 20 mg\B2_inside_1d_nearinjection_50_20mg.rec"),
    dict(key="100nm", size_nm=100, label="100 nm",
         tif=DATA_ROOT / r"100 nm\Hydrogel_20mg_100nm\Inj\100nm_atInj.tif", rec=None),
    dict(key="200nm", size_nm=200, label="200 nm",
         tif=DATA_ROOT / r"200 nm\B2_200nm_Injection_2d_00.tif", rec=None),
    dict(key="500nm", size_nm=500, label="500 nm",
         tif=DATA_ROOT / r"500 nm\B2_500_center_2d_00.tif", rec=None),
    dict(key="1000nm", size_nm=1000, label="1000 nm",
         tif=DATA_ROOT / r"1000 nm\B3_Inside_1000_20mg_bodyofwater_2d_00.tif", rec=None),
]
BEFORE_AFTER = ("50nm_initial", "50nm_1d")   # (vorher, nachher)

SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Hydrogel_Best_Frames_20mg"

CUTOUT_RATIO = 1.42        # width / height
MIN_WIDTH_UM = 40.0
WIDTH_STEP_UM = 5.0
TILE_UM = 2.0              # scoring grid; cutout positions snap to this grid
SMOOTH_SIGMA_PX = 1.0      # scoring only, the rendered frame is not smoothed
BACKGROUND_SIGMA_UM = 3.0  # large-scale background removed for the fine-structure term
STRUCTURE_WEIGHT = 1.0     # weight of fine structure (particles, edges) vs. plain brightness;
                           # < 1 lets bright but featureless haze pull the cutout away from the particles
DARK_LEVEL = 0.15          # tile is dark if its mean < low + 0.15 * (high - low)
DARK_MIN = 0.10            # desired share of dark tiles in the cutout: some dark area ...
DARK_MAX = 0.30            # ... but no large dark areas
DARK_BONUS = 0.2          # score reduction for a cutout without any dark tiles; larger values pull the
                           # cutout toward the dark vignetting at the image border
MAX_FRAMES_EVAL = 120
N_LEVEL_SAMPLE_FRAMES = 40
DISPLAY_GAMMA = 0.6        # same PowerNorm gamma as trajectory_plotter.render_frame_on_ax
DISPLAY_PERCENTILES = (0.5, 99.9)   # display range from the cutout itself
SCALEBAR_UM = 10.0
DOG_RADIUS_UM = 0.75       # 2.5 px at 0.3 um/px, as in SNR_LoG_DoG_Frame.py
SAVE_DPI = 600


# ============================================================================
# Metadata
# ============================================================================

def _resolve_rec(entry: dict) -> tuple[Optional[Path], bool]:
    """(rec path or None, True if the .rec was assigned explicitly in IMAGES)."""
    if entry.get("rec") is not None:
        rec = Path(entry["rec"])
        return (rec if rec.exists() else None), True
    tif = Path(entry["tif"])
    for cand in (tif.with_name(tif.name + ".rec"), tif.with_suffix(".rec")):
        if cand.exists():
            return cand, False
    return None, False


def _rec_comment(rec_path: Optional[Path]) -> str:
    if rec_path is None:
        return ""
    text = rec_path.read_text(encoding=check_text_encoding(rec_path), errors="replace")
    idx = text.find("Comment:")
    if idx == -1:
        return ""
    lines = [ln.strip() for ln in text[idx + len("Comment:"):].splitlines()]
    return "\n".join(ln for ln in lines if ln)


# ============================================================================
# Frame / cutout selection
# ============================================================================

def _intensity_levels(stack: np.ndarray, mpp: float) -> tuple[float, float, float]:
    """(low, high, structure_high) of the movie from evenly spaced sample
    frames: 1st / 99.9th percentile of the smoothed intensity and 99.9th
    percentile of the fine structure."""
    idx = np.linspace(0, stack.shape[0] - 1, min(N_LEVEL_SAMPLE_FRAMES, stack.shape[0])).astype(int)
    frames = [stack[i].astype(np.float32) for i in idx]
    smoothed = np.stack([gaussian_filter(f, SMOOTH_SIGMA_PX) for f in frames])
    structure = np.stack([_structure(f, mpp) for f in frames])
    lo, hi = np.percentile(smoothed, [1.0, 99.9])
    return float(lo), float(max(hi, lo + 1e-6)), float(max(np.percentile(structure, 99.9), 1e-6))


def _tile_stats(img: np.ndarray, tile_px: int) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = img.shape[0] // tile_px, img.shape[1] // tile_px
    tiles = img[:ny * tile_px, :nx * tile_px].reshape(ny, tile_px, nx, tile_px)
    return tiles.max(axis=(1, 3)), tiles.mean(axis=(1, 3))


def _candidate_widths_um(shape: tuple[int, int], mpp: float) -> list[float]:
    h, w = shape
    max_w = min((w - 1) * mpp, (h - 1) * mpp * CUTOUT_RATIO)
    if max_w < MIN_WIDTH_UM:
        raise ValueError(f"Bild zu klein fuer {MIN_WIDTH_UM} um Breite (max. {max_w:.1f} um)")
    widths = list(np.arange(MIN_WIDTH_UM, max_w, WIDTH_STEP_UM))
    if not widths or max_w - widths[-1] > 1e-6:
        widths.append(max_w)
    return [float(x) for x in widths]


def _structure(img: np.ndarray, mpp: float) -> np.ndarray:
    """Fine structure (particles, edges): lightly smoothed image minus its
    large-scale background, negative values clipped."""
    return np.clip(gaussian_filter(img, SMOOTH_SIGMA_PX) - gaussian_filter(img, BACKGROUND_SIGMA_UM / mpp), 0, None)


def _dark_factor(dark_frac: np.ndarray) -> np.ndarray:
    """1 inside [DARK_MIN, DARK_MAX]; rises from 1 - DARK_BONUS below DARK_MIN
    (no dark area), falls linearly to 0 above DARK_MAX (large dark areas)."""
    below = 1.0 - DARK_BONUS * (1.0 - np.clip(dark_frac / DARK_MIN, 0.0, 1.0))
    above = 1.0 - np.clip((dark_frac - DARK_MAX) / (1.0 - DARK_MAX), 0.0, 1.0)
    return np.where(dark_frac < DARK_MIN, below, above)


def select_frame_and_cutout(stack: np.ndarray, tif_path: Path, mpp: float) -> dict:
    scores = compute_sharpness_series(tif_path)
    is_blurry, _ = flag_blurry_frames(scores)
    sharp = np.flatnonzero(~is_blurry[:stack.shape[0]])
    if sharp.size == 0:
        sharp = np.arange(stack.shape[0])
    frames = sharp[np.linspace(0, sharp.size - 1, min(MAX_FRAMES_EVAL, sharp.size)).astype(int)]

    lo, hi, s_hi = _intensity_levels(stack, mpp)
    tile_px = max(2, int(round(TILE_UM / mpp)))
    tile_um = tile_px * mpp
    h, w = stack.shape[1:]
    widths = _candidate_widths_um((h, w), mpp)

    best: Optional[dict] = None
    for frame_idx in frames:
        avg = average_frames(stack, int(frame_idx))
        t_max, t_mean = _tile_stats(gaussian_filter(avg, SMOOTH_SIGMA_PX), tile_px)
        s_max, _ = _tile_stats(_structure(avg, mpp), tile_px)
        brightness = np.clip((t_max - lo) / (hi - lo), 0.0, 1.0)
        structure = np.clip(s_max / s_hi, 0.0, 1.0)
        interest = (1.0 - STRUCTURE_WEIGHT) * brightness + STRUCTURE_WEIGHT * structure
        dark = ((t_mean - lo) / (hi - lo) < DARK_LEVEL).astype(np.float32)
        for width_um in widths:
            cw = int(np.ceil(width_um / tile_um))
            ch = int(np.ceil(width_um / CUTOUT_RATIO / tile_um))
            if ch > t_max.shape[0] or cw > t_max.shape[1]:
                continue
            win_interest = sliding_window_view(interest, (ch, cw)).mean(axis=(-2, -1))
            dark_frac = sliding_window_view(dark, (ch, cw)).mean(axis=(-2, -1))
            score = win_interest * _dark_factor(dark_frac)
            iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
            if best is None or score[iy, ix] > best["score"] + 1e-9:
                best = dict(frame_index=int(frame_idx), width_um=width_um, tile_origin_px=(ix * tile_px, iy * tile_px),
                            score=float(score[iy, ix]), interest=float(win_interest[iy, ix]),
                            dark_fraction=float(dark_frac[iy, ix]))

    width_px = best["width_um"] / mpp
    height_px = width_px / CUTOUT_RATIO
    x0 = min(float(best["tile_origin_px"][0]), (w - 1) - width_px)
    y0 = min(float(best["tile_origin_px"][1]), (h - 1) - height_px)
    best["cutout_bounds"] = (x0, x0 + width_px, y0, y0 + height_px)
    best.update(n_frames_evaluated=int(frames.size), n_blurry=int(is_blurry.sum()), tile_um=tile_um)
    return best


# ============================================================================
# Rendering
# ============================================================================

def _render(ax, frame: np.ndarray, bounds: tuple, mpp: float) -> None:
    """Gray PowerNorm display as in trajectory_plotter.render_frame_on_ax, but
    with the display range taken from the cutout's own percentiles instead of
    [min, 2 * max] of the full frame -- the latter leaves these hydrogel
    movies (high, uneven background) nearly uniformly gray. Negative values
    (DoG output) are clipped to 0 as there."""
    x0, x1, y0, y1 = bounds
    data = frame.astype(np.float32).clip(min=0)
    crop = data[int(y0):int(np.ceil(y1)) + 1, int(x0):int(np.ceil(x1)) + 1]
    vmin, vmax = np.percentile(crop, DISPLAY_PERCENTILES)
    ax.imshow(data, cmap="gray", origin="upper",
              norm=colors.PowerNorm(gamma=DISPLAY_GAMMA, vmin=vmin, vmax=max(vmax, vmin + 1e-6)))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    add_scalebar(ax, frame.shape, length_um=SCALEBAR_UM, mpp=mpp)
    ax.set_axis_off()


def _dog(frame: np.ndarray, mpp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_px = DOG_RADIUS_UM / mpp
    return dog_filter(frame, (r_px, r_px))


def save_single(result: dict, out_dir: Path) -> None:
    for suffix, frame in (("frame", result["frame"]), ("DoG", result["frame_dog"])):
        fig, ax = _make_fig(CUTOUT_RATIO)
        _render(ax, frame, result["cutout_bounds"], result["mpp"])
        safe_savefig(fig, out_dir / f"{result['key']}_{suffix}.png", dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def save_before_after(before: dict, after: dict, out_dir: Path) -> None:
    w = 7.15
    for suffix, frame_key in (("", "frame"), ("_DoG", "frame_dog")):
        fig, axes = plt.subplots(1, 2, figsize=(2 * w, w / CUTOUT_RATIO + 0.6))
        for ax, res in zip(axes, (before, after)):
            _render(ax, res[frame_key], res["cutout_bounds"], res["mpp"])
            ax.set_title(res["label"], fontsize=15)
        fig.tight_layout(w_pad=1.0)
        safe_savefig(fig, out_dir / f"50nm_vorher_nachher{suffix}.png", dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


# ============================================================================
# Text summary
# ============================================================================

def _fmt(value, fmt: str = "{}", missing: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return missing
    return fmt.format(value)


def write_summary(results: list[dict], path: Path) -> None:
    lines = [
        "Hydrogel 20 mg/mL C16 -- repraesentative Einzelbilder",
        "Erstellt von Hydrogel_Best_Frames.py",
        "",
        "Auswahlkriterium: Score = mittlere Helligkeit/Struktur * Dunkel-Faktor, bewertet auf der",
        f"+-1-Frame-Mittelung in Kacheln von ca. {TILE_UM} um. Helligkeit/Struktur: Anteil {1 - STRUCTURE_WEIGHT:.2f}",
        f"normierte Kachelhelligkeit + Anteil {STRUCTURE_WEIGHT:.2f} feine Struktur (Bild minus Hintergrund,",
        f"sigma {BACKGROUND_SIGMA_UM} um). Dunkle Kachel: Mittelwert < {DARK_LEVEL} der Intensitaetsspanne.",
        f"Dunkel-Faktor = 1 bei {DARK_MIN:.0%}-{DARK_MAX:.0%} dunklen Kacheln, {1 - DARK_BONUS:.2f} ohne dunkle Kachel,",
        "faellt gegen 0 bei grossem dunklen Anteil.",
        "Hinweis: Das Aufnahmedatum stammt aus der .rec-Datei (Kamerauhr).",
        f"Darstellung: PowerNorm gamma {DISPLAY_GAMMA}, Bereich = Perzentile {DISPLAY_PERCENTILES} des Ausschnitts.",
        f"Ausschnitt: Breite:Hoehe = {CUTOUT_RATIO}:1, Breite >= {MIN_WIDTH_UM:.0f} um, Scalebar {SCALEBAR_UM:.0f} um.",
        f"DoG: snr_signal_profile.dog_filter, Radius {DOG_RADIUS_UM} um.",
        "",
    ]
    for r in results:
        rec_meta, rec_info = r["rec_meta"], r["rec_info"]
        x0, x1, y0, y1 = r["cutout_bounds"]
        mpp = r["mpp"]
        rec_line = str(r["rec_path"]) if r["rec_path"] else "nicht gefunden"
        if r["rec_path"] and r["rec_assigned"]:
            rec_line += "  (manuell zugeordnet, Dateiname weicht vom TIFF ab)"
        lines += [
            "=" * 78,
            f"{r['key']}  --  {r['size_nm']} nm  --  {r['label']}",
            "=" * 78,
            f"TIFF:              {r['tif']}",
            f"REC:               {rec_line}",
            f"Aufnahme:          {_fmt(rec_meta.get('recorded_at'))}",
            f"Bildgroesse:       {r['shape'][2]} x {r['shape'][1]} px, {r['shape'][0]} Frames",
            f"Pixelgroesse:      {mpp:.4f} um/px ({r['mpp_source']})",
            f"Bildrate:          {_fmt(r['fps'], '{:.2f}')} fps, Belichtung {_fmt(rec_info.get('exposure_ms'), '{:.3f}')} ms",
            f"Laserleistung:     {_fmt(rec_meta.get('laser_power_mW'), '{:.3g}')} mW",
            f"Tiefe:             {_fmt(rec_meta.get('depth_um'), '{:.0f}')} um",
            f"Gewaehlter Frame:  {r['frame_index']} (Mittel aus Frames {max(r['frame_index'] - 1, 0)}-"
            f"{min(r['frame_index'] + 1, r['shape'][0] - 1)}), t = {_fmt(r['frame_index'] / r['fps'] if r['fps'] else None, '{:.2f}')} s",
            f"Ausschnitt (px):   x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f}",
            f"Ausschnitt (um):   {(x1 - x0) * mpp:.1f} x {(y1 - y0) * mpp:.1f} um "
            f"(Gesamtbild {r['shape'][2] * mpp:.1f} x {r['shape'][1] * mpp:.1f} um)",
            f"Score:             {r['score']:.3f} (Helligkeit/Struktur {r['interest']:.3f}, Dunkelanteil {r['dark_fraction']:.3f})",
            f"Bewertet:          {r['n_frames_evaluated']} scharfe Frames ({r['n_blurry']} als unscharf verworfen)",
            f"DoG-Sigmas (px):   {r['dog_sigmas'][0]:.2f} / {r['dog_sigmas'][1]:.2f}",
            f"Dateien:           {r['key']}_frame.png, {r['key']}_DoG.png",
            "Kommentar (.rec):",
        ]
        comment = r["comment"] or "(kein Kommentar)"
        lines += [f"    {ln}" for ln in comment.splitlines()]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# Main
# ============================================================================

def process_image(entry: dict) -> dict:
    tif = Path(entry["tif"])
    rec_path, rec_assigned = _resolve_rec(entry)
    rec_info = parse_rec_file(rec_path) if rec_path else {}
    rec_meta = parse_rec_comment_metadata(rec_path)

    stack = load_tiff_stack(tif)
    mpp, mpp_source = rec_info.get("mpp"), ".rec"
    if not mpp:
        mpp, mpp_source = MPP_BY_IMAGE_WIDTH_PX[stack.shape[2]], "Bildbreite, keine .rec"

    sel = select_frame_and_cutout(stack, tif, mpp)
    frame = average_frames(stack, sel["frame_index"])
    frame_dog, small, large = _dog(frame, mpp)
    return dict(entry, **sel, tif=tif, rec_path=rec_path, rec_assigned=rec_assigned, rec_info=rec_info,
                rec_meta=rec_meta, comment=_rec_comment(rec_path), mpp=mpp, mpp_source=mpp_source,
                fps=rec_info.get("fps"), shape=stack.shape, frame=frame, frame_dog=frame_dog,
                dog_sigmas=(float(np.mean(small)), float(np.mean(large))))


def main() -> None:
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for entry in IMAGES:
        print(f"{entry['key']}: {Path(entry['tif']).name}")
        res = process_image(entry)
        save_single(res, SAVE_PATH)
        x0, x1, _, _ = res["cutout_bounds"]
        print(f"  frame {res['frame_index']}, Breite {(x1 - x0) * res['mpp']:.1f} um, score {res['score']:.3f}")
        results[entry["key"]] = res

    before, after = (results[k] for k in BEFORE_AFTER)
    save_before_after(before, after, SAVE_PATH)
    write_summary(list(results.values()), SAVE_PATH / "Bildinformationen.txt")
    print(f"Gespeichert in {SAVE_PATH}")


if __name__ == "__main__":
    main()
