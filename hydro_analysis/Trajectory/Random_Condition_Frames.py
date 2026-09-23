"""
Random example-frame figures, one set per experimental condition combination.

Conditions come from the two established per-file caches (the "D0" free-
diffusion/water dataset and the "Deff" hydrogel dataset, per the user's own
D0/Deff terminology):
  - water:    cache/msd_d0_results.pkl   (written by MSD_FromTrackmate_D0.py)
              grouped by particle_size_nm only -- water filenames carry no
              A/B chamber code, so there is no loading-type/injection-spot
              split for this medium.
  - hydrogel: cache/msd_20mg_files.pkl   (written by MSD_FromTrackmate_20mg.py,
              "20mg C16" root, per the user's own path pointer)
              grouped by (particle_size_nm, loading_type, injection_spot),
              parsed per file from its base_name via core.io's
              condition_label_from_filename() ("A"->"Surface loading",
              "B"->"Injection") and parse_chamber_day_repeat()['chamber']
              (e.g. "A2" -> spot "2"). On the current machine only 20/50 nm
              hydrogel data is reachable (100/200/500/1000 nm folders are
              absent/empty here) -- the script only emits combinations that
              actually have data, it does not assume all six sizes exist for
              both media.

Pure consumer of these two caches for tracks_df/mpp/fps/tif_path (never
re-pools raw trajectories or refits anything, per CLAUDE.md's compute vs.
consumer convention) -- but it DOES read the raw TIFF pixel data directly
(tifffile), since frame images are not themselves a cached artifact and
there is no way around opening the movie to render one.

Per condition, one (frame, anchor particle) example is chosen by:
  1. Shuffling that condition's candidate files (RANDOM_SEED, reproducible).
  2. Per file: keep only tracks with >= MIN_TRACK_LENGTH points, i.e. the
     user's "at least 5 trajectory length" requirement (groupby('particle')
     .size() filter, the established idiom used elsewhere in this repo,
     e.g. Trajectory/TrajectoryOverlay_SEM.py::select_random_trajectory).
     A file is only considered if it has enough such tracks to reach
     MIN_PARTICLES_IN_VIEW in the first place.
  3. Sharpness: MSD_Trackmate/Validation_Claude/Frame_Sharpness_Immobilized
     .compute_sharpness_series()/flag_blurry_frames() (variance-of-Laplacian,
     per-movie MAD threshold), unmodified, reused as-is -- a frame only
     qualifies if it is not flagged blurry.
  4. Among sharp frames, only those where >= MIN_PARTICLES_IN_VIEW qualifying
     tracks are simultaneously present are tried (user: "at least 3 in the
     cutout"). For each present particle tried as the anchor (in random
     order): visibility is a Michelson peak-vs-local-background contrast at
     its position (same formula as Visualisation/Detection_Demo_Figures.py
     ::pick_best_contrast_particle, duplicated here as a per-candidate
     scalar filter -- that function instead argmaxes over a fixed particle
     list, a different shape than the accept/reject filter needed here).
     A file is tried once requiring MIN_CONTRAST and, only if that yields
     nothing anywhere in the file, once more without the contrast
     requirement (contrast is a preference, not allowed to make an
     otherwise-valid combination silently produce nothing).
  5. For the accepted (frame, anchor), the smallest field of view in
     [FOV_MIN_UM, FOV_MAX_UM] (user: "roughly the same" physical size across
     conditions -- see those constants below for the current range) that
     still contains >= MIN_PARTICLES_IN_VIEW qualifying tracks, centered on
     the anchor, is used as the cutout -- see _fov_for_min_count(). If even
     FOV_MAX_UM cannot fit 3, that anchor/frame is rejected and the next is
     tried.

Rendering reuses Trajectory/trajectory_plotter.py's primitives unchanged
(load_tiff_stack, average_frames, render_frame_on_ax, render_tracks_on_ax,
_make_fig, _history_frames_from_fps, D_MAX) -- see that module's own
docstring for why each exists; nothing there is edited. render_tracks_on_ax
is always called with the FILE'S FULL tracks_df, so every qualifying
particle visible in the cutout gets a trajectory drawn the same way (same
blue in v3, same D colormap in v4) as the anchor -- there is no
special-casing of "the one selected particle" in the rendering step, only
in candidate selection.

v4's trail is colored by diffusion coefficient (user: "the colored trail
should be in the color of the Diffusion coeff"): render_tracks_on_ax's
own color_mode="velocity" (one color per whole track, from its own
whole-track-averaged fitted D via core.analysis.calculate_step_sizes /
fit_gaussian_diffusion_1d / diffusion_2d_from_1d_fits, jet colormap,
log-scaled 0..D_MAX), unchanged -- an earlier version of this script instead
colored each trail SEGMENT by local instantaneous speed, which the user has
now corrected back to trajectory_plotter's existing per-track D convention.
The colorbar (LogNorm(1, 1+D_MAX), same tick values [0.4,0.8,2,4,8,13]) is
copied from trajectory_plotter.save_overview_velocity/save_comparison_grid
for the same look.

Particle markers (v2-v4) are drawn as true-to-scale circles at TrackMate's
own detection radius for that movie (user: "the particle detection circle
should be in the size of the trackmate tracking pixels"), not the fixed
on-screen marker trajectory_plotter.highlight_current_particles() uses.
The raw Tracks.xml exported by this pipeline carries no per-spot radius
(only frame/particle/x/y), so the radius is read from the matching raw
TrackMate *session* XML's Settings/DetectorSettings/RADIUS (the same
per-movie detector setting PSF_Analysis.py's read_detector_values() reads
for its own, unrelated purpose, reused unchanged here) -- see
_find_analysis_xml()/_detection_radius_px(). That session file is located
by scanning sibling folders whose name contains "analy" (matches both the
hydrogel dataset's "analysis" and the water dataset's "Trackmate_Analyses"
folder naming) for one whose own Settings/ImageData references the same
source .tif (Compare_Tracks_vs_RawAnalysis._read_image_data(), unchanged).
If no match is found, falls back to trajectory_plotter's fixed-size marker
and prints a note -- never fabricates a radius.

Crop: a 1 : 1.42 width/height cutout (user requirement), centered on the
anchor particle, sized per point 5 above (FOV_MIN_UM-FOV_MAX_UM). Trajectory
history shown in v3/v4 is capped at TRAIL_HISTORY_SECONDS (not "whole track
since start" -- tracks here can run to 1000+ frames, and an uncapped window
would draw a trail far outside the fixed-size cutout).

Four PNGs per condition subfolder (own folder under SAVE_PATH, one
subfolder per condition, PNG, 600 dpi), each named after its condition
(user: "save the picture with said condition, like 20_hyd_inj_trj_hist.png")
via _variant_filename(): "{size_nm}_{wat|hyd}[_{inj|surf}]_{variant}.png"
-- injection_spot is not repeated in the filename since the subfolder
already encodes it and multiple spots would otherwise collide:
  {..}_frame.png     -- cutout + scalebar only
  {..}_loc.png        -- + current-frame particle position marker(s)
  {..}_trj_hist.png   -- + trajectory history (blue, fading)
  {..}_trj_D.png       -- + trajectory history colored by diffusion coefficient
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle

from hydro_analysis.core.io import condition_label_from_filename, parse_chamber_day_repeat, find_rec_tif_files
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Frame_Sharpness_Immobilized import (
    compute_sharpness_series, flag_blurry_frames,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Compare_Tracks_vs_RawAnalysis import _read_image_data
from hydro_analysis.MSD_Trackmate.Validation_Claude.PSF_Analysis import read_detector_values
from hydro_analysis.Trajectory.trajectory_plotter import (
    load_tiff_stack, average_frames, render_frame_on_ax, render_tracks_on_ax,
    highlight_current_particles, _make_fig, D_MAX,
)

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_MSD_D0 = Path(__file__).parent.parent / "MSD_Trackmate" / "cache" / "msd_d0_results.pkl"
CACHE_20MG = Path(__file__).parent.parent / "MSD_Trackmate" / "cache" / "msd_20mg_files.pkl"

SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Random_Condition_Frames"

RANDOM_SEED = 42
MIN_TRACK_LENGTH = 5          # user requirement: "at least 5 trajectory length"
MIN_HISTORY_INDEX = 3         # prefer anchors with >=3 prior points already visible for a trail
MIN_CONTRAST = 0.10           # Michelson contrast floor for the anchor; dropped file-wide if nothing clears it

CUTOUT_RATIO = 1.42           # user requirement: 1 x 1.42 cutout
MIN_PARTICLES_IN_VIEW = 3     # user requirement: "at least 3 in the cutout"
FOV_MIN_UM = 15.0             # user requirement: "roughly the same, about 15 to 22 um"
FOV_MAX_UM = 35.0
TRAIL_HISTORY_SECONDS = 5.0   # trajectory-history window shown in v3/v4;
                               # capped (not "whole track since start"), since tracks here can run to
                               # 1000+ frames and a full-track window would draw a trail far outside
                               # the fixed-size cutout
DEFAULT_MARKER_RADIUS_PX = 4.0  # fallback circle radius only when no matching session XML is found
SAVE_DPI = 600


# ============================================================================
# Condition grouping
# ============================================================================

def _water_condition_key(rd: dict) -> tuple:
    return ("water", rd.get("particle_size_nm"), None, None)


def _hydrogel_condition_key(rd: dict) -> tuple:
    base_name = rd.get("base_name", "")
    loading = condition_label_from_filename(base_name)          # "Surface loading" / "Injection" / None
    chamber = parse_chamber_day_repeat(base_name).get("chamber")  # e.g. "A2"
    spot = chamber[1:] if chamber else None
    return ("hydrogel", rd.get("particle_size_nm"), loading, spot)


def build_condition_groups(water_cache: dict, hydro_cache: dict) -> dict[tuple, list[dict]]:
    """{(medium, particle_size_nm, loading_type, injection_spot): [result_dict, ...]}."""
    groups: dict[tuple, list[dict]] = {}
    for rd in water_cache.values():
        key = _water_condition_key(rd)
        if key[1] is None:
            continue
        groups.setdefault(key, []).append(rd)
    for rd in hydro_cache.values():
        key = _hydrogel_condition_key(rd)
        if key[1] is None or key[2] is None or key[3] is None:
            continue
        groups.setdefault(key, []).append(rd)
    return groups


def _condition_folder_name(key: tuple) -> str:
    medium, size_nm, loading, spot = key
    size_label = f"{int(size_nm)}nm"
    if medium == "water":
        return f"water_{size_label}"
    loading_slug = (loading or "unknown").replace(" ", "")
    return f"hydrogel_{size_label}_{loading_slug}_spot{spot}"


_LOADING_ABBREV = {"Surface loading": "surf", "Injection": "inj"}


def _variant_filename(key: tuple, variant_slug: str) -> str:
    """Compact condition-coded filename, e.g. "20_hyd_inj_trj_hist.png" (user
    example) or "1000_wat_frame.png". Injection_spot is deliberately left out
    -- the subfolder (_condition_folder_name) already disambiguates it, and
    repeating it here would make an already-long name longer for no benefit."""
    medium, size_nm, loading, _spot = key
    parts = [str(int(size_nm))]
    if medium == "water":
        parts.append("wat")
    else:
        parts.append("hyd")
        parts.append(_LOADING_ABBREV.get(loading, "unk"))
    parts.append(variant_slug)
    return "_".join(parts) + ".png"


# ============================================================================
# Candidate selection: sharp frame, >=3 visible tracks (length >= MIN_TRACK_LENGTH)
# ============================================================================

def _michelson_contrast(frame: np.ndarray, x0: float, y0: float, inner: int = 3, outer: int = 8) -> float:
    """Peak-vs-local-background Michelson contrast at (x0, y0); same formula as
    Visualisation/Detection_Demo_Figures.py::pick_best_contrast_particle,
    duplicated here as a per-candidate scalar (that function argmaxes over a
    fixed particle_ids list, a different shape than the filter needed here)."""
    h, w = frame.shape[:2]
    xi, yi = int(round(x0)), int(round(y0))
    x_lo, x_hi = max(xi - outer, 0), min(xi + outer + 1, w)
    y_lo, y_hi = max(yi - outer, 0), min(yi + outer + 1, h)
    patch = frame[y_lo:y_hi, x_lo:x_hi]
    if patch.size == 0:
        return -np.inf
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    dist = np.hypot(xx - xi, yy - yi)
    inner_mask, outer_mask = dist <= inner, (dist > inner) & (dist <= outer)
    if not inner_mask.any() or not outer_mask.any():
        return -np.inf
    peak = float(patch[inner_mask].max())
    background = float(np.median(patch[outer_mask]))
    return (peak - background) / (peak + background + 1e-6)


def _fov_for_min_count(anchor_xy: tuple[float, float], positions_xy: list[tuple[float, float]],
                        mpp: float, ratio: float, min_count: int,
                        fov_min_um: float, fov_max_um: float) -> Optional[float]:
    """Smallest FOV width (um, clipped to >= fov_min_um) whose 1:ratio box
    centered on anchor_xy contains >= min_count of positions_xy (anchor
    itself included, at distance 0). None if even fov_max_um is not enough."""
    ax0, ay0 = anchor_xy
    needed = []
    for x, y in positions_xy:
        dx_um = abs(x - ax0) * mpp
        dy_um = abs(y - ay0) * mpp
        # box half-width = fov/2, half-height = fov/(2*ratio); point inside <=> fov >= max(2dx, 2*ratio*dy)
        needed.append(max(2.0 * dx_um, 2.0 * ratio * dy_um))
    needed.sort()
    if len(needed) < min_count:
        return None
    fov = max(fov_min_um, needed[min_count - 1])
    return fov if fov <= fov_max_um else None


def _search_file(rd: dict, tracks_df, tif_path: Path, mpp: float, fps: float,
                  rng: random.Random, require_contrast: bool) -> Optional[dict]:
    lengths = tracks_df.groupby("particle").size()
    valid_particles = set(lengths[lengths >= MIN_TRACK_LENGTH].index.tolist())
    if len(valid_particles) < MIN_PARTICLES_IN_VIEW:
        return None

    scores = compute_sharpness_series(tif_path)
    is_blurry, _ = flag_blurry_frames(scores)
    sharp_frames = set(np.flatnonzero(~is_blurry).tolist())
    if not sharp_frames:
        return None

    sub = tracks_df[tracks_df["particle"].isin(valid_particles)]
    frames_by_pid = {pid: g.sort_values("frame")["frame"].to_numpy() for pid, g in sub.groupby("particle")}
    frame_list = sorted(set(sub["frame"].unique().tolist()) & sharp_frames)
    if not frame_list:
        return None
    rng.shuffle(frame_list)

    stack = None
    for frame_idx in frame_list:
        grp = sub[sub["frame"] == frame_idx]
        if len(grp) < MIN_PARTICLES_IN_VIEW:
            continue
        positions = list(zip(grp["particle"].tolist(), grp["x"].tolist(), grp["y"].tolist()))
        if stack is None:
            stack = load_tiff_stack(tif_path)
        if frame_idx >= stack.shape[0]:
            continue
        frame_img = stack[frame_idx]

        anchors = list(positions)
        rng.shuffle(anchors)
        for pid, x, y in anchors:
            pid_frames = frames_by_pid[pid]
            idx_in_track = int(np.searchsorted(pid_frames, frame_idx))
            is_last = idx_in_track == len(pid_frames) - 1
            if not (idx_in_track >= MIN_HISTORY_INDEX or is_last):
                continue
            contrast = _michelson_contrast(frame_img, x, y)
            if require_contrast and contrast < MIN_CONTRAST:
                continue
            fov_um = _fov_for_min_count((x, y), [(px, py) for _, px, py in positions], mpp,
                                         CUTOUT_RATIO, MIN_PARTICLES_IN_VIEW, FOV_MIN_UM, FOV_MAX_UM)
            if fov_um is None:
                continue

            track_start = int(pid_frames[0])
            available_seconds = max((frame_idx - track_start) / fps, 1.0 / fps)
            history_seconds = min(available_seconds, TRAIL_HISTORY_SECONDS)

            return dict(result_dict=rd, particle_id=int(pid), frame_index=int(frame_idx), stack=stack,
                        contrast=contrast, track_length=int(lengths[pid]), history_seconds=history_seconds,
                        fov_um=fov_um, anchor_xy=(float(x), float(y)), n_particles_in_view=len(positions))
    return None


def select_candidate(result_dicts: list[dict], rng: random.Random) -> Optional[dict]:
    """Try each file (shuffled), each once requiring the anchor's Michelson
    contrast to clear MIN_CONTRAST and, only if nothing in the whole file
    qualifies that way, once more without that requirement. Returns a dict
    with keys: result_dict, particle_id, frame_index, stack, contrast,
    track_length, history_seconds, fov_um, anchor_xy, n_particles_in_view --
    or None if nothing in this condition's files qualifies."""
    files = list(result_dicts)
    rng.shuffle(files)

    prepared = []
    for rd in files:
        tracks_df = rd.get("tracks_df")
        tif_path_str = rd.get("tif_path")
        mpp, fps = rd.get("mpp"), rd.get("fps")
        if tracks_df is None or tracks_df.empty or not tif_path_str:
            continue
        if not isinstance(mpp, (int, float)) or mpp <= 0 or not isinstance(fps, (int, float)) or fps <= 0:
            continue
        tif_path = Path(tif_path_str)
        if not tif_path.exists():
            continue
        prepared.append((rd, tracks_df, tif_path, mpp, fps))

    for require_contrast in (True, False):
        for rd, tracks_df, tif_path, mpp, fps in prepared:
            result = _search_file(rd, tracks_df, tif_path, mpp, fps, rng, require_contrast)
            if result is not None:
                if not require_contrast:
                    print(f"    [NOTE] no anchor cleared MIN_CONTRAST={MIN_CONTRAST} anywhere in this "
                          "condition; using the best available sharp candidate instead.")
                return result
    return None


# ============================================================================
# Cutout geometry
# ============================================================================

def build_cutout_bounds(anchor_xy: tuple[float, float], mpp: float,
                         fov_um: float) -> tuple[float, float, float, float]:
    """1:CUTOUT_RATIO box of physical width fov_um (user: 15-22 um), centered
    on the anchor particle's own position."""
    x0, y0 = anchor_xy
    half_w_px = (fov_um / 2.0) / mpp
    half_h_px = half_w_px / CUTOUT_RATIO
    return (x0 - half_w_px, x0 + half_w_px, y0 - half_h_px, y0 + half_h_px)


# ============================================================================
# Particle detection-radius lookup (for true-to-scale markers)
# ============================================================================

def _find_analysis_xml(tracks_xml: Path) -> Optional[Path]:
    """Find the raw TrackMate session XML matching tracks_xml's own source
    .tif, by scanning sibling folders whose name suggests an analysis/session
    folder ("analy" case-insensitive substring -- matches both the hydrogel
    dataset's "analysis" and the water dataset's "Trackmate_Analyses") for
    one whose own Settings/ImageData references the same filename."""
    calib = find_rec_tif_files(tracks_xml)
    tif_name = calib["tiff_file"].name if calib.get("tiff_file") else None
    if tif_name is None:
        return None
    search_root = tracks_xml.parent.parent
    if not search_root.exists():
        return None
    candidate_dirs = [d for d in search_root.iterdir() if d.is_dir() and "analy" in d.name.lower()]
    for d in candidate_dirs:
        for xml_path in d.glob("*.xml"):
            try:
                info = _read_image_data(xml_path)
            except Exception:
                continue
            if info.get("filename") == tif_name:
                return xml_path
    return None


def _detection_radius_px(tracks_xml: Path) -> Optional[float]:
    """TrackMate's own configured detection radius (px) for this movie, from
    the matching raw session XML's Settings/DetectorSettings/RADIUS (read via
    PSF_Analysis.read_detector_values, unchanged). None if no matching
    session XML is found or it cannot be parsed."""
    analysis_xml = _find_analysis_xml(tracks_xml)
    if analysis_xml is None:
        return None
    try:
        values = read_detector_values(analysis_xml)
    except Exception:
        return None
    rx, ry = values.get("detector_radius_x_px"), values.get("detector_radius_y_px")
    if not (isinstance(rx, (int, float)) and isinstance(ry, (int, float)) and np.isfinite(rx) and np.isfinite(ry)):
        return None
    return float((rx + ry) / 2.0)


def draw_particle_markers(ax, tracks_df, frame_index: int, radius_px: Optional[float]) -> None:
    """Circle at TrackMate's own detection radius (data units = raw image
    pixels) per particle present at frame_index, so the ring's size on the
    cutout accurately reflects TrackMate's own tracking pixel size and
    scales correctly with the cutout's zoom level -- falls back to
    trajectory_plotter.highlight_current_particles' fixed on-screen marker
    only when no radius could be determined."""
    if tracks_df is None or tracks_df.empty:
        return
    if radius_px is None:
        highlight_current_particles(ax, tracks_df, frame_index)
        return
    current = tracks_df[tracks_df["frame"] == frame_index][["x", "y"]]
    for _, row in current.iterrows():
        ax.add_patch(Circle((row["x"], row["y"]), radius=radius_px, fill=False,
                             edgecolor="hotpink", linewidth=1.5, zorder=5))


# ============================================================================
# Save the 4 variants for one condition
# ============================================================================

def save_condition_variants(candidate: dict, out_dir: Path, key: tuple) -> None:
    rd = candidate["result_dict"]
    tracks_df = rd["tracks_df"]
    mpp, fps = rd["mpp"], rd["fps"]
    frame_index = candidate["frame_index"]
    history_seconds = candidate["history_seconds"]
    stack = candidate["stack"]

    frame_avg = average_frames(stack, frame_index)
    cutout_bounds = build_cutout_bounds(candidate["anchor_xy"], mpp, candidate["fov_um"])
    radius_px = _detection_radius_px(Path(rd["xml_path"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    # v1: cutout + scalebar only
    fig, ax = _make_fig(CUTOUT_RATIO)
    render_frame_on_ax(ax, frame_avg, cutout_bounds=cutout_bounds, mpp=mpp)
    safe_savefig(fig, out_dir / _variant_filename(key, "frame"), dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # v2: + current-frame particle location marker(s), sized to TrackMate's own detection radius
    fig, ax = _make_fig(CUTOUT_RATIO)
    render_frame_on_ax(ax, frame_avg, cutout_bounds=cutout_bounds, mpp=mpp)
    draw_particle_markers(ax, tracks_df, frame_index, radius_px)
    safe_savefig(fig, out_dir / _variant_filename(key, "loc"), dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # v3: + trajectory history (blue, fading), all particles colored the same way
    fig, ax = _make_fig(CUTOUT_RATIO)
    render_frame_on_ax(ax, frame_avg, cutout_bounds=cutout_bounds, mpp=mpp)
    render_tracks_on_ax(ax, tracks_df, fps, mpp, frame_index, color_mode="blue",
                         history_seconds=history_seconds)
    draw_particle_markers(ax, tracks_df, frame_index, radius_px)
    safe_savefig(fig, out_dir / _variant_filename(key, "trj_hist"), dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # v4: + trajectory history colored by diffusion coefficient (D), all particles colored the same way
    fig, ax = _make_fig(CUTOUT_RATIO)
    render_frame_on_ax(ax, frame_avg, cutout_bounds=cutout_bounds, mpp=mpp)
    render_tracks_on_ax(ax, tracks_df, fps, mpp, frame_index, color_mode="velocity",
                         history_seconds=history_seconds)
    draw_particle_markers(ax, tracks_df, frame_index, radius_px)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=colors.LogNorm(vmin=1, vmax=1 + D_MAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$D$ ($\mathrm{\mu m^2/s}$)", fontsize=9)
    tick_vals = [0.4, 0.8, 2, 4, 8, 13]
    cbar.set_ticks([1 + v for v in tick_vals])
    cbar.set_ticklabels([str(v) for v in tick_vals])
    cbar.ax.tick_params(labelsize=8)
    safe_savefig(fig, out_dir / _variant_filename(key, "trj_D"), dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    if not CACHE_MSD_D0.exists():
        raise FileNotFoundError(f"Nicht gefunden: {CACHE_MSD_D0}\nBitte zuerst MSD_FromTrackmate_D0.py ausfuehren.")
    if not CACHE_20MG.exists():
        raise FileNotFoundError(f"Nicht gefunden: {CACHE_20MG}\nBitte zuerst MSD_FromTrackmate_20mg.py ausfuehren.")

    with open(CACHE_MSD_D0, "rb") as f:
        water_cache = pickle.load(f)
    with open(CACHE_20MG, "rb") as f:
        hydro_cache = pickle.load(f)

    groups = build_condition_groups(water_cache, hydro_cache)
    print(f"{len(groups)} condition combinations found "
          f"({sum(1 for k in groups if k[0] == 'water')} water, "
          f"{sum(1 for k in groups if k[0] == 'hydrogel')} hydrogel).")

    rng = random.Random(RANDOM_SEED)
    n_ok, n_failed = 0, 0

    for key in sorted(groups.keys(), key=lambda k: (k[0], k[1] or 0, k[2] or "", k[3] or "")):
        folder_name = _condition_folder_name(key)
        print(f"\n=== {folder_name} ({len(groups[key])} file(s)) ===")
        candidate = select_candidate(groups[key], rng)
        if candidate is None:
            print(f"  [SKIP] no qualifying sharp frame with >= {MIN_PARTICLES_IN_VIEW} tracks "
                  f"(length >= {MIN_TRACK_LENGTH}) fitting within {FOV_MAX_UM} um.")
            n_failed += 1
            continue

        rd = candidate["result_dict"]
        print(f"  file={rd.get('base_name')} particle={candidate['particle_id']} "
              f"frame={candidate['frame_index']} track_length={candidate['track_length']} "
              f"contrast={candidate['contrast']:.3f} fov={candidate['fov_um']:.1f}um "
              f"n_particles_in_view={candidate['n_particles_in_view']}")

        out_dir = SAVE_PATH / folder_name
        save_condition_variants(candidate, out_dir, key)
        print(f"  [saved] {out_dir}")
        n_ok += 1

    print(f"\nDone: {n_ok} condition(s) rendered, {n_failed} skipped.")


if __name__ == "__main__":
    main()
