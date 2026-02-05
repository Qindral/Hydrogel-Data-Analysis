"""
Trajectory Plotter

- Find 500 nm XML inside Tracks/ folders
- Load XML + REC + TIFF via core.io
- Show frame with tracks colored by D
"""

from pathlib import Path
from typing import Optional
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import font_manager as fm
import numpy as np
import tifffile
from matplotlib_scalebar.scalebar import ScaleBar

from core.analysis import calculate_step_sizes, diffusion_2d_from_1d_fits, fit_gaussian_diffusion_1d
from core.io import extract_particle_size_from_path, single_file_data


ROOT_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Trajectory_Visualisation")

df_cutout = pd.read_csv(r"E:\PhD Data Analysis\SPT 2025 II\Trajectory_Visualisation\Cutout.txt") #<-- Path to frame and cutout
# print(df)

# for idx, row in df.iterrows():
#     print('width/height',(row["cutout1_p1"]-row["cutout2_p1"])/(row["cutout1_p2"]-row["cutout2_p2"]))
DEFAULT_FRAME_INDEX = 277
REQUIRED_CUTOUT_COLUMNS = (
    "particle_nm",
    "frame",
    "cutout1_p1",
    "cutout1_p2",
    "cutout2_p1",
    "cutout2_p2",
)
CUTOUT_WIDTH_OVER_HEIGHT = 1.4
TRACK_HISTORY_SECONDS = 0.5
SCALEBAR_UM = 5.0

def load_tiff_stack(tif_path: Path) -> np.ndarray:
    """Load TIFF as (T, Y, X) or (T, Z/Y, X) stack."""
    with tifffile.TiffFile(tif_path) as tif:
        stack = tif.asarray()

    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    elif stack.ndim == 4:
        # Assume CTYX or CZYX and take first channel/Z
        stack = stack[0, ...]

    return stack


def find_xmls_in_tracks(root: Path) -> list[Path]:
    """Find all XMLs inside Tracks/ folders."""
    return sorted(root.rglob("Tracks/*.xml"))


def show_frame(result_dict, frame_index: int, cutout_bounds=None) -> None:
    tif_path = Path(result_dict.get("tif_path")) if result_dict.get("tif_path") else None
    if tif_path is None or not tif_path.exists():
        print("[!] No TIFF path found from core.io for this XML.")
        return

    fps = result_dict.get("fps")
    mpp = result_dict.get("mpp")
    tracks_df = result_dict.get("tracks_df")

    stack = load_tiff_stack(tif_path)
    print(f"[OK] Stack shape: {stack.shape}")

    if frame_index < 0 or frame_index >= stack.shape[0]:
        print(f"[!] Frame {frame_index} out of range (0..{stack.shape[0] - 1})")
        return

    fig = plt.figure(figsize=(6, 6))
    title_fps = f"{fps:.2f}" if isinstance(fps, (int, float)) else "n/a"
    title_mpp = f"{mpp:.3f}" if isinstance(mpp, (int, float)) else "n/a"
    window_title = f"{tif_path.name} | frame {frame_index} | mpp={title_mpp} | fps={title_fps}"
    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass
    frame = stack[frame_index]
    frame_display = (frame.astype(np.float32)).clip(min=0)
    ax = plt.gca()
    frame_min, frame_max = np.min(frame_display), np.max(frame_display)
    
    ax.imshow(
        frame_display,
        cmap="gray",
        norm=colors.PowerNorm(gamma=0.6, vmin=frame_min, vmax=frame_max * 2),
        origin="upper",
    )

    # Overlay tracks colored by diffusion constant (D)
    if tracks_df is None or tracks_df.empty:
        print("[!] Tracks data is empty; skipping tracks overlay.")
    else:
        history_frames = _history_frames_from_fps(fps, TRACK_HISTORY_SECONDS)
        overlay_tracks_by_diffusion(
            ax,
            tracks_df,
            fps=fps,
            mpp=mpp,
            d_max=15.0,
            frame_index=frame_index,
            history=history_frames,
            alpha_now=0.96,
            alpha_past=0.07,
            linewidth=3.0,
        )
        highlight_current_particles(ax, tracks_df, frame_index=frame_index)
        if cutout_bounds is not None:
            x_min, x_max, y_min, y_max = _enforce_cutout_ratio(
                cutout_bounds, width_over_height=CUTOUT_WIDTH_OVER_HEIGHT
            )
            height, width = frame.shape[:2]
            x_min, x_max = _shift_window_to_bounds(x_min, x_max, 0.0, float(width - 1))
            y_min, y_max = _clamp_vertical_lower_only(y_min, y_max, 0.0, float(height - 1))
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)  # origin="upper"
            print(
                f"[OK] Cutout view: x=({x_min:.1f},{x_max:.1f}) "
                f"y=({y_min:.1f},{y_max:.1f}) frame={frame_index}"
            )
        else:
            crop = compute_crop_window(
                tracks_df,
                frame_index=frame_index,
                image_shape=frame.shape,
                target_count=12,
                aspect=1.7,
                margin_px=5,
            )
            if crop is not None:
                x_min, x_max, y_min, y_max, count = crop
                if isinstance(mpp, (int, float)) and mpp > 0:
                    pad_px = 5.0 / mpp
                    x_min -= pad_px
                    x_max += pad_px
                    y_min -= pad_px
                    y_max += pad_px
                    height, width = frame.shape[:2]
                    x_min, x_max = _shift_window_to_bounds(x_min, x_max, 0.0, float(width - 1))
                    y_min, y_max = _shift_window_to_bounds(y_min, y_max, 0.0, float(height - 1))
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_max, y_min)  # origin="upper"
                print(f"[OK] Cropped view: {count} particles in frame {frame_index}")

    add_scalebar(ax, frame.shape, length_um=SCALEBAR_UM, mpp=mpp)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def add_scalebar(ax, image_shape, length_um: float, mpp: Optional[float]) -> None:
    """Draw a scalebar in the lower-right corner of the current view."""
    if not isinstance(mpp, (int, float)) or mpp <= 0:
        print("[!] mpp not available; scalebar skipped.")
        return

    # Prefer current view limits so the bar stays inside cropped view
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_min, x_max = (min(x0, x1), max(x0, x1))
    y_min, y_max = (min(y0, y1), max(y0, y1))

    width_view = x_max - x_min
    height_view = y_max - y_min
    if not np.isfinite(width_view) or not np.isfinite(height_view) or width_view <= 0 or height_view <= 0:
        height, width = image_shape[:2]
        x_min, x_max = 0.0, float(width - 1)
        y_min, y_max = 0.0, float(height - 1)
        width_view = x_max - x_min

    length_px = length_um / mpp
    pad_px = 10.0
    if length_px > width_view - 2 * pad_px:
        pad_px = 2.0
    if length_px > width_view - 2 * pad_px:
        print("[!] Scalebar does not fit in image; skipped.")
        return

    fig = ax.figure
    if fig is not None and fig.canvas is not None:
        fig.canvas.draw()
        axes_height_px = ax.bbox.height
    else:
        axes_height_px = height_view

    thickness_px = 8.0
    font_size_pt = 15.0
    width_fraction = thickness_px / axes_height_px if axes_height_px > 0 else 0.01

    # ScaleBar requires font_properties as dict or string, not FontProperties object
    try:
        available = {f.name for f in fm.fontManager.ttflist}
    except Exception:
        available = set()
    font_props_dict: dict = {"size": font_size_pt}
    if "Open Sans" in available:
        font_props_dict["family"] = "Open Sans"

    scalebar = ScaleBar(
        mpp,
        units="um",
        dimension="si-length",
        fixed_value=length_um,
        fixed_units="um",
        location="lower right",
        width_fraction=width_fraction,
        sep=2,
        frameon=True,
        color="white",
        box_color="black",
        box_alpha=0.6,
        scale_loc="top",
        label_loc="bottom",
        font_properties=font_props_dict,
    )
    ax.add_artist(scalebar)


def highlight_current_particles(ax, tracks_df, frame_index: int) -> None:
    """Mark particle positions at the current frame with a pink circle."""
    current = tracks_df[tracks_df["frame"] == frame_index][["x", "y"]]
    if current.empty:
        return
    ax.scatter(
        current["x"],
        current["y"],
        s=85,
        facecolors="none",
        edgecolors="hotpink",
        linewidths=1.50,
        zorder=5,
    )


def overlay_tracks_by_diffusion(
    ax,
    tracks_df,
    fps: Optional[float],
    mpp: Optional[float],
    d_max: float,
    frame_index: int,
    history: int,
    alpha_now: float,
    alpha_past: float,
    linewidth: float,
) -> None:
    """Plot tracks visible at frame_index, colored by D using core.analysis."""
    if not isinstance(fps, (int, float)) or fps <= 0:
        print("[!] fps missing; cannot compute diffusion constants.")
        return

    if not isinstance(mpp, (int, float)) or mpp <= 0:
        print("[!] mpp missing; diffusion constants will be in px^2/s.")
        mpp = 1.0

    cmap = plt.get_cmap("jet")

    steps_df = calculate_step_sizes(tracks_df, step_interval=1, sliding=True)

    for particle_id, group in tracks_df.groupby("particle"):
        group = group.sort_values("frame")
        x = group["x"].to_numpy()
        y = group["y"].to_numpy()
        frames = group["frame"].to_numpy()
        if len(x) < 2:
            continue

        # Only plot tracks visible in the current frame
        if frame_index not in set(frames):
            continue

        # Estimate D using core.analysis on per-particle step sizes
        step_p = steps_df[steps_df["particle"] == particle_id]
        if step_p.empty:
            continue

        fit_x = fit_gaussian_diffusion_1d(step_p["dx"], mpp, fps, frame_interval=1, axis="x")
        fit_y = fit_gaussian_diffusion_1d(step_p["dy"], mpp, fps, frame_interval=1, axis="y")
        fit_2d = diffusion_2d_from_1d_fits(fit_x, fit_y)
        d_est = fit_2d.get("D_um2_per_s", np.nan)

        d_clamped = max(0.0, min(float(d_est) if np.isfinite(d_est) else 0.0, d_max))
        log_max = np.log10(1.0 + d_max) if d_max > 0 else 1.0
        log_val = np.log10(1.0 + d_clamped)
        color = cmap((log_val / log_max) if log_max > 0 else 0.0)

        # Use only the last `history` frames up to current frame
        min_frame = frame_index - history
        mask = (frames >= min_frame) & (frames <= frame_index)
        frames = frames[mask]
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            continue

        # Plot segment-by-segment with fading
        for i in range(len(x) - 1):
            age = frame_index - frames[i + 1]
            if age < 0 or age > history:
                continue
            t = age / float(history) if history > 0 else 0.0
            alpha = alpha_now * (1.0 - t) + alpha_past * t

            # Plot twice to allow additive overlap
            ax.plot(
                [x[i], x[i + 1]],
                [y[i], y[i + 1]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle="round",
            )
            ax.plot(
                [x[i], x[i + 1]],
                [y[i], y[i + 1]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle="round",
            )


def compute_crop_window(
    tracks_df,
    frame_index: int,
    image_shape,
    target_count: int = 5,
    aspect: float = 1.7,
    margin_px: float = 5.0,
):
    """Find a crop window (x_min, x_max, y_min, y_max, count) with ~target_count particles."""
    frame_points = tracks_df[tracks_df["frame"] == frame_index][["x", "y"]]
    if frame_points.empty:
        print("[!] No particles in the requested frame; crop skipped.")
        return None

    pts = frame_points.to_numpy()
    n = pts.shape[0]
    if n < target_count:
        print(f"[!] Only {n} particles in frame; crop skipped.")
        return None

    height, width = image_shape[:2]
    best = None  # (count, area, x_min, x_max, y_min, y_max)

    for i in range(n):
        p = pts[i]
        d2 = np.sum((pts - p) ** 2, axis=1)
        idx = np.argsort(d2)[:target_count]
        sel = pts[idx]
        minx, maxx = sel[:, 0].min() - margin_px, sel[:, 0].max() + margin_px
        miny, maxy = sel[:, 1].min() - margin_px, sel[:, 1].max() + margin_px

        w = max(maxx - minx, 1.0)
        h = max(maxy - miny, 1.0)
        if w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect

        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
        x_min = cx - w / 2.0
        x_max = cx + w / 2.0
        y_min = cy - h / 2.0
        y_max = cy + h / 2.0

        x_min, x_max = _shift_window_to_bounds(x_min, x_max, 0.0, width - 1.0)
        y_min, y_max = _shift_window_to_bounds(y_min, y_max, 0.0, height - 1.0)

        count = np.sum(
            (pts[:, 0] >= x_min)
            & (pts[:, 0] <= x_max)
            & (pts[:, 1] >= y_min)
            & (pts[:, 1] <= y_max)
        )
        area = (x_max - x_min) * (y_max - y_min)

        if count >= target_count:
            key = (count, area)
            if best is None or key < (best[0], best[1]):
                best = (count, area, x_min, x_max, y_min, y_max)

    if best is None:
        return None

    _, _, x_min, x_max, y_min, y_max = best
    return (x_min, x_max, y_min, y_max, best[0])


def _shift_window_to_bounds(minv: float, maxv: float, low: float, high: float):
    """Shift [minv, maxv] into [low, high] without resizing."""
    span = maxv - minv
    if span > (high - low):
        return low, high
    if minv < low:
        maxv += (low - minv)
        minv = low
    if maxv > high:
        minv -= (maxv - high)
        maxv = high
    if minv < low:
        minv = low
    if maxv > high:
        maxv = high
    return minv, maxv


def _history_frames_from_fps(fps: Optional[float], seconds: float) -> int:
    if not isinstance(fps, (int, float)) or fps <= 0:
        return 1
    frames = int(round(float(fps) * float(seconds)))
    return max(frames, 1)


def _enforce_cutout_ratio(
    cutout_bounds: tuple[float, float, float, float],
    width_over_height: float,
) -> tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = cutout_bounds
    width = max(x_max - x_min, 1.0)
    target_height = width / float(width_over_height)
    new_y_max = y_min + target_height
    return (x_min, x_max, y_min, new_y_max)


def _clamp_vertical_lower_only(
    y_min: float, y_max: float, low: float, high: float
) -> tuple[float, float]:
    if y_min < low:
        y_min = low
    if y_max > high:
        y_max = high
    if y_max <= y_min:
        y_max = min(high, y_min + 1.0)
    return y_min, y_max


def _coerce_particle_nm(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


def _get_cutout_row(df_cutout: pd.DataFrame, particle_size_nm: Optional[float]) -> Optional[pd.Series]:
    if particle_size_nm is None:
        return None
    if "particle_nm" not in df_cutout.columns:
        print("[!] Cutout table missing 'particle_nm' column.")
        return None

    series = df_cutout["particle_nm"].apply(_coerce_particle_nm)
    diffs = (series - float(particle_size_nm)).abs()
    if diffs.isna().all():
        print(f"[!] Could not parse particle_nm values for size {particle_size_nm} nm.")
        return None
    idx = diffs.idxmin()
    if not np.isfinite(diffs.loc[idx]) or diffs.loc[idx] > 0.5:
        print(f"[!] No close cutout row found for size {particle_size_nm} nm.")
        return None
    return df_cutout.loc[idx]


def _cutout_bounds_from_row(row: pd.Series) -> Optional[tuple[float, float, float, float]]:
    try:
        x1 = float(row["cutout1_p1"])
        y1 = float(row["cutout1_p2"])
        x2 = float(row["cutout2_p1"])
        y2 = float(row["cutout2_p2"])
    except Exception:
        return None

    x_min, x_max = (min(x1, x2), max(x1, x2))
    y_min, y_max = (min(y1, y2), max(y1, y2))
    return (x_min, x_max, y_min, y_max)


def _particles_present_from_row(row: pd.Series) -> Optional[float]:
    if "particles_present" in row:
        return row["particles_present"]
    if "particle_present" in row:
        return row["particle_present"]
    return None


def _count_particles_in_cutout(tracks_df: pd.DataFrame, frame_index: int, cutout_bounds) -> int:
    if tracks_df is None or tracks_df.empty:
        return 0
    x_min, x_max, y_min, y_max = cutout_bounds
    frame_df = tracks_df[tracks_df["frame"] == frame_index]
    if frame_df.empty:
        return 0
    in_cutout = frame_df[
        (frame_df["x"] >= x_min)
        & (frame_df["x"] <= x_max)
        & (frame_df["y"] >= y_min)
        & (frame_df["y"] <= y_max)
    ]
    return int(in_cutout.shape[0])


def main() -> None:
    xml_files = find_xmls_in_tracks(ROOT_PATH)
    if not xml_files:
        print(f"[!] No XML files found under Tracks/ in: {ROOT_PATH}")
        return

    for xml_path in xml_files:
        print(f"\n[OK] Using XML: {xml_path}")
        print(extract_particle_size_from_path(xml_path))
        result = single_file_data(xml_path)
        if result is None:
            print("[!] core.io.single_file_data returned None (missing calibration?). Skipping.")
            continue
        if result.get("mpp") is None or result.get("fps") is None:
            print("[!] Missing mpp/fps from .rec. Skipping.")
            continue
        particle_nm = result.get("particle_size_nm")
        print(f"Particle size: {particle_nm} nm")
        cutout_row = _get_cutout_row(df_cutout, particle_nm)
        if cutout_row is None:
            print(f"[!] No cutout row found for {particle_nm} nm. Using default frame.")
            show_frame(result, DEFAULT_FRAME_INDEX, cutout_bounds=None)
            continue

        missing_cols = [col for col in REQUIRED_CUTOUT_COLUMNS if col not in cutout_row.index]
        if missing_cols:
            print(f"[!] Cutout row missing columns: {missing_cols}. Using default frame.")
            show_frame(result, DEFAULT_FRAME_INDEX, cutout_bounds=None)
            continue

        frame_index = int(float(cutout_row["frame"]))
        cutout_bounds = _cutout_bounds_from_row(cutout_row)
        if cutout_bounds is None:
            print(f"[!] Invalid cutout bounds for {particle_nm} nm. Using default frame.")
            show_frame(result, DEFAULT_FRAME_INDEX, cutout_bounds=None)
            continue

        cutout_bounds = _enforce_cutout_ratio(
            cutout_bounds, width_over_height=CUTOUT_WIDTH_OVER_HEIGHT
        )
        xml_count = _count_particles_in_cutout(result.get("tracks_df"), frame_index, cutout_bounds)
        expected = _particles_present_from_row(cutout_row)
        print(
            f"[OK] {Path(result['xml_path']).name} | size={particle_nm} nm | "
            f"frame={frame_index} | xml particles in cutout={xml_count} | "
            f"particles_present={expected}"
        )

        show_frame(result, frame_index, cutout_bounds=cutout_bounds)


if __name__ == "__main__":
    main()
