"""
Trajectory Plotter

- Find XMLs inside Tracks/ folders
- Load XML + REC + TIFF via core.io
- Save multiple figure variants per particle size:
  1) Frame + Scalebar only
  2) Frame + Scalebar + blue trajectories
  3) Side-by-side: Frame | Trajectories on white
  4) 6-panel overview with velocity colorbar
- Optionally repeat with ±1 frame averaging
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

from hydro_analysis.core.analysis import calculate_step_sizes, diffusion_2d_from_1d_fits, fit_gaussian_diffusion_1d
from hydro_analysis.core.io import single_file_data

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Trajectory_Visualisation")
HYDROGEL_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Trajectory_Visualisation\hydrogel")
CUTOUT_FILE = ROOT_PATH / "Cutout.txt"
CUTOUT_FILE_HYDROGEL = HYDROGEL_PATH / "Cutout.txt"
OUTPUT_DIR = ROOT_PATH / "output"
OUTPUT_AVG_DIR = ROOT_PATH / "output_averaged"
OUTPUT_HYDROGEL_DIR = ROOT_PATH / "output_hydrogel"
OUTPUT_HYDROGEL_AVG_DIR = ROOT_PATH / "output_hydrogel_averaged"

DEFAULT_FRAME_INDEX = 277
REQUIRED_CUTOUT_COLUMNS = (
    "particle_nm", "frame",
    "cutout1_p1", "cutout1_p2", "cutout2_p1", "cutout2_p2",
)
CUTOUT_WIDTH_OVER_HEIGHT = 1.4
TRACK_HISTORY_SECONDS = 0.5
SCALEBAR_UM = 5.0
D_MAX = 15.0
SAVE_DPI = 600

COLOR_BLUE = "#0000da"
COLOR_BLUE_DARK = "#000099"

# ============================================================================
# TIFF / XML helpers
# ============================================================================

def load_tiff_stack(tif_path: Path) -> np.ndarray:
    """Load TIFF as (T, Y, X) stack."""
    with tifffile.TiffFile(tif_path) as tif:
        stack = tif.asarray()
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    elif stack.ndim == 4:
        stack = stack[0, ...]
    return stack


def average_frames(stack: np.ndarray, frame_index: int) -> np.ndarray:
    """Average frame[i-1], frame[i], frame[i+1] (clamped to stack bounds)."""
    i_min = max(0, frame_index - 1)
    i_max = min(stack.shape[0], frame_index + 2)  # exclusive
    return np.mean(stack[i_min:i_max].astype(np.float32), axis=0)


def find_xmls_in_tracks(root: Path) -> list[Path]:
    """Find all XMLs inside Tracks/ folders."""
    return sorted(root.rglob("Tracks/*.xml"))


# ============================================================================
# Rendering primitives
# ============================================================================

def render_frame_on_ax(
    ax, frame: np.ndarray, cutout_bounds=None, mpp=None,
    gamma: float = 0.6,
    scalebar_thickness: float = 8.0, scalebar_fontsize: float = 15.0,
) -> None:
    """Draw a microscopy frame on *ax*, apply cutout limits and scalebar."""
    frame_display = frame.astype(np.float32).clip(min=0)
    fmin, fmax = np.min(frame_display), np.max(frame_display)
    ax.imshow(
        frame_display, cmap="gray",
        norm=colors.PowerNorm(gamma=gamma, vmin=fmin, vmax=fmax * 2),
        origin="upper",
    )
    if cutout_bounds is not None:
        x_min, x_max, y_min, y_max = cutout_bounds
        h, w = frame.shape[:2]
        x_min, x_max = _shift_window_to_bounds(x_min, x_max, 0.0, float(w - 1))
        y_min, y_max = _clamp_vertical_lower_only(y_min, y_max, 0.0, float(h - 1))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # origin="upper"
    add_scalebar(ax, frame.shape, length_um=SCALEBAR_UM, mpp=mpp,
                 thickness_px=scalebar_thickness, font_size_pt=scalebar_fontsize)
    ax.set_axis_off()


def _compute_per_particle_D(tracks_df, fps, mpp):
    """Return dict {particle_id: D_um2_per_s} for all particles."""
    if not isinstance(fps, (int, float)) or fps <= 0:
        return {}
    eff_mpp = mpp if isinstance(mpp, (int, float)) and mpp > 0 else 1.0
    steps_df = calculate_step_sizes(tracks_df, step_interval=1, sliding=True)
    d_map = {}
    for pid, step_p in steps_df.groupby("particle"):
        if step_p.empty:
            continue
        fit_x = fit_gaussian_diffusion_1d(step_p["dx"], eff_mpp, fps, frame_interval=1, axis="x")
        fit_y = fit_gaussian_diffusion_1d(step_p["dy"], eff_mpp, fps, frame_interval=1, axis="y")
        fit_2d = diffusion_2d_from_1d_fits(fit_x, fit_y)
        d_map[pid] = fit_2d.get("D_um2_per_s", np.nan)
    return d_map


def render_tracks_on_ax(
    ax, tracks_df, fps, mpp, frame_index: int,
    color_mode: str = "velocity",
    history_seconds: float = TRACK_HISTORY_SECONDS,
    linewidth: float = 3.0,
    alpha_now: float = 0.96,
    alpha_past: float = 0.07,
    d_map: dict = None,
) -> dict:
    """
    Draw trajectories on *ax*.

    color_mode: "blue" → all blue, "velocity" → jet by D.
    Returns the d_map (particle_id → D) used for coloring.
    """
    if tracks_df is None or tracks_df.empty:
        return {}

    history = _history_frames_from_fps(fps, history_seconds)

    if d_map is None:
        d_map = _compute_per_particle_D(tracks_df, fps, mpp)

    cmap = plt.get_cmap("jet")

    for pid, group in tracks_df.groupby("particle"):
        group = group.sort_values("frame")
        x = group["x"].to_numpy()
        y = group["y"].to_numpy()
        frames = group["frame"].to_numpy()
        if len(x) < 2 or frame_index not in set(frames):
            continue

        # Determine color
        if color_mode == "blue":
            seg_color = COLOR_BLUE
        else:
            d_est = d_map.get(pid, 0.0)
            d_clamped = max(0.0, min(float(d_est) if np.isfinite(d_est) else 0.0, D_MAX))
            log_max = np.log10(1.0 + D_MAX)
            log_val = np.log10(1.0 + d_clamped)
            seg_color = cmap(log_val / log_max)

        # Window to history
        min_frame = frame_index - history
        mask = (frames >= min_frame) & (frames <= frame_index)
        x, y, frames = x[mask], y[mask], frames[mask]
        if len(x) < 2:
            continue

        for i in range(len(x) - 1):
            age = frame_index - frames[i + 1]
            if age < 0 or age > history:
                continue
            t = age / float(history) if history > 0 else 0.0
            alpha = alpha_now * (1.0 - t) + alpha_past * t
            ax.plot(
                [x[i], x[i + 1]], [y[i], y[i + 1]],
                color=seg_color, linewidth=linewidth,
                alpha=alpha, solid_capstyle="round",
            )

    return d_map


def render_tracks_on_white(
    ax, tracks_df, fps, mpp, frame_index: int,
    cutout_bounds=None,
    history_seconds: float = TRACK_HISTORY_SECONDS,
    linewidth: float = 3.0,
    d_map: dict = None,
) -> None:
    """Draw blue trajectories on a white background matching the cutout region."""
    ax.set_facecolor("white")
    if cutout_bounds is not None:
        x_min, x_max, y_min, y_max = cutout_bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # match origin="upper"
        ax.set_aspect("equal")

    render_tracks_on_ax(
        ax, tracks_df, fps, mpp, frame_index,
        color_mode="blue", history_seconds=history_seconds,
        linewidth=linewidth, alpha_now=1.0, alpha_past=0.15,
        d_map=d_map,
    )
    highlight_current_particles(ax, tracks_df, frame_index)
    add_scalebar(ax, None, length_um=SCALEBAR_UM, mpp=mpp)
    ax.set_axis_off()


# ============================================================================
# Scalebar + highlight helpers (preserved from original)
# ============================================================================

def add_scalebar(
    ax, image_shape, length_um: float, mpp: Optional[float],
    thickness_px: float = 8.0, font_size_pt: float = 15.0,
) -> None:
    """Draw a scalebar in the lower-right corner of the current view."""
    if not isinstance(mpp, (int, float)) or mpp <= 0:
        return

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)

    width_view = x_max - x_min
    height_view = y_max - y_min
    if not np.isfinite(width_view) or width_view <= 0:
        if image_shape is not None:
            width_view = float(image_shape[1] - 1)

    length_px = length_um / mpp
    if length_px > width_view - 20.0:
        return

    fig = ax.figure
    if fig is not None and fig.canvas is not None:
        fig.canvas.draw()
        axes_height_px = ax.bbox.height
    else:
        axes_height_px = height_view

    width_fraction = thickness_px / axes_height_px if axes_height_px > 0 else 0.01

    try:
        available = {f.name for f in fm.fontManager.ttflist}
    except Exception:
        available = set()
    font_props_dict: dict = {"size": font_size_pt}
    if "Open Sans" in available:
        font_props_dict["family"] = "Open Sans"

    scalebar = ScaleBar(
        mpp, units="um", dimension="si-length",
        fixed_value=length_um, fixed_units="um",
        location="lower right", width_fraction=width_fraction, sep=2,
        frameon=True, color="white", box_color="black", box_alpha=0.6,
        scale_loc="top", label_loc="bottom",
        font_properties=font_props_dict,
    )
    ax.add_artist(scalebar)


def highlight_current_particles(ax, tracks_df, frame_index: int) -> None:
    """Mark particle positions at the current frame with a pink circle."""
    if tracks_df is None or tracks_df.empty:
        return
    current = tracks_df[tracks_df["frame"] == frame_index][["x", "y"]]
    if current.empty:
        return
    ax.scatter(
        current["x"], current["y"],
        s=85, facecolors="none", edgecolors="hotpink",
        linewidths=1.50, zorder=5,
    )


# ============================================================================
# Save functions (one per variant)
# ============================================================================

def _make_fig(ratio: float = CUTOUT_WIDTH_OVER_HEIGHT):
    """Create a figure with the style-guide ratio."""
    w = 7.15
    h = w / ratio
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def save_frame_only(
    frame: np.ndarray, cutout_bounds, mpp, out_path: Path,
) -> None:
    """Variant 1: Frame + Scalebar, no trajectories."""
    fig, ax = _make_fig()
    render_frame_on_ax(ax, frame, cutout_bounds=cutout_bounds, mpp=mpp)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  [saved] {out_path.name}")


def save_frame_with_blue_tracks(
    frame: np.ndarray, cutout_bounds, tracks_df, fps, mpp,
    frame_index: int, out_path: Path,
    d_map: dict = None,
) -> None:
    """Variant 2: Frame + Scalebar + blue trajectories."""
    fig, ax = _make_fig()
    render_frame_on_ax(ax, frame, cutout_bounds=cutout_bounds, mpp=mpp)
    render_tracks_on_ax(
        ax, tracks_df, fps, mpp, frame_index,
        color_mode="blue", d_map=d_map,
    )
    highlight_current_particles(ax, tracks_df, frame_index)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  [saved] {out_path.name}")


def save_frame_tracks_side_by_side(
    frame: np.ndarray, cutout_bounds, tracks_df, fps, mpp,
    frame_index: int, out_path: Path,
    d_map: dict = None,
) -> None:
    """Variant 3: Left = Frame+Scalebar, Right = Trajectories on white."""
    w = 14.3
    h = w / (2 * CUTOUT_WIDTH_OVER_HEIGHT)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(w, h))

    # Left: microscopy frame
    render_frame_on_ax(ax_l, frame, cutout_bounds=cutout_bounds, mpp=mpp)

    # Right: trajectories on white
    render_tracks_on_white(
        ax_r, tracks_df, fps, mpp, frame_index,
        cutout_bounds=cutout_bounds, d_map=d_map,
    )

    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  [saved] {out_path.name}")


def save_overview_velocity(
    panel_data: list[dict], out_path: Path,
) -> None:
    """
    Variant 4: All particle sizes side-by-side, velocity-colored tracks, shared colorbar.

    panel_data: list of dicts with keys:
        frame, cutout_bounds, tracks_df, fps, mpp, frame_index, size_nm, d_map
    """
    n = len(panel_data)
    if n == 0:
        return

    # 3 columns x 2 rows layout
    nrows, ncols = 2, 3
    panel_w = 4.5
    panel_h = panel_w / CUTOUT_WIDTH_OVER_HEIGHT
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols + 0.8, panel_h * nrows + 0.6))
    axes_flat = axes.flatten()

    # Fixed ordering: 20, 50, 100, 200, 500, 1000 nm
    SIZE_ORDER = [20, 50, 100, 200, 500, 1000]
    # Build lookup by size
    by_size = {int(pd_["size_nm"]): pd_ for pd_ in panel_data}

    for idx, size in enumerate(SIZE_ORDER):
        ax = axes_flat[idx]
        pd_ = by_size.get(size)
        if pd_ is None:
            ax.set_axis_off()
            ax.set_title(f"{size} nm", fontsize=10, pad=4)
            continue
        render_frame_on_ax(
            ax, pd_["frame"], cutout_bounds=pd_["cutout_bounds"], mpp=pd_["mpp"],
            scalebar_thickness=4.0, scalebar_fontsize=9.0,
        )
        render_tracks_on_ax(
            ax, pd_["tracks_df"], pd_["fps"], pd_["mpp"], pd_["frame_index"],
            color_mode="velocity", d_map=pd_["d_map"],
        )
        highlight_current_particles(ax, pd_["tracks_df"], pd_["frame_index"])
        ax.set_title(f"{size} nm", fontsize=10, pad=4)

    # Hide any extra axes beyond 6
    for idx in range(len(SIZE_ORDER), len(axes_flat)):
        axes_flat[idx].set_axis_off()

    # Shared colorbar (jet, log scale 0–D_MAX)
    sm = plt.cm.ScalarMappable(
        cmap="jet",
        norm=colors.LogNorm(vmin=1, vmax=1 + D_MAX),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat.tolist(), fraction=0.02, pad=0.02, shrink=0.8)
    cbar.set_label(r"$D$ (µm²/s)", fontsize=9)
    tick_vals = [0.4, 0.8, 2, 4, 8, 13]
    cbar.set_ticks([1 + v for v in tick_vals])
    cbar.set_ticklabels([str(v) for v in tick_vals])
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  [saved] {out_path.name}")


def save_comparison_grid(
    water_panels: list[dict], hydrogel_panels: list[dict], out_path: Path,
) -> None:
    """
    2x2 comparison: Water vs Hydrogel for shared particle sizes.

    Layout (columns = Water | Hydrogel, rows = particle sizes):
        Water 20 nm  | Hydrogel 20 nm
        Water 50 nm  | Hydrogel 50 nm
    """
    water_by_size = {int(p["size_nm"]): p for p in water_panels}
    hydro_by_size = {int(p["size_nm"]): p for p in hydrogel_panels}

    # Find shared sizes, sorted ascending
    shared = sorted(set(water_by_size) & set(hydro_by_size))
    if not shared:
        print("  [!] No shared particle sizes for comparison grid.")
        return

    nrows = len(shared)
    ncols = 2
    panel_w = 5.0
    panel_h = panel_w / CUTOUT_WIDTH_OVER_HEIGHT
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols + 0.8, panel_h * nrows + 0.6))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["Water", "Hydrogel"]
    for col, (label, by_size) in enumerate(zip(col_labels, [water_by_size, hydro_by_size])):
        for row, size in enumerate(shared):
            ax = axes[row, col]
            pd_ = by_size.get(size)
            if pd_ is None:
                ax.set_axis_off()
                continue
            render_frame_on_ax(
                ax, pd_["frame"], cutout_bounds=pd_["cutout_bounds"], mpp=pd_["mpp"],
                scalebar_thickness=4.0, scalebar_fontsize=9.0,
            )
            render_tracks_on_ax(
                ax, pd_["tracks_df"], pd_["fps"], pd_["mpp"], pd_["frame_index"],
                color_mode="velocity", d_map=pd_["d_map"],
            )
            highlight_current_particles(ax, pd_["tracks_df"], pd_["frame_index"])
            # Column label on top row
            if row == 0:
                ax.set_title(label, fontsize=11, pad=4)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap="jet",
        norm=colors.LogNorm(vmin=1, vmax=1 + D_MAX),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, shrink=0.8)
    cbar.set_label(r"$D$ (µm²/s)", fontsize=9)
    tick_vals = [0.4, 0.8, 2, 4, 8, 13]
    cbar.set_ticks([1 + v for v in tick_vals])
    cbar.set_ticklabels([str(v) for v in tick_vals])
    cbar.ax.tick_params(labelsize=8)

    # Row labels (particle size) left of each row
    fig.canvas.draw()
    for row, size in enumerate(shared):
        ax_left = axes[row, 0]
        bbox = ax_left.get_position()
        fig.text(
            bbox.x0 - 0.02, (bbox.y0 + bbox.y1) / 2,
            f"{size} nm", fontsize=10, fontweight="semibold",
            ha="right", va="center", rotation=90,
        )

    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  [saved] {out_path.name}")


# ============================================================================
# Geometry helpers
# ============================================================================

def _shift_window_to_bounds(minv: float, maxv: float, low: float, high: float):
    span = maxv - minv
    if span > (high - low):
        return low, high
    if minv < low:
        maxv += (low - minv)
        minv = low
    if maxv > high:
        minv -= (maxv - high)
        maxv = high
    return max(minv, low), min(maxv, high)


def _history_frames_from_fps(fps: Optional[float], seconds: float) -> int:
    if not isinstance(fps, (int, float)) or fps <= 0:
        return 1
    return max(int(round(float(fps) * float(seconds))), 1)


def _enforce_cutout_ratio(cutout_bounds, width_over_height: float):
    x_min, x_max, y_min, y_max = cutout_bounds
    width = max(x_max - x_min, 1.0)
    target_height = width / float(width_over_height)
    return (x_min, x_max, y_min, y_min + target_height)


def _clamp_vertical_lower_only(y_min: float, y_max: float, low: float, high: float):
    y_min = max(y_min, low)
    y_max = min(y_max, high)
    if y_max <= y_min:
        y_max = min(high, y_min + 1.0)
    return y_min, y_max


# ============================================================================
# Cutout lookup helpers
# ============================================================================

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
        return None
    series = df_cutout["particle_nm"].apply(_coerce_particle_nm)
    diffs = (series - float(particle_size_nm)).abs()
    if diffs.isna().all():
        return None
    idx = diffs.idxmin()
    if not np.isfinite(diffs.loc[idx]) or diffs.loc[idx] > 0.5:
        return None
    return df_cutout.loc[idx]


def _cutout_bounds_from_row(row: pd.Series) -> Optional[tuple[float, float, float, float]]:
    """Parse cutout corners from Cutout.txt.

    Columns are (x1, y1) and (x2, y2) in image-array orientation
    (origin top-left, x = column, y = row), which matches
    matplotlib imshow(origin='upper').
    """
    try:
        x1, y1 = float(row["cutout1_p1"]), float(row["cutout1_p2"])
        x2, y2 = float(row["cutout2_p1"]), float(row["cutout2_p2"])
    except Exception:
        return None
    return (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))


# ============================================================================
# Main
# ============================================================================

def _size_label(nm: float) -> str:
    """e.g. 20.0 → '20', 100.5 → '100.5'"""
    return str(int(nm)) if nm == int(nm) else str(nm)


def _process_source(
    source_root: Path, cutout_file: Path,
    out_dir: Path, out_avg_dir: Path, source_label: str,
) -> tuple[list[dict], list[dict]]:
    """
    Process all XMLs from a data source. Save variants 1-3 per particle size.
    Returns (panels_normal, panels_averaged) for overview/comparison.
    """
    df_cutout = pd.read_csv(cutout_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_avg_dir.mkdir(parents=True, exist_ok=True)

    xml_files = find_xmls_in_tracks(source_root)
    if not xml_files:
        print(f"[!] No XML files found under Tracks/ in: {source_root}")
        return [], []

    panels = []
    panels_avg = []

    for xml_path in xml_files:
        print(f"\n{'='*60}")
        print(f"[{source_label}] XML: {xml_path}")
        result = single_file_data(xml_path)
        if result is None:
            print("[!] single_file_data returned None. Skipping.")
            continue
        if result.get("mpp") is None or result.get("fps") is None:
            print("[!] Missing mpp/fps. Skipping.")
            continue

        particle_nm = result.get("particle_size_nm")
        if particle_nm is None:
            print("[!] Unknown particle size. Skipping.")
            continue
        print(f"  Particle size: {particle_nm} nm")

        tif_path = Path(result["tif_path"]) if result.get("tif_path") else None
        if tif_path is None or not tif_path.exists():
            print("[!] No TIFF found. Skipping.")
            continue
        stack = load_tiff_stack(tif_path)

        cutout_row = _get_cutout_row(df_cutout, particle_nm)
        if cutout_row is None:
            print(f"[!] No cutout for {particle_nm} nm. Skipping.")
            continue

        missing = [c for c in REQUIRED_CUTOUT_COLUMNS if c not in cutout_row.index]
        if missing:
            print(f"[!] Missing cutout columns: {missing}. Skipping.")
            continue

        frame_index = int(float(cutout_row["frame"]))
        cutout_bounds = _cutout_bounds_from_row(cutout_row)
        if cutout_bounds is None:
            print("[!] Invalid cutout bounds. Skipping.")
            continue
        cutout_bounds = _enforce_cutout_ratio(cutout_bounds, CUTOUT_WIDTH_OVER_HEIGHT)

        if frame_index < 0 or frame_index >= stack.shape[0]:
            print(f"[!] Frame {frame_index} out of range. Skipping.")
            continue

        fps = result["fps"]
        mpp = result["mpp"]
        tracks_df = result.get("tracks_df")
        label = _size_label(particle_nm)

        frame = stack[frame_index]
        frame_avg = average_frames(stack, frame_index)
        d_map = _compute_per_particle_D(tracks_df, fps, mpp) if tracks_df is not None else {}

        print(f"  Saving variants for {label} nm ...")

        # --- Variant 1: Frame only ---
        save_frame_only(frame, cutout_bounds, mpp, out_dir / f"{label}_nm.png")
        save_frame_only(frame_avg, cutout_bounds, mpp, out_avg_dir / f"{label}_nm.png")

        # --- Variant 2: Frame + blue tracks ---
        save_frame_with_blue_tracks(
            frame, cutout_bounds, tracks_df, fps, mpp,
            frame_index, out_dir / f"{label}_nm_traj_b.png", d_map=d_map,
        )
        save_frame_with_blue_tracks(
            frame_avg, cutout_bounds, tracks_df, fps, mpp,
            frame_index, out_avg_dir / f"{label}_nm_traj_b.png", d_map=d_map,
        )

        # --- Variant 3: Side-by-side ---
        save_frame_tracks_side_by_side(
            frame, cutout_bounds, tracks_df, fps, mpp,
            frame_index, out_dir / f"{label}_nm_traj_neben.png", d_map=d_map,
        )
        save_frame_tracks_side_by_side(
            frame_avg, cutout_bounds, tracks_df, fps, mpp,
            frame_index, out_avg_dir / f"{label}_nm_traj_neben.png", d_map=d_map,
        )

        # --- Collect for overview ---
        panel_base = dict(
            tracks_df=tracks_df, fps=fps, mpp=mpp,
            frame_index=frame_index, size_nm=particle_nm,
            cutout_bounds=cutout_bounds, d_map=d_map,
        )
        panels.append({**panel_base, "frame": frame})
        panels_avg.append({**panel_base, "frame": frame_avg})

    return panels, panels_avg


def main() -> None:
    # --- Water ---
    print("\n" + "=" * 60)
    print("PROCESSING: Water")
    print("=" * 60)
    water_panels, water_panels_avg = _process_source(
        ROOT_PATH, CUTOUT_FILE, OUTPUT_DIR, OUTPUT_AVG_DIR, "Water",
    )

    # Water overview (3x2 grid)
    if water_panels:
        water_panels.sort(key=lambda p: p["size_nm"])
        water_panels_avg.sort(key=lambda p: p["size_nm"])
        print(f"\nSaving water overview ({len(water_panels)} panels) ...")
        save_overview_velocity(water_panels, OUTPUT_DIR / "overview_velocity.png")
        save_overview_velocity(water_panels_avg, OUTPUT_AVG_DIR / "overview_velocity.png")

    # --- Hydrogel ---
    hydro_panels, hydro_panels_avg = [], []
    if HYDROGEL_PATH.exists():
        print("\n" + "=" * 60)
        print("PROCESSING: Hydrogel")
        print("=" * 60)
        hydro_panels, hydro_panels_avg = _process_source(
            HYDROGEL_PATH, CUTOUT_FILE_HYDROGEL,
            OUTPUT_HYDROGEL_DIR, OUTPUT_HYDROGEL_AVG_DIR, "Hydrogel",
        )

    # --- Comparison grid: Water vs Hydrogel (2x2) ---
    if water_panels and hydro_panels:
        print("\nSaving comparison grids (Water vs Hydrogel) ...")
        save_comparison_grid(water_panels, hydro_panels, OUTPUT_DIR / "comparison_water_hydrogel.png")
        save_comparison_grid(water_panels_avg, hydro_panels_avg, OUTPUT_AVG_DIR / "comparison_water_hydrogel.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
