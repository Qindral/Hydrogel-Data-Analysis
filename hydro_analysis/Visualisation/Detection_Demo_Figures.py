"""
Detection Demo Figures
======================
Illustrative method figures for one example frame each from the 35 nm
(nominal 20 nm) and 50 nm 20mg-hydrogel measurements (Hydrogel Messung /
20mg C16 dataset, injection condition B).

Rendering (frame contrast, scalebar, track coloring/highlighting) reuses
the functions and color scheme from hydro_analysis.Trajectory.trajectory_plotter
so the demo figures match the project's established trajectory-plot style.

For each particle size, produces 4 panels from a single chosen frame with
several detected particles that are mutually well separated:
  1) Frame + scalebar only
  2) Frame + playful simple arrows pointing at each detected particle
  3) Frame + colorful trajectories (jet, colored by diffusion coefficient) - past -> current frame
  4) Zoom on one particle with a circle + crosshair at its exact position
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib import colors

from scipy.optimize import curve_fit

from hydro_analysis.core.io import single_file_data
from hydro_analysis.Trajectory.trajectory_plotter import (
    add_scalebar,
    render_frame_on_ax,
    render_tracks_on_ax,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder"
    r"\Experiments and Results - Data\Auswertungsbilder\Detection_Demo"
)

SCALEBAR_UM = 5.0
SAVE_DPI = 600

ARROW_COLOR = "#66ff66"  # light green
ZOOM_CIRCLE_COLOR = "hotpink"
ZOOM_CROSS_COLOR = "#0000da"  # project blue
ZOOM_ALPHA = 0.8

DATASETS = [
    {
        "label": "35nm",
        "xml": Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\Tracks\20nm_1d_B2_01_Tracks.xml"),
        "frame": 1290,
        "particle_ids": [84, 85, 86, 87, 88],
    },
    {
        "label": "50nm",
        "xml": Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks_new\50 nm B2_1d_02_Tracks.xml"),
        "frame": 311,
        "particle_ids": [84, 91, 92, 93, 94, 95, 96],
    },
]


# ============================================================================
# Helpers
# ============================================================================

def load_tiff_stack(tif_path: Path) -> np.ndarray:
    with tifffile.TiffFile(tif_path) as tif:
        stack = tif.asarray()
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    return stack


def average_frames(stack: np.ndarray, frame_index: int, radius: int = 1) -> np.ndarray:
    """Average frame[i-radius..i+radius] (clamped to stack bounds) to reduce noise."""
    i_min = max(0, frame_index - radius)
    i_max = min(stack.shape[0], frame_index + radius + 1)
    return np.mean(stack[i_min:i_max].astype(np.float32), axis=0)


def _make_fig(w_over_h: float):
    w = 7.15
    h = w / w_over_h
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def _gaussian_2d(coords, amplitude, x0, y0, sigma, offset):
    x, y = coords
    return offset + amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def fit_particle_radius(frame: np.ndarray, x0: float, y0: float,
                         window: int = 6, default_radius: float = 4.0) -> float:
    """Estimate the apparent particle radius (px) by fitting a 2D Gaussian to
    the local intensity spot and returning its half-width-at-half-maximum."""
    h, w = frame.shape[:2]
    xi0, xi1 = max(int(round(x0)) - window, 0), min(int(round(x0)) + window + 1, w)
    yi0, yi1 = max(int(round(y0)) - window, 0), min(int(round(y0)) + window + 1, h)
    crop = frame[yi0:yi1, xi0:xi1].astype(np.float64)
    yy, xx = np.mgrid[yi0:yi1, xi0:xi1]
    try:
        p0 = [crop.max() - crop.min(), x0, y0, 2.0, crop.min()]
        popt, _ = curve_fit(_gaussian_2d, (xx.ravel(), yy.ravel()), crop.ravel(),
                            p0=p0, maxfev=2000)
        sigma = abs(popt[3])
        if not np.isfinite(sigma) or sigma <= 0.5 or sigma > window:
            return default_radius
        return 1.1774 * sigma  # HWHM
    except Exception:
        return default_radius


def pick_best_contrast_particle(frame: np.ndarray, current_df, particle_ids,
                                 inner: int = 3, outer: int = 8) -> int:
    """Return the particle_id with the highest Michelson contrast (peak vs. local
    background), i.e. the particle that stands out most clearly from the noise -
    not necessarily the absolute brightest one."""
    h, w = frame.shape[:2]
    best_pid, best_contrast = particle_ids[0], -np.inf
    for pid in particle_ids:
        row = current_df[current_df["particle"] == pid]
        if row.empty:
            continue
        x0, y0 = float(row["x"].iloc[0]), float(row["y"].iloc[0])
        xi, yi = int(round(x0)), int(round(y0))
        x_lo, x_hi = max(xi - outer, 0), min(xi + outer + 1, w)
        y_lo, y_hi = max(yi - outer, 0), min(yi + outer + 1, h)
        patch = frame[y_lo:y_hi, x_lo:x_hi]
        if patch.size == 0:
            continue
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        dist = np.hypot(xx - xi, yy - yi)
        inner_mask, outer_mask = dist <= inner, (dist > inner) & (dist <= outer)
        if not inner_mask.any() or not outer_mask.any():
            continue
        peak = float(patch[inner_mask].max())
        background = float(np.median(patch[outer_mask]))
        contrast = (peak - background) / (peak + background + 1e-6)
        if contrast > best_contrast:
            best_contrast, best_pid = contrast, pid
    return best_pid


# ============================================================================
# Panel 1: Frame + scalebar
# ============================================================================

def save_frame_scalebar(frame, mpp, w_over_h, out_path: Path) -> None:
    fig, ax = _make_fig(w_over_h)
    render_frame_on_ax(ax, frame, mpp=mpp)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  [saved] {out_path.name}")


# ============================================================================
# Panel 2: Playful simple arrows pointing at each particle
# ============================================================================

def save_frame_arrows(frame, positions, mpp, w_over_h,
                       out_path: Path, seed: int = 7) -> None:
    """positions: (N, 2) array of (x, y) pixel coords."""
    rng = np.random.default_rng(seed)
    h, w = frame.shape[:2]
    fig, ax = _make_fig(w_over_h)
    render_frame_on_ax(ax, frame, mpp=mpp)

    n = len(positions)
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rng.shuffle(base_angles)
    diag = np.hypot(w, h)
    arrow_len = diag * 0.08

    for (x, y), angle in zip(positions, base_angles):
        angle = angle + rng.uniform(-0.35, 0.35)
        tail_x = x + arrow_len * np.cos(angle)
        tail_y = y + arrow_len * np.sin(angle)
        # keep the tail roughly on-canvas so arrows don't vanish off-frame
        tail_x = np.clip(tail_x, 4, w - 4)
        tail_y = np.clip(tail_y, 4, h - 4)
        gap = 5.5  # leave a small gap so the tip doesn't sit on the particle
        vx, vy = x - tail_x, y - tail_y
        vnorm = np.hypot(vx, vy)
        if vnorm > gap:
            head_x = x - gap * vx / vnorm
            head_y = y - gap * vy / vnorm
        else:
            head_x, head_y = x, y
        ax.annotate(
            "", xy=(head_x, head_y), xytext=(tail_x, tail_y),
            arrowprops=dict(
                arrowstyle="-|>", color=ARROW_COLOR,
                lw=1.4, mutation_scale=11,
                shrinkA=0, shrinkB=0,
            ),
            zorder=6,
        )

    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  [saved] {out_path.name}")


# ============================================================================
# Panel 3: Colorful trajectories (jet, by diffusion coefficient), past -> current frame
# ============================================================================

def save_frame_colorful_tracks(frame, tracks_df, particle_ids, fps, mpp, frame_index,
                                w_over_h, out_path: Path) -> None:
    """Draw each particle's trajectory individually (like trajectory_plotter's
    render_tracks_on_ax), with its own past -> present alpha fade spanning the
    particle's *entire* history rather than a fixed window - so the fade is
    always visible regardless of how short or long the track is."""
    fig, ax = _make_fig(w_over_h)
    render_frame_on_ax(ax, frame, mpp=mpp)

    for pid in particle_ids:
        track = tracks_df[(tracks_df["particle"] == pid) & (tracks_df["frame"] <= frame_index)]
        if track.empty:
            continue
        track_span_frames = int(track["frame"].max() - track["frame"].min())
        history_seconds = max(track_span_frames, 1) / fps
        render_tracks_on_ax(
            ax, tracks_df[tracks_df["particle"] == pid], fps, mpp, frame_index,
            color_mode="velocity", history_seconds=history_seconds,
        )

    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  [saved] {out_path.name}")


# ============================================================================
# Panel 4: Zoom on one particle, circle + crosshair
# ============================================================================

def save_particle_zoom(frame, x0, y0, mpp, out_path: Path,
                        half_window_px: float = 22.0) -> None:
    """Zoom uses render_frame_on_ax's contrast/cutout logic but draws its own
    1 um scalebar - render_frame_on_ax hardcodes the project-standard 5 um bar,
    which is too large relative to this ~13 um-wide zoom window."""
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    frame_display = frame.astype(np.float32).clip(min=0)
    fmin, fmax = float(frame_display.min()), float(frame_display.max())
    ax.imshow(frame_display, cmap="gray",
              norm=colors.PowerNorm(gamma=0.5, vmin=fmin, vmax=fmax * 2), origin="upper")

    h, w = frame.shape[:2]
    win = 2 * half_window_px
    x_min = min(max(x0 - half_window_px, 0.0), max(w - win, 0.0))
    y_min = min(max(y0 - half_window_px, 0.0), max(h - win, 0.0))
    ax.set_xlim(x_min, x_min + win)
    ax.set_ylim(y_min + win, y_min)  # origin='upper'
    ax.set_axis_off()
    add_scalebar(ax, frame.shape, length_um=1.0, mpp=mpp, thickness_px=5.0, font_size_pt=11.0)

    circle_radius_px = fit_particle_radius(frame, x0, y0)
    circle = plt.Circle((x0, y0), circle_radius_px, fill=False,
                         edgecolor=ZOOM_CIRCLE_COLOR, linewidth=2.0,
                         alpha=ZOOM_ALPHA, zorder=6)
    ax.add_patch(circle)

    cross_len = half_window_px * 0.35
    ax.plot([x0 - cross_len, x0 + cross_len], [y0, y0], color=ZOOM_CROSS_COLOR,
             linewidth=1.0, alpha=ZOOM_ALPHA, zorder=5)
    ax.plot([x0, x0], [y0 - cross_len, y0 + cross_len], color=ZOOM_CROSS_COLOR,
             linewidth=1.0, alpha=ZOOM_ALPHA, zorder=5)

    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  [saved] {out_path.name}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    for cfg in DATASETS:
        label = cfg["label"]
        print(f"\n{'=' * 60}\n{label}: {cfg['xml'].name}, frame {cfg['frame']}\n{'=' * 60}")

        result = single_file_data(cfg["xml"])
        if result is None:
            print("  [!] single_file_data failed, skipping.")
            continue

        tracks_df = result["tracks_df"]
        mpp = result["mpp"]
        fps = result["fps"]
        tif_path = Path(result["tif_path"])
        frame_index = cfg["frame"]

        stack = load_tiff_stack(tif_path)
        frame = average_frames(stack, frame_index, radius=2)
        h, w = frame.shape[:2]
        w_over_h = w / h

        current = tracks_df[
            (tracks_df["frame"] == frame_index) & (tracks_df["particle"].isin(cfg["particle_ids"]))
        ]
        positions = current[["x", "y"]].to_numpy()
        print(f"  {len(positions)} particles at frame {frame_index}, mpp={mpp}")

        save_frame_scalebar(
            frame, mpp, w_over_h, SAVE_PATH / f"{label}_1_frame_scalebar.png",
        )
        save_frame_arrows(
            frame, positions, mpp, w_over_h,
            SAVE_PATH / f"{label}_2_frame_arrows.png",
        )
        save_frame_colorful_tracks(
            frame, tracks_df, cfg["particle_ids"], fps, mpp, frame_index, w_over_h,
            SAVE_PATH / f"{label}_3_frame_tracks_colorful.png",
        )

        # Zoom on the particle with the best contrast against the background
        zoom_pid = pick_best_contrast_particle(frame, current, cfg["particle_ids"])
        row = current[current["particle"] == zoom_pid]
        if not row.empty:
            x0, y0 = float(row["x"].iloc[0]), float(row["y"].iloc[0])
            save_particle_zoom(
                frame, x0, y0, mpp, SAVE_PATH / f"{label}_4_particle_zoom.png",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
