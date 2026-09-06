"""
Candidate Frame Viewer
======================
Interactive review of candidate example frames for the 35 nm (nominal 20 nm)
and 50 nm detection-demo figures. Shows each candidate as a grid (matplotlib
window, pan/zoom-able) with detected particles marked by a circle - nothing
is saved to disk.

Edit CANDIDATES below to add/remove/adjust entries (xml path + frame index),
then re-run.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib import colors

from hydro_analysis.core.io import single_file_data

# ============================================================================
# CONFIGURATION
# ============================================================================

# (label, xml path, frame index, note)
CANDIDATES_35NM = [
    # Hydrogel, 20 mg C16, day 1 (A = surface loading/ontop, B = injection)
    ("A_B3_inj_d1",   r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\Tracks\20nm_1d_B3_04_Tracks.xml", 896, "20mg hydrogel, B3 injection day1 (10 sep.)"),
    ("B_B3_inj_d1b",  r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\Tracks\20nm_1d_B3_01_Tracks.xml", 39, "20mg hydrogel, B3 injection day1 (7 sep.)"),
    ("C_B2_inj_d1",   r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\Tracks\20nm_1d_B2_01_Tracks.xml", 1290, "20mg hydrogel, B2 injection day1 (5 sep., dist 45.6px)"),
    ("D_A4_ontop_d1", r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\Tracks\Resultof20nm_1d_A4_01_Tracks.xml", 346, "20mg hydrogel, A4 ontop day1 (5 sep.)"),
    ("E_B4_inj_d1",   r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\27_7_test\20nm_1d_B4_04_Tracks.xml", 1208, "20mg hydrogel, B4 injection day1 (5 sep.)"),
]

CANDIDATES_50NM = [
    # Hydrogel, 20 mg C16, various loading conditions (A = surface loading, B = injection)
    ("A_A3ontop_d1",  r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks\A3_ontop_1d_50_20mg_processed_Tracks.xml", 49, "20mg hydrogel, A3 ontop day1 (10 sep.)"),
    ("B_B2_inj_d2",   r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks_test\50 nm B2_2d_01_processed_Tracks.xml", 36, "20mg hydrogel, B2 injection day2 (9 sep.)"),
    ("C_B2_inj_d1",   r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks_new\50 nm B2_1d_02_Tracks.xml", 311, "20mg hydrogel, B2 injection day1 (7 sep.)"),
    ("D_B2_inj_d1b",  r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Demonstration_Processed\50 nm B2_1d_01_Tracks.xml", 1545, "20mg hydrogel, B2 injection day1 (7 sep.)"),
    ("E_A3ontop_d1b", r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks\A3_ontop_1d_2_50_20mg_processed_Tracks.xml", 50, "20mg hydrogel, A3 ontop day1 run2 (6 sep., dist 45px)"),
    ("F_B3_inj_d2",   r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks_old\50 nm B3_2d_01_processed_Tracks.xml", 184, "20mg hydrogel, B3 injection day2 (6 sep.)"),
]

AVERAGE_RADIUS = 2  # average frame_index +/- this many neighboring frames to reduce noise


# ============================================================================
# Helpers
# ============================================================================

def load_tiff_stack(tif_path: Path) -> np.ndarray:
    with tifffile.TiffFile(tif_path) as tif:
        stack = tif.asarray()
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    return stack


def average_frames(stack: np.ndarray, frame_index: int, radius: int = AVERAGE_RADIUS) -> np.ndarray:
    i_min = max(0, frame_index - radius)
    i_max = min(stack.shape[0], frame_index + radius + 1)
    return np.mean(stack[i_min:i_max].astype(np.float32), axis=0)


def render_frame(ax, frame: np.ndarray, gamma: float = 0.6) -> None:
    """Percentile-based contrast so per-file background offsets don't wash the image out."""
    frame_display = frame.astype(np.float32)
    vmin = float(np.percentile(frame_display, 1.0))
    vmax = float(np.percentile(frame_display, 99.9))
    if vmax <= vmin:
        vmin, vmax = float(frame_display.min()), float(frame_display.max())
    ax.imshow(frame_display, cmap="gray",
              norm=colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax), origin="upper")
    ax.set_axis_off()


def show_candidate_grid(candidates: list[tuple], fig_title: str) -> None:
    n = len(candidates)
    ncols = min(n, 3)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (label, xml_str, frame_index, note) in zip(axes_flat, candidates):
        xml_path = Path(xml_str)
        result = single_file_data(xml_path)
        if result is None:
            ax.set_title(f"{label}: single_file_data failed")
            ax.set_axis_off()
            continue

        tracks_df = result["tracks_df"]
        tif_path = Path(result["tif_path"])
        stack = load_tiff_stack(tif_path)
        if frame_index >= stack.shape[0]:
            ax.set_title(f"{label}: frame {frame_index} out of range ({stack.shape[0]})")
            ax.set_axis_off()
            continue

        frame = average_frames(stack, frame_index)
        current = tracks_df[tracks_df["frame"] == frame_index]
        xy = current[["x", "y"]].to_numpy()

        render_frame(ax, frame)
        ax.scatter(xy[:, 0], xy[:, 1], s=90, facecolors="none",
                   edgecolors="#39ff14", linewidths=1.6, zorder=5)
        ax.set_title(f"{label}  |  {tif_path.name}  frame={frame_index}  n={len(xy)}\n[{note}]",
                    fontsize=9)

    for ax in axes_flat[n:]:
        ax.set_axis_off()

    fig.suptitle(fig_title, fontsize=13, fontweight="semibold")
    fig.tight_layout()


def main() -> None:
    show_candidate_grid(CANDIDATES_35NM, "35 nm (nominal 20 nm) - candidate frames")
    show_candidate_grid(CANDIDATES_50NM, "50 nm - candidate frames")
    plt.show()


if __name__ == "__main__":
    main()
